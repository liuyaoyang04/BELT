"""General multi-domain calibration set: prompts and reasoning traces across five categories.

Categories (sources pinned in data/calib_sources/manifest.json, ``scripts/download_calib_sources.sh``):
- math: AI-MO/NuminaMath-1.5 -- non-synthetic problems marked valid, balanced across sources
- science: m-a-p/SuperGPQA (science, engineering, medicine, agronomy, military science) and MMLU-Pro STEM, multiple
  choice formatted with letter options
- code: deepmind/code_contests (Codeforces, AtCoder, CodeChef, ... before 2022), stdin/stdout Python
- instruction: allenai/tulu-3-sft-personas-instruction-following (verifiable format / keyword constraints)
- knowledge_chat: WildChat-1M first user turns (non-toxic, unredacted; English, Chinese and other languages) and
  humanities / social-science multiple choice from MMLU-Pro and SuperGPQA

Usage:
1. Build prompt requests JSONL from verified local sources (``bash scripts/download_calib_sources.sh``):
    python scripts/calib.py requests \
        --out data/calib/requests.jsonl --heldout data/calib/heldout.jsonl

2. Generate Qwen3.8-27B reasoning traces against one or more ``vllm serve`` instances (e.g. launched via
   ``MAX_MODEL_LEN=32768 bash scripts/serve_vllm.sh /path/to/Qwen3.8-27B 8000``):
    python scripts/calib.py generate \
        --requests data/calib/requests.jsonl \
        --servers http://127.0.0.1:8000 \
        --out data/calib/responses.jsonl

3. Merge Qwen3.8-27B generated reasoning traces into ``docs.jsonl`` for
   ``python -m lrkv.experiment_data --model /path/to/Qwen3.8-27B --source data/calib/docs.jsonl``:
    python scripts/calib.py texts \
        --responses data/calib/responses.jsonl \
        --requests data/calib/requests.jsonl \
        --tokenizer /path/to/Qwen3.8-27B \
        --out data/calib/docs.jsonl
"""

import argparse
import asyncio
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SRC = os.path.join(REPO, "data", "calib_sources")
# (pool, held out) prompts per category
COUNTS = {"math": (700, 50), "science": (800, 50), "code": (550, 50), "instruction": (1500, 50),
          "knowledge_chat": (1900, 60)}
SUPERGPQA_SCIENCE = {"Science", "Engineering", "Medicine", "Agronomy", "Military Science"}
MMLU_PRO_STEM = {"math", "physics", "chemistry", "biology", "engineering", "computer science"}
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
CODE_SUFFIX = ("\n\nWrite a Python 3 program that solves this problem. It should read the input from standard input "
               "and write the answer to standard output. Put the complete program in a single ```python code block "
               "at the end of your answer.")
MC_TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
               "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before "
               "answering.\n\n{question}\n\n{options}")
NGRAM = 13
BOILERPLATE_DF = 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources(sources_dir=SRC):
    manifest = json.load(open(os.path.join(sources_dir, "manifest.json")))
    for m in manifest:
        path = os.path.join(sources_dir, m["repo"].replace("/", "__"), m["file"])
        if not os.path.exists(path):
            raise SystemExit(f"missing calibration source file: {path} (run: bash scripts/download_calib_sources.sh)")
        if sha256(path) != m["sha256"]:
            raise SystemExit(f"sha256 mismatch: {path}")
    return manifest


def src(repo, pattern, sources_dir=SRC):
    return sorted(glob.glob(os.path.join(sources_dir, repo.replace("/", "__"), pattern)))


def mc_prompt(question, options):
    letters = "ABCDEFGHIJ"[:len(options)]
    return MC_TEMPLATE.format(letters=letters, question=question,
                              options="\n".join(f"{c}) {o}" for c, o in zip(letters, options)))


def words(text):
    return re.findall(r"\w+", text.lower())


def ngrams(text):
    w = words(text)
    return {" ".join(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def eval_ngrams(eval_jsonl=None):
    if not eval_jsonl:
        return set(), 0
    texts = [json.loads(line)["text"] for line in open(eval_jsonl)]
    grams = set()
    for t in texts:
        grams |= ngrams(t)
    return grams, len(texts)


# --- candidate pools: lists of (source, source_id, prompt, extra), shuffled with a fixed seed -------------------------

def pool_math(rng, sources_dir=SRC):
    import pyarrow.parquet as pq
    by_source = collections.defaultdict(list)
    for path in src("AI-MO/NuminaMath-1.5", "data/*.parquet", sources_dir):
        for r in pq.read_table(path, columns=["problem", "question_type", "problem_is_valid", "source", "synthetic"]).to_pylist():
            p = (r["problem"] or "").strip()
            if r["synthetic"] or r["problem_is_valid"] != "Yes" or not 50 <= len(p) <= 4000:
                continue
            prompt = p if r["question_type"] == "proof" else p + MATH_SUFFIX
            by_source[r["source"]].append(("numinamath", r["source"], prompt, {"question_type": r["question_type"]}))
    for v in by_source.values():
        rng.shuffle(v)
    out, lists = [], [by_source[s] for s in sorted(by_source)]
    for i in range(max(map(len, lists))):
        out += [lst[i] for lst in lists if i < len(lst)]
    return [(s, f"{sub}-{i}", p, e) for i, (s, sub, p, e) in enumerate(out)]


def supergpqa(disciplines, rng, sources_dir=SRC):
    rows = [json.loads(line) for line in open(src("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl", sources_dir)[0])]
    out = [("supergpqa", r["uuid"], mc_prompt(r["question"], r["options"]),
            {"discipline": r["discipline"], "field": r["field"], "difficulty": r["difficulty"]})
           for r in rows if r["discipline"] in disciplines]
    rng.shuffle(out)
    return out


def mmlu_pro(categories, rng, sources_dir=SRC):
    import pyarrow.parquet as pq
    parquet_path = src("TIGER-Lab/MMLU-Pro", "data/*.parquet", sources_dir)[0]
    out = [("mmlu_pro", str(r["question_id"]), mc_prompt(r["question"], r["options"]), {"mmlu_pro_category": r["category"]})
           for r in pq.read_table(parquet_path).to_pylist() if r["category"] in categories]
    rng.shuffle(out)
    return out


def interleave(parts, shares):
    """Merge shuffled lists so every prefix holds them in the given shares."""
    out, taken = [], [0] * len(parts)
    while any(taken[i] < len(p) for i, p in enumerate(parts)):
        i = min((j for j in range(len(parts)) if taken[j] < len(parts[j])), key=lambda j: taken[j] / shares[j])
        out.append(parts[i][taken[i]])
        taken[i] += 1
    return out


def pool_science(rng, sources_dir=SRC):
    return interleave([supergpqa(SUPERGPQA_SCIENCE, rng, sources_dir), mmlu_pro(MMLU_PRO_STEM, rng, sources_dir)], [0.7, 0.3])


def pool_code(rng, sources_dir=SRC):
    import pyarrow.parquet as pq
    seen, out = set(), []
    for path in src("deepmind/code_contests", "data/train-*.parquet", sources_dir):
        for r in pq.read_table(path, columns=["name", "description", "source", "difficulty", "cf_rating"]).to_pylist():
            d = (r["description"] or "").strip()
            key = " ".join(words(d))[:500]
            if not 200 <= len(d) <= 12000 or key in seen:
                continue
            seen.add(key)
            out.append(("code_contests", r["name"], d + CODE_SUFFIX,
                        {"cc_source": r["source"], "difficulty": r["difficulty"], "cf_rating": r["cf_rating"]}))
    rng.shuffle(out)
    return out


def pool_instruction(rng, sources_dir=SRC):
    import pyarrow.parquet as pq
    out = [("tulu3_personas_if", r["id"], r["prompt"].strip(), {"constraints": r["constraints"]})
           for r in pq.read_table(src("allenai/tulu-3-sft-personas-instruction-following", "data/*.parquet", sources_dir)[0],
                                  columns=["id", "prompt", "constraints"]).to_pylist()
           if r["prompt"] and 20 <= len(r["prompt"].strip()) <= 6000]
    rng.shuffle(out)
    return out


def wildchat(rng, sources_dir=SRC):
    import pyarrow.parquet as pq
    seen, groups = set(), collections.defaultdict(list)
    for path in src("allenai/WildChat-1M", "data/*.parquet", sources_dir):
        for r in pq.read_table(path, columns=["conversation_hash", "conversation", "language", "toxic", "redacted"]).to_pylist():
            if r["toxic"] or r["redacted"] or not r["conversation"]:
                continue
            first = r["conversation"][0]
            text = (first.get("content") or "").strip()
            key = " ".join(text.lower().split())
            if first.get("role") != "user" or not 20 <= len(text) <= 3000 or key in seen:
                continue
            seen.add(key)
            lang = r["language"] if r["language"] in ("English", "Chinese") else "other"
            groups[lang].append(("wildchat", r["conversation_hash"], text, {"language": r["language"]}))
    for v in groups.values():
        rng.shuffle(v)
    return interleave([groups["English"], groups["Chinese"], groups["other"]], [0.55, 0.30, 0.15])


def pool_knowledge_chat(rng, sources_dir=SRC):
    mc = interleave([mmlu_pro({"law", "history", "psychology", "business", "economics", "health", "philosophy", "other"}, rng, sources_dir),
                     supergpqa({"Literature and Arts", "Economics", "History", "Law", "Management", "Education",
                                "Philosophy", "Sociology"}, rng, sources_dir)], [0.5, 0.5])
    return interleave([wildchat(rng, sources_dir), mc], [2 / 3, 1 / 3])


POOLS = {"math": pool_math, "science": pool_science, "code": pool_code, "instruction": pool_instruction,
         "knowledge_chat": pool_knowledge_chat}


def _select_prompts(sources_dir=SRC, eval_jsonl=None):
    manifest = verify_sources(sources_dir)
    grams, n_eval = eval_ngrams(eval_jsonl)
    if n_eval:
        print(f"evaluation 13-grams: {len(grams)} from {n_eval} texts", flush=True)
    stats, chosen = collections.Counter(), {}
    for cat, (n_pool, n_held) in COUNTS.items():
        rng = random.Random(f"calib/{cat}")
        cands = POOLS[cat](rng, sources_dir)[:3 * (n_pool + n_held)]
        shared = [ngrams(c[2]) & grams for c in cands] if grams else [set() for _ in cands]
        df = collections.Counter(g for s in shared for g in s)
        keep, dropped = [], []
        for c, s in zip(cands, shared):
            (dropped if any(df[g] <= BOILERPLATE_DF for g in s) else keep).append(c)
        stats[f"{cat}:candidates"] = len(cands)
        stats[f"{cat}:contaminated"] = len(dropped)
        if len(keep) < n_pool + n_held:
            raise SystemExit(f"{cat}: only {len(keep)} clean prompts")
        chosen[cat] = keep[:n_pool + n_held]
    return manifest, stats, chosen


def requests(args):
    manifest, stats, chosen = _select_prompts(args.sources, args.eval_texts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.heldout)), exist_ok=True)
    with open(args.out, "w") as pool_f, open(args.heldout, "w") as held_f:
        for cat, items in chosen.items():
            n_pool = COUNTS[cat][0]
            for k, (source, sid, prompt, extra) in enumerate(items):
                held = k >= n_pool
                bench = f"calib_heldout_{cat}" if held else f"calib_{cat}"
                qid = re.sub(r"[^\w.-]", "_", f"{source}-{sid}")
                rec = {"uid": f"{bench}/{qid}/0", "bench": bench, "qid": qid, "sample": 0, **extra, "category": cat,
                       "source": source, "messages": [{"role": "user", "content": prompt}]}
                (held_f if held else pool_f).write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[f"{cat}:{'heldout' if held else 'pool'}:{source}"] += 1
    uids = [json.loads(line)["uid"] for path in (args.out, args.heldout) for line in open(path)]
    assert len(uids) == len(set(uids)), "duplicate uids"
    meta = {"sources": manifest, "counts": COUNTS, "ngram": NGRAM, "boilerplate_df": BOILERPLATE_DF,
            "stats": dict(stats), "requests_sha256": sha256(args.out), "heldout_sha256": sha256(args.heldout)}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(dict(stats), indent=1, ensure_ascii=False))


def texts(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    reqs = {json.loads(line)["uid"]: json.loads(line) for line in open(args.requests)}
    latest = {}
    for line in open(args.responses):
        r = json.loads(line)
        if r["uid"] in reqs:
            latest[r["uid"]] = r
    missing = set(reqs) - set(latest)
    if missing:
        raise SystemExit(f"{len(missing)} requests have no completion, e.g. {sorted(missing)[:3]}")
    docs, stats = [], collections.Counter()
    for uid in sorted(latest):
        r, q = latest[uid], reqs[uid]
        finished = r["finish_reason"] == "stop" and "</think>" in r["completion"]
        prompt = tok.apply_chat_template(q["messages"], tokenize=False, add_generation_prompt=True,
                                         reasoning_effort=r.get("reasoning_effort"))
        docs.append({"uid": uid, "category": q["category"], "source": q["source"], "finished": finished,
                     "text": prompt + r["completion"], "prompt_chars": len(prompt),
                     "tokens": r["prompt_tokens"] + r["completion_tokens"]})
        stats[f"{q['category']}:docs"] += 1
        stats[f"{q['category']}:finished"] += finished
        stats[f"{q['category']}:tokens"] += docs[-1]["tokens"]
    random.Random(0).shuffle(docs)
    with open(args.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=1), f"\n{len(docs)} documents -> {args.out} (sha256 {sha256(args.out)})")


def repeat(args):
    """Every request ``--samples`` times (uid .../<sample>, so each sample has its own seed)."""
    with open(args.out, "w") as f:
        for line in open(args.requests):
            q = json.loads(line)
            for k in range(args.samples):
                f.write(json.dumps({**q, "uid": f"{q['bench']}/{q['qid']}/{k}", "sample": k}, ensure_ascii=False) + "\n")
    print(f"{args.samples} samples of each request in {args.requests} -> {args.out}")


# Qwen3.8 model-card thinking-mode sampling (= generation_config.json + explicit zero/unit penalties).
MODEL_CARD_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                       "presence_penalty": 0.0, "repetition_penalty": 1.0}


def _seed_of(uid):
    return zlib.crc32(uid.encode()) & 0x7FFFFFFF


def _done_uids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("finish_reason") in ("stop", "length"):
                    done.add(rec["uid"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def _generate_async(args):
    import httpx
    reqs = [json.loads(line) for line in open(args.requests) if line.strip()]
    done = _done_uids(args.out)
    todo = [r for r in reqs if r["uid"] not in done]
    print(f"{len(reqs)} requests, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    servers = [s.rstrip("/") for s in args.servers.split(",") if s.strip()]
    n_workers = args.concurrency * len(servers)
    limits = httpx.Limits(max_connections=n_workers + 8, max_keepalive_connections=n_workers + 8)
    queue: asyncio.Queue = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)
    attempts = collections.Counter()
    stats = collections.Counter()
    t0 = time.time()

    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        served_models = {}
        for base in servers:
            if args.model:
                served_models[base] = args.model
            else:
                resp = await client.get(f"{base}/v1/models", timeout=30.0)
                resp.raise_for_status()
                served_models[base] = resp.json()["data"][0]["id"]

        with open(args.out, "a") as out_f, open(args.out + ".errors.jsonl", "a") as err_f:
            async def worker(base):
                while True:
                    req = await queue.get()
                    body = {"model": served_models[base], "messages": req["messages"],
                            "seed": _seed_of(req["uid"]), "n": 1, **MODEL_CARD_SAMPLING}
                    if args.reasoning_effort is not None:
                        body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
                    if args.max_tokens is not None and args.max_tokens > 0:
                        body["max_tokens"] = args.max_tokens
                    started = time.time()
                    try:
                        r = await client.post(f"{base}/v1/chat/completions", json=body)
                        if r.status_code >= 500:
                            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                        r.raise_for_status()
                        data = r.json()
                        choice = data["choices"][0]
                        if choice["finish_reason"] not in ("stop", "length"):
                            raise RuntimeError(f"unexpected finish_reason {choice['finish_reason']!r}")
                        msg = choice["message"]
                        content = msg.get("content") or ""
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                        if reasoning and "</think>" not in content:
                            completion = f"<think>\n{reasoning}</think>\n\n{content}"
                        else:
                            completion = content
                        rec = {k: v for k, v in req.items() if k != "messages"}
                        rec.update({"completion": completion, "finish_reason": choice["finish_reason"],
                                    "completion_tokens": data["usage"]["completion_tokens"],
                                    "prompt_tokens": data["usage"]["prompt_tokens"], "seed": body["seed"],
                                    "latency_s": round(time.time() - started, 1), "server": base,
                                    "max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort,
                                    "sampling": MODEL_CARD_SAMPLING})
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        stats["done"] += 1
                        stats["tokens"] += rec["completion_tokens"]
                    except (RuntimeError, httpx.TransportError) as e:
                        attempts[req["uid"]] += 1
                        if attempts[req["uid"]] < args.max_attempts:
                            await asyncio.sleep(2.0)
                            queue.put_nowait(req)
                            stats["requeued"] += 1
                        else:
                            err_f.write(json.dumps({"uid": req["uid"], "error": repr(e)[:2000],
                                                    "attempts": attempts[req["uid"]], "t": time.time()}) + "\n")
                            err_f.flush()
                            stats["errors"] += 1
                    except Exception as e:
                        err_f.write(json.dumps({"uid": req["uid"], "error": repr(e)[:2000],
                                                "attempts": attempts[req["uid"]] + 1, "t": time.time()}) + "\n")
                        err_f.flush()
                        stats["errors"] += 1
                    finally:
                        queue.task_done()

            async def reporter():
                while True:
                    await asyncio.sleep(30.0)
                    el = max(time.time() - t0, 1e-6)
                    print(f"[{time.strftime('%H:%M:%S')}] done {stats['done']}/{len(todo)} "
                          f"errors={stats['errors']} requeued={stats['requeued']} "
                          f"tokens={stats['tokens']} ({stats['tokens'] / el:.0f} tok/s)", flush=True)

            tasks = [asyncio.create_task(worker(servers[i % len(servers)])) for i in range(n_workers)]
            rep = asyncio.create_task(reporter())
            await queue.join()
            rep.cancel()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, rep, return_exceptions=True)

    el = max(time.time() - t0, 1e-6)
    print(f"finished: {stats['done']} done, {stats['errors']} errors, {stats['tokens']} tokens in {el / 60:.1f} min")
    if stats["errors"]:
        raise SystemExit(f"{stats['errors']} requests failed (see {args.out}.errors.jsonl)")


def generate(args):
    asyncio.run(_generate_async(args))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("requests")
    q.add_argument("--sources", default=SRC)
    q.add_argument("--eval-texts", default=None)
    q.add_argument("--out", required=True)
    q.add_argument("--heldout", required=True)
    q = sub.add_parser("generate")
    q.add_argument("--requests", required=True, help="JSONL produced by `calib.py requests`")
    q.add_argument("--servers", default="http://127.0.0.1:8000", help="Comma-separated vLLM server base URLs")
    q.add_argument("--model", default=None, help="Served model name (default: auto-detect from /v1/models)")
    q.add_argument("--concurrency", type=int, default=128, help="Concurrent requests in flight per server")
    q.add_argument("--max-tokens", type=int, default=None,
                   help="Optional explicit output token cap (default: server context limit)")
    q.add_argument("--reasoning-effort", default=None, choices=("xhigh", "medium", "low"))
    q.add_argument("--max-attempts", type=int, default=3)
    q.add_argument("--out", required=True, help="Output responses JSONL (resumes if file exists)")
    q = sub.add_parser("texts")
    q.add_argument("--responses", required=True)
    q.add_argument("--requests", required=True)
    q.add_argument("--tokenizer", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("repeat")
    q.add_argument("--requests", required=True)
    q.add_argument("--samples", type=int, required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args()
    {"requests": requests, "generate": generate, "texts": texts, "repeat": repeat}[args.cmd](args)


if __name__ == "__main__":
    main()
