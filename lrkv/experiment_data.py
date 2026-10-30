"""Shared offline token artifact for basis decomposition and rank allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

FORMAT = "lrkv_shared_experiment_tokens"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenizer_sha256(base_model: Path) -> str:
    hashes = {
        name: _sha256(base_model / name)
        for name in TOKENIZER_FILES
        if (base_model / name).is_file()
    }
    if "tokenizer_config.json" not in hashes:
        raise FileNotFoundError(f"base tokenizer is incomplete: {base_model}")
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def calibration_content_sha256(batches: list[dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for batch_index, batch in enumerate(batches):
        digest.update(batch_index.to_bytes(8, "little"))
        for name in sorted(batch):
            tensor = batch[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DATASETS_VERBOSITY"] = "error"


def identity(model):
    model = Path(model)
    return {
        "model": str(model.resolve()),
        "base_config_sha256": _sha256(model / "config.json"),
        "tokenizer_sha256": _tokenizer_sha256(model),
    }


def content_hash(payload):
    """Bind tokens, split boundaries, tokenizer and source in one semantic hash."""
    metadata = {
        k: v
        for k, v in payload.items()
        if k not in ("input_ids", "calibration_input_ids", "content_sha256")
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    )
    for key in ("input_ids", "calibration_input_ids"):
        rows = payload[key].contiguous().cpu()
        digest.update(key.encode())
        digest.update(str(rows.dtype).encode())
        digest.update(json.dumps(list(rows.shape)).encode())
        digest.update(rows.numpy().tobytes())
    return digest.hexdigest()


def load(path, model_identity, vocab_size):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(
            "expected shared experiment data; "
            "regenerate with python -m lrkv.experiment_data"
        )
    for key in ("model", "base_config_sha256", "tokenizer_sha256"):
        if payload.get(key) != model_identity[key]:
            raise ValueError(f"experiment data model/tokenizer mismatch: {key}")
    rows = payload.get("input_ids")
    train, val, length = (
        payload.get(k) for k in ("train_samples", "validation_samples", "sequence_length")
    )
    if (
        any(type(v) is not int for v in (train, val, length))
        or train < 1
        or val < 1
        or length < 3
        or not isinstance(rows, torch.Tensor)
        or rows.dtype != torch.int32
        or rows.shape != (train + val, length)
        or int(rows.min()) < 0
        or int(rows.max()) >= vocab_size
    ):
        raise ValueError("invalid shared experiment token rows or split boundaries")
    calibration_rows = payload.get("calibration_input_ids")
    count, width = payload.get("calibration_samples"), payload.get("calibration_length")
    if (
        type(count) is not int
        or type(width) is not int
        or count < 1
        or width < 3
        or not isinstance(calibration_rows, torch.Tensor)
        or calibration_rows.dtype != torch.int32
        or calibration_rows.shape != (count, width)
        or int(calibration_rows.min()) < 0
        or int(calibration_rows.max()) >= vocab_size
    ):
        raise ValueError("invalid decomposition calibration rows")
    if payload.get("content_sha256") != content_hash(payload):
        raise ValueError("experiment data content hash mismatch")
    return payload


def metadata(payload, path, split="calibration"):
    if split == "calibration":
        start, stop, length = 0, payload["calibration_samples"], payload["calibration_length"]
    elif split == "train":
        start, stop, length = 0, payload["train_samples"], payload["sequence_length"]
    elif split == "validation":
        start, stop, length = (
            payload["train_samples"],
            len(payload["input_ids"]),
            payload["sequence_length"],
        )
    else:
        raise ValueError("invalid data split")
    return {
        **{k: v for k, v in payload.items() if k not in ("input_ids", "calibration_input_ids")},
        "experiment_data_sha256": _sha256(Path(path)),
        "path": str(Path(path).resolve()),
        "used_split": split,
        "row_range": [start, stop],
        "samples": stop - start,
        "used_sequence_length": length,
        "sequence_length": length,
        "rank_sequence_length": payload["sequence_length"],
    }


def calibration(path, model):
    from .arch import read_text_config

    config = read_text_config(model)
    payload = load(path, identity(model), config["vocab_size"])
    rows = payload["calibration_input_ids"]
    batches = [
        {
            "input_ids": row.long()[None],
            "attention_mask": torch.ones_like(row, dtype=torch.long)[None],
        }
        for row in rows
    ]
    meta = metadata(payload, path)
    meta["batch_content_sha256"] = calibration_content_sha256(batches)
    return batches, meta


def require_same_basis(calibration_metadata, payload, path):
    if (
        calibration_metadata.get("experiment_data_sha256") != _sha256(Path(path))
        or calibration_metadata.get("content_sha256") != payload["content_sha256"]
        or calibration_metadata.get("used_split") != "calibration"
        or calibration_metadata.get("row_range") != [0, payload["calibration_samples"]]
        or calibration_metadata.get("used_sequence_length") != payload["calibration_length"]
    ):
        raise ValueError("basis decomposition and rank training must use the same experiment data")


def add_argument(parser):
    parser.add_argument(
        "--experiment-data",
        required=True,
        help="shared token artifact from lrkv.experiment_data",
    )


def organize_documents(
    docs,
    tokenizer,
    *,
    max_train_samples,
    validation_per_category,
    sequence_length,
    calibration_samples,
    calibration_length,
    document_budget=None,
):
    """Carry-over document rows balanced across categories.

    Within each category, documents (each tokenized as prompt + completion + EOS)
    are packed into rows of ``sequence_length`` in order. When a document crosses
    a row boundary, it continues in the next row with its prompt (the first
    ``prompt_chars`` characters of ``text``, tokenized separately from the
    completion as at generation) repeated first, so every row opens at a prompt
    and every finished trace contributes its ending. With ``document_budget``
    (tokens), a longer finished document keeps its first ``document_budget // 2``
    tokens and its prompt followed by its last completion tokens (the other half);
    an unfinished document (``finished`` false) keeps only its first
    ``document_budget`` tokens. The validation rows' last document stops at their
    end, so no part of it reaches a training row.

    The first ``validation_per_category`` rows of each category are held out; the
    remaining rows are interleaved round-robin across categories until the
    smallest category runs out, truncated to a multiple of 4 (at most
    ``max_train_samples``). Calibration rows are the first ``calibration_samples``
    training rows sliced to ``calibration_length``.
    """
    sizes = (
        max_train_samples,
        validation_per_category,
        sequence_length,
        calibration_samples,
        calibration_length,
    )
    if any(type(n) is not int or n < 1 for n in sizes) or min(sequence_length, calibration_length) < 3:
        raise ValueError("positive sample counts and sequence lengths >= 3 are required")
    if calibration_length > sequence_length:
        raise ValueError("calibration rows are training-row prefixes: calibration_length <= sequence_length")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer requires EOS")
    if document_budget is not None and (type(document_budget) is not int or document_budget < 4):
        raise ValueError("document_budget must be an integer >= 4")
    if not docs:
        raise ValueError("document source is empty")
    if any("prompt_chars" not in d for d in docs):
        raise ValueError("carry packing needs prompt_chars in every document")
    by_cat = {}
    for d in docs:
        by_cat.setdefault(d["category"], []).append(d)
    rows, audit = {}, {}
    for cat, cat_docs in by_cat.items():
        rows[cat], audit[cat] = _carry_rows(
            cat_docs, tokenizer, sequence_length, validation_per_category, document_budget
        )
    cats = sorted(rows)
    per_cat = min(len(rows[c]) for c in cats) - validation_per_category
    if per_cat < 1:
        raise ValueError("a category has too few rows for its validation rows and one training row")
    train = [rows[c][validation_per_category + i] for i in range(per_cat) for c in cats]
    train = train[: min(max_train_samples, len(train)) // 4 * 4]
    if calibration_samples > len(train) or not train:
        raise ValueError(f"{len(train)} training rows: too few for {calibration_samples} calibration rows")
    validation = [rows[c][i] for c in cats for i in range(validation_per_category)]
    tensor = lambda r: torch.tensor(r, dtype=torch.int32)
    rank_rows = torch.cat((tensor(train), tensor(validation)))
    calibration_rows = tensor(train[:calibration_samples])[:, :calibration_length].contiguous()
    sampling = {
        "policy": "document_rows_carry_over_prompt_repeated_categories_round_robin_validation_rows_per_category_first",
        "categories": cats,
        "per_category": audit,
        "train_rows_per_category": len(train) / len(cats),
        "validation_rows_per_category": validation_per_category,
        "rank_training_reused_tokens": 0,
        "calibration_reused_tokens": 0,
        "packing": "carry",
        "document_budget": document_budget,
    }
    return rank_rows, calibration_rows, len(train), len(validation), sampling


def _carry_rows(docs, tokenizer, length, validation_rows, budget):
    """One category's carry-over rows (see organize_documents)."""
    eos = tokenizer.eos_token_id
    out, row = [], []
    stats = {
        "documents": len(docs),
        "documents_over_budget": 0,
        "budget_tokens_dropped": 0,
        "documents_long_prompt_skipped": 0,
        "documents_continued": 0,
        "prompt_tokens_repeated": 0,
        "validation_tail_tokens_dropped": 0,
    }
    for d in docs:
        cut = d["prompt_chars"]
        prompt = tokenizer(d["text"][:cut], add_special_tokens=False)["input_ids"]
        completion = tokenizer(d["text"][cut:], add_special_tokens=False)["input_ids"] + [eos]
        if len(prompt) >= length // 2 or (budget and len(prompt) >= budget // 2):
            stats["documents_long_prompt_skipped"] += 1
            continue
        pieces = [completion]
        if budget and len(prompt) + len(completion) > budget:
            stats["documents_over_budget"] += 1
            if d.get("finished", True):
                head, tail = budget // 2 - len(prompt), budget - budget // 2 - len(prompt)
                pieces = [completion[:head], completion[-tail:]]
            else:
                pieces = [completion[: budget - len(prompt)]]
            stats["budget_tokens_dropped"] += len(completion) - sum(map(len, pieces))
        placed, stop = 0, False
        for n, piece in enumerate(pieces):
            k = 0
            while not stop and k < len(piece):
                segment = prompt + piece[k:]
                take = min(length - len(row), len(segment))
                row += segment[:take]
                if placed:
                    stats["prompt_tokens_repeated"] += min(take, len(prompt))
                placed += 1
                k += max(0, take - len(prompt))
                if len(row) == length:
                    out.append(row)
                    row = []
                    if len(out) == validation_rows and (k < len(piece) or n + 1 < len(pieces)):
                        stats["validation_tail_tokens_dropped"] += len(piece) - k + sum(
                            map(len, pieces[n + 1 :])
                        )
                        stop = True
        stats["documents_continued"] += placed > 1
    stats["partial_row_tokens_dropped"] = len(row)
    stats["rows"] = len(out)
    return out, stats


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if (
        min(args.train_samples, args.validation_per_category, args.calibration_samples) < 1
        or min(args.sequence_length, args.calibration_length) < 3
    ):
        raise ValueError("positive split sizes and sequence length >= 3 are required")
    source = Path(args.source)
    digest = _sha256(source)
    if args.source_sha256 and digest != args.source_sha256:
        raise ValueError("document source SHA256 mismatch")
    with open(source, encoding="utf-8") as handle:
        docs = [json.loads(line) for line in handle if line.strip()]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    budget = None if args.document_budget == 0 else args.document_budget
    rows, calibration_rows, train_samples, validation_samples, sampling = organize_documents(
        docs,
        tokenizer,
        max_train_samples=args.train_samples,
        validation_per_category=args.validation_per_category,
        sequence_length=args.sequence_length,
        calibration_samples=args.calibration_samples,
        calibration_length=args.calibration_length,
        document_budget=budget,
    )
    source_meta = {
        "dataset": "docs",
        "source": str(source.resolve()),
        "source_sha256": digest,
        "documents": len(docs),
        "source_split": "train",
    }
    layout = {
        "stream": "document_rows_eos_after_document_prompt_repeated_on_continuation",
        "calibration_split": "training_rows_prefix",
        "validation_split": "first_rows_of_each_category",
    }
    payload = {
        "format": FORMAT,
        **identity(args.model),
        **source_meta,
        "input_ids": rows,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "sequence_length": args.sequence_length,
        **layout,
        "calibration_input_ids": calibration_rows,
        "calibration_samples": args.calibration_samples,
        "calibration_length": args.calibration_length,
        "sampling": sampling,
        "source_code_sha256": {
            n: _sha256(Path(__file__).with_name(n)) for n in ("experiment_data.py", "allocate.py")
        },
    }
    payload["content_sha256"] = content_hash(payload)
    from .allocate import _save

    _save(payload, output)
    print(
        f"wrote docs: calibration {args.calibration_samples}x{args.calibration_length}, "
        f"rank training {train_samples}x{args.sequence_length}, "
        f"validation {validation_samples}x{args.sequence_length}: {output}"
    )
    for cat, stats in sampling.get("per_category", {}).items():
        print(f"  {cat}: {json.dumps(stats)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local Qwen3.8-27B model directory")
    parser.add_argument(
        "--source",
        required=True,
        help="document JSONL with text, category, prompt_chars, finished (scripts/calib.py)",
    )
    parser.add_argument("--source-sha256", help="optional expected SHA256 of --source")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4000,
        help="maximum training rows (balanced across categories, multiple of 4)",
    )
    parser.add_argument(
        "--validation-per-category",
        type=int,
        default=2,
        help="held-out validation rows per category",
    )
    parser.add_argument(
        "--document-budget",
        type=int,
        default=16384,
        help="max tokens per document (first half + prompt and last half; 0 disables cap)",
    )
    parser.add_argument("--sequence-length", "--train-length", type=int, default=8192)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-length", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    configure_offline()
    prepare(args)


if __name__ == "__main__":
    main()
