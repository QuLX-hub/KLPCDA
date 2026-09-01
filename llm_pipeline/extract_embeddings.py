"""Extract and cache frozen sentence-embedding vectors for a text
classification dataset.

Usage (requires the `embeddings` extra: `pip install -e ".[embeddings]"`):

    python -m llm_pipeline.extract_embeddings \
        --dataset mteb/banking77 --model sentence-transformers/all-MiniLM-L6-v2 \
        --out cache/banking77_minilm.npz

This is a one-time, GPU-optional step (small models run fine on CPU). The
cached .npz feeds any later step that needs embeddings (e.g. a k-shot
`KLPCDA.fit`/`transform` run) without re-downloading the dataset or
re-running the encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mteb/banking77")
    parser.add_argument("--config", default=None, help="dataset config/subset name, e.g. 'plus' for clinc/clinc_oos")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--exclude-label-name",
        default=None,
        help="drop rows whose label name equals this (e.g. 'oos' for clinc/clinc_oos, "
        "for a closed-set classification setup with no reject/unknown class)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="load the encoder in this precision (e.g. float16 to roughly halve RAM for "
        "billion-parameter encoders on memory-constrained CPU-only machines); "
        "default leaves the model at its stored precision",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="required by some larger instruct-embedding models (e.g. gte-Qwen2 family)",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="e5-mistral-style instruction, prepended to every text (train and test alike -- "
        "this is symmetric classification embedding, not asymmetric retrieval, so there is no "
        "separate query/passage side to leave unprefixed) as 'Instruct: {instruction}\\nQuery: "
        "{text}'. Instruction-tuned decoder embedding models (e.g. e5-mistral) are documented to "
        "need this on every input to reach their benchmarked quality, even for non-retrieval "
        "tasks -- leave unset for models that don't expect this convention (MiniLM, BGE-large).",
    )
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "missing optional dependencies; install with `pip install -e \".[embeddings]\"`"
        ) from exc

    from .data import load_classification_dataset

    train_split, test_split, label_names = load_classification_dataset(
        args.dataset, args.config, args.text_field, args.label_field, args.exclude_label_name
    )

    print(f"loading encoder {args.model!r} (dtype={args.dtype!r}) ...")
    model_kwargs = {"torch_dtype": args.dtype} if args.dtype else {}
    model = SentenceTransformer(
        args.model, trust_remote_code=args.trust_remote_code, model_kwargs=model_kwargs
    )

    def encode(split):
        texts = split[args.text_field]
        if args.instruction:
            texts = [f"Instruct: {args.instruction}\nQuery: {t}" for t in texts]
        labels = np.asarray(split[args.label_field])
        embeddings = model.encode(
            texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings, labels

    print(f"encoding {len(train_split)} train examples ...")
    X_train, y_train = encode(train_split)
    print(f"encoding {len(test_split)} test examples ...")
    X_test, y_test = encode(test_split)

    # Cache embeddings at float32 regardless of --dtype: --dtype only
    # controls the encoder's *compute* precision (e.g. float16 to fit a
    # multi-billion-parameter model's activations in GPU memory); the
    # downstream KLPCDA estimator calls numpy.linalg.eig, which does not
    # support float16 storage.
    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=np.array(label_names if label_names is not None else [], dtype=object),
    )
    print(f"saved {out_path} ({X_train.shape=}, {X_test.shape=})")


if __name__ == "__main__":
    main()
