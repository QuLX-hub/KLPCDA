"""Shared dataset-loading logic for few-shot text classification.

Factored out of `extract_embeddings.py` so a text-based baseline (which
needs the raw text, not pre-extracted embeddings) can load exactly the same
train/test splits without duplicating the HuggingFace-`datasets` and
label-exclusion logic.
"""

from __future__ import annotations


def load_classification_dataset(
    dataset: str,
    config: str | None,
    text_field: str,
    label_field: str,
    exclude_label_name: str | None,
):
    """Load a HF `datasets` classification dataset's train/test splits,
    optionally dropping rows with a given human-readable label name (e.g.
    an out-of-scope/reject class not wanted in a closed-set classification
    setup).

    Returns `(train_split, test_split, label_names)`; `label_names` is
    `None` if the label field isn't a HuggingFace `ClassLabel`.
    """
    from datasets import load_dataset

    print(f"loading dataset {dataset!r} (config={config!r}) ...")
    ds = load_dataset(dataset, config) if config else load_dataset(dataset)
    train_split = ds["train"]
    test_split = ds["test"] if "test" in ds else ds["validation"]

    label_names = (
        train_split.features[label_field].names
        if hasattr(train_split.features[label_field], "names")
        else None
    )

    if exclude_label_name is not None:
        if label_names is None:
            raise SystemExit("--exclude-label-name given but label field has no human-readable names")
        exclude_id = label_names.index(exclude_label_name)
        print(f"excluding label {exclude_label_name!r} (id={exclude_id})")
        train_split = train_split.filter(lambda ex: ex[label_field] != exclude_id)
        test_split = test_split.filter(lambda ex: ex[label_field] != exclude_id)

    return train_split, test_split, label_names
