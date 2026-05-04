import os
import csv
import json
import re
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Set

import pandas as pd
import spacy


import spacy_transformers 
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="grad and param do not obey the gradient layout contract"
)

DATA_DIR = Path("path/to/data/directory")
TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV   = DATA_DIR / "dev.csv"
OUTDIR    = Path("output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# regex
INPUT_TEXT_RE = re.compile(
    r'###\s*Input\s*Text:\s*"(?P<txt>.*?)"\s*$',
    flags=re.DOTALL
)

OUTPUT_BLOCK_RE = re.compile(
    r'###\s*Output\s*Text:\s*(?P<block>\[.*\])',
    flags=re.DOTALL
)

TAG_RE = re.compile(
    r"<(?P<term>.*?)\s*\|\s*(?P<label>[A-Z_]+)>",
    flags=re.DOTALL
)

# loading and parsing CSV
def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [
        c.replace("\ufeff", "").strip().strip('"').strip("'").lower()
        for c in df.columns
    ]
    if not {"unprocessed", "processed"} <= set(df.columns):
        raise ValueError(
            f"{path}: must have 'unprocessed' and 'processed' columns. Got {df.columns}"
        )
    return df

# parsing helpers
def text_from_unprocessed(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    m = INPUT_TEXT_RE.search(s.strip())
    if m:
        return m.group("txt")
    return s.strip()

def split_output_variants(processed: str) -> List[str]:
    """
    Extract the content inside:
        ### Output Text:
        [
           ...
        ]

    Then split variants on semicolons. Each variant should contain exactly one
    annotation like <term | LABEL>.
    """
    if not isinstance(processed, str):
        return []

    m = OUTPUT_BLOCK_RE.search(processed)
    if not m:
        return []

    block = m.group("block").strip()

    # remove outer [ ... ]
    if block.startswith("["):
        block = block[1:]
    if block.endswith("]"):
        block = block[:-1]

    # split on semicolons, trim whitespace, drop empties
    parts = [p.strip() for p in block.split(";")]
    return [p for p in parts if p]

def parse_variant_to_span(
    original_text: str,
    variant: str,
) -> Tuple[int, int, str] | None:
    """
    Convert one variant like:
        You go into the <kitchen | LOCATION>
    into:
        (start_char, end_char, "LOCATION")
    """
    m = TAG_RE.search(variant)
    if not m:
        return None

    term = m.group("term")
    label = m.group("label").strip()

    prefix = variant[:m.start()]
    suffix = variant[m.end():]

    clean_variant = prefix + term + suffix

    # Be tolerant of surrounding whitespace around the whole variant
    clean_variant = clean_variant.strip()

    if clean_variant != original_text:
        # fallback: sometimes whitespace at edges differs
        if clean_variant.strip() != original_text.strip():
            return None

        # If only outer whitespace differs, align with stripped original
        original_text = original_text.strip()
        clean_variant = clean_variant.strip()
        prefix = prefix.lstrip()

    start = len(prefix)
    end = start + len(term)

    if original_text[start:end] != term:
        return None

    return (start, end, label)

def spans_from_processed(
    processed: str,
    original_text: str,
) -> List[Tuple[int, int, str]]:
    """
    Parse all variants from processed text and return unique span tuples.
    """
    spans: List[Tuple[int, int, str]] = []
    seen: Set[Tuple[int, int, str]] = set()

    variants = split_output_variants(processed)
    for variant in variants:
        parsed = parse_variant_to_span(original_text, variant)
        if parsed is None:
            continue
        if parsed not in seen:
            seen.add(parsed)
            spans.append(parsed)

    return spans

# build Examples 
def build_examples(df: pd.DataFrame, spans_key: str = "sc") -> List[Example]:
    nlp_tmp = spacy.blank("en")
    exs: List[Example] = []
    total_anns = 0
    made_spans = 0
    failed_rows = 0

    for _, row in df.iterrows():
        unproc = row["unprocessed"]
        processed = row["processed"]

        text = text_from_unprocessed(unproc)
        if text is None:
            failed_rows += 1
            continue

        anns = spans_from_processed(processed, text)
        total_anns += len(anns)

        pred = nlp_tmp.make_doc(text)
        gold = nlp_tmp.make_doc(text)
        spans = []

        for s, e, lb in anns:
            if 0 <= s < e <= len(text):
                sp = gold.char_span(s, e, label=lb, alignment_mode="expand")
                if sp is not None:
                    spans.append(sp)
                    made_spans += 1

        gold.spans[spans_key] = spans
        exs.append(Example(pred, gold))

    coverage = (made_spans / total_anns) if total_anns else 0.0
    print(
        f"[build_examples] requested_ann_spans={total_anns} "
        f"created_spans={made_spans} coverage={coverage:.3f} "
        f"failed_rows={failed_rows}"
    )
    return exs

# collect labels 
def collect_labels(df_list: List[pd.DataFrame]) -> List[str]:
    labels: Set[str] = set()
    for df in df_list:
        for _, row in df.iterrows():
            text = text_from_unprocessed(row["unprocessed"])
            if text is None:
                continue
            for _, _, lb in spans_from_processed(row["processed"], text):
                labels.add(lb)
    return sorted(labels)

# main
def main():
    # hyperparams
    NUM_EPOCHS = 50
    LR = 3e-5
    BATCH_SIZE = 4
    EFFECTIVE_BATCH = BATCH_SIZE
    DROPOUT = 0.1
    random.seed(1337)

    # CPU by default; comment out to use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    df_train = load_split(TRAIN_CSV)
    df_dev = load_split(DEV_CSV)

    labels = collect_labels([df_train, df_dev])
    (OUTDIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print("Labels:", labels)

    train_examples = build_examples(df_train, spans_key="sc")
    dev_examples = build_examples(df_dev, spans_key="sc")

    if all(len(ex.reference.spans.get("sc", [])) == 0 for ex in dev_examples):
        print("[WARN] No gold spans in DEV after parsing/alignment. Dev F1 will be ~0.0.")

    nlp = spacy.blank("en")
    spancat_cfg = {
        "spans_key": "sc",
        "threshold": 0.35,
        "suggester": {
            "@misc": "spacy.ngram_suggester.v1",
            "sizes": [1, 2, 3, 4, 5, 6],
        },
        "model": {
            "@architectures": "spacy.SpanCategorizer.v1",
            "tok2vec": {
                "@architectures": "spacy-transformers.Tok2VecTransformer.v3",
                "name": "emilyalsentzer/Bio_ClinicalBERT",
                "grad_factor": 1.0,
                "tokenizer_config": {"use_fast": True},
                "pooling": {"@layers": "reduce_mean.v1"},
                "get_spans": {
                    "@span_getters": "spacy-transformers.strided_spans.v1",
                    "window": 128,
                    "stride": 96,
                },
            },
        },
    }

    try:
        nlp.add_pipe("span_categorizer", config=spancat_cfg)
        comp_name = "span_categorizer"
    except Exception:
        nlp.add_pipe("spancat", config=spancat_cfg)
        comp_name = "spancat"

    sc = nlp.get_pipe(comp_name)
    for lb in labels:
        sc.add_label(lb)

    optimizer = nlp.initialize(lambda: train_examples)
    optimizer.learn_rate = LR

    best_f1 = -1.0
    rows = []

    for epoch in range(1, NUM_EPOCHS + 1):
        random.shuffle(train_examples)
        losses = {}

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        total_batches = max(1, (len(train_examples) + EFFECTIVE_BATCH - 1) // EFFECTIVE_BATCH)

        for batch in tqdm(
            minibatch(train_examples, size=EFFECTIVE_BATCH),
            total=total_batches,
            desc=f"Training epoch {epoch}",
        ):
            nlp.update(batch, sgd=optimizer, drop=DROPOUT, losses=losses)

        best_th = None
        best_epoch_f = -1.0
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

        for th in thresholds:
            sc.cfg["threshold"] = th
            s = nlp.evaluate(dev_examples)
            f = (
                s.get("spans_sc_f")
                or s.get("spans_f")
                or s.get("spans_micro_f")
                or s.get("spans_sc_micro_f")
            )
            if f is not None and f > best_epoch_f:
                best_epoch_f = f
                best_th = th

        sc.cfg["threshold"] = best_th
        print(f"  -> Losses: {losses} | Dev spans F1: {best_epoch_f:.4f} @ threshold={best_th}")

        s = nlp.evaluate(dev_examples)
        per_type = s.get("spans_sc_per_type") or s.get("spans_per_type") or {}
        short = {
            k: {"p": round(v["p"], 3), "r": round(v["r"], 3), "f": round(v["f"], 3)}
            for k, v in per_type.items()
        }
        print("  Per-label:", short)

        rows.append(
            {
                "epoch": epoch,
                "losses": json.dumps(losses),
                "dev_spans_f1": best_epoch_f,
                "threshold": best_th,
            }
        )

        ckpt = OUTDIR / f"model-epoch{epoch:02d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(ckpt)

        if best_epoch_f is not None and best_epoch_f > best_f1:
            best_f1 = best_epoch_f
            nlp.to_disk(OUTDIR / "model-best")

    pd.DataFrame(rows).to_csv(OUTDIR / "epoch_metrics.csv", index=False)
    print(f"\nDone. Saved per-epoch models in {OUTDIR}, best in {OUTDIR / 'model-best'}")

if __name__ == "__main__":
    main()