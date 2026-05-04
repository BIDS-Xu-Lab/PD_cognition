import os
import re
import csv
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import spacy
import spacy_transformers 
from tqdm import tqdm

TEST_CSV    = Path("/path/to/test.csv")
MODEL_DIR   = Path("/path/to/model-best") 
OUTDIR      = Path("/path/to/output/directory")
RAW_TXT_DIR = OUTDIR / "raw_outputs"

SPANS_KEY   = "sc"
BATCH_SIZE  = 50
USE_CPU     = True   # set False if you want GPU

COMBINED_CSV = OUTDIR / "all_outputs.csv"

warnings.filterwarnings(
    "ignore",
    message="Can't initialize NVML"
)

# Regex/parsing and match training-time format
INPUT_TEXT_RE = re.compile(
    r'###\s*Input\s*Text:\s*"(?P<txt>.*?)"\s*$',
    flags=re.DOTALL
)

def text_from_unprocessed(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = INPUT_TEXT_RE.search(s.strip())
    if m:
        return m.group("txt")
    return s.strip()

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

    if "unprocessed" not in df.columns:
        raise ValueError(
            "{}: must have an 'unprocessed' column. Got {}".format(path, list(df.columns))
        )

    if "processed" not in df.columns:
        df["processed"] = ""

    return df

# Prediction formatting
def spans_to_annotations(spans) -> List[Dict[str, Any]]:
    ann: List[Dict[str, Any]] = []
    seen = set()

    for sp in spans:
        key = (sp.start_char, sp.end_char, sp.text, sp.label_)
        if key in seen:
            continue
        seen.add(key)
        ann.append({
            "start": sp.start_char,
            "end": sp.end_char,
            "term": sp.text,
            "label": sp.label_,
        })

    ann.sort(key=lambda x: (x["start"], x["end"], x["label"], x["term"]))
    return ann

def annotation_to_variant(text: str, ann: Dict[str, Any]) -> str:
    start = ann["start"]
    end = ann["end"]
    term = ann["term"]
    label = ann["label"]

    prefix = text[:start]
    suffix = text[end:]
    return "{}<{} | {}>{}".format(prefix, term, label, suffix)

def annotations_to_bracket_list(text: str, annotations: List[Dict[str, Any]]) -> str:
    if not annotations:
        return "[]"

    variants = [annotation_to_variant(text, ann) for ann in annotations]
    joined = ";\n\n".join(variants)
    return "[\n" + joined + "\n]"

def build_output_text(text: str, annotations: List[Dict[str, Any]]) -> str:
    bracket_list = annotations_to_bracket_list(text, annotations)
    return '### Input Text: "{}"\n\n### Output Text:\n{}\n'.format(text, bracket_list)

def save_raw_output(idx: int, content: str) -> None:
    out_path = RAW_TXT_DIR / "output_{}.txt".format(idx)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

# Main
def main() -> None:
    if USE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    OUTDIR.mkdir(parents=True, exist_ok=True)
    RAW_TXT_DIR.mkdir(parents=True, exist_ok=True)

    print("[info] spaCy version: {}".format(spacy.__version__))
    print("[info] loading model from: {}".format(MODEL_DIR))

    try:
        nlp = spacy.load(MODEL_DIR)
    except Exception as e:
        print("\n[ERROR] Failed to load model.")
        print("[ERROR] {}".format(str(e)))
        print("\nMost likely causes:")
        print("1. spacy-transformers is not installed in this environment")
        print("2. spaCy/spacy-transformers versions do not match the training environment")
        print("3. The saved model directory is incomplete")
        raise

    print("[info] loading test csv: {}".format(TEST_CSV))
    df = load_split(TEST_CSV)
    print("[info] total rows in test set: {}".format(len(df)))

    texts: List[str] = []
    kept_indices: List[int] = []

    for idx, row in df.iterrows():
        txt = text_from_unprocessed(row["unprocessed"])
        if txt is None:
            txt = ""
        texts.append(txt)
        kept_indices.append(idx)

    print("[info] running inference with best model on {} rows...".format(len(texts)))

    all_rows: List[Dict[str, Any]] = []

    docs = nlp.pipe(texts, batch_size=BATCH_SIZE)
    for row_idx, doc in tqdm(zip(kept_indices, docs), total=len(texts)):
        spans = doc.spans.get(SPANS_KEY, []) if hasattr(doc, "spans") else []
        annotations = spans_to_annotations(spans)

        formatted_output = build_output_text(doc.text, annotations)
        save_raw_output(row_idx, formatted_output)

        all_rows.append({
            "idx": row_idx,
            "text": doc.text,
            "generated_output": formatted_output,
        })

    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(COMBINED_CSV, index=False)

    print("\nInference complete.")
    print("Raw per-example outputs saved in: {}".format(RAW_TXT_DIR))
    print("Combined CSV saved to: {}".format(COMBINED_CSV))

if __name__ == "__main__":
    main()