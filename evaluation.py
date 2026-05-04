import os
import re
import pandas as pd

GROUND_TRUTH_CSV = "path/to/test.csv"
PREDICTIONS_CSV  = "path/to/model/specific/all_outputs.csv"
EVAL_OUTPUT_DIR  = "path/for/eval/outputs"

os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

VALID_LABELS = {
    "ACTION": "action",
    "SENSORY": "sensory",
    "EMOTION": "emotion",
    "TIME": "time",
    "LOCATION": "location",
    "SOCIAL_INTERACTION": "social_interaction",
    "THOUGHT": "thought",
}

CATEGORY_ORDER = [
    "action",
    "sensory",
    "emotion",
    "time",
    "location",
    "social_interaction",
    "thought",
]

def strip_outer_quotes(s):
    if s is None:
        return ""

    s = str(s).strip()

    changed = True
    while changed and len(s) >= 2:
        changed = False
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
            changed = True

    return s

def normalize_whitespace(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def normalize_text_key(s):
    """
    Normalize enough to align GT and prediction rows reliably,
    but do not aggressively alter content.
    """
    s = strip_outer_quotes(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s).strip()
    return s

def normalize_term(s):
    """
    Normalize entity term for matching.
    """
    s = strip_outer_quotes(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = normalize_whitespace(s)
    return s

def extract_input_text_from_unprocessed(unprocessed_text):
    """
    Extract actual input text from:
      ### Input Text: "...."
    or return stripped text if wrapper is absent.
    """
    s = str(unprocessed_text)

    if "### Input Text:" in s:
        s = s.split("### Input Text:", 1)[1].strip()

    if "### Output Text:" in s:
        s = s.split("### Output Text:", 1)[0].strip()

    s = strip_outer_quotes(s)
    return normalize_text_key(s)

def extract_bracket_block(text):
    """
    Keep only the bracketed annotation block if present.
    Handles cases where generated output contains:
      ### Input Text: ...
      ### Output Text:
      [ ... ]
    """
    if text is None:
        return ""

    s = str(text).strip()

    if "### Output Text:" in s:
        s = s.split("### Output Text:")[-1].strip()

    start = s.find("[")
    end = s.rfind("]")

    if start != -1 and end != -1 and end > start:
        return s[start:end + 1].strip()

    return s.strip()

def canonicalize_label(label):
    if label is None:
        return None
    label = normalize_whitespace(label).upper()
    return VALID_LABELS.get(label)

def parse_annotations_from_bracket_text(text):
    """
    Parse annotations of the form:
      <entity text | LABEL>

    Returns a list of (term, canonical_label).
    """
    block = extract_bracket_block(text)
    matches = re.findall(r"<\s*(.*?)\s*\|\s*([A-Za-z_]+)\s*>", block, flags=re.DOTALL)

    annotations = []
    for term, label in matches:
        canon_label = canonicalize_label(label)
        if canon_label is None:
            continue
        term = normalize_term(term)
        if term:
            annotations.append((term, canon_label))

    return annotations

# Ground truth / prediction loaders
def load_ground_truth(csv_path):
    """
    Returns:
      ground_truth[text] = set((term, label), ...)
    using:
      - text from 'unprocessed'
      - annotations from 'processed'
    """
    df = pd.read_csv(csv_path)
    ground_truth = {}

    required_cols = {"unprocessed", "processed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError("Ground truth CSV missing required columns: {}".format(missing))

    for _, row in df.iterrows():
        text = extract_input_text_from_unprocessed(row["unprocessed"])
        annotations = set(parse_annotations_from_bracket_text(row["processed"]))

        if text not in ground_truth:
            ground_truth[text] = set()
        ground_truth[text].update(annotations)

    return ground_truth


def load_predictions(pred_csv_path):
    """
    Returns:
      predictions[text] = set((term, label), ...)
    using:
      - text from 'text'
      - annotations from 'generated_output'
    """
    df = pd.read_csv(pred_csv_path)
    predictions = {}

    required_cols = {"text", "generated_output"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError("Prediction CSV missing required columns: {}".format(missing))

    for _, row in df.iterrows():
        text = normalize_text_key(row["text"])
        annotations = set(parse_annotations_from_bracket_text(row["generated_output"]))

        if text not in predictions:
            predictions[text] = set()
        predictions[text].update(annotations)

    return predictions

# Matching helpers
def token_set(text):
    """
    Tokenize lightly for lenient matching.
    Keeps alphanumerics + apostrophes.
    """
    text = text.lower()
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    return set(tokens)


def overlap_score(term_a, term_b):
    """
    Overlap score for lenient matching:
      |intersection(tokens)| / max(len(tokens_a), len(tokens_b))
    """
    a = token_set(term_a)
    b = token_set(term_b)

    if not a or not b:
        return 0.0

    return len(a & b) / max(len(a), len(b))


def greedy_lenient_match(pred_entities, gold_entities, threshold=0.5):
    """
    One-to-one lenient matching, label-aware.
    """
    pred_entities = list(pred_entities)
    gold_entities = list(gold_entities)

    candidates = []
    for pi, (p_term, p_label) in enumerate(pred_entities):
        for gi, (g_term, g_label) in enumerate(gold_entities):
            if p_label != g_label:
                continue
            score = overlap_score(p_term, g_term)
            if score > threshold:
                candidates.append((score, pi, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_pred = set()
    used_gold = set()
    matched_pred = set()
    matched_gold = set()

    for score, pi, gi in candidates:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matched_pred.add(pred_entities[pi])
        matched_gold.add(gold_entities[gi])

    return matched_pred, matched_gold

# Metrics
def init_counts():
    return {
        category: {
            "tp_strict": 0, "fp_strict": 0, "fn_strict": 0,
            "tp_lenient": 0, "fp_lenient": 0, "fn_lenient": 0,
        }
        for category in CATEGORY_ORDER
    }


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_macro_micro(aggregated_counts, strict=True):
    metric_type = "strict" if strict else "lenient"

    total_tp, total_fp, total_fn = 0, 0, 0
    macro_precision, macro_recall, macro_f1 = [], [], []

    for category in CATEGORY_ORDER:
        counts = aggregated_counts[category]
        tp = counts["tp_{}".format(metric_type)]
        fp = counts["fp_{}".format(metric_type)]
        fn = counts["fn_{}".format(metric_type)]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision, recall, f1 = compute_metrics(tp, fp, fn)
        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)

    macro_avg = {
        "precision": sum(macro_precision) / len(macro_precision),
        "recall": sum(macro_recall) / len(macro_recall),
        "f1-score": sum(macro_f1) / len(macro_f1),
    }

    micro_precision, micro_recall, micro_f1 = compute_metrics(total_tp, total_fp, total_fn)
    micro_avg = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1-score": micro_f1,
    }

    return macro_avg, micro_avg

# Evaluate
def evaluate_predictions(ground_truth, predictions, lenient_threshold=0.5):
    aggregated_counts = init_counts()

    all_texts = sorted(set(ground_truth.keys()) | set(predictions.keys()))

    missing_in_predictions = []
    missing_in_ground_truth = []

    for text in all_texts:
        gold = set(ground_truth.get(text, set()))
        pred = set(predictions.get(text, set()))

        if text in ground_truth and text not in predictions:
            missing_in_predictions.append(text)
        if text in predictions and text not in ground_truth:
            missing_in_ground_truth.append(text)

        # Strict
        strict_matched = gold & pred

        for term, label in strict_matched:
            aggregated_counts[label]["tp_strict"] += 1

        for term, label in pred - strict_matched:
            aggregated_counts[label]["fp_strict"] += 1

        for term, label in gold - strict_matched:
            aggregated_counts[label]["fn_strict"] += 1

        # Lenient
        matched_pred, matched_gold = greedy_lenient_match(pred, gold, threshold=lenient_threshold)

        for term, label in matched_pred:
            aggregated_counts[label]["tp_lenient"] += 1

        for term, label in pred - matched_pred:
            aggregated_counts[label]["fp_lenient"] += 1

        for term, label in gold - matched_gold:
            aggregated_counts[label]["fn_lenient"] += 1

    macro_strict, micro_strict = compute_macro_micro(aggregated_counts, strict=True)
    macro_lenient, micro_lenient = compute_macro_micro(aggregated_counts, strict=False)

    overall_df = pd.DataFrame([
        {
            "Metric": "Micro Average",
            "Strict Precision": micro_strict["precision"],
            "Strict Recall": micro_strict["recall"],
            "Strict F1-score": micro_strict["f1-score"],
            "Lenient Precision": micro_lenient["precision"],
            "Lenient Recall": micro_lenient["recall"],
            "Lenient F1-score": micro_lenient["f1-score"],
        },
        {
            "Metric": "Macro Average",
            "Strict Precision": macro_strict["precision"],
            "Strict Recall": macro_strict["recall"],
            "Strict F1-score": macro_strict["f1-score"],
            "Lenient Precision": macro_lenient["precision"],
            "Lenient Recall": macro_lenient["recall"],
            "Lenient F1-score": macro_lenient["f1-score"],
        }
    ])

    per_entity_rows = []
    for category in CATEGORY_ORDER:
        counts = aggregated_counts[category]

        sp, sr, sf = compute_metrics(
            counts["tp_strict"], counts["fp_strict"], counts["fn_strict"]
        )
        lp, lr, lf = compute_metrics(
            counts["tp_lenient"], counts["fp_lenient"], counts["fn_lenient"]
        )

        per_entity_rows.append({
            "Entity": category,
            "TP Strict": counts["tp_strict"],
            "FP Strict": counts["fp_strict"],
            "FN Strict": counts["fn_strict"],
            "Strict Precision": sp,
            "Strict Recall": sr,
            "Strict F1-score": sf,
            "TP Lenient": counts["tp_lenient"],
            "FP Lenient": counts["fp_lenient"],
            "FN Lenient": counts["fn_lenient"],
            "Lenient Precision": lp,
            "Lenient Recall": lr,
            "Lenient F1-score": lf,
        })

    per_entity_df = pd.DataFrame(per_entity_rows)

    diagnostics = {
        "num_ground_truth_texts": len(ground_truth),
        "num_prediction_texts": len(predictions),
        "missing_in_predictions": len(missing_in_predictions),
        "missing_in_ground_truth": len(missing_in_ground_truth),
    }

    diagnostics_df = pd.DataFrame([
        {"Metric": k, "Value": v} for k, v in diagnostics.items()
    ])

    missing_pred_df = pd.DataFrame({"text_missing_prediction": missing_in_predictions})
    missing_gt_df = pd.DataFrame({"text_missing_ground_truth": missing_in_ground_truth})

    return {
        "aggregated_counts": aggregated_counts,
        "overall_df": overall_df,
        "per_entity_df": per_entity_df,
        "diagnostics_df": diagnostics_df,
        "missing_pred_df": missing_pred_df,
        "missing_gt_df": missing_gt_df,
        "micro_strict": micro_strict,
        "micro_lenient": micro_lenient,
        "macro_strict": macro_strict,
        "macro_lenient": macro_lenient,
    }


# Main
def main():
    print("Loading ground truth...")
    ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
    print("Loaded {} unique ground-truth texts.".format(len(ground_truth)))

    print("\nLoading predictions...")
    print("Prediction CSV: {}".format(PREDICTIONS_CSV))
    predictions = load_predictions(PREDICTIONS_CSV)
    print("Loaded {} unique prediction texts.".format(len(predictions)))

    print("\nEvaluating predictions...")
    results = evaluate_predictions(ground_truth, predictions, lenient_threshold=0.5)

    overall_path = os.path.join(EVAL_OUTPUT_DIR, "overall_metrics.csv")
    per_entity_path = os.path.join(EVAL_OUTPUT_DIR, "per_entity_metrics.csv")
    diagnostics_path = os.path.join(EVAL_OUTPUT_DIR, "diagnostics.csv")
    missing_pred_path = os.path.join(EVAL_OUTPUT_DIR, "missing_predictions.csv")
    missing_gt_path = os.path.join(EVAL_OUTPUT_DIR, "missing_ground_truth.csv")
    summary_path = os.path.join(EVAL_OUTPUT_DIR, "summary.csv")

    results["overall_df"].to_csv(overall_path, index=False)
    results["per_entity_df"].to_csv(per_entity_path, index=False)
    results["diagnostics_df"].to_csv(diagnostics_path, index=False)
    results["missing_pred_df"].to_csv(missing_pred_path, index=False)
    results["missing_gt_df"].to_csv(missing_gt_path, index=False)

    summary_df = pd.DataFrame([{
        "micro_strict_precision": results["micro_strict"]["precision"],
        "micro_strict_recall": results["micro_strict"]["recall"],
        "micro_strict_f1": results["micro_strict"]["f1-score"],
        "micro_lenient_precision": results["micro_lenient"]["precision"],
        "micro_lenient_recall": results["micro_lenient"]["recall"],
        "micro_lenient_f1": results["micro_lenient"]["f1-score"],
        "macro_strict_precision": results["macro_strict"]["precision"],
        "macro_strict_recall": results["macro_strict"]["recall"],
        "macro_strict_f1": results["macro_strict"]["f1-score"],
        "macro_lenient_precision": results["macro_lenient"]["precision"],
        "macro_lenient_recall": results["macro_lenient"]["recall"],
        "macro_lenient_f1": results["macro_lenient"]["f1-score"],
    }])
    summary_df.to_csv(summary_path, index=False)

    print("\nOverall metrics:")
    print(results["overall_df"].to_string(index=False))

    print("\nSaved outputs to:")
    print(overall_path)
    print(per_entity_path)
    print(diagnostics_path)
    print(missing_pred_path)
    print(missing_gt_path)
    print(summary_path)


if __name__ == "__main__":
    main()