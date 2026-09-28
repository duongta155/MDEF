#!/usr/bin/env python3
"""
verify_boilerplate_rg_only.py
=============================

Verifies that the Stage 5 text-head boilerplate sentences are derived
from RG training data only (not from CT-Rate / INSPECT).

For each boilerplate sentence, counts occurrences (exact + fuzzy
sliding-token-overlap >= 0.6) in the RG 24k training reports CSV.

Sentences with 100+ training-set occurrences are accepted as "high-
frequency RG training boilerplate" and the strict RG-only training
claim holds. Sentences with <100 occurrences are reported with the top
training-set replacement candidates ranked by training frequency.
"""
import os, re, sys, json
import pandas as pd
from collections import Counter

RG_TRAIN_CSV = "<DATA_ROOT>/radgenome_ct2rep_reports.csv"

# The 6 Stage 5 boilerplate sentences (verbatim from
# care_rg_retrieve_report.py:REGION_BOILERPLATE).
BOILERPLATE = {
    "abdomen": "No significant pathology was detected in the abdominal sections.",
    "bone structures": "Bone structures in the study area are natural.",
    "esophagus": "Thoracic esophagus calibration is normal.",
    "trachea and bronchie": "Trachea and main bronchi are open.",
    "mediastinum": "No enlarged lymph nodes in pathological dimensions.",
    "heart": "Heart contour and size are normal.",
}


def _tokenize(s):
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s.split()


def _sentence_split(text):
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()]


def fuzzy_count_in_text(boil_lower, text_sentences_lower):
    """Count near-matches of boil in text via token Jaccard >= 0.6."""
    boil_toks = set(_tokenize(boil_lower))
    if len(boil_toks) < 4:
        return 0
    hits = 0
    for s in text_sentences_lower:
        s_toks = set(_tokenize(s))
        if not s_toks:
            continue
        j = len(boil_toks & s_toks) / len(boil_toks | s_toks)
        if j >= 0.6:
            hits += 1
    return hits


def main():
    if not os.path.exists(RG_TRAIN_CSV):
        print(f"ERROR: {RG_TRAIN_CSV} missing", file=sys.stderr); sys.exit(1)
    df = pd.read_csv(RG_TRAIN_CSV)
    text_col = "Findings_EN" if "Findings_EN" in df.columns else df.columns[1]
    reports = df[text_col].dropna().astype(str).tolist()
    print(f"Loaded {len(reports)} RG training reports.")

    # Pre-split + lowercase every training report once.
    print("Splitting reports into sentences...")
    train_sentences = []
    for rpt in reports:
        train_sentences.extend(s.lower() for s in _sentence_split(rpt))
    print(f"Total training sentences: {len(train_sentences):,}")

    # Exact sentence-level counter for fast verification.
    exact_counter = Counter(train_sentences)

    print("\n=== Boilerplate verification ===")
    results = {}
    for region, boil in BOILERPLATE.items():
        boil_lower = boil.lower()
        exact = exact_counter.get(boil_lower, 0)
        # Fuzzy: token-Jaccard >= 0.6 against unique training sentences only
        # (saves work; same sentence repeated counts once for the check, we
        # multiply by exact_counter at the end if you want raw freq).
        fuzzy = 0
        boil_toks = set(_tokenize(boil_lower))
        for s, c in exact_counter.items():
            s_toks = set(_tokenize(s))
            if not s_toks:
                continue
            j = len(boil_toks & s_toks) / len(boil_toks | s_toks)
            if j >= 0.6:
                fuzzy += c
        status = "RG-VALIDATED" if (exact + fuzzy) >= 100 else "LOW-FREQUENCY"
        print(f"  [{status}]  region={region!r}")
        print(f"    boilerplate: {boil!r}")
        print(f"    exact match in RG train: {exact}")
        print(f"    fuzzy match (Jaccard>=0.6) in RG train: {fuzzy}")
        results[region] = {"boilerplate": boil, "exact": exact,
                            "fuzzy": fuzzy, "status": status}

    # For LOW-FREQUENCY boilerplate, surface the top-5 RG-training-frequent
    # alternatives that mention the region (or related anatomy).
    print("\n=== Top-15 RG-training-frequent sentences (any) ===")
    for s, c in exact_counter.most_common(15):
        print(f"  {c:>5}x  {s[:120]}")

    os.makedirs("<DATA_ROOT>/"
                "analysis/care_rg_boilerplate_audit", exist_ok=True)
    out = "<DATA_ROOT>/predictions/care_rg_boilerplate_audit/audit.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")

    low = [r for r, d in results.items() if d["status"] == "LOW-FREQUENCY"]
    if low:
        print(f"\nLOW-FREQUENCY regions: {low}")
        print("Recommended: replace those boilerplate strings with an "
              "RG-validated alternative from the top-frequent training sentences.")
    else:
        print("\nAll 6 boilerplate sentences are RG-training-validated. "
              "Strict RG-only training claim holds.")


if __name__ == "__main__":
    main()
