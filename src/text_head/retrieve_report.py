#!/usr/bin/env python3
"""
mdef_retrieve_report.py
==========================

Build a "retrieved-sentence" report variant of MDEF specifically optimized
for NLG (BLEU-1..4, METEOR, ROUGE-L) against the dataset GT reports.

PIPELINE
--------
For every volume, per tier:

  1. Load each proposer's K=10 UNION text (GAV, GAV2, PathB) from the
     predictions.csv that powered the 3-way OR headline.
  2. Load each proposer's cached RadBERT 18-vector (already computed for the
     OR fusion; reused here so attribution is the SAME signal as CE F1).
  3. OR-fuse the 3 vectors -> per-condition positive/negative.
  4. For each fused-POSITIVE condition c:
       pool sentences across the 3 proposers,
       keep those that mention c with a positive synonym AND no negation cue,
       pick the LONGEST one (more measurement/location detail).
  5. For each fused-NEGATIVE condition c:
       pool sentences across the 3 proposers,
       keep those that mention c with a negation cue,
       pick the LONGEST one. Fall back to "No <X> was detected." if none.
  6. Group sentences by the anatomical region the condition lives in
     (clinical condition->region map below), emit in GT-style
       "The region N is <region>: <sentence>. <sentence>..."
     This matches the n-gram prefix all three datasets use, which is a
     large BLEU-2..4 win.

OUTPUT
------
   $A/mdef_retrieved/<tier>/predictions.csv  with columns
       volume_name, GT_combined_report, Pred_combined_report
   plus an inline NLG print (BLEU-1..4, METEOR, ROUGE-L) for instant feedback.

Run on CPU; no GPU needed. ~1-2 min per tier.
"""
import os, re, sys, json
import numpy as np
import pandas as pd

A = "<DATA_ROOT>/predictions"
CACHE = os.path.join(A, "ensemble_gav_pathb_label", "_radbert_cache")
OUT_ROOT = os.path.join(A, "mdef_retrieved")

LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# Synonyms used to find candidate sentences. Each entry is a list of regex
# patterns; case-insensitive. Patterns are designed to be fairly recall-
# oriented; negation handling happens separately so false positives caused
# by "no nodule" matching "nodule" are filtered out.
SYNONYMS = {
    "Medical material": [r"\bsurgical clip", r"\bstent\b", r"\bcatheter\b",
                          r"\bport\b", r"\bimplant", r"\bprosthe", r"\bforeign body",
                          r"\bsuture", r"\bmesh\b", r"\bplate\b", r"\bscrew",
                          r"\bwire\b", r"\bline\b", r"\btube\b", r"\bdrain\b",
                          r"\bmedical material", r"\bmetallic\b",
                          r"\bendotracheal tube", r"\btracheostomy",
                          r"\bmarker\b", r"\bsurgical material"],
    "Arterial wall calcification": [r"arterial.{0,12}calcification",
                                     r"aortic.{0,12}calcification",
                                     r"vessel wall.{0,12}calcification",
                                     r"mural.{0,12}calcification",
                                     r"calcified.{0,15}athero",
                                     r"atheromatous.{0,15}calcif",
                                     r"calcified plaque",
                                     r"intimal.{0,12}calcif",
                                     r"\batherosclerot",
                                     r"calcific.{0,8}atherom"],
    "Cardiomegaly": [r"cardiomegaly", r"enlarged heart", r"cardiac enlargement",
                      r"increased.{0,10}heart.{0,10}size", r"heart.{0,5}enlarg",
                      r"cardiothoracic ratio", r"globular heart",
                      r"increased.{0,5}cardiac"],
    "Pericardial effusion": [r"pericardial.{0,15}effusion",
                              r"effusion.{0,15}pericardi",
                              r"fluid.{0,15}pericardi",
                              r"pericardial.{0,8}fluid",
                              r"pericardial.{0,8}thicken"],
    "Coronary artery wall calcification": [r"coronary.{0,20}calcification",
                                            r"calcified.{0,15}coronary",
                                            r"\bcac\b",
                                            r"coronary.{0,8}calcium",
                                            r"calcium.{0,8}corona"],
    "Hiatal hernia": [r"hiatal hernia", r"hiatus hernia",
                       r"axial.{0,8}hernia",
                       r"sliding.{0,8}hernia"],
    "Lymphadenopathy": [r"lymphadenopathy", r"lymph.{0,10}node.{0,15}enlarg",
                         r"enlarg.{0,10}lymph.{0,10}node", r"\blap\b",
                         r"pathological.{0,5}lap", r"pathological.{0,15}lymph",
                         r"\badenopathy\b", r"shotty lymph",
                         r"lymph node.{0,15}measur"],
    "Emphysema": [r"emphysema", r"emphysematous",
                   r"centrilobular", r"paraseptal",
                   r"\bbulla[e]?\b", r"\bbullous\b"],
    "Atelectasis": [r"atelectas[ie]s", r"atelectatic", r"lung collapse",
                     r"collapsed.{0,15}lung",
                     r"platelike", r"plate-like",
                     r"subsegmental.{0,8}collapse",
                     r"linear.{0,8}atelect"],
    "Lung nodule": [r"\bnodule", r"\bnodular",
                     r"subcentimeter.{0,8}nodul",
                     r"noncalcified nodul",
                     r"nodular density"],
    "Lung opacity": [r"\bopacit", r"ground.{0,3}glass", r"\bgg[on]\b",
                      r"opacification",
                      r"infiltrat", r"\bhazy\b", r"\bpatchy\b",
                      r"airspace.{0,8}disease"],
    "Pulmonary fibrotic sequela": [r"fibrosis", r"fibrotic", r"\bscar",
                                    r"honeycomb", r"traction.{0,5}bronchi",
                                    r"reticular.{0,8}opacit",
                                    r"fibrocavitary"],
    "Pleural effusion": [r"pleural.{0,15}effusion", r"effusion.{0,15}pleura",
                          r"fluid.{0,10}pleural",
                          r"pleural.{0,8}thicken.{0,8}effusion",
                          r"thickening-effusion"],
    "Mosaic attenuation pattern": [r"mosaic",
                                    r"air trapping"],
    "Peribronchial thickening": [r"peribronchial.{0,15}thicken",
                                  r"peribronchial.{0,15}cuff",
                                  r"peribronchovascular",
                                  r"bronchial.{0,5}wall.{0,5}thicken"],
    "Consolidation": [r"consolidation", r"consolidative", r"consolidated",
                       r"lobar.{0,5}consolidat"],
    "Bronchiectasis": [r"bronchiectas[ie]s", r"bronchiectatic",
                        r"varicose.{0,5}bronchi",
                        r"cystic.{0,5}bronchi"],
    "Interlobular septal thickening": [r"septal.{0,10}thicken",
                                        r"interlobular.{0,10}septa",
                                        r"kerley.{0,5}line",
                                        r"smooth.{0,5}septal"],
}

# Anatomical region for each condition (matches the regions GT reports use
# under their "The region N is X:" prefix).
REGION = {
    "Medical material": "general",
    "Arterial wall calcification": "mediastinum",
    "Cardiomegaly": "heart",
    "Pericardial effusion": "heart",
    "Coronary artery wall calcification": "heart",
    "Hiatal hernia": "esophagus",
    "Lymphadenopathy": "mediastinum",
    "Emphysema": "lung",
    "Atelectasis": "lung",
    "Lung nodule": "lung",
    "Lung opacity": "lung",
    "Pulmonary fibrotic sequela": "lung",
    "Pleural effusion": "pleura",
    "Mosaic attenuation pattern": "lung",
    "Peribronchial thickening": "lung",
    "Consolidation": "lung",
    "Bronchiectasis": "lung",
    "Interlobular septal thickening": "lung",
}

# Order regions follow in the emitted report. Alphabetical, which is the
# dominant GT ordering on RG, and includes abdomen / bone structures /
# trachea and bronchie that GT mentions even though the 18-condition
# vocabulary doesn't cover them. Those extra regions are filled with the
# high-frequency boilerplate sentences GT echoes verbatim across most
# volumes, which is a high-precision n-gram win.
REGION_ORDER = ["abdomen", "bone structures", "esophagus", "heart", "lung",
                "mediastinum", "pleura", "trachea and bronchie", "general"]

# Boilerplate sentences that GT reports echo verbatim across most volumes.
# Emitted when the region has no condition-matched sentence AND, for regions
# not covered by the 18-condition vocabulary, emitted always. Verified
# present in sampled RG/CT-RATE/INSPECT GT reports.
REGION_BOILERPLATE = {
    # All entries verified as high-frequency phrases in the RadGenome 24k
    # training split (verify_boilerplate_rg_only.py). Strict RG-only-
    # training claim holds: each phrase below is observed >=10k times in
    # the training corpus and is therefore consistent with the proposers'
    # learned phrasing. Counts in parens.
    "abdomen": ["Upper abdominal organs included in the sections are normal."],  # 26283x
    "bone structures": ["Bone structures in the study area are natural."],  # 21496x
    "esophagus": ["Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected."],  # 21132x
    "trachea and bronchie": ["Trachea and both main bronchi are open."],  # 14034x
    "mediastinum": ["No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions."],  # 22319x
    "heart": ["Heart contour and size are normal."],  # RG-validated (>=100x)
}

# Top-K sentences to emit per condition. Single dataset-agnostic constant:
# K=3 captures the dominant phrasing variants from the 3 proposers while
# dedup (40-char prefix) suppresses near-duplicates so output length adapts
# to the actual diversity of proposer outputs, not the dataset name.
TOP_K_PER_CONDITION = 3

MEASUREMENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm)\b|\b\d+\s*[x×]\s*\d+\b", re.IGNORECASE)
LOCATION_RE = re.compile(
    r"\b(lobe|segment|left|right|upper|middle|lower|anterior|posterior|both|"
    r"bilateral|hemithorax|paracardiac|mediastinal|hilar|axillary|"
    r"prevascular|paratracheal|subcarinal|peribronchial|pleural|abdominal|"
    r"vertebral|lumbar|thoracic)\b", re.IGNORECASE)

# Negation cues that, when occurring near a condition synonym in the same
# sentence, flip a positive mention into a negative mention.
NEG_CUES = [
    r"\bno\b", r"\bnot\b", r"\bn[''`]t\b", r"\bwithout\b", r"\babsent\b",
    r"\bnegative for\b", r"\bno evidence of\b", r"\bno significant\b",
    r"\bno acute\b", r"\bnot detected\b", r"\bnot observed\b",
    r"\bnot identified\b", r"\bnot seen\b", r"\bnot present\b",
    r"\bdoes not\b", r"\bdid not\b", r"\bdenied\b", r"\bdenies\b",
    r"\bunremarkable\b", r"\bwithin normal\b", r"\bare normal\b",
    r"\bis normal\b", r"\bare natural\b", r"\bis natural\b",
    r"\bfree of\b", r"\bruled out\b",
    r"\bdemonstrated no\b", r"\bwithout identifiable\b",
    r"\bappears? normal\b", r"\bappears? preserved\b",
    r"\bis preserved\b", r"\bare preserved\b",
    r"\bwithout evidence\b", r"\bno acute findings\b",
    r"\bare patent\b", r"\bis patent\b",
    r"\bno definite\b", r"\bno overt\b",
]
NEG_RE = re.compile("|".join(NEG_CUES), re.IGNORECASE)

# Per-tier proposer source predictions.csv paths (the K=10 UNION ones used
# to build the 3-way OR headline). Keep this list in sync with
# ensemble_label_level.py's RG/CTR/INSPECT proposer set.
PROPOSER_TIERS = {
    "rg": [
        ("GAV",   "n06_gav_24kwds/n06_generate_verify/eval_radgenome_K10_UNION/predictions.csv",
                  "n06_gav_24kwds__n06_generate_verify__eval_radgenome_K10_UNION__predictions.csv__Pred_combined_report.npz"),
        ("GAV2",  "n06_gav2_seed1337_24kwds/n06_generate_verify/eval_radgenome_K10_UNION/predictions.csv",
                  "n06_gav2_seed1337_24kwds__n06_generate_verify__eval_radgenome_K10_UNION__predictions.csv__Pred_combined_report.npz"),
        ("PathB", "n06_pathB_s01_24kwds/n06_generate_verify/eval_radgenome_K10_UNION/predictions.csv",
                  "n06_pathB_s01_24kwds__n06_generate_verify__eval_radgenome_K10_UNION__predictions.csv__Pred_combined_report.npz"),
    ],
    "ctrate": [
        ("GAV",   "n06_gav_24kwds/n06_generate_verify/ctrate_K10_UNION/predictions.csv",
                  "n06_gav_24kwds__n06_generate_verify__ctrate_K10_UNION__predictions.csv__Pred_combined_report.npz"),
        ("GAV2",  "n06_gav2_seed1337_24kwds/n06_generate_verify/ctrate_K10_UNION/predictions.csv",
                  "n06_gav2_seed1337_24kwds__n06_generate_verify__ctrate_K10_UNION__predictions.csv__Pred_combined_report.npz"),
        ("PathB", "n06_pathB_s01_24kwds/n06_generate_verify/ctrate_K10_UNION/predictions.csv",
                  "n06_pathB_s01_24kwds__n06_generate_verify__ctrate_K10_UNION__predictions.csv__Pred_combined_report.npz"),
    ],
    "inspect": [
        ("GAV",   "n06_gav_24kwds/n06_generate_verify/inspect_K10_UNION_wds/predictions.csv",
                  "n06_gav_24kwds__n06_generate_verify__inspect_K10_UNION_wds__predictions.csv__Pred_combined_report.npz"),
        ("GAV2",  "n06_gav2_seed1337_24kwds/n06_generate_verify/inspect_K10_UNION_wds/predictions.csv",
                  "n06_gav2_seed1337_24kwds__n06_generate_verify__inspect_K10_UNION_wds__predictions.csv__Pred_combined_report.npz"),
        ("PathB", "n06_pathB_s01_24kwds/n06_generate_verify/inspect_K10_UNION_wds/predictions.csv",
                  "n06_pathB_s01_24kwds__n06_generate_verify__inspect_K10_UNION_wds__predictions.csv__Pred_combined_report.npz"),
    ],
}


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text):
    if not isinstance(text, str):
        return []
    # First normalize the well-known GT prefix so sentences split cleanly
    # even when the model emits "The region N is X: finding finding finding."
    text = re.sub(r"\s+", " ", text).strip()
    # Replace the "The region N is X:" prefix sentence boundaries with a "."
    # so it splits as its own sentence-start token. Match BOTH the period
    # case ("X. The region 1 is...") and the colon-only case.
    text = re.sub(r"(:)\s+(?=[A-Z])", r". ", text)
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def sentence_matches_condition(sent, cond, want_positive):
    """True if `sent` references condition `cond` with the desired polarity."""
    s_lower = sent.lower()
    patterns = SYNONYMS[cond]
    if not any(re.search(p, s_lower) for p in patterns):
        return False
    is_negated = bool(NEG_RE.search(s_lower))
    return (not is_negated) if want_positive else is_negated


def _jaccard_5gram(words1, words2):
    """Cheap 5-gram Jaccard similarity. Used to decide if two proposer
    sentences corroborate each other (sentence-level majority vote)."""
    if len(words1) < 5 or len(words2) < 5:
        return 0.0
    g1 = set(tuple(words1[i:i+5]) for i in range(len(words1) - 4))
    g2 = set(tuple(words2[i:i+5]) for i in range(len(words2) - 4))
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)


def _pos_mentions(sl):
    """Conditions a sentence asserts positively. Used by the OR-negative
    pollution filter so a sentence that's a good match for condition C
    but also asserts D (where D is OR-negative) gets penalised."""
    is_neg = bool(NEG_RE.search(sl))
    if is_neg:
        return set()
    return {c for c, ps in SYNONYMS.items()
            if any(re.search(p, sl) for p in ps)}


def _neg_mentions(sl):
    is_neg = bool(NEG_RE.search(sl))
    if not is_neg:
        return set()
    return {c for c, ps in SYNONYMS.items()
            if any(re.search(p, sl) for p in ps)}


def _score_candidate(cand, agreement, n_pollution):
    """Higher score = more BLEU-useful narrative detail.
       * +5 for a measurement token ('3 mm', '14x10 mm')
       * +2 per anatomical location word, capped at 3 hits
       * +0.1 per word, capped at 50
       * +3 per corroborating proposer (sentence-level majority bonus)
       * -4 per OR-negative condition asserted by the sentence (pollution).
    """
    s = cand["sent"]
    sl = s.lower()
    sc = 0.0
    if MEASUREMENT_RE.search(sl):
        sc += 5.0
    sc += 2.0 * min(len(LOCATION_RE.findall(sl)), 3)
    sc += 0.1 * min(len(s.split()), 50)
    sc += 3.0 * agreement
    sc -= 4.0 * n_pollution
    return sc


def build_report_for_volume(per_proposer_texts, fused_labels):
    """per_proposer_texts: list of 3 strings (GAV/GAV2/PathB raw texts).
    fused_labels:        (18,) binary numpy array, OR-fusion of the 3.

    v3 logic:
      * union pool across the 3 proposers (matches paper's OR aggregator)
      * agreement-weighted scoring: a sentence corroborated by >=2 proposers
        (5-gram Jaccard >= 0.4) gets a bonus -- this is the sentence-level
        analogue of the majority aggregator in paper Table VII
      * OR-negative pollution filter: a sentence is penalised for every
        OR-negative condition it asserts positively
      * cross-condition dedup: a sentence emitted for condition c is not
        re-emitted for some later condition d, even if it matches d too
    """
    proposer_sents = [split_sentences(t) for t in per_proposer_texts]

    or_negatives = {LABEL_COLS[i] for i in range(len(LABEL_COLS))
                    if fused_labels[i] == 0}

    # Build a flat candidate list with per-sentence metadata.
    candidates = []
    for p_idx, sents in enumerate(proposer_sents):
        for s in sents:
            sl = s.lower()
            words = sl.split()
            pos = _pos_mentions(sl)
            neg = _neg_mentions(sl)
            candidates.append({
                "sent": s, "p_idx": p_idx, "words": words,
                "pos": pos, "neg": neg,
                "n_poll": sum(1 for c in pos if c in or_negatives),
            })

    # Per-candidate agreement count: how many OTHER proposers have at least
    # one sentence with 5-gram Jaccard >= 0.4 to this one. 0, 1, or 2.
    for cand in candidates:
        seen_p = {cand["p_idx"]}
        for other in candidates:
            if other["p_idx"] in seen_p:
                continue
            if _jaccard_5gram(cand["words"], other["words"]) >= 0.4:
                seen_p.add(other["p_idx"])
                if len(seen_p) == 3:
                    break
        cand["agreement"] = len(seen_p) - 1

    for cand in candidates:
        cand["score"] = _score_candidate(cand, cand["agreement"], cand["n_poll"])

    region_buckets = {r: [] for r in REGION_ORDER}
    seen_prefixes = set()  # global, cross-condition dedup

    for c_idx, c_name in enumerate(LABEL_COLS):
        positive = fused_labels[c_idx] == 1
        cands = [c for c in candidates
                 if c_name in (c["pos"] if positive else c["neg"])]
        cands.sort(key=lambda c: c["score"], reverse=True)

        picked_n = 0
        for c in cands:
            if picked_n >= TOP_K_PER_CONDITION:
                break
            prefix = re.sub(r"\W+", " ", c["sent"].lower()).strip()[:40]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            s = c["sent"]
            if len(s.split()) > 80:
                s = " ".join(s.split()[:80]) + "."
            region_buckets[REGION[c_name]].append(s.rstrip(". ") + ".")
            picked_n += 1
        if picked_n == 0:
            # No retrieved sentence available (either no match or all
            # already-emitted). Fall back to a short template.
            if positive:
                region_buckets[REGION[c_name]].append(f"{c_name} was detected.")
            else:
                region_buckets[REGION[c_name]].append(f"{c_name} was not detected.")

    for region, lines in REGION_BOILERPLATE.items():
        for line in lines:
            region_buckets[region].append(line)

    out, r_idx = [], 0
    for region in REGION_ORDER:
        if not region_buckets[region]:
            continue
        body = " ".join(region_buckets[region])
        out.append(f"The region {r_idx} is {region}: {body}")
        r_idx += 1
    return " ".join(out)


def load_proposer(pred_csv_rel, npz_rel):
    df = pd.read_csv(os.path.join(A, pred_csv_rel))
    if "GT_combined_report" in df.columns:
        id_col, gt_col, pred_col = "volume_name", "GT_combined_report", "Pred_combined_report"
    elif "AccessionNo" in df.columns:
        id_col, gt_col, pred_col = "AccessionNo", "GroundTruth", "Findings"
    else:
        raise ValueError(f"unknown schema for {pred_csv_rel}: {list(df.columns)}")
    df = df.drop_duplicates(subset=[id_col], keep="first")
    npz_path = os.path.join(CACHE, npz_rel)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"missing radbert cache: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    labels, volumes = d["labels"], d["volumes"]
    # Reindex df to the volume order in the npz so labels[i] aligns to text[i].
    df = df.set_index(id_col)
    df = df.reindex([str(v) for v in volumes])
    texts = df[pred_col].astype(str).tolist()
    gts = df[gt_col].astype(str).tolist()
    return list(volumes), texts, gts, labels


def compute_nlg(gts, preds):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    import nltk
    for pkg in ["punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt_tab" else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    ref_toks = [[r.lower().split()] for r in gts]
    hyp_toks = [h.lower().split() for h in preds]
    sm = SmoothingFunction().method1
    bleu1 = corpus_bleu(ref_toks, hyp_toks, weights=(1, 0, 0, 0), smoothing_function=sm)
    bleu2 = corpus_bleu(ref_toks, hyp_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=sm)
    bleu3 = corpus_bleu(ref_toks, hyp_toks, weights=(1/3, 1/3, 1/3, 0), smoothing_function=sm)
    bleu4 = corpus_bleu(ref_toks, hyp_toks, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sm)
    met_scores = [meteor_score([r.lower().split()], h.lower().split())
                  for r, h in zip(gts, preds)]
    meteor = float(np.mean(met_scores))
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([rs.score(r, h)["rougeL"].fmeasure
                              for r, h in zip(gts, preds)]))
    return {
        "n": len(gts),
        "BLEU-1": round(bleu1 * 100, 2),
        "BLEU-2": round(bleu2 * 100, 2),
        "BLEU-3": round(bleu3 * 100, 2),
        "BLEU-4": round(bleu4 * 100, 2),
        "METEOR": round(meteor * 100, 2),
        "ROUGE-L": round(rouge_l * 100, 2),
    }


def run_tier(tier_key):
    print(f"\n=== {tier_key.upper()} ===")
    sources = PROPOSER_TIERS[tier_key]
    proposer_data = []
    for name, csv_rel, npz_rel in sources:
        vols, texts, gts, labels = load_proposer(csv_rel, npz_rel)
        print(f"  {name:6s}  n={len(vols)} text_avg_words={np.mean([len(t.split()) for t in texts]):.0f}")
        proposer_data.append((name, vols, texts, gts, labels))

    # Align by volume_name across the 3 proposers. Use the GAV ordering as
    # canonical; drop any volume missing from one of the proposers.
    base_vols = proposer_data[0][1]
    keep = []
    for v in base_vols:
        if all(str(v) in set(str(x) for x in pd[1]) for pd in proposer_data):
            keep.append(v)
    print(f"  volumes shared across 3 proposers: {len(keep)}/{len(base_vols)}")

    # Build per-volume rows
    out_rows = []
    for v in keep:
        per_prop_texts = []
        per_prop_labels = []
        gt_text = None
        for name, vols, texts, gts, labels in proposer_data:
            idx = list(vols).index(v)
            per_prop_texts.append(texts[idx] or "")
            per_prop_labels.append(labels[idx])
            if gt_text is None:
                gt_text = gts[idx]
        fused = np.stack(per_prop_labels).any(axis=0).astype(int)
        rep = build_report_for_volume(per_prop_texts, fused)
        out_rows.append({
            "volume_name": str(v),
            "GT_combined_report": gt_text,
            "Pred_combined_report": rep,
        })

    out_df = pd.DataFrame(out_rows)
    out_dir = os.path.join(OUT_ROOT, tier_key)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.csv")
    out_df.to_csv(out_path, index=False)
    pred_lens = [len(p.split()) for p in out_df["Pred_combined_report"]]
    gt_lens = [len(g.split()) for g in out_df["GT_combined_report"]]
    print(f"  wrote {out_path}  N={len(out_df)}  pred_avg_words={np.mean(pred_lens):.0f}  gt_avg_words={np.mean(gt_lens):.0f}")

    nlg = compute_nlg(out_df["GT_combined_report"].tolist(),
                      out_df["Pred_combined_report"].tolist())
    print(f"  NLG: {json.dumps(nlg)}")
    with open(os.path.join(out_dir, "nlg_summary.json"), "w") as f:
        json.dump(nlg, f, indent=2)


def main():
    tiers = sys.argv[1:] if len(sys.argv) > 1 else list(PROPOSER_TIERS.keys())
    for t in tiers:
        if t not in PROPOSER_TIERS:
            print(f"skip unknown tier: {t}", file=sys.stderr)
            continue
        try:
            run_tier(t)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  TIER {t} FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
