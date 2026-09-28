#!/usr/bin/env python3
"""
Generate the LaTeX replacement body of Table 3 (tab:per_cond_all) showing
per-condition Precision, Recall, F1 for CARE-RG vs MARCH only, across
RG / CTR / INS, with values in SEPARATE columns. Bolds the winner per
(condition, dataset, metric) cell.

Reads the 6 ce_per_condition.csv files we have on disk; writes the
LaTeX <tr>-style rows to stdout.
"""
import csv
import os
import sys

A = "<DATA_ROOT>/predictions"

SYSTEMS = ["Reg2RG", "CT-GR.", "MARCH", "CARE"]  # column order, left to right
SRC = {
    "RG":  {"Reg2RG":  "n_march_resident_24kwds/n_march_resident/eval_radgenome/ce_per_condition.csv",
            "CT-GR.":  "n_ct_graph_24kwds/n_ct_graph/eval_radgenome/ce_per_condition.csv",
            "MARCH":   "n_march_resident_24kwds/n_march_resident/march_full_pipeline/ce_per_condition.csv",
            "CARE":    "ensemble_gav_pathb_label/rg_3way_K10U__or/ce_per_condition.csv"},
    "CTR": {"Reg2RG":  "n_march_resident_24kwds/n_march_resident/ctrate/ce_per_condition.csv",
            "CT-GR.":  "n_ct_graph_24kwds/n_ct_graph/ctrate/ce_per_condition.csv",
            "MARCH":   "n_march_resident_24kwds/n_march_resident/march_full_pipeline_ctrate/ce_per_condition.csv",
            "CARE":    "ensemble_gav_pathb_label/ctrate_3way_K10U__or/ce_per_condition.csv"},
    "INS": {"Reg2RG":  "n_march_resident_24kwds/n_march_resident/inspect_wds/ce_per_condition.csv",
            "CT-GR.":  "n_ct_graph_24kwds/n_ct_graph/inspect_wds/ce_per_condition.csv",
            "MARCH":   "n_march_resident_24kwds/n_march_resident/march_full_pipeline_inspect/ce_per_condition.csv",
            "CARE":    "ensemble_gav_pathb_label/inspect_3way_K10U__or/ce_per_condition.csv"},
}

# Condition order + abbreviated row labels matching the existing Table 3.
ORDER = [
    ("High prevalence", [
        ("Arterial wall calcification",       r"Arterial wall calc."),
        ("Coronary artery wall calcification", r"Coronary art.\ calc."),
        ("Lung opacity",                       r"Lung opacity"),
        ("Lung nodule",                        r"Lung nodule"),
        ("Cardiomegaly",                       r"Cardiomegaly"),
        ("Pleural effusion",                   r"Pleural effusion"),
    ]),
    ("Medium prevalence", [
        ("Consolidation",                       r"Consolidation"),
        ("Emphysema",                           r"Emphysema"),
        ("Atelectasis",                         r"Atelectasis"),
        ("Pulmonary fibrotic sequela",          r"Pulm.\ fibrotic seq."),
        ("Mosaic attenuation pattern",          r"Mosaic atten.\ patt."),
        ("Lymphadenopathy",                     r"Lymphadenopathy"),
    ]),
    ("Low prevalence", [
        ("Interlobular septal thickening",      r"Interlob.\ sept.\ th."),
        ("Hiatal hernia",                       r"Hiatal hernia"),
        ("Bronchiectasis",                      r"Bronchiectasis"),
        ("Medical material",                    r"Medical material"),
        ("Peribronchial thickening",            r"Peribronchial th."),
        ("Pericardial effusion",                r"Pericardial effusion"),
    ]),
]


def load(rel):
    d = {}
    with open(os.path.join(A, rel)) as f:
        for row in csv.DictReader(f):
            d[row["Condition"]] = {
                "P":  float(row["Precision"]),
                "R":  float(row["Recall"]),
                "F1": float(row["F1"]),
            }
    return d


def fmt(v, bold):
    s = f"{v:.3f}".lstrip("0")  # ".591"; strip leading 0
    if s.startswith("-"):
        s = "-" + s[1:].lstrip("0")
    return f"\\textbf{{{s}}}" if bold else s


def main():
    data = {tier: {s: load(p) for s, p in m.items()} for tier, m in SRC.items()}
    n_num_cols = len(SYSTEMS) * 3 * len(SRC)  # systems * (P,R,F1) * tiers
    total_cols = n_num_cols + 1                # plus Condition column
    out = []
    out.append(r"% --- begin per-condition P/R/F1 table body (auto-generated) ---")
    out.append(r"\toprule")
    # Outer header row: dataset bands.
    band = []
    for i, tier_pretty in enumerate([(r"\textbf{RG} (in-distribution)",),
                                       (r"\textbf{CTR} (cross-dataset)",),
                                       (r"\textbf{INS} (cross-institution)",)]):
        band.append(r"\multicolumn{" + str(len(SYSTEMS) * 3) + r"}{c"
                     + ("|" if i < 2 else "") + r"}{" + tier_pretty[0] + r"}")
    out.append("& " + " & ".join(band) + r" \\")
    # Middle header row: system labels under each dataset band.
    sys_band = []
    for tier_idx in range(len(SRC)):
        for si, s in enumerate(SYSTEMS):
            label = r"\textbf{\ours{}}" if s == "CARE" else r"\textbf{" + s + r"}"
            sep = "|" if not (tier_idx == len(SRC) - 1 and si == len(SYSTEMS) - 1) else ""
            sys_band.append(r"\multicolumn{3}{c" + sep + r"}{" + label + r"}")
    out.append("& " + " & ".join(sys_band) + r" \\")
    # Inner header row: P R F1 under each system.
    inner = []
    for _ in range(len(SRC) * len(SYSTEMS)):
        inner += ["P", "R", "F1"]
    out.append(r"\textbf{Condition} & " + " & ".join(inner) + r" \\")
    out.append(r"\midrule")

    for bucket_name, conds in ORDER:
        out.append(r"\multicolumn{" + str(total_cols) + r"}{@{}l}{\textit{" + bucket_name + r"}} \\")
        for cond_key, cond_label in conds:
            cells = []
            for tier in ("RG", "CTR", "INS"):
                # Find winner per metric across the 4 systems for bolding.
                vals = {s: data[tier][s].get(cond_key, {"P": 0, "R": 0, "F1": 0})
                        for s in SYSTEMS}
                wins = {met: max(vals[s][met] for s in SYSTEMS)
                        for met in ("P", "R", "F1")}
                for s in SYSTEMS:
                    for met in ("P", "R", "F1"):
                        v = vals[s][met]
                        cells.append(fmt(v, v == wins[met] and v > 0))
            out.append(cond_label + " & " + " & ".join(cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"% --- end auto-generated body ---")
    print("\n".join(out))


if __name__ == "__main__":
    main()
