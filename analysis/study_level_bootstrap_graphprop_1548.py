"""
Study-level holdout za graph-propagation nalaz (RRF-3 -> RRF-4), ISTA
metodologija kao study_level_bootstrap_1548.py (RRF-3 vs cosine/BLAST) --
resampling PO IZVORU (reference kolona) umesto po pojedinacnom paru, da se
proveri da li "graph-propagation znacajno prevazilazi RRF-3" tvrdnja
prezivljava kad se uvazi da mnogi parovi dele isti izvor (nisu nezavisni
statisticki uzorci).

Koristi POSTOJECI graph_propagation_signal_1548_per_query.csv (primenjivi
upiti samo, ima_graph_signal=True) -- ista evidencija kao originalni nalaz.

Izlaz:
    output/study_level_bootstrap_graphprop_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

PER_QUERY = Path("/home/lana/ALERGRAF/output/graph_propagation_signal_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/study_level_bootstrap_graphprop_1548_summary.txt")

N_BOOTSTRAP = 2000
SEED = 42

df = pd.read_csv(PER_QUERY)
applicable = df[df["has_graph_signal"] == True].copy()  # noqa: E712 -- ista logika kao original
gold = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
merged = applicable.merge(gold, on="pair_id", how="left")
n_missing_ref = merged["reference"].isna().sum()
merged = merged.dropna(subset=["reference"])

n_studies = merged["reference"].nunique()
n_pairs = merged["pair_id"].nunique()
top_study_frac = merged["reference"].value_counts().iloc[0] / len(merged)

summary_lines = [
    "=" * 80, "Study-level bootstrap: graph-propagation (RRF-4) vs RRF-3, resampling PO IZVORU",
    "=" * 80, "",
    f"Primenjivih upita: {len(merged)} (od {len(applicable)}, {n_missing_ref} bez reference izbaceno), "
    f"parova: {n_pairs}, razlicitih izvora: {n_studies}",
    f"Najveca studijska grupa drzi {top_study_frac:.1%} svih primenjivih upita", "",
]


def paired_bootstrap_graphprop(sub, group_col, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    groups = sub[group_col].unique()
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        d = (np.average(1.0 / resampled["rrf4_rank"], weights=w)
             - np.average(1.0 / resampled["rrf3_rank"], weights=w))
        deltas.append(d)
    return np.array(deltas)


for label, group_col in [("PO PARU (pair_id, postojeca metodologija)", "pair_id"),
                          ("PO IZVORU (reference, novo -- studijski nivo)", "reference")]:
    deltas = paired_bootstrap_graphprop(merged, group_col, N_BOOTSTRAP, SEED)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"--- {label} ---")
    summary_lines.append(f"  RRF-4 vs RRF-3: mean delta = {deltas.mean():+.4f}, "
                          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")
    summary_lines.append("")

non_inferred = merged[merged["evidence_level"] != "Inferred (family-level homology)"]
if len(non_inferred) > 0:
    summary_lines.append(f"--- Samo NE-Inferred (n={len(non_inferred)} upita, "
                          f"{non_inferred['pair_id'].nunique()} parova), study-level ---")
    deltas = paired_bootstrap_graphprop(non_inferred, "reference", N_BOOTSTRAP, SEED)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"  RRF-4 vs RRF-3: mean delta = {deltas.mean():+.4f}, "
                          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")
else:
    summary_lines.append("Nema NE-Inferred primenjivih upita sa graph signalom.")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
