"""
EXP-6.5b: BLAST wins vs RRF wins vs Tie, ogranicheno na CROSS-FAMILY parove
(family_1 != family_2), preko CELOG dataseta (ne samo Tier A - Tier A skoro
da nema cross-family parova, pa originalna hipoteza "RRF resava cross-family
slucajeve koje BLAST ne vidi" tamo nije mogla da se testira).

Ista kategorizacija kao blast_vs_rrf_case_analysis_1548.py:
  BLAST wins: blast_rank <= rrf_rank / 2
  RRF wins:   rrf_rank <= blast_rank / 2
  Tie:        ni jedno ni drugo

Koristi vec sacuvane rezultate (rank_fusion_1548_per_query.csv), nema novog
racunanja.

Izlaz:
    output/blast_vs_rrf_cross_family_case_analysis_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANK_FUSION = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/blast_vs_rrf_cross_family_case_analysis_1548_summary.txt")

WIN_RATIO = 2.0
TARGET_FAMILIES = {"nsLTP", "Profilin"}

df = pd.read_csv(RANK_FUSION)
gold = pd.read_csv(GOLD)[["pair_id", "allergen_id_1", "allergen_id_2", "family_1", "family_2",
                           "evidence_level", "sequence_identity_pct"]].drop_duplicates(subset="pair_id")
clean = pd.read_csv(CLEAN_ALLERGENS)[["official_name", "sequence_length"]].drop_duplicates(subset="official_name")

merged = df.merge(gold, on="pair_id", how="left")

cross_family = merged[merged["family_1"] != merged["family_2"]].copy()
print(f"Cross-family queries: {len(cross_family)} / {len(merged)} total ({len(cross_family)/len(merged):.1%})")
print(f"Cross-family unique pairs: {cross_family['pair_id'].nunique()}")

# categorize
cross_family["category"] = "Tie"
cross_family.loc[cross_family["blast_rank"] <= cross_family["rrf_rank"] / WIN_RATIO, "category"] = "BLAST wins"
cross_family.loc[cross_family["rrf_rank"] <= cross_family["blast_rank"] / WIN_RATIO, "category"] = "RRF wins"
print(cross_family["category"].value_counts())

len_map = dict(zip(clean["official_name"], clean["sequence_length"]))
cross_family["len_1"] = cross_family["allergen_id_1"].map(len_map)
cross_family["len_2"] = cross_family["allergen_id_2"].map(len_map)
cross_family["avg_len"] = (cross_family["len_1"] + cross_family["len_2"]) / 2

cross_family["involves_target_fam"] = (
    cross_family["family_1"].isin(TARGET_FAMILIES) | cross_family["family_2"].isin(TARGET_FAMILIES)
)
cross_family["family_pair"] = cross_family.apply(
    lambda r: " / ".join(sorted([str(r["family_1"]), str(r["family_2"])])), axis=1
)

# overall MRR comparison on this subset (context: does RRF actually beat BLAST here in aggregate?)
overall_lines = []
for rank_col, label in [("cosine_rank", "cosine"), ("blast_rank", "BLAST"),
                         ("foldseektm_rank", "FoldseekTM"), ("rrf_rank", "RRF-3")]:
    mrr = (1.0 / cross_family[rank_col]).mean()
    overall_lines.append(f"  {label:12s} MRR={mrr:.4f}")

summary_lines = ["=" * 70,
                  "EXP-6.5b: BLAST wins vs RRF wins vs Tie, CROSS-FAMILY pairs only (full dataset)",
                  "=" * 70, "",
                  f"Cross-family queries: {len(cross_family)} / {len(merged)} total ({len(cross_family)/len(merged):.1%})",
                  f"Cross-family unique pairs: {cross_family['pair_id'].nunique()}", "",
                  "--- Aggregate MRR on cross-family subset (context) ---"] + overall_lines + [""]

for cat in ["BLAST wins", "RRF wins", "Tie"]:
    sub = cross_family[cross_family["category"] == cat]
    summary_lines.append(f"--- {cat} (n={len(sub)}, {len(sub)/len(cross_family):.1%} of cross-family) ---")
    summary_lines.append(f"  involves nsLTP/Profilin: {sub['involves_target_fam'].mean():.2%}")
    summary_lines.append(f"  avg protein length: {sub['avg_len'].mean():.1f}")
    evid_dist = sub["evidence_level"].str.extract(r"^(Confirmed|Strong evidence|Suspected|Inferred)")[0]
    evid_dist = evid_dist.value_counts(normalize=True)
    summary_lines.append(f"  evidence split: {dict(evid_dist.round(3))}")
    top_pairs = sub["family_pair"].value_counts().head(8)
    summary_lines.append(f"  top family-pairs involved: {dict(top_pairs)}")
    seq_id_numeric = pd.to_numeric(
        sub["sequence_identity_pct"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    if seq_id_numeric.notna().sum() > 0:
        summary_lines.append(f"  sequence_identity_pct (where numeric, n={seq_id_numeric.notna().sum()}): "
                              f"mean={seq_id_numeric.mean():.1f}, median={seq_id_numeric.median():.1f}")
    else:
        summary_lines.append("  sequence_identity_pct: no numeric values in this group")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
