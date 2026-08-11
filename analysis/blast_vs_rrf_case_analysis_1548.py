"""
EXP-6.5: koje slucajeve BLAST resava a RRF ne (i obrnuto), na Tier A
(Confirmed+Strong - najpouzdaniji ground truth).

Kategorizacija po upitu:
  BLAST wins: blast_rank <= rrf_rank / 2 (BLAST bar 2x bolji rang)
  RRF wins:   rrf_rank <= blast_rank / 2
  Tie:        ni jedno ni drugo

Za svaku grupu gleda: familija (same vs cross-family), sequence_identity_pct,
duzina proteina, evidence_level, da li je nsLTP/Profilin ukljucen.

Koristi vec sacuvane rezultate (rank_fusion_1548_per_query.csv +
rrf_ablation_1548_per_query.csv za blast rank), nema novog racunanja.

Izlaz:
    output/blast_vs_rrf_case_analysis_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANK_FUSION = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/blast_vs_rrf_case_analysis_1548_summary.txt")

WIN_RATIO = 2.0

df = pd.read_csv(RANK_FUSION)
gold = pd.read_csv(GOLD)[["pair_id", "allergen_id_1", "allergen_id_2", "family_1", "family_2",
                           "evidence_level", "sequence_identity_pct"]].drop_duplicates(subset="pair_id")
clean = pd.read_csv(CLEAN_ALLERGENS)[["official_name", "sequence_length"]].drop_duplicates(subset="official_name")

merged = df.merge(gold, on="pair_id", how="left")

is_confirmed_strong = merged["evidence_level"].str.startswith(("Confirmed", "Strong evidence"), na=False)
tier_a = merged[is_confirmed_strong].copy()
print(f"Tier A queries: {len(tier_a)}")

# categorize
tier_a["category"] = "Tie"
tier_a.loc[tier_a["blast_rank"] <= tier_a["rrf_rank"] / WIN_RATIO, "category"] = "BLAST wins"
tier_a.loc[tier_a["rrf_rank"] <= tier_a["blast_rank"] / WIN_RATIO, "category"] = "RRF wins"

print(tier_a["category"].value_counts())

# add protein lengths (query + target, avg)
len_map = dict(zip(clean["official_name"], clean["sequence_length"]))
tier_a["len_1"] = tier_a["allergen_id_1"].map(len_map)
tier_a["len_2"] = tier_a["allergen_id_2"].map(len_map)
tier_a["avg_len"] = (tier_a["len_1"] + tier_a["len_2"]) / 2

tier_a["same_family"] = tier_a["family_1"] == tier_a["family_2"]

TARGET_FAMILIES = {"nsLTP", "Profilin"}
tier_a["involves_target_fam"] = tier_a["family_1"].isin(TARGET_FAMILIES) | tier_a["family_2"].isin(TARGET_FAMILIES)

summary_lines = ["=" * 70, "EXP-6.5: BLAST wins vs RRF wins vs Tie, Tier A case analysis", "=" * 70, ""]

for cat in ["BLAST wins", "RRF wins", "Tie"]:
    sub = tier_a[tier_a["category"] == cat]
    summary_lines.append(f"--- {cat} (n={len(sub)}, {len(sub)/len(tier_a):.1%} of Tier A) ---")
    summary_lines.append(f"  same-family fraction: {sub['same_family'].mean():.2%}")
    summary_lines.append(f"  involves nsLTP/Profilin: {sub['involves_target_fam'].mean():.2%}")
    summary_lines.append(f"  avg protein length: {sub['avg_len'].mean():.1f}")
    evid_dist = sub["evidence_level"].str.extract(r"^(Confirmed|Strong evidence)")[0].value_counts(normalize=True)
    summary_lines.append(f"  evidence split: {dict(evid_dist.round(3))}")
    top_fams = pd.concat([sub["family_1"], sub["family_2"]]).value_counts().head(5)
    summary_lines.append(f"  top families involved: {dict(top_fams)}")
    # sequence identity, parse the messy free-text field defensively
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
