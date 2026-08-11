"""
EXP-9: da li performanse opadaju sa smanjenjem pouzdanosti dokaza?

Tri benchmark nivoa (kumulativni):
  A) Confirmed + Strong evidence SAMO (najpouzdaniji dokaz)
  B) A + Suspected
  C) A + B + Inferred (family-level homology) -- sve, isto sto smo do sad
     koristile kao "pun" dataset

Koristi VEC SACUVANE rezultate iz rank_fusion_cosine_blast_foldseek_1548.py
(output/rank_fusion_1548_per_query.csv) - nema novog racunanja, samo presek
po evidence_level iz gold fajla. Direktno produbljuje rank-correlation nalaz
(cosine rho=0.191, RF+BLAST rho=0.204, p<1e-13) na jeziku MRR-a.

Izlaz:
    output/evidence_tier_benchmark_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

PER_QUERY = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/evidence_tier_benchmark_1548_summary.txt")

TOP_K = [1, 5, 10, 20]
METHODS = [("cosine_rank", "cosine"), ("blast_rank", "BLAST"),
           ("foldseektm_rank", "FoldseekTM"), ("rrf_rank", "RRF-3")]


def mrr_hits(sub, rank_col):
    ranks = sub[rank_col]
    mrr = (1.0 / ranks).mean()
    hits = {k: (ranks <= k).mean() for k in TOP_K}
    return mrr, hits


print("Loading data...")
df = pd.read_csv(PER_QUERY)
gold = pd.read_csv(GOLD)
gold = gold[["pair_id", "evidence_level"]].drop_duplicates(subset="pair_id")

merged = df.merge(gold, on="pair_id", how="left")
print(f"Total queries: {len(merged)}, unmatched evidence_level: {merged['evidence_level'].isna().sum()}")

is_confirmed_strong = merged["evidence_level"].str.startswith(("Confirmed", "Strong evidence"), na=False)
is_suspected = merged["evidence_level"].str.startswith("Suspected", na=False)
is_inferred = merged["evidence_level"].str.startswith("Inferred", na=False)

tier_a = merged[is_confirmed_strong]
tier_b = merged[is_confirmed_strong | is_suspected]
tier_c = merged  # everything (matches all prior "full dataset" results)

tiers = [
    ("A: Confirmed+Strong only", tier_a),
    ("B: A + Suspected", tier_b),
    ("C: A + B + Inferred (= full dataset, matches prior results)", tier_c),
]

summary_lines = [
    "=" * 70,
    "EXP-9: performance by evidence-tier benchmark (1548 dataset)",
    "=" * 70,
    "",
]

for tier_name, sub in tiers:
    summary_lines.append(f"--- {tier_name} --- (n={len(sub)} queries, {sub['pair_id'].nunique()} pairs)")
    for rank_col, label in METHODS:
        mrr, hits = mrr_hits(sub, rank_col)
        hits_str = "  ".join(f"Hits@{k}={hits[k]:.4f}" for k in TOP_K)
        summary_lines.append(f"  {label:12s} MRR={mrr:.4f}  {hits_str}")
    summary_lines.append("")

# focused cosine-vs-RRF trend across tiers, side by side
summary_lines.append("--- Trend: MRR by tier (cosine vs RRF-3) ---")
summary_lines.append(f"{'Tier':<45}{'cosine':<10}{'RRF-3':<10}")
for tier_name, sub in tiers:
    cos_mrr, _ = mrr_hits(sub, "cosine_rank")
    rrf_mrr, _ = mrr_hits(sub, "rrf_rank")
    summary_lines.append(f"{tier_name:<45}{cos_mrr:<10.4f}{rrf_mrr:<10.4f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
