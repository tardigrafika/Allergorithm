"""
EXP-9.1/9.2: paired bootstrap (cosine vs RRF-3, BLAST vs RRF-3) na sva tri
evidence-tier-a (A: Confirmed+Strong, B: +Suspected, C: sve).

Pitanje koje resava: da li je RRF-3 STATISTICKI razluciv od pojedinacnih
signala na svakom tier-u, posebno na Tier A gde je BLAST sam (0.2634)
nominalno ispred RRF-3 (0.2564) - da li je to signal ili sum?

Koristi vec sacuvane rezultate (rank_fusion_1548_per_query.csv), nema novog
racunanja.

Izlaz:
    output/evidence_tier_bootstrap_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

PER_QUERY = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/evidence_tier_bootstrap_1548_summary.txt")

N_BOOTSTRAP = 2000
SEED = 42

df = pd.read_csv(PER_QUERY)
gold = pd.read_csv(GOLD)[["pair_id", "evidence_level"]].drop_duplicates(subset="pair_id")
merged = df.merge(gold, on="pair_id", how="left")

is_confirmed_strong = merged["evidence_level"].str.startswith(("Confirmed", "Strong evidence"), na=False)
is_suspected = merged["evidence_level"].str.startswith("Suspected", na=False)

tiers = [
    ("A: Confirmed+Strong only", merged[is_confirmed_strong]),
    ("B: A + Suspected", merged[is_confirmed_strong | is_suspected]),
    ("C: full dataset", merged),
]

comparisons = [("cosine_rank", "cosine"), ("blast_rank", "BLAST")]

rng = np.random.default_rng(SEED)
summary_lines = [
    "=" * 70,
    "EXP-9.1/9.2: paired bootstrap, method vs RRF-3, by evidence tier",
    "=" * 70,
    "",
]

for tier_name, sub in tiers:
    summary_lines.append(f"--- {tier_name} (n={len(sub)} queries, {sub['pair_id'].nunique()} pairs) ---")
    pair_ids = sub["pair_id"].unique()
    for rank_col, label in comparisons:
        deltas = []
        for _ in range(N_BOOTSTRAP):
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            counts = pd.Series(sampled).value_counts()
            resampled = sub.merge(counts.rename("w"), left_on="pair_id", right_index=True)
            w = resampled["w"].to_numpy()
            d = (np.average(1.0 / resampled["rrf_rank"], weights=w)
                 - np.average(1.0 / resampled[rank_col], weights=w))
            deltas.append(d)
        deltas = np.array(deltas)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        frac_rrf_better = (deltas > 0).mean()
        mean_delta = deltas.mean()
        significant = (ci_lo > 0) or (ci_hi < 0)
        verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
        summary_lines.append(
            f"  RRF-3 vs {label:8s}: mean delta = {mean_delta:+.4f}, "
            f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], RRF-3 bolji u {frac_rrf_better:.1%} resample-ova -- {verdict}"
        )
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
