"""
Paired bootstrap CI za MI/hypergraph LSE-pooling rezultat (mi_lse_pooling_1548.py),
ISTA metodologija kao ml/graph_propagation_signal_1548.py -- 2000 resample-ova
po pair_id (ne po upitu, jer su oba smera istog para korelisana), delta MRR
distribucija, 95% percentile CI.

Ulaz: output/mi_lse_pooling_1548_per_query.csv (vec sacuvan iz prethodnog run-a)
Izlaz: output/mi_lse_bootstrap_ci_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

PER_QUERY = Path("/home/lana/ALERGRAF/output/mi_lse_pooling_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mi_lse_bootstrap_ci_1548_summary.txt")
SEED = 42
N_BOOTSTRAP = 2000

df = pd.read_csv(PER_QUERY)
rng = np.random.default_rng(SEED)

summary_lines = ["=" * 80, "MI/hypergraph LSE-pooling -- paired bootstrap CI (po pair_id, 2000 resample)",
                  "=" * 80, ""]

for scope, sub in [("UKUPNO (nsLTP+Profilin+PR-10)", df)] + [(fam, df[df["family"] == fam])
                                                                for fam in ["nsLTP", "Profilin", "PR-10"]]:
    pair_ids = sub["pair_id"].unique()
    deltas = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on="pair_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["lse_rr"], weights=w) - np.average(resampled["cosine_rr"], weights=w)
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    frac_better = (deltas > 0).mean()
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"{scope} (n={len(sub)} upita, {len(pair_ids)} parova): "
                          f"mean delta={deltas.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
                          f"LSE bolje u {frac_better:.1%} resample-ova -- {verdict}")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
