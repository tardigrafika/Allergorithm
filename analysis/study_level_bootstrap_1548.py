"""
Study-level holdout/bootstrap -- mentorov predlog: da li su projektove
"znacajne" bootstrap CI tvrdnje (RRF-3 vs cosine, itd.) robusne kad se
resampling radi po IZVORU (reference kolona), ne po pojedinacnom paru?

Motivacija: paired bootstrap svuda u projektu (evidence_tier_bootstrap_1548.py,
graph_propagation_signal_1548.py, itd.) resampluje PO pair_id, tretirajuci
svaki par kao nezavisan uzorak. Ali VELIKI delovi dataseta dele ISTI izvor
(reference kolona) -- 1106/1922 parova (57.5%) potice od JEDNOG blanket
citata (Radauer & Breiteneder 2018 + Aalberse 2000, Inferred tier). Ako je
ta inferenciona logika sistematski pristrasna (u bilo kom pravcu), ona je
pristrasna za SVIH 1106 odjednom -- ne 1106 nezavisnih dokaza. Pair-level
bootstrap ovo tretira kao 1106 nezavisnih uzoraka i moze dati LAZNO usko
(preoptimisticno) CI, isti tip greske koji je vec uhvacen i ispravljen kod
real-world pacijentskog testa (naivni Mann-Whitney -> cluster-permutacija
jer trial-ovi nisu nezavisni).

Napomena o prirodi korelacije: Inferred blanket-citat NIJE "ista
eksperimentalna studija" u tradicionalnom smislu (nema 1106 stvarnih
eksperimenata) -- to je 1106 razlicitih bioloskih tvrdnji koje DELE ISTU
inferencionu logiku (familija-nivo homologija) i ISTI generisan citat. I
dalje predstavlja pravu (ne)nezavisnost koju bootstrap treba da uvazi --
ako je ta inferenciona logika pristrasna, pristrasna je za sve odjednom.

Koristi POSTOJECE rank_fusion_1548_per_query.csv (ista tacna evidencija
kao originalni evidence_tier_bootstrap_1548.py -- izoluje SAMO promenu u
resampling jedinici, ne mesa sa pitanjem "da li prosireni dataset menja
rezultat").

Izlaz:
    output/study_level_bootstrap_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

PER_QUERY = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/study_level_bootstrap_1548_summary.txt")

N_BOOTSTRAP = 2000
SEED = 42

df = pd.read_csv(PER_QUERY)
gold = pd.read_csv(GOLD)[["pair_id", "reference", "evidence_level"]].drop_duplicates(subset="pair_id")
merged = df.merge(gold, on="pair_id", how="inner")

n_studies = merged["reference"].nunique()
n_pairs = merged["pair_id"].nunique()
top_study_frac = merged["reference"].value_counts().iloc[0] / len(merged)

summary_lines = [
    "=" * 80, "Study-level bootstrap: RRF-3 vs cosine, resampling PO IZVORU (reference) "
    "umesto po paru", "=" * 80, "",
    f"Ukupno upita: {len(merged)}, parova: {n_pairs}, razlicitih izvora (studies): {n_studies}",
    f"Najveca studijska grupa (Inferred blanket-citat) drzi {top_study_frac:.1%} svih upita "
    f"-- ekstremna koncentracija, ocekivano SIROKO CI pod study-level resampling-om", "",
]


def paired_bootstrap(sub, rank_col, group_col, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    groups = sub[group_col].unique()
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        d = (np.average(1.0 / resampled["rrf_rank"], weights=w)
             - np.average(1.0 / resampled[rank_col], weights=w))
        deltas.append(d)
    return np.array(deltas)


for label, group_col in [("PO PARU (pair_id, postojeca metodologija)", "pair_id"),
                          ("PO IZVORU (reference, novo -- studijski nivo)", "reference")]:
    summary_lines.append(f"--- Resampling {label} ---")
    for rank_col, cmp_label in [("cosine_rank", "cosine"), ("blast_rank", "BLAST")]:
        deltas = paired_bootstrap(merged, rank_col, group_col, N_BOOTSTRAP, SEED)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        significant = (ci_lo > 0) or (ci_hi < 0)
        verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
        summary_lines.append(f"  RRF-3 vs {cmp_label:8s}: mean delta = {deltas.mean():+.4f}, "
                              f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")
    summary_lines.append("")

# -------------------------------------------------------
# Dodatno: study-level bootstrap SA IZUZETIM Inferred blanket-citatom (samo
# na direktnije potvrdjenim parovima) -- da li se nalaz drzi na "cistijem"
# podskupu gde je studijska nezavisnost realisticnija pretpostavka?
# -------------------------------------------------------
non_inferred = merged[merged["evidence_level"] != "Inferred (family-level homology)"]
summary_lines.append(f"--- Samo NE-Inferred parovi (n={len(non_inferred)} upita, "
                      f"{non_inferred['pair_id'].nunique()} parova, "
                      f"{non_inferred['reference'].nunique()} izvora), study-level resampling ---")
for rank_col, cmp_label in [("cosine_rank", "cosine"), ("blast_rank", "BLAST")]:
    deltas = paired_bootstrap(non_inferred, rank_col, "reference", N_BOOTSTRAP, SEED)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"  RRF-3 vs {cmp_label:8s}: mean delta = {deltas.mean():+.4f}, "
                          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
