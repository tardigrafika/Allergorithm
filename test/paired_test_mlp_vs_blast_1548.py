"""
ISPRAVAN upareni test: MLP(hadamard) SAM naspram BLAST SAM, na ISTIM
pacijentima/upitima -- za razliku od dva odvojena testa protiv permutovane
nulte hipoteze (koji dokazuju SAMO da svaki model pojedinacno nosi signal,
NE da je jedan bolji od drugog).

Tri testa, sva uparena po (patient_id, hidden_protein):
  1. Patient-level Wilcoxon signed-rank -- po pacijentu, MRR(MLP) - MRR(BLAST)
  2. Cluster-permutacija -- permutuje OZNAKU MODELA (koji je MLP, koji BLAST)
     UNUTAR svakog pacijenta (cuva par upita, ne pojedinacne vrednosti),
     testira da li je posmatrana razlika veca nego sto permutacija daje.
  3. Patient-level bootstrap -- resempluje PACIJENTE (ne pojedinacne upite),
     racuna CI za MRR(MLP) - MRR(BLAST).

Izlaz:
    output/paired_test_mlp_vs_blast_1548_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MLP_RAW = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
BLAST_RAW = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/paired_test_mlp_vs_blast_1548_summary.txt")

SEED = 42
N_PERM = 10000
N_BOOTSTRAP = 10000

mlp = pd.read_json(MLP_RAW)
blast = pd.read_json(BLAST_RAW)

mlp["rr"] = 1.0 / mlp["rank"]
blast["rr"] = 1.0 / blast["rank"]

merged = mlp[["patient_id", "hidden_protein", "true_result", "verification_status", "rr", "percentile"]].merge(
    blast[["patient_id", "hidden_protein", "rr", "percentile"]],
    on=["patient_id", "hidden_protein"], suffixes=("_mlp", "_blast"))
assert len(merged) == len(mlp) == len(blast), "Spajanje nije 1:1 -- proveri kljuceve"

summary_lines = ["=" * 80, "Upareni test: MLP(hadamard) SAM vs BLAST SAM (iste 54-pacijentske trials)",
                  "=" * 80, "", f"Ukupno uparenih upita: {len(merged)}, pacijenata: {merged['patient_id'].nunique()}",
                  ""]


def run_all_tests(sub, label):
    lines = [f"--- {label} (n={len(sub)} upita, {sub['patient_id'].nunique()} pacijenata) ---"]

    # 1) Patient-level Wilcoxon signed-rank na MRR(MLP)-MRR(BLAST) po pacijentu
    per_patient = sub.groupby("patient_id").agg(mrr_mlp=("rr_mlp", "mean"), mrr_blast=("rr_blast", "mean"))
    diffs = per_patient["mrr_mlp"] - per_patient["mrr_blast"]
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) >= 5:
        stat, pval = wilcoxon(diffs_nonzero)
        lines.append(f"  1) Patient-level Wilcoxon (MRR_MLP - MRR_BLAST, n={len(diffs_nonzero)} pacijenata "
                      f"sa razlikom != 0): mean diff={diffs.mean():+.4f}, p={pval:.4f} "
                      f"-- {'ZNACAJNO' if pval < 0.05 else 'nije znacajno'}")
    else:
        lines.append(f"  1) Patient-level Wilcoxon: n={len(diffs_nonzero)} < 5, test nepouzdan/nije izvrsen")

    # 2) Cluster-permutacija: permutuj OZNAKU MODELA unutar pacijenta
    rng = np.random.default_rng(SEED)
    observed = (sub["rr_mlp"] - sub["rr_blast"]).mean()
    patient_ids = sub["patient_id"].unique()
    perm_diffs = np.empty(N_PERM)
    sub_by_patient = {pid: g[["rr_mlp", "rr_blast"]].to_numpy() for pid, g in sub.groupby("patient_id")}
    for i in range(N_PERM):
        total, n = 0.0, 0
        for pid, arr in sub_by_patient.items():
            flip = rng.random() < 0.5  # ceo pacijent zajedno -- cuva within-patient strukturu
            a = arr if not flip else arr[:, ::-1]
            total += (a[:, 0] - a[:, 1]).sum()
            n += len(a)
        perm_diffs[i] = total / n
    p_perm = (np.abs(perm_diffs) >= np.abs(observed)).mean()
    lines.append(f"  2) Cluster-permutacija (permutuj MLP/BLAST oznaku unutar pacijenta, N={N_PERM}): "
                  f"observed mean(rr_mlp-rr_blast)={observed:+.4f}, p={p_perm:.4f} "
                  f"-- {'ZNACAJNO' if p_perm < 0.05 else 'nije znacajno'}")

    # 3) Patient-level bootstrap CI za MRR(MLP)-MRR(BLAST)
    rng2 = np.random.default_rng(SEED)
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng2.choice(patient_ids, size=len(patient_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on="patient_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["rr_mlp"], weights=w) - np.average(resampled["rr_blast"], weights=w)
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    sig_boot = (ci_lo > 0) or (ci_hi < 0)
    lines.append(f"  3) Patient-level bootstrap (N={N_BOOTSTRAP}, resample po pacijentu): "
                  f"mean diff={boot_diffs.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                  f"-- {'ZNACAJNO' if sig_boot else 'nije znacajno'}")
    lines.append("")
    return lines


for label, sub in [("SVI upiti", merged),
                    ("SAMO hard (full_text_verified)", merged[merged["verification_status"] == "full_text_verified"])]:
    summary_lines.extend(run_all_tests(sub, label))

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
