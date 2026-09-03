"""
Deo A ablacione studije (rad/ablacioni_test.md): svodi vec izracunate weighted-
evidence i strict-evidence rezultate (koji su racunati na 68 pacijenata, posle
Giuffrida prosirenja) na TACNO isti (patient_id, hidden_protein) skup od 176
proba / 57 pacijenata koji vec stoji kao headline broj u radu (test/evaluation_
results_raw_blastonly.json je autoritativan izvor tog skupa kljuceva -- isti
trik kao test/evaluate_cosine_only_patients_1548.py). NE trenira ponovo
nijedan model -- samo filtrira vec postojece per-trial CSV-ove i ponovo racuna
uparene testove na podskupu.

Izlaz: stdout (summary teksta), koji se rucno prepisuje u rad/ablacioni_test.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 42
N_PERM = 10000
N_BOOTSTRAP = 10000

EXISTING_BLAST = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
WEIGHTED_CSV = Path("/home/lana/ALERGRAF/output/weighted_evidence_mlp_1548_per_trial.csv")
STRICT_CSV = Path("/home/lana/ALERGRAF/output/strict_evidence_mlp_1548_per_trial.csv")

canonical = pd.read_json(EXISTING_BLAST)[["patient_id", "hidden_protein"]].drop_duplicates()
print(f"Kanonicni skup (headline n=176/54): {len(canonical)} kljuceva, "
      f"{canonical['patient_id'].nunique()} pacijenata", flush=True)


def restrict(df):
    before = len(df)
    out = df.merge(canonical, on=["patient_id", "hidden_protein"], how="inner")
    print(f"  {before} -> {len(out)} trial-ova posle filtriranja na 57-pacijentski skup", flush=True)
    return out


weighted = restrict(pd.read_csv(WEIGHTED_CSV))
strict = restrict(pd.read_csv(STRICT_CSV))


def run_all_tests(sub, label, col_a, col_b, name_a, name_b):
    lines = [f"--- {label} (n={len(sub)} upita, {sub['patient_id'].nunique()} pacijenata) ---"]
    per_patient = sub.groupby("patient_id").agg(mrr_a=(col_a, "mean"), mrr_b=(col_b, "mean"))
    diffs = per_patient["mrr_a"] - per_patient["mrr_b"]
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) >= 5:
        stat, pval = wilcoxon(diffs_nonzero)
        lines.append(f"  1) Patient-level Wilcoxon (MRR_{name_a}-MRR_{name_b}, n={len(diffs_nonzero)}): "
                      f"mean diff={diffs.mean():+.4f}, p={pval:.4f} "
                      f"-- {'ZNACAJNO' if pval < 0.05 else 'nije znacajno'}")
    else:
        lines.append(f"  1) Patient-level Wilcoxon: n={len(diffs_nonzero)} < 5, nepouzdan")

    rng = np.random.default_rng(SEED)
    observed = (sub[col_a] - sub[col_b]).mean()
    sub_by_patient = {pid: g[[col_a, col_b]].to_numpy() for pid, g in sub.groupby("patient_id")}
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        total, n = 0.0, 0
        for pid, arr in sub_by_patient.items():
            flip = rng.random() < 0.5
            a = arr if not flip else arr[:, ::-1]
            total += (a[:, 0] - a[:, 1]).sum()
            n += len(a)
        perm_diffs[i] = total / n
    p_perm = (np.abs(perm_diffs) >= np.abs(observed)).mean()
    lines.append(f"  2) Cluster-permutacija (N={N_PERM}): observed mean(rr_{name_a}-rr_{name_b})={observed:+.4f}, "
                  f"p={p_perm:.4f} -- {'ZNACAJNO' if p_perm < 0.05 else 'nije znacajno'}")

    rng2 = np.random.default_rng(SEED)
    patient_ids = sub["patient_id"].unique()
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng2.choice(patient_ids, size=len(patient_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on="patient_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled[col_a], weights=w) - np.average(resampled[col_b], weights=w)
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    sig_boot = (ci_lo > 0) or (ci_hi < 0)
    lines.append(f"  3) Patient-level bootstrap (N={N_BOOTSTRAP}): mean diff={boot_diffs.mean():+.4f}, "
                  f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {'ZNACAJNO' if sig_boot else 'nije znacajno'}")
    lines.append("")
    return lines


CROWDED = {"nsLTP", "profilin", "PR-10"}

summary = ["=" * 100, "Deo A: weighted/strict-evidence preracunato na TACNO 57-pacijentski (n=176) skup",
           "=" * 100, ""]

# per-family breakdown (57-only)
for name, df, col in [("WEIGHTED", weighted, "percentile_weighted"), ("STRICT", strict, "percentile_strict")]:
    summary.append(f"### {name} -- per-familija (57-pacijentski skup) ###")
    for fam in sorted(df["protein_family"].dropna().unique()):
        sub = df[df["protein_family"] == fam]
        tag = " [CROWDED]" if fam in CROWDED else ""
        line = [f"  {fam}{tag} (n={len(sub)}):"]
        for res in ("positive", "negative"):
            rsub = sub[sub["true_result"] == res]
            if len(rsub) == 0:
                continue
            line.append(f" {res}={rsub[col].median():.1f}%(baseline={rsub['percentile_baseline'].median():.1f}%)")
        summary.append("".join(line))
    summary.append("")

for comp_label, df, col_a, name_a, col_b, name_b in [
    ("WEIGHTED vs Baseline", weighted, "rr_weighted", "weighted", "rr_baseline", "baseline"),
    ("WEIGHTED vs BLAST", weighted, "rr_weighted", "weighted", "rr_blast", "blast"),
    ("STRICT vs Baseline", strict, "rr_strict", "strict", "rr_baseline", "baseline"),
    ("STRICT vs BLAST", strict, "rr_strict", "strict", "rr_blast", "blast"),
]:
    summary.append(f"### {comp_label} (57-pacijentski skup) ###")
    summary.extend(run_all_tests(df, comp_label, col_a, col_b, name_a, name_b))

for comp_label, df, col_a, name_a, col_b, name_b in [
    ("WEIGHTED vs Baseline, SAMO crowded", weighted[weighted["family_crowding"]],
     "rr_weighted", "weighted", "rr_baseline", "baseline"),
    ("STRICT vs Baseline, SAMO crowded", strict[strict["family_crowding"]],
     "rr_strict", "strict", "rr_baseline", "baseline"),
]:
    summary.append(f"### {comp_label} ###")
    summary.extend(run_all_tests(df, comp_label, col_a, col_b, name_a, name_b))

summary_text = "\n".join(summary)
print("\n" + summary_text)
with open("/home/lana/ALERGRAF/output/ablation_partA_57patients_summary.txt", "w") as f:
    f.write(summary_text + "\n")
print("\nSaved: /home/lana/ALERGRAF/output/ablation_partA_57patients_summary.txt")
