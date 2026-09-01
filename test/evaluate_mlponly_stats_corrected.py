"""
Ispravka statistike iz test/evaluate_test_cases.py -- Mann-Whitney U na 76
(ili 61) leave-one-out trials NIJE validan test, jer trials NISU nezavisni:
dolaze iz 33 pacijenta, vise trials po pacijentu deli isti "poznat profil"
i isti candidate pool. Efektivan N je blizu 33, ne 76 -- MWU na "ravnim"
trials-ima je anti-konzervativan (p-vrednost preterano optimisticna).

Ovde tri ispravke:
  1) PRIMARNA: cluster-permutaciona verzija testa -- permutuj pozitivno/
     negativno labele SAMO UNUTAR svakog pacijenta (cuva broj poz/neg po
     pacijentu i medju-pacijent varijaciju), preracunaj test-statistiku
     (mean percentile negativnih - mean percentile pozitivnih) B puta,
     p-vrednost = udeo permutacija >= posmatranoj vrednosti. Koristi SVE
     podatke (ne gubi trials koji nemaju par u istom pacijentu).
  2) SEKUNDARNA: patient-level Wilcoxon signed-rank -- samo pacijenti sa
     BAR JEDNIM hidden-pozitivnim I BAR JEDNIM hidden-negativnim trial-om,
     testira medijan po-pacijent razlike (median_neg - median_pos).
  3) SENSITIVITY: ponovi oba testa BEZ Mothes-Luksch kohorte (13/33
     pacijenata iz JEDNOG rada/laboratorije/metoda -- ne 13 nezavisnih
     "glasova").

Zahteva da je test/evaluate_test_cases.py vec pokrenut (koristi njegov
test/evaluation_results_raw.json izlaz).

Izlaz:
    test/evaluation_stats_corrected_summary.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RAW_RESULTS = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluate_mlponly_stats_corrected_summary.txt")
SEED = 42
N_PERM = 10000

df = pd.read_json(RAW_RESULTS)
df["is_mothesluksch"] = df["patient_id"].str.startswith("motheslucksch2017")

summary_lines = ["=" * 70, "Statisticka ispravka: cluster-permutacija umesto Mann-Whitney na 'ravnim' trials-ima",
                  "=" * 70, ""]


def cluster_permutation_test(sub_df, n_perm=N_PERM, seed=SEED):
    """Permutuj pozitivno/negativno labele UNUTAR svakog pacijenta (cuva
    marginalni broj poz/neg po pacijentu), test-statistika = mean(percentile|neg)
    - mean(percentile|pos), zelimo POZITIVNU vrednost (negativni imaju visi
    percentil = ispravno deprioritizovani)."""
    rng = np.random.default_rng(seed)
    pos_mask = sub_df["true_result"] == "positive"
    neg_mask = sub_df["true_result"] == "negative"
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None
    observed = sub_df.loc[neg_mask, "percentile"].mean() - sub_df.loc[pos_mask, "percentile"].mean()

    labels = sub_df["true_result"].to_numpy().copy()
    patient_ids = sub_df["patient_id"].to_numpy()
    percentiles = sub_df["percentile"].to_numpy()

    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = labels.copy()
        for pid in np.unique(patient_ids):
            idx = np.where(patient_ids == pid)[0]
            perm_labels[idx] = rng.permutation(labels[idx])
        p_mask = perm_labels == "positive"
        n_mask = perm_labels == "negative"
        if p_mask.sum() == 0 or n_mask.sum() == 0:
            perm_stats[i] = np.nan
            continue
        perm_stats[i] = percentiles[n_mask].mean() - percentiles[p_mask].mean()

    valid_perm = perm_stats[~np.isnan(perm_stats)]
    p_value = (valid_perm >= observed).mean()
    return observed, p_value, len(valid_perm)


def patient_level_wilcoxon(sub_df):
    """Samo pacijenti sa >=1 hidden-pozitivan I >=1 hidden-negativan trial."""
    per_patient_diff = []
    for pid, g in sub_df.groupby("patient_id"):
        pos = g.loc[g["true_result"] == "positive", "percentile"]
        neg = g.loc[g["true_result"] == "negative", "percentile"]
        if len(pos) == 0 or len(neg) == 0:
            continue
        per_patient_diff.append(neg.median() - pos.median())
    per_patient_diff = np.array(per_patient_diff)
    if len(per_patient_diff) < 5:
        return per_patient_diff, None
    stat, pval = wilcoxon(per_patient_diff, alternative="greater")
    return per_patient_diff, pval


for label, sub in [
    ("SVI 76 trials (hard + soft)", df),
    ("SAMO hard (full_text_verified)", df[df["verification_status"] == "full_text_verified"]),
    ("SVI 76, BEZ Mothes-Luksch kohorte (sensitivity)", df[~df["is_mothesluksch"]]),
    ("SAMO hard, BEZ Mothes-Luksch kohorte (sensitivity)",
     df[(df["verification_status"] == "full_text_verified") & (~df["is_mothesluksch"])]),
]:
    n_patients = sub["patient_id"].nunique()
    n_pos = (sub["true_result"] == "positive").sum()
    n_neg = (sub["true_result"] == "negative").sum()
    summary_lines.append(f"--- {label} ---")
    summary_lines.append(f"  n_trials={len(sub)} (pos={n_pos}, neg={n_neg}), n_pacijenata={n_patients}")

    perm_result = cluster_permutation_test(sub)
    if perm_result is None:
        summary_lines.append("  Cluster-permutacija: nedovoljno podataka (nema i pozitivnih i negativnih)")
    else:
        observed, pval, n_valid = perm_result
        sig = "ZNACAJNO" if pval < 0.05 else "nije znacajno"
        summary_lines.append(f"  Cluster-permutacija (N={N_PERM}, within-patient shuffle): "
                              f"observed diff={observed:+.2f}pp, p={pval:.4f} -- {sig}")

    per_patient_diff, wpval = patient_level_wilcoxon(sub)
    summary_lines.append(f"  Patient-level parovi (>=1 poz I >=1 neg hidden trial): n_pacijenata={len(per_patient_diff)}")
    if wpval is not None:
        sig = "ZNACAJNO" if wpval < 0.05 else "nije znacajno"
        summary_lines.append(f"  Wilcoxon signed-rank (median_neg - median_pos > 0): p={wpval:.4f} -- {sig}")
    else:
        summary_lines.append("  Wilcoxon: n<5 parova, test nije pouzdan/nije izvrsen")
    summary_lines.append("")

summary_lines.append("=" * 70)
summary_lines.append("NAPOMENA o graph-propagation nalazu (+0.006 MRR)")
summary_lines.append("=" * 70)
summary_lines.append("Taj nalaz (ml/graph_propagation_signal_1548.py, ml/weighted_rrf4_fusion_1548.py)")
summary_lines.append("je INTERNI -- racunat leave-one-edge-out na GOLD datasetu (cross_reactive_1548.csv,")
summary_lines.append("1537 kurianih parova), NEMA veze sa ovih 33 real-world pacijenata iz literature.")
summary_lines.append("Taj interni nalaz JESTE prosao bootstrap CI (2000 resample po pair_id): CI")
summary_lines.append("[+0.0037,+0.0123], znacajno. To je odvojena, vec ranije zavrsena provera --")
summary_lines.append("ova skripta ispravlja SAMO statistiku eksternog 33-pacijentskog test-suite-a.")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
