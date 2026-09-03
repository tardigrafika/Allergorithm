"""
MLP(hadamard) na ESM-2 3B (retunovan: standardize=True, isti fer config
potvrdjen na gold LOCO -- ml/loco_esm2_3b_retuned_mlp_hadamard_1548.py)
SAM (bez RRF fuzije), testiran na svih 57 CRD pacijenata, uporedjen sa
VEC POSTOJECIM BLAST SAM i MLP(hadamard)-650M SAM rezultatima (test/
evaluation_results_raw_blastonly.json / _mlponly.json -- NE racunaju se
ponovo, ista disciplina kao ESM-1b/3B gold LOCO eksperimenti).

Isti leave-one-out mehanizam (RRF-K suma preko poznatih pozitiva, JEDAN
signal) kao test/evaluate_mlp_only_vs_blast_only_patients_1548.py, samo
embedding izvor promenjen na embeddings_esm2_3b.pkl + retunovan config
(h32, standardize=True -- minimalna izmena od 650M config-a, isti
kandidat izabran kao "najcistiji" u gold LOCO potvrdi).

Nakon leave-one-out generisanja: DVA uparena testa (ista metodologija kao
test/paired_test_mlp_vs_blast_1548.py -- patient-level Wilcoxon,
cluster-permutacija, patient-level bootstrap):
  1. MLP(hadamard)-3B vs BLAST
  2. MLP(hadamard)-3B vs MLP(hadamard)-650M

Izlaz:
    test/evaluation_results_raw_mlp3bonly.json
    output/evaluate_mlp3b_patients_1548_summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/test")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.pkl")
METADATA_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
RAW_OUTPUT_MLP3B = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlp3bonly.json")
EXISTING_MLP650M = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
EXISTING_BLAST = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/evaluate_mlp3b_patients_1548_summary.txt")

for f in (EMBEDDINGS_3B, METADATA_3B, EXISTING_MLP650M, EXISTING_BLAST):
    if not f.exists():
        raise FileNotFoundError(f"{f} ne postoji.")

SEED = 42
NEG_PER_POS = 10
N_PERM = 10000
N_BOOTSTRAP = 10000

MLP_3B_PARAMS = dict(input_encoding="hadamard", standardize=True, hidden_dims=[32], dropout=[0.3],
                       learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                       max_epochs=300, patience=20, val_fraction=0.15)

print("Loading ESM-2 3B dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS_3B, METADATA_3B, GOLD)
train_pairs_clean = training_eligible_pairs(dataset.gold_pairs)
print(f"  Trening-podobnih (bez Inferred): {len(train_pairs_clean)}", flush=True)

train_negatives = sample_negative_pairs(dataset.all_ids, len(train_pairs_clean) * NEG_PER_POS, SEED,
                                          dataset.positive_pair_set)
print("\nTreniram produkcioni MLP(hadamard)-3B (h32, standardize=True)...", flush=True)
mlp = MLPPairClassifier(params=MLP_3B_PARAMS, seed=SEED)
mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print("Trening gotov.", flush=True)

print("\nUcitavam CrossReactivityRanker (samo za pool/name lookup)...", flush=True)
ranker = CrossReactivityRanker()

assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])


def mlp_scores_in_ranker_order(aid):
    raw_scores = mlp.score_all(aid)
    return raw_scores[perm_dataset_to_ranker]


def make_single_signal_ranker(score_fn):
    def rank_for_patient(known_positive_names, known_negative_names=None):
        def resolve(names):
            ids = []
            for name in names or []:
                a = ranker.name_to_id.get(name)
                if a is None or a not in ranker.id_to_index:
                    continue
                ids.append(a)
            return ids

        positive_ids = resolve(known_positive_names)
        negative_ids = resolve(known_negative_names)
        if not positive_ids:
            raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

        exclude_idx = {ranker.id_to_index[aid] for aid in positive_ids + negative_ids}
        combined = np.zeros(ranker.n_pool, dtype=np.float64)

        for aid in positive_ids:
            scores = score_fn(aid)
            order = np.argsort(scores)[::-1]
            ranks = np.empty(ranker.n_pool, dtype=np.int64)
            ranks[order] = np.arange(1, ranker.n_pool + 1)
            combined += 1.0 / (RRF_K + ranks)

        for idx in exclude_idx:
            combined[idx] = -np.inf

        order = np.argsort(combined)[::-1]
        result = pd.DataFrame({
            "candidate_id": [ranker.pool[i] for i in order],
            "candidate_name": [ranker.id_to_name.get(ranker.pool[i], ranker.pool[i]) for i in order],
            "priority_score": combined[order],
        })
        result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result
    return rank_for_patient


rank_mlp3b_only = make_single_signal_ranker(mlp_scores_in_ranker_order)

pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} pacijenata", flush=True)


def run_leave_one_out(rank_fn, label):
    records = []
    for case in cases:
        pid = case["patient_id"]
        verif_status = case["verification"]["status"]
        resolvable = []
        for c in case["components"]:
            if c["result"] not in ("positive", "negative"):
                continue
            resolved = resolve_protein(c["protein"])
            if resolved is None:
                continue
            resolvable.append({"json_name": c["protein"], "pool_name": resolved, "result": c["result"]})

        if len(resolvable) < 2:
            continue

        for i, hidden in enumerate(resolvable):
            others = resolvable[:i] + resolvable[i + 1:]
            known_pos = [o["pool_name"] for o in others if o["result"] == "positive"]
            known_neg = [o["pool_name"] for o in others if o["result"] == "negative"]
            if not known_pos:
                continue

            result_df = rank_fn(known_pos, known_negative_names=known_neg)
            row = result_df[result_df["candidate_name"] == hidden["pool_name"]]
            if len(row) == 0:
                continue
            rank = int(row.iloc[0]["rank"])
            n_cand = len(result_df)
            percentile = rank / n_cand * 100

            records.append({
                "patient_id": pid, "hidden_protein": hidden["pool_name"],
                "true_result": hidden["result"], "rank": rank, "n_candidates": n_cand,
                "percentile": percentile, "verification_status": verif_status,
            })
    df = pd.DataFrame(records)
    print(f"  {label}: {len(df)} trials", flush=True)
    return df


print("\n--- MLP(hadamard)-3B SAM ---", flush=True)
df_mlp3b = run_leave_one_out(rank_mlp3b_only, "MLP-3B-only")
df_mlp3b.to_json(RAW_OUTPUT_MLP3B, orient="records", indent=2)

# ---------------------------------------------------------------------------
# Upareni testovi: MLP-3B vs BLAST, MLP-3B vs MLP-650M -- PONOVO KORISCENI
# postojeci raw rezultati, ne racunaju se ponovo.
# ---------------------------------------------------------------------------
mlp3b = df_mlp3b.copy()
mlp3b["rr"] = 1.0 / mlp3b["rank"]

blast = pd.read_json(EXISTING_BLAST)
blast["rr"] = 1.0 / blast["rank"]
mlp650m = pd.read_json(EXISTING_MLP650M)
mlp650m["rr"] = 1.0 / mlp650m["rank"]

summary_lines = ["=" * 80, "MLP(hadamard)-3B (retunovan) SAM vs BLAST SAM vs MLP(hadamard)-650M SAM "
                  "-- 57-pacijentski suite", "=" * 80, ""]


def run_all_tests(sub, label, col_a, col_b, name_a, name_b):
    lines = [f"--- {label} (n={len(sub)} upita, {sub['patient_id'].nunique()} pacijenata) ---"]

    per_patient = sub.groupby("patient_id").agg(mrr_a=(col_a, "mean"), mrr_b=(col_b, "mean"))
    diffs = per_patient["mrr_a"] - per_patient["mrr_b"]
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) >= 5:
        stat, pval = wilcoxon(diffs_nonzero)
        lines.append(f"  1) Patient-level Wilcoxon (MRR_{name_a} - MRR_{name_b}, n={len(diffs_nonzero)} "
                      f"pacijenata sa razlikom != 0): mean diff={diffs.mean():+.4f}, p={pval:.4f} "
                      f"-- {'ZNACAJNO' if pval < 0.05 else 'nije znacajno'}")
    else:
        lines.append(f"  1) Patient-level Wilcoxon: n={len(diffs_nonzero)} < 5, test nepouzdan/nije izvrsen")

    rng = np.random.default_rng(SEED)
    observed = (sub[col_a] - sub[col_b]).mean()
    patient_ids = sub["patient_id"].unique()
    perm_diffs = np.empty(N_PERM)
    sub_by_patient = {pid: g[[col_a, col_b]].to_numpy() for pid, g in sub.groupby("patient_id")}
    for i in range(N_PERM):
        total, n = 0.0, 0
        for pid, arr in sub_by_patient.items():
            flip = rng.random() < 0.5
            a = arr if not flip else arr[:, ::-1]
            total += (a[:, 0] - a[:, 1]).sum()
            n += len(a)
        perm_diffs[i] = total / n
    p_perm = (np.abs(perm_diffs) >= np.abs(observed)).mean()
    lines.append(f"  2) Cluster-permutacija (permutuj {name_a}/{name_b} oznaku unutar pacijenta, N={N_PERM}): "
                  f"observed mean(rr_{name_a}-rr_{name_b})={observed:+.4f}, p={p_perm:.4f} "
                  f"-- {'ZNACAJNO' if p_perm < 0.05 else 'nije znacajno'}")

    rng2 = np.random.default_rng(SEED)
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
    lines.append(f"  3) Patient-level bootstrap (N={N_BOOTSTRAP}, resample po pacijentu): "
                  f"mean diff={boot_diffs.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                  f"-- {'ZNACAJNO' if sig_boot else 'nije znacajno'}")
    lines.append("")
    return lines


for comp_label, other_df, name_other in [("MLP-3B vs BLAST", blast, "blast"), ("MLP-3B vs MLP-650M", mlp650m, "650m")]:
    merged = mlp3b[["patient_id", "hidden_protein", "true_result", "verification_status", "rr", "percentile"]].merge(
        other_df[["patient_id", "hidden_protein", "rr", "percentile"]],
        on=["patient_id", "hidden_protein"], suffixes=("_3b", f"_{name_other}"))
    assert len(merged) == len(mlp3b) == len(other_df), f"Spajanje ({comp_label}) nije 1:1 -- proveri kljuceve"

    summary_lines.append(f"### {comp_label} ###")
    summary_lines.append(f"Ukupno uparenih upita: {len(merged)}, pacijenata: {merged['patient_id'].nunique()}")
    summary_lines.append("")
    for label, sub in [("SVI upiti", merged),
                        ("SAMO hard (full_text_verified)", merged[merged["verification_status"] == "full_text_verified"])]:
        summary_lines.extend(run_all_tests(sub, label, "rr_3b", f"rr_{name_other}", "3b", name_other))
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
