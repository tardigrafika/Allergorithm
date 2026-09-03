"""
Deo C ablacije (rad/ablacioni_test.md): leave-one-component-out na arhitekturi
MLP(hadamard)-650M, svaka komponenta NEZAVISNO zamenjena trivijalnijom
alternativom (ostale ostaju iste kao produkcioni baseline), testirano na
istom kanonicnom 57-pacijentskom/176-trial skupu kao Deo A/B.

  C.1 -- reprezentacija:  ESM-2 embedding (1280-dim)  ->  aminokiselinski
          sastav (20-dim frekvencijski vektor po proteinu, iz FASTA
          sekvence, output/clean_allergens.csv). I dalje Hadamard produkt +
          MLP, samo je ULAZNI VEKTOR PROTEINA trivijalniji.
  C.2 -- kombinovanje:    Hadamard produkt (u*v)  ->  apsolutna razlika
          (|u-v|), ESM-2 embedding i MLP arhitektura ostaju isti. (LOCO
          verzija ovoga vec postoji i dosledno je losija od cosine-a --
          ovo je prvi put da se testira NA PACIJENTIMA.)
  C.3 -- model:           MLP (skriveni sloj [32], nelinearan)  ->
          logisticka regresija (hidden_dims=[], JEDAN linearni sloj,
          identican Hadamard ulaz). PairMLP sa praznim hidden_dims se
          svodi tacno na jedan nn.Linear(input_dim,1) -- logisticka
          regresija trenirana gradient descent-om, ne zatvorena forma, ali
          funkcionalno identicna.

Sve tri porede se protiv ISTOG produkcionog baseline-a (MLP-hadamard-650M,
standardize=False, hidden_dims=[32]), treniran ovde iznova radi
samodovoljnosti skripte.

Izlaz:
    output/component_ablation_1548_per_trial.csv
    output/component_ablation_1548_summary.txt
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

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
EXISTING_BLAST = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
PER_TRIAL_OUTPUT = Path("/home/lana/ALERGRAF/output/component_ablation_1548_per_trial.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/component_ablation_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
N_PERM = 10000
N_BOOTSTRAP = 10000
AA20 = "ACDEFGHIKLMNPQRSTVWY"

BASE_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                     learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                     max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset (ESM-2 650M)...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_pairs_clean = training_eligible_pairs(dataset.gold_pairs)
train_negatives = sample_negative_pairs(dataset.all_ids, len(train_pairs_clean) * NEG_PER_POS, SEED,
                                          dataset.positive_pair_set)

# ---------------------------------------------------------------------------
# C.1: aminokiselinski sastav (20-dim), poravnat sa dataset.all_ids poretkom.
# ---------------------------------------------------------------------------
allergens = pd.read_csv(ALLERGENS)
seq_by_id = dict(zip(allergens["allergen_id"], allergens["fasta_sequence"]))


def aa_composition(seq):
    seq = str(seq) if pd.notna(seq) else ""
    vec = np.zeros(20, dtype=np.float64)
    if not seq:
        return vec
    for ch in seq:
        idx = AA20.find(ch)
        if idx >= 0:
            vec[idx] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec


aa_matrix = np.array([aa_composition(seq_by_id.get(aid)) for aid in dataset.all_ids], dtype=np.float64)
n_missing = sum(1 for aid in dataset.all_ids if not str(seq_by_id.get(aid, "")).strip())
print(f"  AA-composition matrica: {aa_matrix.shape}, {n_missing} proteina bez sekvence (nula-vektor)", flush=True)

print("\nTreniram BASELINE (ESM-2 + hadamard + MLP)...", flush=True)
mlp_baseline = MLPPairClassifier(params=BASE_PARAMS, seed=SEED)
mlp_baseline.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print(f"  Gotovo. best_val_auc={mlp_baseline.best_val_auc:.4f}", flush=True)

print("\nTreniram C.1 (AA-sastav + hadamard + MLP)...", flush=True)
mlp_c1 = MLPPairClassifier(params=BASE_PARAMS, seed=SEED)
mlp_c1.fit(train_pairs_clean, train_negatives, aa_matrix, dataset.id_to_index)
print(f"  Gotovo. best_val_auc={mlp_c1.best_val_auc:.4f}", flush=True)

print("\nTreniram C.2 (ESM-2 + absdiff + MLP)...", flush=True)
c2_params = {**BASE_PARAMS, "input_encoding": "absdiff", "standardize": True}
mlp_c2 = MLPPairClassifier(params=c2_params, seed=SEED)
mlp_c2.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print(f"  Gotovo. best_val_auc={mlp_c2.best_val_auc:.4f}", flush=True)

print("\nTreniram C.3 (ESM-2 + hadamard + logisticka regresija)...", flush=True)
c3_params = {**BASE_PARAMS, "hidden_dims": [], "dropout": []}
mlp_c3 = MLPPairClassifier(params=c3_params, seed=SEED)
mlp_c3.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print(f"  Gotovo. best_val_auc={mlp_c3.best_val_auc:.4f}", flush=True)

ranker = CrossReactivityRanker()
assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])


def make_score_fn(model):
    def f(aid):
        return model.score_all(aid)[perm_dataset_to_ranker]
    return f


def make_ranker(score_fn):
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


RANKERS = {
    "baseline": make_ranker(make_score_fn(mlp_baseline)),
    "c1_aacomp": make_ranker(make_score_fn(mlp_c1)),
    "c2_absdiff": make_ranker(make_score_fn(mlp_c2)),
    "c3_linear": make_ranker(make_score_fn(mlp_c3)),
}

pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    all_cases = json.load(f)

canonical = pd.read_json(EXISTING_BLAST)[["patient_id", "hidden_protein"]].drop_duplicates()
canonical_keys = set(zip(canonical["patient_id"], canonical["hidden_protein"]))
canonical_patients = set(canonical["patient_id"])
cases = [c for c in all_cases if c["patient_id"] in canonical_patients]
print(f"\n{len(cases)} pacijenata (kanonicni 57-pacijentski skup)", flush=True)


def run_leave_one_out(rank_fn):
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
            if (pid, hidden["pool_name"]) not in canonical_keys:
                continue
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
            records.append({"patient_id": pid, "hidden_protein": hidden["pool_name"], "true_result": hidden["result"],
                             "rank": rank, "n_candidates": n_cand, "percentile": rank / n_cand * 100,
                             "verification_status": verif_status})
    return pd.DataFrame(records)


dfs = {}
for name, rank_fn in RANKERS.items():
    print(f"--- {name} ---", flush=True)
    dfs[name] = run_leave_one_out(rank_fn)
    print(f"  {len(dfs[name])} trials", flush=True)
    dfs[name]["rr"] = 1.0 / dfs[name]["rank"]

merged = dfs["baseline"][["patient_id", "hidden_protein", "true_result", "verification_status", "rr"]].rename(
    columns={"rr": "rr_baseline"})
for name in ("c1_aacomp", "c2_absdiff", "c3_linear"):
    merged = merged.merge(dfs[name][["patient_id", "hidden_protein", "rr"]].rename(columns={"rr": f"rr_{name}"}),
                            on=["patient_id", "hidden_protein"])
assert len(merged) == len(dfs["baseline"]), "Spajanje nije 1:1"
merged.to_csv(PER_TRIAL_OUTPUT, index=False)


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


summary = ["=" * 100, "Deo C: komponentna ablacija (leave-one-component-out) -- 57-pacijentski kanonicni skup",
           "=" * 100, "",
           f"best_val_auc: baseline={mlp_baseline.best_val_auc:.4f}  c1_aacomp={mlp_c1.best_val_auc:.4f}  "
           f"c2_absdiff={mlp_c2.best_val_auc:.4f}  c3_linear={mlp_c3.best_val_auc:.4f}", ""]

for comp_label, col_a, name_a in [
    ("C.1 (AA-sastav umesto ESM-2) vs baseline", "rr_c1_aacomp", "c1"),
    ("C.2 (absdiff umesto hadamard) vs baseline", "rr_c2_absdiff", "c2"),
    ("C.3 (linearni model umesto MLP-a) vs baseline", "rr_c3_linear", "c3"),
]:
    summary.append(f"### {comp_label} ###")
    summary.extend(run_all_tests(merged, comp_label, col_a, "rr_baseline", name_a, "baseline"))

summary_text = "\n".join(summary)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {PER_TRIAL_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
