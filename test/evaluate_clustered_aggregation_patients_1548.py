"""
Testira ideju (1)+(3) iz diskusije 2026-09-02: umesto sabiranja RRF-K doprinosa
SVIH poznatih pozitiva pacijenta u jedan pool (trenutni mehanizam --
'jedna mreza'), grupisati poznate pozitive u klastere nezavisnih senzibilizacija
i uzeti MAX preko klastera:

    score(c|P) = max_i  sum_{p in C_i} 1/(K + r_p(c))     [klasterovano]
    score(c|P) = sum_{p in P} 1/(K + r_p(c))               [baseline, pooled]

Klasterovanje je NAMERNO leakage-safe: koristi ISKLJUCIVO poznate pozitive
(nikad sakriveni cilj), preko cosine slicnosti NJIHOVIH SOPSTVENIH ESM-2
embeddinga (union-find, prag tau). Sakriveni cilj nikad ne ucestvuje u
formiranju klastera -- max se uzima preko VEC formiranih klastera, ne
pogadja se kom klasteru sakriveni cilj pripada.

Prag tau=0.95 nije proizvoljan: gold (istinski cross-reaktivni) parovi imaju
cosine sim p10=0.93/p50=0.99, nasumicni parovi p90=0.95/p95=0.96 (test/
threshold provera, 2026-09-02) -- 0.95 sedi na "lakat" tacki gde vecina gold
parova (>=90%) prelazi prag a velika vecina nasumicnih parova (>=95%) ne.

Testira se i za MLP(hadamard)-650M i za BLAST kao osnovni signal, na tacno
istom 57-pacijentskom/176-trial kanonicnom skupu kao Deo A ablacije.

Izlaz:
    output/clustered_aggregation_1548_per_trial.csv
    output/clustered_aggregation_1548_summary.txt
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
from ml.pipeline.common.features import load_blast_matrices  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
EXISTING_BLAST = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
PER_TRIAL_OUTPUT = Path("/home/lana/ALERGRAF/output/clustered_aggregation_1548_per_trial.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/clustered_aggregation_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
N_PERM = 10000
N_BOOTSTRAP = 10000
TAU = 0.95  # klaster-prag, obrazlozenje u docstring-u

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset (ESM-2 650M)...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_pairs_clean = training_eligible_pairs(dataset.gold_pairs)
train_negatives = sample_negative_pairs(dataset.all_ids, len(train_pairs_clean) * NEG_PER_POS, SEED,
                                          dataset.positive_pair_set)

print("Treniram produkcioni MLP(hadamard)-650M...", flush=True)
mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED)
mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print("Trening gotov.", flush=True)

blast = load_blast_matrices(BLAST_MATRIX)

ranker = CrossReactivityRanker()
assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])

perm_blast = np.array([blast["id_to_index"].get(aid, -1) for aid in ranker.pool])
valid_blast_idx = np.where(perm_blast >= 0)[0]
blast_matrix_ranker_order = np.zeros((ranker.n_pool, ranker.n_pool), dtype=np.float32)
blast_matrix_ranker_order[np.ix_(valid_blast_idx, valid_blast_idx)] = \
    blast["score_matrix"][np.ix_(perm_blast[valid_blast_idx], perm_blast[valid_blast_idx])]

# Normalizovana embedding matrica u RANKER poretku, za klaster-kosinus (SAMO
# medju poznatim pozitivima, nikad ne dodiruje sakriveni cilj).
emb_ranker_order = dataset.embedding_matrix[perm_dataset_to_ranker]
_norms = np.linalg.norm(emb_ranker_order, axis=1, keepdims=True)
emb_normed = emb_ranker_order / np.clip(_norms, 1e-12, None)


def mlp_scores_in_ranker_order(aid):
    return mlp.score_all(aid)[perm_dataset_to_ranker]


def blast_scores_in_ranker_order(aid):
    return blast_matrix_ranker_order[ranker.id_to_index[aid]]


def cluster_positive_ids(positive_ids):
    """Union-find preko poznatih pozitiva SAMO, cosine(sopstveni embeddingi) >= TAU.
    Sakriveni cilj NIKAD ne ucestvuje ovde -- leakage-safe po konstrukciji."""
    n = len(positive_ids)
    if n <= 1:
        return [positive_ids]
    idxs = [ranker.id_to_index[aid] for aid in positive_ids]
    vecs = emb_normed[idxs]
    sim = vecs @ vecs.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= TAU:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(positive_ids[i])
    return list(groups.values())


def make_ranker(score_fn, mode):
    """mode: 'pooled' (baseline, trenutni mehanizam) ili 'clustered' (max preko klastera)."""
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

        if mode == "pooled":
            groups = [positive_ids]
        else:
            groups = cluster_positive_ids(positive_ids)

        cluster_scores = []
        for group in groups:
            combined = np.zeros(ranker.n_pool, dtype=np.float64)
            for aid in group:
                scores = score_fn(aid)
                order = np.argsort(scores)[::-1]
                ranks = np.empty(ranker.n_pool, dtype=np.int64)
                ranks[order] = np.arange(1, ranker.n_pool + 1)
                combined += 1.0 / (RRF_K + ranks)
            cluster_scores.append(combined)

        if mode == "pooled":
            final = cluster_scores[0]
        else:
            final = np.maximum.reduce(cluster_scores)

        for idx in exclude_idx:
            final[idx] = -np.inf

        order = np.argsort(final)[::-1]
        result = pd.DataFrame({
            "candidate_id": [ranker.pool[i] for i in order],
            "candidate_name": [ranker.id_to_name.get(ranker.pool[i], ranker.pool[i]) for i in order],
            "priority_score": final[order],
        })
        result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result
    return rank_for_patient


RANKERS = {
    "mlp_pooled": make_ranker(mlp_scores_in_ranker_order, "pooled"),
    "mlp_clustered": make_ranker(mlp_scores_in_ranker_order, "clustered"),
    "blast_pooled": make_ranker(blast_scores_in_ranker_order, "pooled"),
    "blast_clustered": make_ranker(blast_scores_in_ranker_order, "clustered"),
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

merged = dfs["mlp_pooled"][["patient_id", "hidden_protein", "true_result", "verification_status", "rr"]].rename(
    columns={"rr": "rr_mlp_pooled"})
for name in ("mlp_clustered", "blast_pooled", "blast_clustered"):
    merged = merged.merge(dfs[name][["patient_id", "hidden_protein", "rr"]].rename(columns={"rr": f"rr_{name}"}),
                            on=["patient_id", "hidden_protein"])
assert len(merged) == len(dfs["mlp_pooled"]), "Spajanje nije 1:1"
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


summary = ["=" * 100, f"Klasterovana (max preko klastera, tau={TAU}) vs pooled (sum) agregacija -- "
           f"57-pacijentski kanonicni skup", "=" * 100, ""]

for comp_label, col_a, name_a, col_b, name_b in [
    ("MLP clustered vs pooled", "rr_mlp_clustered", "clustered", "rr_mlp_pooled", "pooled"),
    ("BLAST clustered vs pooled", "rr_blast_clustered", "clustered", "rr_blast_pooled", "pooled"),
]:
    summary.append(f"### {comp_label} ###")
    summary.extend(run_all_tests(merged, comp_label, col_a, col_b, name_a, name_b))

summary_text = "\n".join(summary)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {PER_TRIAL_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
