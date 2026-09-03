"""
Testira klasterovanu (max preko klastera nezavisnih senzibilizacija, tau=0.95)
naspram pooled (trenutni, sve poznate pozitive u jedan zbir) agregacije --
BAS na standalone slucajevima (Nadja, Lana) koji su namerno raznovrsniji
(vise nepovezanih mehanizama senzibilizacije) od 57-pacijentskog literaturnog
suite-a, gde je efekat ispao zanemarljiv (test/evaluate_clustered_aggregation_
patients_1548.py, 2026-09-02) jer vecina tih pacijenata vec ima samo JEDAN
mehanizam/klaster.

Isti leakage-safe klaster mehanizam: klasterovanje SAMO nad poznatim
pozitivima (cosine njihovih sopstvenih embeddinga >= tau), sakriveni cilj
nikad ne ucestvuje u formiranju klastera.

NAMERNO odvojeno od glavnog suite-a, bez agregatne statistike (n=2 pacijenta) --
samo per-trial poredjenje.

Izlaz: stdout + output/standalone_clustered_1548_per_trial.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
STANDALONE_CASES = Path("/home/lana/ALERGRAF/test/standalone_cases.json")
PER_TRIAL_OUTPUT = Path("/home/lana/ALERGRAF/output/standalone_clustered_1548_per_trial.csv")

SEED = 42
NEG_PER_POS = 10
TAU = 0.95

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

emb_ranker_order = dataset.embedding_matrix[perm_dataset_to_ranker]
_norms = np.linalg.norm(emb_ranker_order, axis=1, keepdims=True)
emb_normed = emb_ranker_order / np.clip(_norms, 1e-12, None)


def mlp_scores_in_ranker_order(aid):
    return mlp.score_all(aid)[perm_dataset_to_ranker]


def blast_scores_in_ranker_order(aid):
    return blast_matrix_ranker_order[ranker.id_to_index[aid]]


def cluster_positive_ids(positive_ids):
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
        groups = [positive_ids] if mode == "pooled" else cluster_positive_ids(positive_ids)

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

        final = cluster_scores[0] if mode == "pooled" else np.maximum.reduce(cluster_scores)
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
        return result, len(groups)
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


with open(STANDALONE_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} standalone pacijenata", flush=True)

records = []
for case in cases:
    pid = case["patient_id"]
    resolvable = []
    for c in case["components"]:
        if c["result"] not in ("positive", "negative"):
            continue
        resolved = resolve_protein(c["protein"])
        if resolved is None:
            continue
        resolvable.append({"json_name": c["protein"], "pool_name": resolved, "result": c["result"]})
    if len(resolvable) < 2:
        print(f"\n--- {pid}: preskoceno (< 2 resolvovane komponente) ---")
        continue

    print(f"\n--- {pid} ({len(resolvable)} komponenti) ---")
    for i, hidden in enumerate(resolvable):
        others = resolvable[:i] + resolvable[i + 1:]
        known_pos = [o["pool_name"] for o in others if o["result"] == "positive"]
        known_neg = [o["pool_name"] for o in others if o["result"] == "negative"]
        if not known_pos:
            continue

        row = {"patient_id": pid, "hidden_protein": hidden["pool_name"], "true_result": hidden["result"]}
        line = f"  Sakriven: {hidden['pool_name']:20s} ({hidden['result']:8s})"
        n_clusters_seen = None
        for model_name, rank_fn in RANKERS.items():
            result_df, n_clusters = rank_fn(known_pos, known_negative_names=known_neg)
            if "mlp" in model_name:
                n_clusters_seen = n_clusters
            match = result_df[result_df["candidate_name"] == hidden["pool_name"]]
            if len(match) == 0:
                continue
            rank = int(match.iloc[0]["rank"])
            n_cand = len(result_df)
            percentile = rank / n_cand * 100
            row[f"rank_{model_name}"] = rank
            row[f"percentile_{model_name}"] = percentile
            line += f"  |  {model_name}: top {percentile:.1f}%"
        line += f"   [broj klastera medju {len(known_pos)} poznatih: {n_clusters_seen}]"
        records.append(row)
        print(line)

df = pd.DataFrame(records)
df.to_csv(PER_TRIAL_OUTPUT, index=False)
print(f"\nSaved: {PER_TRIAL_OUTPUT}")
