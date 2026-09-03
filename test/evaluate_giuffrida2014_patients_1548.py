"""
Rezultati SAMO za 11 novododatih Giuffrida et al. 2014 pacijenata
(test_cases.json, patient_id prefiks "giuffrida2014_pt") -- MLP(hadamard)-650M
vs BLAST, isti leave-one-out mehanizam i isti produkcioni config kao
test/evaluate_mlp_only_vs_blast_only_patients_1548.py. Ne racuna agregatnu
statistiku za ceo 68-pacijentski suite (to je zaseban korak/odluka kada
korisnica to zatrazi -- ovde samo per-trial ispis za nove pacijente).
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
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
PER_TRIAL_OUTPUT = Path("/home/lana/ALERGRAF/output/giuffrida2014_evaluation_1548_per_trial.csv")

SEED = 42
NEG_PER_POS = 10

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


def mlp_scores_in_ranker_order(aid):
    return mlp.score_all(aid)[perm_dataset_to_ranker]


def blast_scores_in_ranker_order(aid):
    idx = ranker.id_to_index[aid]
    return blast_matrix_ranker_order[idx]


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


RANKERS = {
    "mlp_hadamard_650m": make_single_signal_ranker(mlp_scores_in_ranker_order),
    "blast": make_single_signal_ranker(blast_scores_in_ranker_order),
}

pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    all_cases = json.load(f)
cases = [c for c in all_cases if c["patient_id"].startswith("giuffrida2014_pt")]
print(f"\n{len(cases)} Giuffrida 2014 pacijenata pronadjeno (od ukupno {len(all_cases)})", flush=True)

records = []
print(f"\n{'='*100}")
for case in cases:
    pid = case["patient_id"]
    resolvable = []
    for c in case["components"]:
        if c["result"] not in ("positive", "negative"):
            continue
        resolved = resolve_protein(c["protein"])
        if resolved is None:
            print(f"  [{pid}] NEREZOLVOVANO: '{c['protein']}'")
            continue
        resolvable.append({"json_name": c["protein"], "pool_name": resolved, "result": c["result"]})

    known_summary = ", ".join(f"{r['json_name']}={r['result']}" for r in resolvable)
    print(f"\n--- {pid} --- ({known_summary})")
    for i, hidden in enumerate(resolvable):
        others = resolvable[:i] + resolvable[i + 1:]
        known_pos = [o["pool_name"] for o in others if o["result"] == "positive"]
        known_neg = [o["pool_name"] for o in others if o["result"] == "negative"]
        if not known_pos:
            continue

        line = f"  Sakriven: {hidden['json_name']:10s}->{hidden['pool_name']:15s} ({hidden['result']:8s})"
        row = {"patient_id": pid, "hidden_json_name": hidden["json_name"], "hidden_pool_name": hidden["pool_name"],
               "true_result": hidden["result"]}
        skipped = False
        for model_name, rank_fn in RANKERS.items():
            result_df = rank_fn(known_pos, known_negative_names=known_neg)
            match = result_df[result_df["candidate_name"] == hidden["pool_name"]]
            if len(match) == 0:
                skipped = True
                continue
            rank = int(match.iloc[0]["rank"])
            n_cand = len(result_df)
            percentile = rank / n_cand * 100
            row[f"rank_{model_name}"] = rank
            row[f"percentile_{model_name}"] = percentile
            line += f"  |  {model_name}: rank {rank}/{n_cand} (top {percentile:.1f}%)"
        if skipped:
            line += "  [PRESKOCENO -- sakriveni protein isti pool-zapis kao poznati (Pen a1/Pen m1 kolizija)]"
        else:
            records.append(row)
        print(line)

df = pd.DataFrame(records)
df.to_csv(PER_TRIAL_OUTPUT, index=False)
print(f"\n{'='*100}")
print(f"Validnih proba (bez preskocenih): {len(df)}")
if len(df):
    for model_name in RANKERS:
        col = f"percentile_{model_name}"
        pos = df[df["true_result"] == "positive"][col]
        neg = df[df["true_result"] == "negative"][col]
        print(f"\n{model_name}:")
        if len(pos):
            print(f"  Pozitivni ({len(pos)}): medijan percentil = {pos.median():.1f}%")
        if len(neg):
            print(f"  Negativni ({len(neg)}): medijan percentil = {neg.median():.1f}%")
print(f"\nSaved: {PER_TRIAL_OUTPUT}")
