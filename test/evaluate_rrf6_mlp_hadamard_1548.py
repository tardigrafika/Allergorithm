"""
RRF-6: RRF-4 + MLP(hadamard) kao dodatni GLOBALNI signal (ne familijski-
ogranicen kao LSE u RRF-5 -- MLP(hadamard) je pod LOCO tacno izjednacio
cosine preko CELOG dataseta, nema dokaza da je familijski-specifican, pa se
dodaje svuda, isto kao u ml/rrf_mlp_hadamard_fusion_1548.py testu).

Trening: training_eligible_pairs() (BEZ preostalih Inferred parova, odluka
2026-08-29) -- ISTA disciplina kao evaluate_rrf5_cleantrain_1548.py, da
provera bude poštena (ne poredimo "prljav MLP" protiv "cist LSE").

Testira se na ISTOM 49-pacijentskom test/test_cases.json, ISTA leave-one-out
+ cluster-permutacija/patient-level Wilcoxon metodologija, direktno uporedivo
sa RRF-4 baznom linijom i oba RRF-5 varijante (original, cleantrain).

Izlaz:
    test/evaluation_results_raw_rrf6.json
    test/evaluate_rrf6_mlp_hadamard_1548_summary.txt
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
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
RAW_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_rrf6.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluate_rrf6_mlp_hadamard_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
print(f"  {len(dataset.all_ids)} proteina, {len(dataset.gold_pairs)} gold parova", flush=True)

train_pairs_clean = training_eligible_pairs(dataset.gold_pairs)
print(f"  Trening-podobnih (bez Inferred): {len(train_pairs_clean)}", flush=True)

train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
train_ids |= set(dataset.all_ids)  # ceo pool dostupan za negative (produkcioni model, ne LOCO)
n_train_neg = len(train_pairs_clean) * NEG_PER_POS
train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED, dataset.positive_pair_set)

print(f"\nTreniram produkcioni MLP(hadamard) ({len(train_pairs_clean)} poz + {len(train_negatives)} neg)...",
      flush=True)
mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED)
mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print("Trening gotov.", flush=True)

print("\nUcitavam CrossReactivityRanker (RRF-4 osnova)...", flush=True)
ranker = CrossReactivityRanker()

# KRITICNO: dataset.all_ids (parquet red) i ranker.pool (sortiran) NEMAJU isti
# redosled (isti skup proteina, razlicit poredak) -- mlp.score_all() vraca
# skorove indeksirane po dataset.id_to_index. Bez ove permutacije, mlp_ranks
# bi bio tiho pogresno poravnat sa ranker.pool indeksima (razlicit protein na
# svakoj poziciji) -- uhvaceno PRE pokretanja direktnom proverom redosleda.
assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina izmedju dataset i ranker!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])


def mlp_scores_in_ranker_order(aid):
    raw_scores = mlp.score_all(aid)  # indeksirano po dataset.id_to_index
    return raw_scores[perm_dataset_to_ranker]  # preslikano u ranker.pool redosled


def rank_for_patient_rrf6(known_positive_names, known_negative_names=None):
    def resolve(names):
        ids = []
        for name in names or []:
            aid = ranker.name_to_id.get(name)
            if aid is None or aid not in ranker.id_to_index:
                continue
            ids.append(aid)
        return ids

    positive_ids = resolve(known_positive_names)
    negative_ids = resolve(known_negative_names)
    if not positive_ids:
        raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

    exclude_idx = {ranker.id_to_index[aid] for aid in positive_ids + negative_ids}
    combined = np.zeros(ranker.n_pool, dtype=np.float64)

    for aid in positive_ids:
        idx = ranker.id_to_index[aid]
        score_vec = ranker._rrf3_score_vector(idx)
        order = np.argsort(score_vec)[::-1]
        ranks = np.empty(ranker.n_pool, dtype=np.int64)
        ranks[order] = np.arange(1, ranker.n_pool + 1)
        combined += 1.0 / (RRF_K + ranks)

        # RRF-6 dodatak: MLP(hadamard) skor, GLOBALNO (svi poznati pozitivi)
        mlp_scores = mlp_scores_in_ranker_order(aid)
        mlp_order = np.argsort(mlp_scores)[::-1]
        mlp_ranks = np.empty(ranker.n_pool, dtype=np.int64)
        mlp_ranks[mlp_order] = np.arange(1, ranker.n_pool + 1)
        combined += 1.0 / (RRF_K + mlp_ranks)

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


pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} pacijenata za RRF-6 evaluaciju", flush=True)

records = []
skipped_no_positive_left = 0

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
            skipped_no_positive_left += 1
            continue

        result_df = rank_for_patient_rrf6(known_pos, known_negative_names=known_neg)
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
    if len(records) % 20 == 0 and len(records) > 0:
        print(f"  ...{len(records)} trials obradjeno", flush=True)

df = pd.DataFrame(records)
print(f"\nLeave-one-out trials (RRF-6): {len(df)}", flush=True)
df.to_json(RAW_OUTPUT, orient="records", indent=2)
print(f"Saved: {RAW_OUTPUT}", flush=True)

summary_lines = ["=" * 70, "RRF-6 (RRF-4 + MLP(hadamard), cist trening) -- leave-one-out rezultati",
                  "=" * 70, "", f"Ukupno trials: {len(df)}", ""]
hard = df[df["verification_status"] == "full_text_verified"]
for label, sub in [("SVI", df), ("HARD (full_text_verified)", hard)]:
    pos = sub[sub["true_result"] == "positive"]["percentile"]
    neg = sub[sub["true_result"] == "negative"]["percentile"]
    summary_lines.append(f"--- {label} ---")
    summary_lines.append(f"  Pozitivne mete (n={len(pos)}): medijan percentil = {pos.median():.1f}%")
    summary_lines.append(f"  Negativne mete (n={len(neg)}): medijan percentil = {neg.median():.1f}%")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"Saved: {SUMMARY_OUTPUT}")
