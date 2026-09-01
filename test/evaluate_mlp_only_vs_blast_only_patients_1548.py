"""
Dva DODATNA pacijentska rankera, oba BEZ RRF fuzije (za direktno poredjenje
sa RRF-6 i sa dosadasnjim BLAST-vs-MLP LOCO nalazom, ali sada na pravim
pacijentima):
  1. MLP(hadamard) SAM -- da li embedding-based klasifikator sam po sebi
     radi bolje na pacijentima nego u LOCO testu na gold datasetu (gde je
     bio izjednacen sa BLAST-om)?
  2. BLAST SAM -- referentna tacka bez ikakvog embeddinga, da RRF-6 (vec
     testiran protiv RRF-4) moze da se uporedi i direktno protiv NAJPROSTIJE
     moguce bazne linije.

Isti leave-one-out mehanizam kao CrossReactivityRanker.rank_for_patient()
(RRF-K suma preko poznatih pozitiva), ali sa SAMO JEDNIM signalom po
rankeru (ne RRF-3+X). MLP(hadamard) treniran na training_eligible_pairs()
(cist trening, ista disciplina kao RRF-6/RRF-4-MLP LOCO test).

Testira se na test/test_cases.json (54 pacijenta), ista leave-one-out +
cluster-permutacija/patient-level Wilcoxon metodologija.

Izlaz:
    test/evaluation_results_raw_mlponly.json
    test/evaluation_results_raw_blastonly.json
    test/evaluate_mlp_only_vs_blast_only_1548_summary.txt
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
RAW_OUTPUT_MLP = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
RAW_OUTPUT_BLAST = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluate_mlp_only_vs_blast_only_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_pairs_clean = training_eligible_pairs(dataset.gold_pairs)
print(f"  Trening-podobnih (bez Inferred): {len(train_pairs_clean)}", flush=True)

train_negatives = sample_negative_pairs(dataset.all_ids, len(train_pairs_clean) * NEG_PER_POS, SEED,
                                          dataset.positive_pair_set)
print(f"\nTreniram produkcioni MLP(hadamard)...", flush=True)
mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED)
mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print("Trening gotov.", flush=True)

blast = load_blast_matrices(BLAST_MATRIX)

print("\nUcitavam CrossReactivityRanker (samo za pool/name lookup)...", flush=True)
ranker = CrossReactivityRanker()

assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])

perm_blast = np.array([blast["id_to_index"].get(aid, -1) for aid in ranker.pool])
valid_blast_idx = np.where(perm_blast >= 0)[0]
blast_matrix_ranker_order = np.zeros((ranker.n_pool, ranker.n_pool), dtype=np.float32)
blast_matrix_ranker_order[np.ix_(valid_blast_idx, valid_blast_idx)] = \
    blast["score_matrix"][np.ix_(perm_blast[valid_blast_idx], perm_blast[valid_blast_idx])]


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


rank_mlp_only = make_single_signal_ranker(mlp_scores_in_ranker_order)
rank_blast_only = make_single_signal_ranker(lambda aid: blast_matrix_ranker_order[ranker.id_to_index[aid]])

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


print("\n--- MLP(hadamard) SAM ---", flush=True)
df_mlp = run_leave_one_out(rank_mlp_only, "MLP-only")
df_mlp.to_json(RAW_OUTPUT_MLP, orient="records", indent=2)

print("\n--- BLAST SAM ---", flush=True)
df_blast = run_leave_one_out(rank_blast_only, "BLAST-only")
df_blast.to_json(RAW_OUTPUT_BLAST, orient="records", indent=2)

summary_lines = ["=" * 70, "MLP(hadamard) SAM vs BLAST SAM -- leave-one-out na pacijentima", "=" * 70, ""]
for label, df in [("MLP(hadamard) SAM", df_mlp), ("BLAST SAM", df_blast)]:
    hard = df[df["verification_status"] == "full_text_verified"]
    summary_lines.append(f"--- {label} ---")
    for sub_label, sub in [("SVI", df), ("HARD", hard)]:
        pos = sub[sub["true_result"] == "positive"]["percentile"]
        neg = sub[sub["true_result"] == "negative"]["percentile"]
        summary_lines.append(f"  {sub_label}: Pozitivne (n={len(pos)}) medijan={pos.median():.1f}%  "
                              f"Negativne (n={len(neg)}) medijan={neg.median():.1f}%")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"Saved: {SUMMARY_OUTPUT}")
