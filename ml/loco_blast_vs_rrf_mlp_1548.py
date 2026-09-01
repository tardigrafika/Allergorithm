"""
Centralno naucno pitanje teze: da li metoda zasnovana na ESM embeddinzima
(RRF-4-MLP: cosine + BLAST + FoldseekTM + MLP(hadamard)) STATISTICKI ZNACAJNO
prevazilazi CIST BLAST (referentnu tacku bez embeddinga) -- puna LOCO (40
folda) validacija + bootstrap CI na OBA nivoa (po paru I po izvoru/studiji,
posle otkrica 2026-08-29 da pair-level bootstrap moze biti preoptimistican
kad mnogo parova deli isti citat).

MLP(hadamard) treniran na training_eligible_pairs() (BEZ preostalih Inferred
parova) SVAKI FOLD ISPOCETKA -- ista "cist trening" disciplina kao RRF-6
(test/evaluate_rrf6_mlp_hadamard_1548.py), koja je pokazala prvo pravo
poboljsanje na pravim pacijentima. Graph-propagation NIJE ukljucen (strukturno
nekompatibilan sa component-level LOCO -- izdvojena komponenta nema vidljive
susede), isto obrazlozenje kao svaki raniji RRF-4-MLP test ove sesije.

Izlaz:
    output/loco_blast_vs_rrf_mlp_1548_per_query.csv
    output/loco_blast_vs_rrf_mlp_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.features import load_blast_matrices  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_blast_vs_rrf_mlp_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_blast_vs_rrf_mlp_1548_summary.txt")

SEED = 42
RRF_K = 20
NEG_PER_POS = 10
N_BOOTSTRAP = 2000

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
blast = load_blast_matrices(BLAST_MATRIX)

import pickle  # noqa: E402
with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)
foldseek_matrix = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in dataset.id_to_index and b in dataset.id_to_index:
        i, j = dataset.id_to_index[a], dataset.id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score

perm = np.array([blast["id_to_index"].get(aid, -1) for aid in dataset.all_ids])
valid_idx = np.where(perm >= 0)[0]
blast_score_matrix_full = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
blast_score_matrix_full[np.ix_(valid_idx, valid_idx)] = blast["score_matrix"][np.ix_(perm[valid_idx], perm[valid_idx])]


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}", flush=True)

records = []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_pairs_clean = training_eligible_pairs(train_pairs)
    train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
    train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED + fold_idx,
                                              dataset.positive_pair_set)

    if len(train_pairs_clean) < 5:
        # premalo cistih pozitiva u ovom foldu za smisleno treniranje MLP-a
        mlp = None
    else:
        mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED + fold_idx)
        mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            cos_rank = ranks_from_scores(cosine_matrix[qi], qi)
            blast_rank = ranks_from_scores(blast_score_matrix_full[qi], qi)
            fs_rank = ranks_from_scores(foldseek_matrix[qi], qi)

            rrf3 = 1.0 / (RRF_K + cos_rank) + 1.0 / (RRF_K + blast_rank) + 1.0 / (RRF_K + fs_rank)
            if mlp is not None:
                mlp_scores = mlp.score_all(query_id)
                mlp_rank = ranks_from_scores(mlp_scores, qi)
                rrf4mlp = rrf3 + 1.0 / (RRF_K + mlp_rank)
            else:
                rrf4mlp = rrf3

            blast_final_rank = int(blast_rank[ti])
            rrf4mlp_order = np.argsort(rrf4mlp)[::-1]
            rrf4mlp_final_rank = int(np.where(rrf4mlp_order == ti)[0][0]) + 1

            records.append({
                "fold": fold_idx, "pair_id": p["pair_id"],
                "blast_rank": blast_final_rank, "rrf4mlp_rank": rrf4mlp_final_rank,
                "blast_rr": 1.0 / blast_final_rank, "rrf4mlp_rr": 1.0 / rrf4mlp_final_rank,
            })

    elapsed = time.time() - overall_start
    fold_df_tmp = pd.DataFrame([r for r in records if r["fold"] == fold_idx])
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(fold_df_tmp)}, "
          f"clean_train={len(train_pairs_clean)}) -- blast={fold_df_tmp['blast_rr'].mean():.4f} "
          f"rrf4mlp={fold_df_tmp['rrf4mlp_rr'].mean():.4f} ({elapsed/60:.1f} min)", flush=True)

df = pd.DataFrame(records)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - overall_start
print(f"All {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min", flush=True)


def paired_bootstrap(sub, group_col, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    groups = sub[group_col].dropna().unique()
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["rrf4mlp_rr"], weights=w) - np.average(resampled["blast_rr"], weights=w)
        deltas.append(d)
    return np.array(deltas)


summary_lines = ["=" * 80, f"LOCO ({K_FOLDS} folds, cist trening): RRF-4-MLP(hadamard) vs cist BLAST",
                  "=" * 80, "", f"Ukupno runtime: {total_elapsed/60:.1f} min", f"Ukupno upita: {len(df)}", "",
                  f"BLAST MRR (micro): {df['blast_rr'].mean():.4f}",
                  f"RRF-4-MLP MRR (micro): {df['rrf4mlp_rr'].mean():.4f}",
                  f"Delta: {df['rrf4mlp_rr'].mean() - df['blast_rr'].mean():+.4f}", ""]

for label, group_col in [("PO PARU (pair_id)", "pair_id"), ("PO IZVORU (reference, studijski nivo)", "reference")]:
    deltas = paired_bootstrap(df, group_col, N_BOOTSTRAP, SEED)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"{label}: mean delta={deltas.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] "
                          f"-- {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
