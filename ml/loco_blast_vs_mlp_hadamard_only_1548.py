"""
Centralno naucno pitanje teze, CIST oblik (bez fuzije): da li SAM MLP(hadamard)
(nauceno preko ESM embeddinga, bez ikakvog doprinosa BLAST-a/cosine-a/
FoldseekTM-a u samom skoru) STATISTICKI ZNACAJNO prevazilazi CIST BLAST
(referenca bez embeddinga) -- puna LOCO (40 folda) + bootstrap CI na OBA
nivoa (po paru I po izvoru/studiji).

RAZLIKA od loco_blast_vs_rrf_mlp_1548.py (koji je RRF-FUZIJU, dakle vec
ukljucuje BLAST kao sastojak, testirao protiv BLAST-a -- manje cist test
"da li embeddinzi sami dodaju vrednost", jer fuzija po konstrukciji nikad
ne moze biti gora od BLAST-a unutar RRF formule). Ovde: MLP(hadamard) skor
SAM, nezavisno, upoređen direktno sa BLAST skorom SAMIM.

MLP(hadamard) treniran na training_eligible_pairs() (BEZ preostalih Inferred
parova) SVAKI FOLD ISPOCETKA -- ista "cist trening" disciplina kao RRF-6.

Izlaz:
    output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv
    output/loco_blast_vs_mlp_hadamard_only_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_blast_vs_mlp_hadamard_only_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
N_BOOTSTRAP = 2000

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
blast = load_blast_matrices(BLAST_MATRIX)

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
        mlp = None
    else:
        mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED + fold_idx)
        mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            blast_rank = ranks_from_scores(blast_score_matrix_full[qi], qi)
            blast_final_rank = int(blast_rank[ti])

            if mlp is not None:
                mlp_scores = mlp.score_all(query_id)
                mlp_rank = ranks_from_scores(mlp_scores, qi)
                mlp_final_rank = int(mlp_rank[ti])
            else:
                mlp_final_rank = None

            records.append({
                "fold": fold_idx, "pair_id": p["pair_id"],
                "blast_rank": blast_final_rank, "mlp_rank": mlp_final_rank,
                "blast_rr": 1.0 / blast_final_rank,
                "mlp_rr": (1.0 / mlp_final_rank) if mlp_final_rank is not None else None,
            })

    elapsed = time.time() - overall_start
    fold_df_tmp = pd.DataFrame([r for r in records if r["fold"] == fold_idx])
    mlp_mrr_str = f"{fold_df_tmp['mlp_rr'].mean():.4f}" if fold_df_tmp["mlp_rr"].notna().any() else "N/A"
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(fold_df_tmp)}, "
          f"clean_train={len(train_pairs_clean)}) -- blast={fold_df_tmp['blast_rr'].mean():.4f} "
          f"mlp_hadamard={mlp_mrr_str} ({elapsed/60:.1f} min)", flush=True)

df = pd.DataFrame(records)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - overall_start
print(f"All {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min", flush=True)

df_valid = df.dropna(subset=["mlp_rr"]).copy()
n_skipped = len(df) - len(df_valid)


def paired_bootstrap(sub, group_col, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    groups = sub[group_col].dropna().unique()
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["mlp_rr"], weights=w) - np.average(resampled["blast_rr"], weights=w)
        deltas.append(d)
    return np.array(deltas)


summary_lines = ["=" * 80, f"LOCO ({K_FOLDS} folds, cist trening): SAM MLP(hadamard) vs SAM BLAST "
                  "(bez fuzije)", "=" * 80, "",
                  f"Ukupno runtime: {total_elapsed/60:.1f} min",
                  f"Ukupno upita: {len(df)} (izbaceno, prazan fold trening: {n_skipped})", "",
                  f"BLAST MRR (micro): {df_valid['blast_rr'].mean():.4f}",
                  f"MLP(hadamard) MRR (micro): {df_valid['mlp_rr'].mean():.4f}",
                  f"Delta: {df_valid['mlp_rr'].mean() - df_valid['blast_rr'].mean():+.4f}", ""]

for label, group_col in [("PO PARU (pair_id)", "pair_id"), ("PO IZVORU (reference, studijski nivo)", "reference")]:
    deltas = paired_bootstrap(df_valid, group_col, N_BOOTSTRAP, SEED)
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
