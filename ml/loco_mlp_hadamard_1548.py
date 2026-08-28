"""
LOCO (44 folda): Cosine vs Hadamard bilinear vs MLP(hadamard) -- 1548 dataset.

Zasto: analysis/mlp_hadamard_pipeline_sensitivity_1548.py je pokazao da MLP
na Hadamard-produktu (standardize=False, l2_lambda=1e-3, 32 skrivene
jedinice) dostize paritet sa cosine-om (MRR=0.1745, delta+0.0059, CI cist
oko nule) na JEDNOM 80/20 split-u -- ista vrsta rezultata (parni sa cosine,
ne pobeda) kao cist Hadamard bilinear. Ovo je LOCO provera da li ta slika
ostaje dosledna preko svih 44 nezavisnih povezanih komponenti, isti
protokol kao za RF max_depth (ml/loco_rf_blast_maxdepth6_1548.py).

Koristi pipeline klasifikatore direktno (HadamardBilinearClassifier,
MLPPairClassifier sa input_encoding="hadamard") -- ne duplira trening kod.

Izlaz:
    output/loco_mlp_hadamard_1548_per_fold.csv
    output/loco_mlp_hadamard_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import retrieval_evaluate  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.hadamard import HadamardBilinearClassifier  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
PER_FOLD_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_mlp_hadamard_1548_per_fold.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_mlp_hadamard_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10

HADAMARD_BILINEAR_PARAMS = dict(optimizer="adamw", max_epochs=300, patience=15, learning_rate=1e-2,
                                  weight_decay=0.0, l2_lambda=1e-3, val_fraction=0.15)
MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)


def sample_negatives(protein_pool, n_needed, seed, positive_pair_set):
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    unlabeled = set()
    max_attempts = n_needed * 50 + 2000
    attempts = 0
    while len(unlabeled) < n_needed and attempts < max_attempts:
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1
        if pair in positive_pair_set or pair in unlabeled:
            continue
        unlabeled.add(pair)
    return sorted(unlabeled)


print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}")

all_dfs = {"hadamard_bilinear": [], "mlp_hadamard": []}
per_fold_rows = []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_ids = {pid for p in train_pairs for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = len(train_pairs) * NEG_PER_POS
    train_negatives = sample_negatives(train_ids, n_train_neg, SEED + fold_idx, dataset.positive_pair_set)

    hb = HadamardBilinearClassifier(params=HADAMARD_BILINEAR_PARAMS, seed=SEED + fold_idx)
    hb.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
    hb_df = retrieval_evaluate(test_pairs, hb, dataset.embedding_matrix, dataset.id_to_index,
                                 cosine_matrix=cosine_matrix)

    mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED + 500 + fold_idx)
    mlp.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
    mlp_df = retrieval_evaluate(test_pairs, mlp, dataset.embedding_matrix, dataset.id_to_index,
                                  cosine_matrix=cosine_matrix)

    hb_df["fold"] = fold_idx
    mlp_df["fold"] = fold_idx
    all_dfs["hadamard_bilinear"].append(hb_df)
    all_dfs["mlp_hadamard"].append(mlp_df)

    per_fold_rows.append({
        "fold": fold_idx, "component_size": len(test_ids), "n_queries": len(hb_df),
        "cosine_mrr": hb_df["cosine_reciprocal_rank"].mean(),
        "hadamard_bilinear_mrr": hb_df["model_reciprocal_rank"].mean(),
        "mlp_hadamard_mrr": mlp_df["model_reciprocal_rank"].mean(),
    })
    elapsed = time.time() - overall_start
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(hb_df)}) -- "
          f"cosine={per_fold_rows[-1]['cosine_mrr']:.4f} hb={per_fold_rows[-1]['hadamard_bilinear_mrr']:.4f} "
          f"mlp_h={per_fold_rows[-1]['mlp_hadamard_mrr']:.4f} ({elapsed/60:.1f} min)", flush=True)

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)

hb_all = pd.concat(all_dfs["hadamard_bilinear"], ignore_index=True)
mlp_all = pd.concat(all_dfs["mlp_hadamard"], ignore_index=True)

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
hb_macro = per_fold_df["hadamard_bilinear_mrr"].to_numpy()
mlp_macro = per_fold_df["mlp_hadamard_mrr"].to_numpy()

delta_hb = hb_macro - cos_macro
delta_mlp = mlp_macro - cos_macro
se_hb = float(delta_hb.std(ddof=1) / np.sqrt(K_FOLDS))
se_mlp = float(delta_mlp.std(ddof=1) / np.sqrt(K_FOLDS))

summary_lines = [
    "=" * 70,
    f"LOCO ({K_FOLDS} folds): Cosine vs Hadamard bilinear vs MLP(hadamard) (1548)",
    "=" * 70,
    f"Total runtime: {total_elapsed/60:.1f} min", "",
    "MACRO (unweighted mean across component-folds):",
    f"  cosine              MRR: {cos_macro.mean():.4f} +/- {cos_macro.std(ddof=1):.4f}",
    f"  Hadamard bilinear    MRR: {hb_macro.mean():.4f} +/- {hb_macro.std(ddof=1):.4f}",
    f"  MLP(hadamard)        MRR: {mlp_macro.mean():.4f} +/- {mlp_macro.std(ddof=1):.4f}", "",
    "MICRO (query-weighted, pooled preko svih foldova -- najpouzdaniji broj):",
    f"  cosine              MRR: {hb_all['cosine_reciprocal_rank'].mean():.4f}",
    f"  Hadamard bilinear    MRR: {hb_all['model_reciprocal_rank'].mean():.4f}",
    f"  MLP(hadamard)        MRR: {mlp_all['model_reciprocal_rank'].mean():.4f}", "",
    f"Paired delta Hadamard bilinear vs cosine: {delta_hb.mean():+.4f} (SE {se_hb:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_hb.mean()) > 2*se_hb else '- unutar suma'}",
    f"Paired delta MLP(hadamard) vs cosine:     {delta_mlp.mean():+.4f} (SE {se_mlp:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_mlp.mean()) > 2*se_mlp else '- unutar suma'}",
    f"Paired wins (Hadamard bilinear > cosine): {int((delta_hb > 0).sum())}/{K_FOLDS}",
    f"Paired wins (MLP(hadamard) > cosine):     {int((delta_mlp > 0).sum())}/{K_FOLDS}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
