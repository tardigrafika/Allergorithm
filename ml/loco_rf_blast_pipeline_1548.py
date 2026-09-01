"""
LOCO (Cosine vs RF+BLAST) preko pipeline-a (load_dataset sa CCD+negative
filterima) -- zamenjuje stariji ml/loco_rf_blast_foldseektm_1548.py koji
NIJE znao za ccd_flag kolonu (pisan pre nego sto je ona postojala). Koristi
prosireni dataset (output/cross_reactive_1548.csv, sad 1910 redova/1884
posle filtera) da se proveri da li vece/kvalitetnije podaci menjaju raniji
"RF+BLAST ne prevazilazi cosine" LOCO nalaz.

Izlaz:
    output/loco_rf_blast_pipeline_1548_per_fold.csv
    output/loco_rf_blast_pipeline_1548_summary.txt
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
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.random_forest import RandomForestPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
PER_FOLD_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_rf_blast_pipeline_1548_per_fold.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_rf_blast_pipeline_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", n_jobs=-1)

print("Loading dataset (sa CCD+negative filterima)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
print(f"Gold parova posle filtera: {len(dataset.gold_pairs)}")
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}")

per_fold_rows = []
all_cos_rr, all_rf_rr = [], []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_ids = {pid for p in train_pairs for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = len(train_pairs) * NEG_PER_POS
    train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED + fold_idx, dataset.positive_pair_set)

    rf = RandomForestPairClassifier(params=RF_PARAMS, extra_features=["blast_identity", "blast_score"],
                                      blast_matrix_path=BLAST_MATRIX, seed=SEED + fold_idx)
    rf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    rf_df = retrieval_evaluate(test_pairs, rf, dataset.embedding_matrix, dataset.id_to_index,
                                 cosine_matrix=cosine_matrix)

    all_cos_rr.extend(rf_df["cosine_reciprocal_rank"].tolist())
    all_rf_rr.extend(rf_df["model_reciprocal_rank"].tolist())

    per_fold_rows.append({
        "fold": fold_idx, "component_size": len(test_ids), "n_queries": len(rf_df),
        "cosine_mrr": rf_df["cosine_reciprocal_rank"].mean(),
        "rf_blast_mrr": rf_df["model_reciprocal_rank"].mean(),
    })
    elapsed = time.time() - overall_start
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(rf_df)}) -- "
          f"cosine={per_fold_rows[-1]['cosine_mrr']:.4f} rf_blast={per_fold_rows[-1]['rf_blast_mrr']:.4f} "
          f"({elapsed/60:.1f} min)", flush=True)

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
rf_macro = per_fold_df["rf_blast_mrr"].to_numpy()
delta = rf_macro - cos_macro
se = float(delta.std(ddof=1) / np.sqrt(K_FOLDS))

summary_lines = [
    "=" * 70, f"LOCO ({K_FOLDS} folds): Cosine vs RF+BLAST, PROSIRENI dataset (pipeline, CCD+neg filtrirano)", "=" * 70,
    f"Total runtime: {total_elapsed/60:.1f} min", "",
    "MACRO:",
    f"  cosine    MRR: {cos_macro.mean():.4f} +/- {cos_macro.std(ddof=1):.4f}",
    f"  RF+BLAST  MRR: {rf_macro.mean():.4f} +/- {rf_macro.std(ddof=1):.4f}", "",
    "MICRO (query-weighted, najpouzdaniji broj):",
    f"  cosine    MRR: {np.mean(all_cos_rr):.4f}",
    f"  RF+BLAST  MRR: {np.mean(all_rf_rr):.4f}", "",
    f"Paired delta (RF+BLAST - cosine): {delta.mean():+.4f} (SE {se:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta.mean()) > 2*se else '- unutar suma'}",
    f"Paired wins (RF+BLAST > cosine): {int((delta > 0).sum())}/{K_FOLDS}",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
