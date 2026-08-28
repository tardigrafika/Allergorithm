"""
Random Forest: senzitivnost na hiperparametre (n_estimators, max_depth,
min_samples_leaf, max_features) + puna feature importance analiza -- 1548
dataset, RF+BLAST konfiguracija (do sad najbolja isprobana feature kombinacija).

Zasto ovo, a ne jos jedan feature: ceo dosadasnji RF due diligence (BLAST,
kmer, TM-score/FoldseekTM, same_family, hard negativi, PU bagging, ensemble
sa cosine -- videti ml/random_forest_*_1443.py, ml/loco_*rf*_1512.py,
ml/loco_rf_blast_foldseektm_1548.py) je testirao FEATURE-e, ali nikad same
RF hiperparametre -- svaka skripta kopira isti default (300 stabala,
max_depth=12, min_samples_leaf=3) bez provere da li je to uopste dobar izbor.
Ovo je direktan RF-analog onoga sto je uradjeno za Hadamard model
(analysis/hadamard_sensitivity_1548.py, analysis/hadamard_normalize_cosinit_1548.py).

Izlaz:
    output/rf_hyperparam_sensitivity_1548_summary.txt
    output/rf_hyperparam_sensitivity_1548_results.csv
    output/rf_feature_importance_1548_summary.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import bootstrap_ci, retrieval_evaluate, summarize_retrieval  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import group_aware_split, split_pairs  # noqa: E402
from ml.pipeline.models.classifiers.random_forest import RandomForestPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/rf_hyperparam_sensitivity_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/rf_hyperparam_sensitivity_1548_results.csv")
IMPORTANCE_OUTPUT = Path("/home/lana/ALERGRAF/output/rf_feature_importance_1548_summary.txt")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

GRID = [
    # label, n_estimators, max_depth, min_samples_leaf, max_features
    ("baseline_300_d12_leaf3_sqrt", 300, 12,   3,  "sqrt"),
    ("n_estimators_100",            100, 12,   3,  "sqrt"),
    ("n_estimators_600",            600, 12,   3,  "sqrt"),
    ("max_depth_6",                 300, 6,    3,  "sqrt"),
    ("max_depth_none",              300, None, 3,  "sqrt"),
    ("min_leaf_1",                  300, 12,   1,  "sqrt"),
    ("min_leaf_10",                 300, 12,   10, "sqrt"),
    ("max_features_log2",           300, 12,   3,  "log2"),
    ("max_features_all",            300, 12,   3,  1.0),
]

print("Loading dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
n_train_neg = len(train_pairs) * NEG_PER_POS
train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

print(f"\nProlazim kroz {len(GRID)} konfiguracija...\n")
results = []
best_clf = None
for label, n_estimators, max_depth, min_samples_leaf, max_features in GRID:
    params = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                  max_features=max_features, class_weight="balanced", n_jobs=-1)
    clf = RandomForestPairClassifier(params=params, extra_features=["blast_identity", "blast_score"],
                                       blast_matrix_path=BLAST_MATRIX, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    results.append({
        "label": label, "n_estimators": n_estimators, "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf, "max_features": max_features,
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:32s}  MRR={summary['mrr']:.4f}  delta={delta_stats['mean']:+.4f}  "
          f"CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}", flush=True)

    if label == "baseline_300_d12_leaf3_sqrt":
        best_clf = clf

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "Random Forest (+BLAST): senzitivnost na hiperparametre", "=" * 80, "",
                  f"Cosine baseline MRR: {results_df['cosine_mrr'].iloc[0]:.4f}", ""]
for _, r in results_df.sort_values("mrr", ascending=False).iterrows():
    summary_lines.append(
        f"{r['label']:<32}{r['mrr']:<10.4f}delta={r['delta']:+.4f}  "
        f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{'  ZNACAJNO' if r['significant'] else ''}"
    )
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")

# =====================================================
# FEATURE IMPORTANCE -- na baseline RF+BLAST modelu
# =====================================================
print("\nRacunam feature importance na baseline modelu...")
importances = best_clf.feature_importances()  # 1283: abs_diff[1280] + cosine[1] + blast_identity[1] + blast_score[1]
feature_names = [f"abs_diff_dim{i}" for i in range(1280)] + ["cosine", "blast_identity", "blast_score"]
order = np.argsort(importances)[::-1]

imp_lines = ["=" * 80, "Random Forest (+BLAST) baseline: feature importance (1283 feature-a)", "=" * 80, ""]
imp_lines.append(f"Ukupna vaznost svih 1280 abs_diff embedding dimenzija: {importances[:1280].sum():.4f}")
imp_lines.append(f"Vaznost cosine feature-a:          {importances[1280]:.4f}  (rank {int((importances > importances[1280]).sum()) + 1}/1283)")
imp_lines.append(f"Vaznost blast_identity feature-a:  {importances[1281]:.4f}  (rank {int((importances > importances[1281]).sum()) + 1}/1283)")
imp_lines.append(f"Vaznost blast_score feature-a:      {importances[1282]:.4f}  (rank {int((importances > importances[1282]).sum()) + 1}/1283)")
imp_lines.append(f"Prosecna vaznost po feature-u (referentno): {importances.mean():.5f}")
imp_lines.append("")
imp_lines.append("Top 20 feature-a po vaznosti:")
for rank, idx in enumerate(order[:20], 1):
    imp_lines.append(f"  {rank:2d}. {feature_names[idx]:<20s} {importances[idx]:.5f}")

imp_text = "\n".join(imp_lines)
print("\n" + imp_text)
with open(IMPORTANCE_OUTPUT, "w") as f:
    f.write(imp_text + "\n")
print(f"\nSaved: {IMPORTANCE_OUTPUT}")
