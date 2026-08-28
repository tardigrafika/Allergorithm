"""
Sensitivity sweep za MLPPairClassifier(input_encoding="hadamard") -- SADA
kroz pravu pipeline integraciju (ml/pipeline/models/classifiers/mlp.py), ne
standalone dijagnosticku skriptu (analysis/mlp_hadamard_input_1548.py).

Pipeline verzija koristi drugaciju trening proceduru od standalone
dijagnostike (mini-batch umesto full-batch, standardizacija feature-a,
rana zaustavljanja preko val ROC-AUC umesto val loss) -- ovaj sweep prvo
PROVERAVA da integracija i dalje daje dobre rezultate (fidelity check), pa
onda radi pravi due-diligence sweep (arhitektura x regularizacija x LR) na
ispravnom (Hadamard) enkodingu.

Izlaz:
    output/mlp_hadamard_pipeline_sensitivity_1548_summary.txt
    output/mlp_hadamard_pipeline_sensitivity_1548_results.csv
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import bootstrap_ci, retrieval_evaluate, summarize_retrieval  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import group_aware_split, split_pairs  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_pipeline_sensitivity_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_pipeline_sensitivity_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

GRID = [
    # label, hidden_dims, dropout, learning_rate, weight_decay, l2_lambda
    ("h16_lr1e-2",              [16],      [0.2],      1e-2, 1e-4, 0.0),
    ("h16_lr1e-3",              [16],      [0.2],      1e-3, 1e-4, 0.0),
    ("h32_lr1e-2",              [32],      [0.3],      1e-2, 1e-4, 0.0),
    ("h32_lr1e-2_l2loss",       [32],      [0.3],      1e-2, 0.0,  1e-3),
    ("h32_lr1e-3_l2loss",       [32],      [0.3],      1e-3, 0.0,  1e-3),
    ("h64_lr1e-2_l2loss",       [64],      [0.3],      1e-2, 0.0,  1e-3),
    ("h32_16_lr1e-2_l2loss",    [32, 16],  [0.3, 0.2], 1e-2, 0.0,  1e-3),
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
for label, hidden_dims, dropout, lr, wd, l2l in GRID:
    params = dict(input_encoding="hadamard", standardize=False, hidden_dims=hidden_dims, dropout=dropout,
                  learning_rate=lr, weight_decay=wd, l2_lambda=l2l,
                  batch_size=64, max_epochs=300, patience=20)
    clf = MLPPairClassifier(params=params, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    results.append({
        "label": label, "hidden_dims": str(hidden_dims), "lr": lr, "weight_decay": wd, "l2_lambda": l2l,
        "stopped_epoch": getattr(clf, "stopped_epoch", None),
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:24s}  epoch={results[-1]['stopped_epoch']:4}  MRR={summary['mrr']:.4f}  "
          f"delta={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}",
          flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "MLP(hadamard) kroz pipeline: senzitivnost na arhitekturu/LR/regularizaciju", "=" * 80, "",
                  f"Cosine baseline MRR: {results_df['cosine_mrr'].iloc[0]:.4f}",
                  "Referenca -- standalone dijagnostika (mlp_hadamard_input_1548.py), h32+l2loss: MRR=0.1791, delta=+0.0100",
                  ""]
for _, r in results_df.sort_values("mrr", ascending=False).iterrows():
    summary_lines.append(
        f"{r['label']:<24}{r['mrr']:<10.4f}delta={r['delta']:+.4f}  "
        f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{'  ZNACAJNO' if r['significant'] else ''}"
    )
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
