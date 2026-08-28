"""
MLP: senzitivnost na arhitekturu (velicina/dubina skrivenih slojeva) i
regularizaciju - 1548 dataset. MLP je jedini od tri glavna modela (Hadamard,
RF, MLP) koji JOS NIJE dobio sistematski due diligence u ovoj rundi.

Zasto bas ovo prvo testirati: trenutni MLP baseline (1281->256->64->1) ima
~344,000 parametara na svega ~1269 trening parova / 44 nezavisne povezane
komponente -- ogroman disbalans kapaciteta naspram efektivne velicine podataka.
Ovo je TACNO mentorova originalna zamerka koja je pokrenula Hadamard bilinear
eksperiment (y=sigmoid(w.(u*v)), ~1280 parametara). Ovaj sweep testira da li
manja MLP arhitektura -- most izmedju Hadamard-ovih ~1280 parametara i
trenutnih 344k -- radi bolje, plus standardnu regularizacionu senzitivnost
(dropout, weight_decay, eksplicitan L2-u-loss-u kao kod Hadamard-a).

NAPOMENA: isti feature vektor kao originalni mlp_baseline.py (abs_diff[1280]
+ cosine[1] = 1281 dim, BEZ BLAST-a) -- verno originalu, ne dodajemo novi
signal ovde, samo arhitekturu/regularizaciju.

Izlaz:
    output/mlp_sensitivity_1548_summary.txt
    output/mlp_sensitivity_1548_results.csv
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_sensitivity_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_sensitivity_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

GRID = [
    # label, hidden_dims, dropout, weight_decay, l2_lambda
    ("baseline_256_64",        [256, 64], [0.3, 0.2], 1e-4, 0.0),
    ("small_64",                [64],      [0.3],       1e-4, 0.0),
    ("small_32",                [32],      [0.3],       1e-4, 0.0),
    ("tiny_16",                 [16],      [0.2],       1e-4, 0.0),
    ("small_64_16",             [64, 16],  [0.3, 0.2],  1e-4, 0.0),
    ("baseline_high_weightdecay", [256, 64], [0.3, 0.2], 1e-3, 0.0),
    ("baseline_l2_in_loss",     [256, 64], [0.3, 0.2], 0.0,  1e-3),
    ("small_32_high_dropout",   [32],      [0.5],       1e-4, 0.0),
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
for label, hidden_dims, dropout, weight_decay, l2_lambda in GRID:
    n_params_approx = 0
    prev = 1281
    for h in hidden_dims:
        n_params_approx += prev * h + h
        prev = h
    n_params_approx += prev * 1 + 1

    params = dict(hidden_dims=hidden_dims, dropout=dropout, weight_decay=weight_decay, l2_lambda=l2_lambda,
                  batch_size=64, max_epochs=200, patience=20, learning_rate=1e-3)
    clf = MLPPairClassifier(params=params, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    stopped_epoch = getattr(clf, "stopped_epoch", None)
    results.append({
        "label": label, "hidden_dims": str(hidden_dims), "dropout": str(dropout),
        "weight_decay": weight_decay, "l2_lambda": l2_lambda, "n_params_approx": n_params_approx,
        "stopped_epoch": stopped_epoch,
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:28s}  params~{n_params_approx:6d}  epoch={stopped_epoch}  MRR={summary['mrr']:.4f}  "
          f"delta={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}",
          flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "MLP: senzitivnost na arhitekturu (kapacitet) i regularizaciju", "=" * 80, "",
                  f"Cosine baseline MRR: {results_df['cosine_mrr'].iloc[0]:.4f}", ""]
for _, r in results_df.sort_values("mrr", ascending=False).iterrows():
    summary_lines.append(
        f"{r['label']:<28}params~{r['n_params_approx']:<8}{r['mrr']:<10.4f}delta={r['delta']:+.4f}  "
        f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{'  ZNACAJNO' if r['significant'] else ''}"
    )
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
