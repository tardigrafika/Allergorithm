"""
Osetljivost Hadamard bilinear modela (y=sigmoid(w.(u*v))) na optimizator
(AdamW vs SGD), learning rate i L2 regularizaciju u loss-u -- mentorov
zahtev, PRE prelaska na dublju Random Forest analizu.

Koristi NOVOIZGRADJEN pipeline (ml/pipeline/) umesto duplirane logike --
dataset/split/negativi se ucitaju/generisu JEDNOM, pa se model trenira
za svaku kombinaciju parametara (isti split, isti negativi -- razlike u
rezultatu dolaze SAMO od hiperparametara modela, ne od suma u split-u).

Izlaz:
    output/hadamard_sensitivity_1548_summary.txt
    output/hadamard_sensitivity_1548_results.csv
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
from ml.pipeline.models.classifiers.hadamard import HadamardBilinearClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/hadamard_sensitivity_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/hadamard_sensitivity_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

# =====================================================
# GRID -- namerno umeren obim (due diligence, ne iscrpna pretraga)
# =====================================================
GRID = [
    # (label, optimizer, learning_rate, l2_lambda, weight_decay, momentum)
    ("adamw_lr1e-3_l2=0",    "adamw", 1e-3, 0.0,  0.0, 0.9),
    ("adamw_lr1e-2_l2=0",    "adamw", 1e-2, 0.0,  0.0, 0.9),
    ("adamw_lr1e-2_l2=1e-3", "adamw", 1e-2, 1e-3, 0.0, 0.9),
    ("adamw_lr1e-2_l2=1e-4", "adamw", 1e-2, 1e-4, 0.0, 0.9),
    ("sgd_lr1e-2_l2=0",      "sgd",   1e-2, 0.0,  0.0, 0.9),
    ("sgd_lr1e-1_l2=0",      "sgd",   1e-1, 0.0,  0.0, 0.9),
    ("sgd_lr1e0_l2=0",       "sgd",   1.0,  0.0,  0.0, 0.9),
    ("sgd_lr1e-1_l2=1e-3",   "sgd",   1e-1, 1e-3, 0.0, 0.9),
    ("sgd_lr1e-1_l2=1e-4",   "sgd",   1e-1, 1e-4, 0.0, 0.9),
]

print("Loading dataset (jednom, deli se za sve konfiguracije)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
print(f"Gold pairs: {len(dataset.gold_pairs)}  |  Candidate pool: {len(dataset.all_ids)}")

train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
print(f"Train proteina: {len(train_ids)}  Test proteina: {len(test_ids)}")
print(f"Train parova: {len(train_pairs)}  Test parova: {len(test_pairs)}")

n_train_neg = len(train_pairs) * NEG_PER_POS
n_test_neg = len(test_pairs) * NEG_PER_POS
train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)
test_negatives = sample_negative_pairs(test_ids, n_test_neg, SEED + 1, dataset.positive_pair_set)

print("\nRacunam cosine baseline matricu (deljena za sve konfiguracije)...")
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

print(f"\nProlazim kroz {len(GRID)} konfiguracija...\n")
results = []
for label, optimizer, lr, l2_lambda, weight_decay, momentum in GRID:
    params = dict(optimizer=optimizer, learning_rate=lr, l2_lambda=l2_lambda,
                  weight_decay=weight_decay, momentum=momentum, max_epochs=300, patience=15)
    clf = HadamardBilinearClassifier(params=params, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    stopped_epoch = getattr(clf, "stopped_epoch", None)
    best_val_loss = getattr(clf, "best_val_loss", None)

    results.append({
        "label": label, "optimizer": optimizer, "learning_rate": lr, "l2_lambda": l2_lambda,
        "momentum": momentum if optimizer == "sgd" else None,
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:24s}  MRR={summary['mrr']:.4f}  delta={delta_stats['mean']:+.4f}  "
          f"CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}", flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "Hadamard bilinear: osetljivost na optimizator / learning rate / L2", "=" * 80, "",
                  f"Cosine baseline MRR (isti split za sve): {results_df['cosine_mrr'].iloc[0]:.4f}", ""]
summary_lines.append(f"{'Config':<24}{'Optim':<8}{'LR':<10}{'L2':<10}{'MRR':<10}{'Delta':<12}{'95% CI':<24}{'Sig?'}")
for _, r in results_df.sort_values("mrr", ascending=False).iterrows():
    summary_lines.append(
        f"{r['label']:<24}{r['optimizer']:<8}{r['learning_rate']:<10}{r['l2_lambda']:<10}"
        f"{r['mrr']:<10.4f}{r['delta']:<+12.4f}[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
        f"{'  ZNACAJNO' if r['significant'] else ''}"
    )

best = results_df.loc[results_df["mrr"].idxmax()]
summary_lines.append("")
summary_lines.append(f"Najbolja konfiguracija po MRR: {best['label']} (MRR={best['mrr']:.4f}, "
                      f"delta={best['delta']:+.4f}, {'ZNACAJNO' if best['significant'] else 'nije znacajno'})")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
