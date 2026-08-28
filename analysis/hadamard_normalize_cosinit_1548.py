"""
Nastavak hadamard_sensitivity_1548.py -- testira normalize="l2" i cosine_init,
ukljucujuci da li normalizacija popravlja SGD nestabilnost dijagnostikovanu
ranije (loss oscilacija, MRR=0.0018 identican za sve LR).

Izlaz:
    output/hadamard_normalize_cosinit_1548_summary.txt
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/hadamard_normalize_cosinit_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/hadamard_normalize_cosinit_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

GRID = [
    # label, optimizer, lr, normalize, cosine_init, max_epochs, patience
    ("adamw_lr1e-2_raw_randinit",       "adamw", 1e-2, "none", False, 300,  15),
    ("adamw_lr1e-2_l2norm_randinit",    "adamw", 1e-2, "l2",   False, 300,  15),
    ("adamw_lr1e-2_l2norm_cosinit",     "adamw", 1e-2, "l2",   True,  300,  15),
    ("sgd_lr1e-2_l2norm_randinit",      "sgd",   1e-2, "l2",   False, 300,  15),
    ("sgd_lr1e-2_l2norm_cosinit",       "sgd",   1e-2, "l2",   True,  300,  15),
    ("sgd_lr1e-1_l2norm_cosinit",       "sgd",   1e-1, "l2",   True,  300,  15),
    ("sgd_lr1e0_l2norm_cosinit",        "sgd",   1.0,  "l2",   True,  300,  15),
]

print("Loading dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
n_train_neg = len(train_pairs) * NEG_PER_POS
n_test_neg = len(test_pairs) * NEG_PER_POS
train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

print(f"\nProlazim kroz {len(GRID)} konfiguracija...\n")
results = []
for label, optimizer, lr, normalize, cosine_init, max_epochs, patience in GRID:
    params = dict(optimizer=optimizer, learning_rate=lr, momentum=0.9, weight_decay=0.0, l2_lambda=0.0,
                  normalize=normalize, cosine_init=cosine_init, max_epochs=max_epochs, patience=patience)
    clf = HadamardBilinearClassifier(params=params, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    results.append({
        "label": label, "optimizer": optimizer, "learning_rate": lr, "normalize": normalize,
        "cosine_init": cosine_init, "stopped_epoch": clf.stopped_epoch,
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:32s}  epoch={clf.stopped_epoch:4d}  MRR={summary['mrr']:.4f}  delta={delta_stats['mean']:+.4f}  "
          f"CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}", flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "Hadamard bilinear: normalizacija + cosine-informisana inicijalizacija", "=" * 80, "",
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
