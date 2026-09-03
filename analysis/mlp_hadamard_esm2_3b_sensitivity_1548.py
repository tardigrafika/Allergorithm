"""
Senzitivnost sweep za MLPPairClassifier(input_encoding="hadamard") NA ESM-2
3B EMBEDDINZIMA (dim 2560, duplo vise od 1280 na kom su postojeci
MLP_HADAMARD_PARAMS (hidden_dims=[32], lr=1e-2, l2_lambda=1e-3,
standardize=False) originalno tunovani -- output/mlp_hadamard_pipeline_
sensitivity_1548_summary.txt).

Korisnica je ispravno primetila da poredjenje 3B vs 650M (ml/loco_esm2_3b_
vs_esm2_650m_1548.py) nije bilo fer -- ISTI hiperparametri kopirani na
duplo veci ulaz bez ikakvog retuning-a, katastrofalan kolaps (MRR=0.0395)
je verovatnije artefakt neuskladjene arhitekture nego dokaz da je 3B losiji
model. Ovaj sweep testira da li RETUNOVANA arhitektura (veci hidden layer
i/ili standardize=True za veci ulaz) popravlja 3B rezultat, PRE nego sto
se bilo sta zakljuci o 3B embeddinzima samim.

Ista brza single-split metodologija kao originalni sensitivity sweep
(group_aware_split 80/20, NE puni LOCO -- LOCO se radi tek za finalnog
pobednika, ne za svih N config-a u sweep-u, isto pravilo kao svuda u
projektu). training_eligible_pairs() koriscen (trenutna "cist trening"
disciplina, originalni sweep je predatirao tu odluku).

Izlaz:
    output/mlp_hadamard_esm2_3b_sensitivity_1548_summary.txt
    output/mlp_hadamard_esm2_3b_sensitivity_1548_results.csv
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.evaluation import bootstrap_ci, retrieval_evaluate, summarize_retrieval  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import group_aware_split, split_pairs  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_esm2_3b_sensitivity_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_esm2_3b_sensitivity_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

# label, hidden_dims, dropout, learning_rate, weight_decay, l2_lambda, standardize
GRID = [
    ("h32_lr1e-2_stdFalse (=650M config, referenca)", [32],      [0.3],      1e-2, 0.0, 1e-3, False),
    ("h64_lr1e-2_stdFalse",                            [64],      [0.3],      1e-2, 0.0, 1e-3, False),
    ("h128_lr1e-2_stdFalse",                           [128],     [0.3],      1e-2, 0.0, 1e-3, False),
    ("h64_lr1e-3_stdFalse",                             [64],      [0.3],      1e-3, 0.0, 1e-3, False),
    ("h32_lr1e-2_stdTrue",                              [32],      [0.3],      1e-2, 0.0, 1e-3, True),
    ("h64_lr1e-2_stdTrue",                              [64],      [0.3],      1e-2, 0.0, 1e-3, True),
    ("h64_32_lr1e-2_stdFalse",                          [64, 32],  [0.3, 0.2], 1e-2, 0.0, 1e-3, False),
]

print("Loading ESM-2 3B dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
train_pairs = training_eligible_pairs(train_pairs)
n_train_neg = len(train_pairs) * NEG_PER_POS
train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
print(f"Train pairs (clean): {len(train_pairs)}, test pairs: {len(test_pairs)}, dim: {dataset.embedding_matrix.shape[1]}")

print(f"\nProlazim kroz {len(GRID)} konfiguracija...\n")
results = []
for label, hidden_dims, dropout, lr, wd, l2l, standardize in GRID:
    params = dict(input_encoding="hadamard", standardize=standardize, hidden_dims=hidden_dims, dropout=dropout,
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
        "label": label, "hidden_dims": str(hidden_dims), "lr": lr, "l2_lambda": l2l, "standardize": standardize,
        "stopped_epoch": getattr(clf, "stopped_epoch", None),
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:45s}  epoch={results[-1]['stopped_epoch']:4}  MRR={summary['mrr']:.4f}  "
          f"delta={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}",
          flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "MLP(hadamard) na ESM-2 3B: senzitivnost na arhitekturu/LR/standardize "
                  "(single-split screening, LOCO tek za pobednika)", "=" * 80, "",
                  f"Cosine baseline MRR (3B, ovaj split): {results_df['cosine_mrr'].iloc[0]:.4f}",
                  f"Referenca -- 650M konfiguracija na 650M embeddinzima (LOCO, drugi run): MRR=0.1259",
                  f"Referenca -- ISTI (650M) config na 3B embeddinzima BEZ retuninga (LOCO): MRR=0.0395 (kolaps)",
                  ""]
for _, r in results_df.sort_values("mrr", ascending=False).iterrows():
    summary_lines.append(
        f"{r['label']:<45}{r['mrr']:<10.4f}delta={r['delta']:+.4f}  "
        f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{'  ZNACAJNO' if r['significant'] else ''}"
    )
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
