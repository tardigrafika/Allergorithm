"""
Learning curve za MLP: da li je slab MLP rezultat (svih 8 konfiguracija
znacajno gore od cosine-a u analysis/mlp_sensitivity_1548.py) posledica
NEDOVOLJNO PODATAKA za mrezu ove velicine, ili je isti "plafon reprezentacije"
koji smo vec potvrdili za RF+BLAST (analysis/learning_curve_1548.py, ravna
kriva 25%->100%)?

Testiramo DVE MLP konfiguracije preko 4 frakcije (25/50/75/100% train parova):
  - baseline_256_64 (~344.7k parametara, najgori rezultat u sweep-u) --
    ako joj VELIKA arhitektura treba vise podataka da bi generalizovala,
    ocekujemo da MRR jasno RASTE sa N (za razliku od RF-a).
  - baseline_l2_in_loss (ista arhitektura, ali L2-u-loss-u -- najbolji MLP
    rezultat u sweep-u) -- da vidimo da li regularizacija menja OBLIK krive,
    ne samo njen nivo.

Podskupovi su UGNJEZDENI (isti seed/shuffle kao learning_curve_1548.py) radi
direktne uporedivosti sa RF krivom.

Izlaz:
    output/mlp_learning_curve_1548_summary.txt
    output/mlp_learning_curve_1548_results.csv
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
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_learning_curve_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_learning_curve_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10
FRACTIONS = [0.25, 0.5, 0.75, 1.0]

CONFIGS = [
    ("baseline_256_64", dict(hidden_dims=[256, 64], dropout=[0.3, 0.2], weight_decay=1e-4, l2_lambda=0.0)),
    ("baseline_l2_in_loss", dict(hidden_dims=[256, 64], dropout=[0.3, 0.2], weight_decay=0.0, l2_lambda=1e-3)),
]

print("Loading dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs_full, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
print(f"Train parova (100%): {len(train_pairs_full)}  Test parova: {len(test_pairs)}")

cosine_matrix = cosine_similarity(dataset.embedding_matrix)
rng = np.random.default_rng(SEED)
shuffled_order = rng.permutation(len(train_pairs_full))

print(f"\nProlazim kroz {len(CONFIGS)} konfiguracije x {len(FRACTIONS)} frakcije...\n")
results = []
cosine_mrr_ref = None
for cfg_label, cfg_params in CONFIGS:
    for frac in FRACTIONS:
        n_take = max(2, int(round(frac * len(train_pairs_full))))
        subset_idx = shuffled_order[:n_take]
        train_pairs = [train_pairs_full[i] for i in subset_idx]

        n_train_neg = len(train_pairs) * NEG_PER_POS
        train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)

        params = dict(cfg_params, batch_size=64, max_epochs=200, patience=20, learning_rate=1e-3)
        clf = MLPPairClassifier(params=params, seed=SEED)
        clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

        retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                            cosine_matrix=cosine_matrix)
        summary = summarize_retrieval(retrieval_df)
        delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                     n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")
        cosine_mrr_ref = summary["cosine_mrr"]

        results.append({
            "config": cfg_label, "fraction": frac, "n_train_pairs": len(train_pairs),
            "stopped_epoch": getattr(clf, "stopped_epoch", None),
            "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
            "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
            "significant": delta_stats["significant"],
        })
        sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
        print(f"  {cfg_label:22s} frac={frac:.2f}  n_train={len(train_pairs):4d}  MRR={summary['mrr']:.4f}  "
              f"delta={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}",
              flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "MLP learning curve: MRR vs kolicina trening parova (2 konfiguracije)", "=" * 80, "",
                  f"Cosine baseline MRR (fiksno): {cosine_mrr_ref:.4f}", ""]
for cfg_label, _ in CONFIGS:
    summary_lines.append(f"\n{cfg_label}:")
    sub = results_df[results_df["config"] == cfg_label].sort_values("fraction")
    for _, r in sub.iterrows():
        summary_lines.append(
            f"  frac={r['fraction']:.2f}  n_train={r['n_train_pairs']:4.0f}  MRR={r['mrr']:.4f}  "
            f"delta={r['delta']:+.4f}  CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
            f"{'  ZNACAJNO' if r['significant'] else ''}"
        )
    mrr_25 = sub.loc[sub["fraction"] == 0.25, "mrr"].iloc[0]
    mrr_100 = sub.loc[sub["fraction"] == 1.0, "mrr"].iloc[0]
    summary_lines.append(f"  MRR 25%->100%: {mrr_25:.4f} -> {mrr_100:.4f}  (razlika {mrr_100-mrr_25:+.4f})")
    if mrr_100 - mrr_25 > 0.02:
        summary_lines.append("  Kriva JASNO RASTE - konzistentno sa 'nedostaje podataka' hipotezom za ovu konfiguraciju.")
    else:
        summary_lines.append("  Kriva JE RAVNA/blaga - NE izgleda kao cist problem kolicine podataka za ovu konfiguraciju.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
