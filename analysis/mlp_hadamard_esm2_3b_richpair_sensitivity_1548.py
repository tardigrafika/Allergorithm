"""
Da li veci embedding (ESM-2 3B, dim 2560) moze biti koristan uz ODGOVARAJUCU
pairwise arhitekturu, umesto trenutnog plain Hadamard->MLP pristupa (koji na
3B, cak i retunovan standardize=True, ostaje znacajno GORI od BLAST-a i od
650M -- ml/loco_esm2_3b_retuned_mlp_hadamard_1548.py). Korisnicki zahtev,
3 prioriteta:
  1. richconcat: [eA, eB, |eA-eB|, eA*eB] -> MLP (umesto samo eA*eB)
  2. Naucena projekcija 2560->256/512 -> ISTI pairwise feature-i -> MLP
  3. LayerNorm/standardizacija PRE pairwise operacije (ne posle, kao
     postojeci standardize=True koji z-score-uje FINALNI feature vektor)

Isti dataset/split/protokol kao postojeci 3B benchmark (ml/loco_esm2_3b_
vs_esm2_650m_1548.py, analysis/mlp_hadamard_esm2_3b_sensitivity_1548.py) --
group_aware_split 80/20, training_eligible_pairs(), NEG_PER_POS=10, isti
seed. Evaluacija (retrieval_evaluate/bootstrap_ci) NIJE menjana.

Brz single-split screening OVDE (ne LOCO -- LOCO tek za pobednika, ista
disciplina kao svuda). Cilj: da li ijedna kombinacija prevazilazi trenutni
MRR ~0.113 (LOCO broj plain-hadamard-retunovan) na fer (standardize/
retunovan) baznoj liniji.

Izlaz:
    output/mlp_hadamard_esm2_3b_richpair_sensitivity_1548_summary.txt
    output/mlp_hadamard_esm2_3b_richpair_sensitivity_1548_results.csv
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
from ml.pipeline.models.classifiers.projected_mlp import ProjectedMLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_esm2_3b_richpair_sensitivity_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_esm2_3b_richpair_sensitivity_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10

print("Loading ESM-2 3B dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
train_pairs = training_eligible_pairs(train_pairs)
n_train_neg = len(train_pairs) * NEG_PER_POS
train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
print(f"Train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}, dim: {dataset.embedding_matrix.shape[1]}\n")

BASE_MLP = dict(hidden_dims=[32], dropout=[0.3], learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3,
                  batch_size=64, max_epochs=300, patience=20, val_fraction=0.15)
BASE_PROJ = dict(hidden_dims=[32], dropout=[0.3], learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3,
                   batch_size=64, max_epochs=300, patience=20, val_fraction=0.15)

# (label, classifier_class, params)
GRID = [
    # --- Referenca: fer-retunovan plain hadamard (LOCO potvrdjeno MRR=0.1131) ---
    ("ref_hadamard_stdTrue", MLPPairClassifier,
     dict(input_encoding="hadamard", standardize=True, **BASE_MLP)),

    # --- Prioritet 1: richconcat [eA,eB,|eA-eB|,eA*eB] ---
    ("richconcat_stdFalse_preL2False", MLPPairClassifier,
     dict(input_encoding="richconcat", standardize=False, pre_l2_normalize=False, **BASE_MLP)),
    ("richconcat_stdTrue_preL2False", MLPPairClassifier,
     dict(input_encoding="richconcat", standardize=True, pre_l2_normalize=False, **BASE_MLP)),
    # --- Prioritet 3 (na richconcat): pre-L2-normalizacija UMESTO post-standardize ---
    ("richconcat_stdFalse_preL2True", MLPPairClassifier,
     dict(input_encoding="richconcat", standardize=False, pre_l2_normalize=True, **BASE_MLP)),
    # --- Prioritet 3 (na hadamard): pre-L2-normalizacija na postojecem hadamard enkodingu ---
    ("hadamard_stdFalse_preL2True", MLPPairClassifier,
     dict(input_encoding="hadamard", standardize=False, pre_l2_normalize=True, **BASE_MLP)),

    # --- Prioritet 2: naucena projekcija 2560->{256,512}, hadamard combine ---
    ("proj256_hadamard_noLN", ProjectedMLPPairClassifier,
     dict(proj_dim=256, combine="hadamard", pre_layernorm=False, **BASE_PROJ)),
    ("proj256_hadamard_LN", ProjectedMLPPairClassifier,
     dict(proj_dim=256, combine="hadamard", pre_layernorm=True, **BASE_PROJ)),
    ("proj512_hadamard_noLN", ProjectedMLPPairClassifier,
     dict(proj_dim=512, combine="hadamard", pre_layernorm=False, **BASE_PROJ)),

    # --- Prioritet 2 + 1 kombinovano: projekcija + richconcat combine ---
    ("proj256_richconcat_noLN", ProjectedMLPPairClassifier,
     dict(proj_dim=256, combine="richconcat", pre_layernorm=False, **BASE_PROJ)),
    ("proj512_richconcat_LN", ProjectedMLPPairClassifier,
     dict(proj_dim=512, combine="richconcat", pre_layernorm=True, **BASE_PROJ)),
]

print(f"Prolazim kroz {len(GRID)} konfiguracija...\n")
results = []
for label, cls, params in GRID:
    clf = cls(params=params, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    results.append({
        "label": label, "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "stopped_epoch": getattr(clf, "stopped_epoch", None),
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:32s}  epoch={results[-1]['stopped_epoch']:4}  MRR={summary['mrr']:.4f}  "
          f"delta={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}",
          flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 90, "MLP na ESM-2 3B: richconcat / naucena projekcija / pre-pairwise standardizacija "
                  "(single-split screening)", "=" * 90, "",
                  f"Cosine baseline MRR (ovaj split): {results_df['cosine_mrr'].iloc[0]:.4f}",
                  f"Referenca -- fer-retunovan plain hadamard, LOCO potvrdjeno: MRR=0.1131 (h32) / 0.1136 (h64)",
                  ""]
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
