"""
Poslednji pokusaj da se ESM-2 3B (2560-dim) iskoristi bolje od plain
hadamard-a: FIKSNA (ne-naucena) PCA redukcija dimenzija PRE hadamard
kombinovanja, umesto naucene projekcije (koja je katastrofalno kolabirala
-- analysis/mlp_hadamard_esm2_3b_richpair_sensitivity_1548.py, proj256/512
hadamard config-i, MRR 0.016-0.042).

Zasto ovo moze biti drugacije: naucena projekcija (nn.Linear 2560->256,
trenirana END-TO-END sa MLP glavom) ima ~655K parametara koje mora da
nauci SAMO iz ~785 obelezenih trening parova -- ozbiljno underdetermined,
otud kolaps. PCA se fituje NA CELOM POOL-u (1535 proteina, BEZ labela,
unsupervised) -- mnogo vise podataka za stabilnu procenu projekcionih
pravaca, bez rizika od overfitting-a specificno na labela.

VAZNO ogranicenje unapred: cosine na sirovom 3B TACNO izjednacuje cosine
na 650M (ml/loco_esm2_3b_vs_esm2_650m_1548.py, delta=-0.0007, CI ukljucuje
0) -- 3B ne nosi VISE sirovog signala od 650M. Ako PCA cosine (dole) NE
prevazidje sirovi cosine, malo je verovatno da ce bilo koja dalja obrada
(hadamard, MLP) magicno izvuci nesto sto vec nije tamo -- ocekivanja
namerno skromna.

Isti dataset/split/protokol kao svi ostali 3B eksperimenti -- group_aware_
split 80/20, training_eligible_pairs(), NEG_PER_POS=10.

Izlaz:
    output/mlp_hadamard_esm2_3b_pca_sensitivity_1548_summary.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_esm2_3b_pca_sensitivity_1548_summary.txt")

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
print(f"Train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}\n")

# PCA fitovan na CEO pool (1535 proteina), BEZ labela -- unsupervised, ne trening-set-limited.
raw_cosine_matrix = cosine_similarity(dataset.embedding_matrix)
raw_summary_ref = None  # racunamo referentni broj u istoj petlji ispod za fer poredjenje istim kodom

BASE_MLP = dict(hidden_dims=[32], dropout=[0.3], learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3,
                  batch_size=64, max_epochs=300, patience=20, val_fraction=0.15)

results = []
summary_lines = ["=" * 90, "MLP(hadamard) na ESM-2 3B posle FIKSNE PCA redukcije (256/512) -- poslednji pokusaj",
                  "=" * 90, ""]

configs = [("raw_2560_stdTrue_ref", None)] + [(f"pca_{d}", d) for d in (256, 512)]

for label, pca_dim in configs:
    if pca_dim is None:
        emb_matrix = dataset.embedding_matrix
        standardize = True  # fer-retunovana referenca, isti config kao vec potvrdjeno
    else:
        pca = PCA(n_components=pca_dim, random_state=SEED)
        emb_matrix = pca.fit_transform(dataset.embedding_matrix).astype(np.float32)
        explained = pca.explained_variance_ratio_.sum()
        print(f"PCA({pca_dim}): objasnjena varijansa = {explained:.3f}", flush=True)
        standardize = True  # isti standardize=True kao fer-retunovan hadamard config

    cosine_matrix_this = cosine_similarity(emb_matrix)

    clf = MLPPairClassifier(params=dict(input_encoding="hadamard", standardize=standardize, **BASE_MLP), seed=SEED)
    clf.fit(train_pairs, train_negatives, emb_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, emb_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix_this)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    results.append({
        "label": label, "pca_dim": pca_dim, "mlp_mrr": summary["mrr"], "cosine_mrr_this_space": summary["cosine_mrr"],
        "delta_vs_own_cosine": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    line = (f"  {label:24s}  MLP_MRR={summary['mrr']:.4f}  cosine_MRR(ovaj prostor)={summary['cosine_mrr']:.4f}  "
            f"MLP_delta_vs_cosine={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}")
    print(line, flush=True)
    summary_lines.append(line)

summary_lines.append("")
summary_lines.append("Referenca -- raw 2560 cosine (ceo dataset, LOCO ranije potvrdjeno): MRR=0.1204, "
                      "tacno izjednaceno sa 650M cosine (delta=-0.0007, CI ukljucuje 0)")
summary_lines.append("Referenca -- fer-retunovan hadamard 2560, LOCO potvrdjeno: MRR=0.1131 (h32)")
summary_lines.append("")
summary_lines.append("Kljucno pitanje: da li PCA cosine_MRR (bilo koja dimenzija) PREVAZILAZI raw 2560 cosine "
                      "(0.1204 LOCO / referentni broj na ovom splitu iznad za 'raw_2560_stdTrue_ref' red) -- "
                      "ako ne, PCA ne dodaje signal, samo gubi informaciju kompresijom.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
