"""
Dijagnostika: da li je losiji LOCO rezultat fer-retunovanog MLP(hadamard)
na ESM-2 3B (MRR=0.1131, znacajno gore od BLAST-a/650M) posledica
OVERFITTING-a, ili neceg drugog (npr. genuine train->held-out-familija
transfer gap). Korisnicki zahtev: "kako da znam da nije overfitovao".

Za svaki od 40 LOCO folda, loguje se best_val_auc (postignut tokom
early-stopping-a, VEC racunat unutar MLPPairClassifier.fit() ali NIGDE
ranije ispisan/sacuvan -- videti ml/pipeline/models/classifiers/mlp.py
linije 151-187) ZAJEDNO sa test-fold MRR. Logika: ako je val_auc visok
(model dobro razdvaja pozitive/negative na held-out VALIDACIONOM skupu iz
ISTE trening-distribucije) ali test MRR i dalje los, to NIJE klasican
overfitting (model generalise dobro na sopstvenu val distribuciju) --
ukazuje na transfer gap SPECIFICNO ka potpuno nevidjenoj familiji (LOCO
test), razlicit i suptilniji problem od overfitting-a. Ako je val_auc i
sam nizak, to je konzistentnije sa overfitting/underfitting na samu
trening distribuciju.

Isti config kao vec potvrdjen h32_stdTrue (ml/loco_esm2_3b_retuned_mlp_
hadamard_1548.py) -- SAMO dodato logovanje, ne menja rezultat.

Izlaz:
    output/loco_esm2_3b_valauc_diagnostic_1548_per_fold.csv
    output/loco_esm2_3b_valauc_diagnostic_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.pkl")
METADATA_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
PER_FOLD_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm2_3b_valauc_diagnostic_1548_per_fold.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm2_3b_valauc_diagnostic_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10

MLP_PARAMS = dict(input_encoding="hadamard", standardize=True, hidden_dims=[32], dropout=[0.3],
                    learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                    max_epochs=300, patience=20, val_fraction=0.15)

print("Loading ESM-2 3B dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS_3B, METADATA_3B, GOLD)

folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}", flush=True)


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


fold_records = []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_pairs_clean = training_eligible_pairs(train_pairs)
    train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
    train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED + fold_idx,
                                              dataset.positive_pair_set)

    if len(train_pairs_clean) < 5:
        fold_records.append({"fold": fold_idx, "n_train_clean": len(train_pairs_clean), "n_test_pairs": len(test_pairs),
                              "best_val_auc": None, "stopped_epoch": None, "test_mrr": None})
        continue

    mlp = MLPPairClassifier(params=MLP_PARAMS, seed=SEED + fold_idx)
    mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    rrs = []
    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]
            scores = mlp.score_all(query_id)
            rank = ranks_from_scores(scores, qi)
            rrs.append(1.0 / int(rank[ti]))
    test_mrr = float(np.mean(rrs)) if rrs else None

    fold_records.append({
        "fold": fold_idx, "n_train_clean": len(train_pairs_clean), "n_test_pairs": len(test_pairs),
        "best_val_auc": mlp.best_val_auc, "stopped_epoch": mlp.stopped_epoch, "test_mrr": test_mrr,
    })
    elapsed = time.time() - overall_start
    print(f"  fold {fold_idx + 1}/{K_FOLDS}: n_train={len(train_pairs_clean)}, "
          f"val_auc={mlp.best_val_auc:.4f}, stopped_epoch={mlp.stopped_epoch}, "
          f"test_mrr={test_mrr:.4f} ({elapsed/60:.1f} min)", flush=True)

df = pd.DataFrame(fold_records)
df.to_csv(PER_FOLD_OUTPUT, index=False)
total_elapsed = time.time() - overall_start
print(f"\nSaved: {PER_FOLD_OUTPUT}", flush=True)
print(f"Ukupno: {total_elapsed/60:.1f} min", flush=True)

df_valid = df.dropna(subset=["best_val_auc", "test_mrr"])
corr = df_valid["best_val_auc"].corr(df_valid["test_mrr"])

summary_lines = [
    "=" * 80, "Dijagnostika: validation AUC vs LOCO test MRR (ESM-2 3B, fer-retunovan hadamard h32)",
    "=" * 80, "",
    f"Foldova sa validnim fit-om: {len(df_valid)}/{K_FOLDS}",
    f"Val AUC:  mean={df_valid['best_val_auc'].mean():.4f}  median={df_valid['best_val_auc'].median():.4f}  "
    f"min={df_valid['best_val_auc'].min():.4f}  max={df_valid['best_val_auc'].max():.4f}",
    f"Test MRR: mean={df_valid['test_mrr'].mean():.4f}  median={df_valid['test_mrr'].median():.4f}  "
    f"min={df_valid['test_mrr'].min():.4f}  max={df_valid['test_mrr'].max():.4f}",
    f"Stopped epoch: mean={df_valid['stopped_epoch'].mean():.1f}  "
    f"(max_epochs=300, patience=20 -- rano zaustavljanje ako << 300)",
    "",
    f"Korelacija (val_auc, test_mrr) preko foldova: {corr:+.3f}",
    "",
    "Tumacenje:",
    "  - Ako je val_auc VISOK (npr. >0.85) skoro svuda, a test_mrr i dalje nizak/varijabilan: "
    "NIJE klasican overfitting (model dobro fituje sopstvenu val distribuciju) -- ukazuje na "
    "transfer gap ka held-out familiji specificno, ne na memorisanje trening skupa.",
    "  - Ako je val_auc I SAM nizak/nestabilan: konzistentnije sa underfitting/nestabilnim treningom "
    "na visoko-dimenzionom (2560) ulazu sa malo trening parova po foldu.",
    "  - Rano zaustavljanje (stopped_epoch << 300) je OCEKIVANO i zdravo (early stopping radi), "
    "NE samo po sebi znak problema.",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
