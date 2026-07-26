"""
MLP classifier (Pristup A) 1443
  +++evidence-level sample weighting tokom treniranja.

Confirmed / Strong evidence: 244 → weight 1.00
Suspected: 76 → weight 0.65
Inferred / family-level: 1.112 → weight 0.45

Ovo sprecava da oko 78% dataseta koji cine samo inferred parovi dominira treningom

Weights se koriste iskljucivo tokom treniranja u loss funkciji!!
validation loss, ROC-AUC.. racunaju se bez weightinga
Izlaz:
    output/mlp_model_1443.pt
    output/mlp_retrieval_results_1443.csv
    output/mlp_summary_1443.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
RF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_retrieval_results_1443.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODEL_OUTPUT = OUTPUT_DIR / "mlp_model_1443.pt"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "mlp_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mlp_summary_1443.txt"

OLD_MLP_RESULTS = Path("/home/lana/ALERGRAF/output/mlp_retrieval_results.csv")  # 296-pair version


# =====================================================
# CONFIGURATION (identical to the 296-pair MLP classifier script)
# =====================================================

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]

VAL_FRACTION = 0.15
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EVIDENCE_WEIGHTS = {
    "Confirmed/Strong": 1.00,
    "Suspected": 0.65,
    "Inferred/family-level": 0.45,
}

np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================================================
# LOAD DATA
# =====================================================

print("\n==============================")
print("LOADING DATA")
print("==============================")

with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Proteins in embeddings: {len(embeddings_dict)}")

metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()
print(f"Metadata rows with embeddings: {len(metadata)}")

gold_raw = pd.read_csv(GOLD)
print(f"Rows in gold file: {len(gold_raw)}")


# =====================================================
# EVIDENCE-LEVEL FILTERING + WEIGHT BUCKETING
# =====================================================

print("\n==============================")
print("EVIDENCE-LEVEL FILTERING + SAMPLE WEIGHTS")
print("==============================")

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()

print(f"Rows excluded as negative/contested/risky: {len(excluded)}")
print(f"Positive gold-standard pairs retained    : {len(gold)}")


def evidence_bucket(value):
    if value.startswith("Confirmed") or value.startswith("Strong evidence"):
        return "Confirmed/Strong"
    if value.startswith("Suspected"):
        return "Suspected"
    if value.startswith("Inferred"):
        return "Inferred/family-level"
    return "UNMAPPED"


gold["evidence_bucket"] = gold["evidence_level"].map(evidence_bucket)
unmapped_count = int((gold["evidence_bucket"] == "UNMAPPED").sum())
if unmapped_count:
    print(f"WARNING: {unmapped_count} rows had an unrecognized evidence_level prefix "
          f"and will default to weight 0.45 (Inferred-level, most conservative).")

print("\nEvidence-level bucket counts (sample weight used in training):")
for bucket, weight in EVIDENCE_WEIGHTS.items():
    count = int((gold["evidence_bucket"] == bucket).sum())
    print(f"  {bucket:<24}: {count:5d} pairs  (weight={weight})")


# =====================================================
# NAME -> ALLERGEN_ID MAPPING
# =====================================================

name_to_id = {}
duplicate_names = 0
for _, row in metadata.iterrows():
    official_name = str(row["official_name"]).strip()
    if official_name == "" or official_name.lower() == "nan":
        continue
    if official_name in name_to_id:
        duplicate_names += 1
        continue
    name_to_id[official_name] = row["allergen_id"]

print(f"\nOfficial names mapped : {len(name_to_id)}")


# =====================================================
# EMBEDDING MATRIX + COSINE MATRIX
# =====================================================

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array(
    [embeddings_dict[allergen_id] for allergen_id in all_ids],
    dtype=np.float64,
)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

cosine_similarity_matrix = cosine_similarity(embedding_matrix)


# =====================================================
# MAP GOLD STANDARD PAIRS TO ALLERGEN IDS (weight carried per pair)
# =====================================================

gold_pairs = []
missing_pairs = 0

for _, row in gold.iterrows():
    name_1 = str(row["allergen_id_1"]).strip()
    name_2 = str(row["allergen_id_2"]).strip()

    if name_1 not in name_to_id or name_2 not in name_to_id:
        missing_pairs += 1
        continue

    id_1 = name_to_id[name_1]
    id_2 = name_to_id[name_2]

    if id_1 not in id_to_index or id_2 not in id_to_index or id_1 == id_2:
        missing_pairs += 1
        continue

    bucket = row["evidence_bucket"]
    weight = EVIDENCE_WEIGHTS.get(bucket, 0.45)

    gold_pairs.append(
        {
            "pair_id": row["pair_id"],
            "id_1": id_1,
            "id_2": id_2,
            "name_1": name_1,
            "name_2": name_2,
            "family_1": row["family_1"],
            "family_2": row["family_2"],
            "evidence_bucket": bucket,
            "weight": weight,
        }
    )

print(f"Mapped gold pairs : {len(gold_pairs)}")
print(f"Missing/unmapped  : {missing_pairs}")

positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}


# =====================================================
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm to other scripts)
# =====================================================

print("\n==============================")
print("PROTEIN-LEVEL SPLIT (leakage prevention)")
print("==============================")

parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for p in gold_pairs:
    union(p["id_1"], p["id_2"])

components = {}
for protein_id in parent:
    root = find(protein_id)
    components.setdefault(root, set()).add(protein_id)

component_list = list(components.values())
print(f"Connected components in gold-standard graph: {len(component_list)}")
print(f"Proteins covered by gold-standard graph     : {sum(len(c) for c in component_list)}")

rng = np.random.default_rng(SEED)
order = rng.permutation(len(component_list))
gold_protein_count = sum(len(c) for c in component_list)
target_component_test = round(TEST_FRACTION * gold_protein_count)

train_ids, test_ids = set(), set()
running_test = 0
for idx in order:
    component = component_list[idx]
    if running_test < target_component_test:
        test_ids |= component
        running_test += len(component)
    else:
        train_ids |= component

free_proteins = [pid for pid in all_ids if pid not in train_ids and pid not in test_ids]
free_proteins = list(free_proteins)
rng.shuffle(free_proteins)
n_free_test = round(TEST_FRACTION * len(free_proteins))
test_ids |= set(free_proteins[:n_free_test])
train_ids |= set(free_proteins[n_free_test:])

assert train_ids.isdisjoint(test_ids), "protein split is not disjoint (bug)"
assert len(train_ids) + len(test_ids) == len(all_ids), "split does not cover all proteins"

print(f"Train proteins: {len(train_ids)}  ({len(train_ids)/len(all_ids):.1%})")
print(f"Test proteins : {len(test_ids)}  ({len(test_ids)/len(all_ids):.1%})")

train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
cross_split_pairs = len(gold_pairs) - len(train_positive_pairs) - len(test_positive_pairs)

print(f"Train positive pairs: {len(train_positive_pairs)}")
print(f"Test positive pairs : {len(test_positive_pairs)}")
print(f"Cross-split pairs (must be 0): {cross_split_pairs}")
assert cross_split_pairs == 0, "a gold pair spans both splits (leakage, should be impossible)"


# =====================================================
# NEGATIVE SAMPLING (identical function; negatives always get weight 1.0)
# =====================================================

def sample_negative_pairs(protein_pool, n_needed, seed):
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    negatives = set()
    max_attempts = n_needed * 50 + 2000
    attempts = 0
    while len(negatives) < n_needed and attempts < max_attempts:
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1
        if pair in positive_pair_set or pair in negatives:
            continue
        negatives.add(pair)
    if len(negatives) < n_needed:
        print(f"WARNING: only sampled {len(negatives)}/{n_needed} negatives")
    return sorted(negatives)


n_train_neg = len(train_positive_pairs) * NEG_PER_POS
n_test_neg = len(test_positive_pairs) * NEG_PER_POS

train_negative_pairs = sample_negative_pairs(train_ids, n_train_neg, seed=SEED)
test_negative_pairs = sample_negative_pairs(test_ids, n_test_neg, seed=SEED + 1)

print(f"\nTrain negative pairs sampled: {len(train_negative_pairs)} (target ratio {NEG_PER_POS}:1)")
print(f"Test negative pairs sampled : {len(test_negative_pairs)} (target ratio {NEG_PER_POS}:1)")


# =====================================================
# FEATURE CONSTRUCTION (identical function, PLUS per-example weight array)
# =====================================================

def pairwise_features(emb_a, emb_b):
    emb_a = np.atleast_2d(emb_a)
    emb_b = np.atleast_2d(emb_b)
    abs_diff = np.abs(emb_a - emb_b)
    dot = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    cosine = dot / (norm_a * norm_b + 1e-12)
    return np.hstack([abs_diff, cosine.reshape(-1, 1)])


def build_feature_matrix(positive_pairs, negative_pairs):
    rows_a, rows_b, labels, weights = [], [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        labels.append(1)
        weights.append(p["weight"])
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        labels.append(0)
        weights.append(1.0)  # sampled negatives have no evidence_level -> full weight
    X = pairwise_features(np.array(rows_a), np.array(rows_b))
    y = np.array(labels, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)
    return X, y, w


print("\n==============================")
print("BUILDING FEATURE MATRICES")
print("==============================")

X_train, y_train, w_train = build_feature_matrix(train_positive_pairs, train_negative_pairs)
X_test, y_test, w_test = build_feature_matrix(test_positive_pairs, test_negative_pairs)

print(f"Train: X={X_train.shape}, positives={int(y_train.sum())}, negatives={int((y_train == 0).sum())}")
print(f"Test : X={X_test.shape}, positives={int(y_test.sum())}, negatives={int((y_test == 0).sum())}")
print(f"Train weight range: [{w_train.min():.2f}, {w_train.max():.2f}], mean={w_train.mean():.3f}")


# =====================================================
# FEATURE STANDARDIZATION (fit on TRAIN only)
# =====================================================

feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)
feature_std[feature_std < 1e-8] = 1.0


def scale(X):
    return (X - feature_mean) / feature_std


X_train_scaled = scale(X_train).astype(np.float32)
X_test_scaled = scale(X_test).astype(np.float32)


# =====================================================
# FIT/VALIDATION SPLIT FOR EARLY STOPPING (weights carried through,
# but NOT used for the val loss/AUC computed below -- see docstring)
# =====================================================

X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
    X_train_scaled, y_train, w_train,
    test_size=VAL_FRACTION, random_state=SEED, stratify=y_train,
)

print(f"\nFit examples        : {X_fit.shape[0]} (positives={int(y_fit.sum())})")
print(f"Validation examples : {X_val.shape[0]} (positives={int(y_val.sum())})")


# =====================================================
# MODEL (identical architecture)
# =====================================================

class PairMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


input_dim = X_train.shape[1]
model = PairMLP(input_dim)

print("\n==============================")
print("MODEL ARCHITECTURE")
print("==============================")
print(model)

n_pos_fit = float(y_fit.sum())
n_neg_fit = float((y_fit == 0).sum())
pos_weight = torch.tensor(n_neg_fit / n_pos_fit, dtype=torch.float32)
print(f"\npos_weight (neg/pos on fit set, class-imbalance correction): {pos_weight.item():.4f}")
print("(applied together with, but independently of, the evidence-level sample weight)")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

X_fit_t = torch.from_numpy(X_fit)
y_fit_t = torch.from_numpy(y_fit)
w_fit_t = torch.from_numpy(w_fit)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

val_criterion = nn.BCEWithLogitsLoss()  # UNWEIGHTED -- used only for early-stopping monitoring


# =====================================================
# TRAINING LOOP WITH EARLY STOPPING
# =====================================================

print("\n==============================")
print("TRAINING MLP (evidence-weighted loss)")
print("==============================")
print(f"Max epochs: {MAX_EPOCHS}, patience: {PATIENCE}, batch size: {BATCH_SIZE}, "
      f"lr: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY}")

n_fit = X_fit_t.shape[0]
batch_rng = np.random.default_rng(SEED)

best_val_auc = -np.inf
best_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    permutation = batch_rng.permutation(n_fit)

    epoch_loss = 0.0
    for start in range(0, n_fit, BATCH_SIZE):
        batch_idx = permutation[start:start + BATCH_SIZE]
        xb = X_fit_t[batch_idx]
        yb = y_fit_t[batch_idx]
        wb = w_fit_t[batch_idx]

        optimizer.zero_grad()
        logits = model(xb)
        per_example_loss = criterion(logits, yb)          # class-imbalance-weighted, per example
        weighted_loss = (per_example_loss * wb).mean()      # + evidence-level weight
        weighted_loss.backward()
        optimizer.step()

        epoch_loss += weighted_loss.item() * len(batch_idx)

    epoch_loss /= n_fit

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_loss = val_criterion(val_logits, y_val_t).item()  # unweighted
    val_auc = roc_auc_score(y_val, val_probs)                  # unweighted

    improved = val_auc > best_val_auc
    if improved:
        best_val_auc = val_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0 or improved:
        marker = " *" if improved else ""
        print(f"Epoch {epoch:3d}: fit_loss={epoch_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}{marker}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no val_auc improvement for {PATIENCE} epochs).")
        break

model.load_state_dict(best_state)
model.eval()
print(f"\nBest validation ROC-AUC: {best_val_auc:.4f}")


# =====================================================
# A) CLASSIFICATION METRICS (UNWEIGHTED, held-out test pairs)
# =====================================================

print("\n==============================")
print("CLASSIFICATION METRICS (test split, unweighted)")
print("==============================")

X_test_t = torch.from_numpy(X_test_scaled)
with torch.no_grad():
    test_logits = model(X_test_t)
    y_proba = torch.sigmoid(test_logits).numpy()
y_pred = (y_proba >= 0.5).astype(int)

clf_metrics = {
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": average_precision_score(y_test, y_proba),
}
conf_matrix = confusion_matrix(y_test, y_pred)

for name, value in clf_metrics.items():
    print(f"{name:10s}: {value:.4f}")
print("\nConfusion matrix (rows=true, cols=predicted, order=[0,1]):")
print(conf_matrix)


# =====================================================
# B) RETRIEVAL EVALUATION
# =====================================================

print("\n==============================")
print("RETRIEVAL EVALUATION (Hits@K / MRR)")
print("==============================")
print(f"Test-split gold pairs: {len(test_positive_pairs)}  "
      f"-> up to {2 * len(test_positive_pairs)} retrieval queries")

full_embedding_batch = embedding_matrix
retrieval_results = []

model.eval()
with torch.no_grad():
    for p in test_positive_pairs:
        directions = [
            (p["id_1"], p["id_2"], p["name_1"], p["name_2"], p["family_1"], p["family_2"]),
            (p["id_2"], p["id_1"], p["name_2"], p["name_1"], p["family_2"], p["family_1"]),
        ]
        for query_id, target_id, query_name, target_name, family_q, family_t in directions:
            query_index = id_to_index[query_id]
            target_index = id_to_index[target_id]

            query_vec = embedding_matrix[query_index]
            query_batch = np.tile(query_vec, (len(all_ids), 1))

            X_candidates = pairwise_features(query_batch, full_embedding_batch)
            X_candidates_scaled = scale(X_candidates).astype(np.float32)

            logits = model(torch.from_numpy(X_candidates_scaled))
            mlp_scores = torch.sigmoid(logits).numpy()
            mlp_scores[query_index] = -np.inf

            mlp_ranked = np.argsort(mlp_scores)[::-1]
            mlp_rank = int(np.where(mlp_ranked == target_index)[0][0]) + 1
            mlp_reciprocal_rank = 1.0 / mlp_rank
            mlp_true_pair_probability = mlp_scores[target_index]

            cos_scores = cosine_similarity_matrix[query_index].copy()
            cos_scores[query_index] = -np.inf
            cos_ranked = np.argsort(cos_scores)[::-1]
            cos_rank = int(np.where(cos_ranked == target_index)[0][0]) + 1
            cos_reciprocal_rank = 1.0 / cos_rank

            retrieval_results.append({
                "pair_id": p["pair_id"],
                "query_allergen": query_name,
                "target_allergen": target_name,
                "query_allergen_id": query_id,
                "target_allergen_id": target_id,
                "query_family": family_q,
                "target_family": family_t,
                "evidence_bucket": p["evidence_bucket"],
                "mlp_probability": mlp_true_pair_probability,
                "mlp_rank": mlp_rank,
                "mlp_reciprocal_rank": mlp_reciprocal_rank,
                "mlp_hits_at_1": int(mlp_rank <= 1),
                "mlp_hits_at_5": int(mlp_rank <= 5),
                "mlp_hits_at_10": int(mlp_rank <= 10),
                "mlp_hits_at_20": int(mlp_rank <= 20),
                "cosine_rank": cos_rank,
                "cosine_reciprocal_rank": cos_reciprocal_rank,
                "cosine_hits_at_1": int(cos_rank <= 1),
                "cosine_hits_at_5": int(cos_rank <= 5),
                "cosine_hits_at_10": int(cos_rank <= 10),
                "cosine_hits_at_20": int(cos_rank <= 20),
            })

retrieval_df = pd.DataFrame(retrieval_results)

mlp_hits = {k: retrieval_df[f"mlp_hits_at_{k}"].mean() for k in TOP_K}
mlp_mrr = retrieval_df["mlp_reciprocal_rank"].mean()
cosine_test_hits = {k: retrieval_df[f"cosine_hits_at_{k}"].mean() for k in TOP_K}
cosine_test_mrr = retrieval_df["cosine_reciprocal_rank"].mean()

print(f"Retrieval queries evaluated: {len(retrieval_df)}")
for k in TOP_K:
    print(f"Hits@{k:<2d}: MLP={mlp_hits[k]:.4f}  Cosine={cosine_test_hits[k]:.4f}")
print(f"MRR    : MLP={mlp_mrr:.4f}  Cosine={cosine_test_mrr:.4f}")


# =====================================================
# SAVE OUTPUTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nRetrieval results saved to: {RETRIEVAL_OUTPUT}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "architecture": "1281 -> Linear(256) -> ReLU -> Dropout(0.3) -> "
                         "Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)",
        "evidence_weights": EVIDENCE_WEIGHTS,
        "seed": SEED,
    },
    MODEL_OUTPUT,
)
print(f"Model saved to: {MODEL_OUTPUT}")


# =====================================================
# CROSS-CHECK + LOAD RANDOM FOREST'S SAVED RESULTS (same test set)
# =====================================================

rf_available = RF_RETRIEVAL_RESULTS.exists()
rf_hits = {k: float("nan") for k in TOP_K}
rf_mrr = float("nan")
split_consistent = None

if rf_available:
    rf_df = pd.read_csv(RF_RETRIEVAL_RESULTS)
    if len(rf_df) == len(retrieval_df):
        merged = retrieval_df.merge(
            rf_df[["pair_id", "query_allergen", "cosine_reciprocal_rank"]],
            on=["pair_id", "query_allergen"], suffixes=("_mlp_script", "_rf_script"),
        )
        split_consistent = bool(np.allclose(
            merged["cosine_reciprocal_rank_mlp_script"], merged["cosine_reciprocal_rank_rf_script"]
        ))
        rf_hits = {k: rf_df[f"rf_hits_at_{k}"].mean() for k in TOP_K}
        rf_mrr = rf_df["rf_reciprocal_rank"].mean()
    else:
        print(f"WARNING: RF file has {len(rf_df)} queries, this script has {len(retrieval_df)} "
              f"-- skipping RF comparison.")
        rf_available = False

print(f"\nRF (1443) results file found: {rf_available}")
if rf_available:
    print(f"Split consistency check vs RF_1443 script: {split_consistent}")


# =====================================================
# COMPARISON WITH OLD 296-PAIR MLP CLASSIFIER
# =====================================================

old_available = OLD_MLP_RESULTS.exists()
old_mlp_hits = {k: float("nan") for k in TOP_K}
old_mlp_mrr = float("nan")
if old_available:
    old_df = pd.read_csv(OLD_MLP_RESULTS)
    old_mlp_hits = {k: old_df[f"mlp_hits_at_{k}"].mean() for k in TOP_K}
    old_mlp_mrr = old_df["mlp_reciprocal_rank"].mean()


# =====================================================
# FINAL SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("MLP CLASSIFIER (1443 dataset, evidence-weighted) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Random seed              : {SEED}")
summary_lines.append(f"Rows in gold file (1443) : {len(gold_raw)}")
summary_lines.append(f"Excluded (negative/contested/risky): {len(excluded)}")
summary_lines.append(f"Positive gold-standard pairs retained: {len(gold)}")
summary_lines.append("")
summary_lines.append("Evidence-level sample weights used in TRAINING LOSS only "
                      "(validation/test metrics are always unweighted):")
for bucket, weight in EVIDENCE_WEIGHTS.items():
    count = sum(1 for p in gold_pairs if p["evidence_bucket"] == bucket)
    summary_lines.append(f"  {bucket:<24}: {count:5d} pairs  (weight={weight})")
summary_lines.append("  Sampled negative pairs   : weight=1.0 (no evidence_level)")
summary_lines.append("")
summary_lines.append("Split strategy: group-aware, protein-level split (same algorithm/seed "
                      "as the 296-pair scripts, applied to the larger positive-pair graph).")
summary_lines.append(f"  Train proteins        : {len(train_ids)} ({len(train_ids)/len(all_ids):.1%})")
summary_lines.append(f"  Test proteins         : {len(test_ids)} ({len(test_ids)/len(all_ids):.1%})")
summary_lines.append(f"  Train positive pairs  : {len(train_positive_pairs)}")
summary_lines.append(f"  Test positive pairs   : {len(test_positive_pairs)}")
summary_lines.append(f"  Train examples (total): {len(y_train)}  (fit={len(y_fit)}, val={len(y_val)})")
summary_lines.append(f"  Test examples (total) : {len(y_test)}")
if rf_available:
    summary_lines.append(f"  Split consistency check vs RF_1443: {split_consistent}")
summary_lines.append("")
summary_lines.append(f"Training stopped at epoch {epoch} (best val ROC-AUC: {best_val_auc:.4f})")
summary_lines.append("")
summary_lines.append("Classification metrics (test split, unweighted):")
for name, value in clf_metrics.items():
    summary_lines.append(f"  {name:10s}: {value:.4f}")
summary_lines.append(f"  confusion matrix [ [TN FP] [FN TP] ]: {conf_matrix.tolist()}")
summary_lines.append("")
summary_lines.append(f"Retrieval evaluation: {len(retrieval_df)} queries "
                      f"({len(test_positive_pairs)} test pairs x 2 directions)")
summary_lines.append("")

header = f"{'Metric':<10}{'Cosine (same test)':<20}{'RF (1443)':<20}{'MLP (1443, weighted)':<20}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
for k in TOP_K:
    summary_lines.append(
        f"{'Hits@' + str(k):<10}{cosine_test_hits[k]:<20.4f}{rf_hits[k]:<20.4f}{mlp_hits[k]:<20.4f}"
    )
summary_lines.append(f"{'MRR':<10}{cosine_test_mrr:<20.4f}{rf_mrr:<20.4f}{mlp_mrr:<20.4f}")

if old_available:
    summary_lines.append("")
    summary_lines.append("Comparison with the OLD 296-pair MLP classifier (unweighted, "
                          "each on its own held-out test split):")
    header2 = f"{'Metric':<10}{'MLP (296, old)':<20}{'MLP (1443, new)':<20}"
    summary_lines.append(header2)
    summary_lines.append("-" * len(header2))
    for k in TOP_K:
        summary_lines.append(f"{'Hits@' + str(k):<10}{old_mlp_hits[k]:<20.4f}{mlp_hits[k]:<20.4f}")
    summary_lines.append(f"{'MRR':<10}{old_mlp_mrr:<20.4f}{mlp_mrr:<20.4f}")
else:
    summary_lines.append(f"\nNOTE: {OLD_MLP_RESULTS} not found -- run ml/mlp_baseline.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
