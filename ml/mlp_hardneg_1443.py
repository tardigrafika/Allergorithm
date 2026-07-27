"""
MLP klasifikator (Pristup A) sa mesavinom hard/easy negativa - 1443 dataset.

Ne menja mlp_baseline_1443.py (uniform-random negativi ostaju odvojena verzija).
30% negativa je hard (cross-family, geometrijski najslicniji), ostatak
nasumican iz celog pool-a (za razliku od triplet-hardneg, ovde NIJE 100% hard -
cuva sirinu distribucije). Evidence-level weighting isto kao u baseline verziji.

Rezultat: LOSIJE od baseline-a (MRR 0.174 -> 0.144) - hard negative mining
dosledno ne pomaze u ovom projektu.

Izlaz:
    output/mlp_hardneg_model_1443.pt
    output/mlp_hardneg_retrieval_results_1443.csv
    output/mlp_hardneg_summary_1443.txt
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
RF_HARDNEG_RETRIEVAL_RESULTS = Path(
    "/home/lana/ALERGRAF/output/random_forest_hardneg_retrieval_results_1443.csv"
)
BASELINE_MLP_RESULTS = Path("/home/lana/ALERGRAF/output/mlp_retrieval_results_1443.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODEL_OUTPUT = OUTPUT_DIR / "mlp_hardneg_model_1443.pt"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "mlp_hardneg_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mlp_hardneg_summary_1443.txt"


# =====================================================
# CONFIGURATION (identical to ml/mlp_baseline_1443.py plus the new
# HARD_FRAC / HARD_CANDIDATES_PER_DRAW negative-mining knobs)
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

HARD_FRAC = 0.3
HARD_CANDIDATES_PER_DRAW = 10

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
# EVIDENCE-LEVEL FILTERING + WEIGHT BUCKETING (identical to mlp_baseline_1443.py)
# =====================================================

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()
print(f"\nExcluded negative/contested/risky rows: {len(excluded)}")
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
# MAP GOLD STANDARD PAIRS TO ALLERGEN IDS (weight + family carried per pair)
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

family_map = {}
for p in gold_pairs:
    family_map.setdefault(p["id_1"], p["family_1"])
    family_map.setdefault(p["id_2"], p["family_2"])
print(f"Proteins with a known family label: {len(family_map)} (hard-negative eligible)")


# =====================================================
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm to other *_1443 scripts)
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
assert cross_split_pairs == 0, "a gold pair spans both splits (leakage, should be impossible)"


# =====================================================
# NEGATIVE SAMPLING -- MIXED HARD/EASY (identical logic to
# ml/random_forest_hardneg_1443.py)
# =====================================================

def cosine_sim_raw(id_a, id_b):
    ea = embedding_matrix[id_to_index[id_a]]
    eb = embedding_matrix[id_to_index[id_b]]
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12))


def sample_negative_pairs_mixed(protein_pool, n_needed, seed):
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)

    labeled = [pid for pid in pool if pid in family_map]
    fam_buckets = {}
    for pid in labeled:
        fam_buckets.setdefault(family_map[pid], []).append(pid)
    fam_names = list(fam_buckets.keys())

    negatives = set()
    n_hard_target = int(n_needed * HARD_FRAC)
    max_attempts = n_needed * 50 + 2000
    attempts = 0

    while len(negatives) < n_hard_target and len(fam_names) >= 2 and attempts < max_attempts:
        best_pair, best_similarity = None, -2.0
        for _ in range(HARD_CANDIDATES_PER_DRAW):
            attempts += 1
            fam_idx = local_rng.choice(len(fam_names), size=2, replace=False)
            fam_a, fam_b = fam_names[fam_idx[0]], fam_names[fam_idx[1]]
            a = fam_buckets[fam_a][local_rng.integers(0, len(fam_buckets[fam_a]))]
            b = fam_buckets[fam_b][local_rng.integers(0, len(fam_buckets[fam_b]))]
            pair = tuple(sorted((a, b)))
            if pair in positive_pair_set or pair in negatives:
                continue
            similarity = cosine_sim_raw(a, b)
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = pair
        if best_pair is not None:
            negatives.add(best_pair)

    n_hard_actual = len(negatives)

    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        if pair in positive_pair_set or pair in negatives:
            continue
        negatives.add(pair)

    if len(negatives) < n_needed:
        print(f"WARNING: only sampled {len(negatives)}/{n_needed} negatives")

    return sorted(negatives), n_hard_actual, len(negatives) - n_hard_actual


n_train_neg = len(train_positive_pairs) * NEG_PER_POS
n_test_neg = len(test_positive_pairs) * NEG_PER_POS

train_negative_pairs, n_train_hard, n_train_easy = sample_negative_pairs_mixed(train_ids, n_train_neg, seed=SEED)
test_negative_pairs, n_test_hard, n_test_easy = sample_negative_pairs_mixed(test_ids, n_test_neg, seed=SEED + 1)

print(f"\nTrain negatives: {len(train_negative_pairs)} "
      f"({n_train_hard} hard_cross_family_geometric / {n_train_easy} easy_random)")
print(f"Test negatives : {len(test_negative_pairs)} "
      f"({n_test_hard} hard_cross_family_geometric / {n_test_easy} easy_random)")


# =====================================================
# FEATURE CONSTRUCTION (identical function, plus per-example weight array)
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
        weights.append(1.0)
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
# FIT/VALIDATION SPLIT FOR EARLY STOPPING
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
print(f"\npos_weight (neg/pos on fit set): {pos_weight.item():.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

X_fit_t = torch.from_numpy(X_fit)
y_fit_t = torch.from_numpy(y_fit)
w_fit_t = torch.from_numpy(w_fit)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

val_criterion = nn.BCEWithLogitsLoss()  # unweighted, early-stopping monitor only


# =====================================================
# TRAINING LOOP WITH EARLY STOPPING
# =====================================================

print("\n==============================")
print("TRAINING MLP (hard negatives, evidence-weighted loss)")
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
        per_example_loss = criterion(logits, yb)
        weighted_loss = (per_example_loss * wb).mean()
        weighted_loss.backward()
        optimizer.step()

        epoch_loss += weighted_loss.item() * len(batch_idx)

    epoch_loss /= n_fit

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_loss = val_criterion(val_logits, y_val_t).item()
    val_auc = roc_auc_score(y_val, val_probs)

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
# A) CLASSIFICATION METRICS (unweighted)
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
        "hard_frac": HARD_FRAC,
        "seed": SEED,
    },
    MODEL_OUTPUT,
)
print(f"Model saved to: {MODEL_OUTPUT}")


# =====================================================
# COMPARISON WITH BASELINE (uniform-random-negative) MLP CLASSIFIER
# =====================================================

baseline_available = BASELINE_MLP_RESULTS.exists()
baseline_hits = {k: float("nan") for k in TOP_K}
baseline_mrr = float("nan")
split_consistent = None

if baseline_available:
    baseline_df = pd.read_csv(BASELINE_MLP_RESULTS)
    if len(baseline_df) == len(retrieval_df):
        merged = retrieval_df.merge(
            baseline_df[["pair_id", "query_allergen", "cosine_reciprocal_rank"]],
            on=["pair_id", "query_allergen"], suffixes=("_this_script", "_baseline_script"),
        )
        split_consistent = bool(np.allclose(
            merged["cosine_reciprocal_rank_this_script"], merged["cosine_reciprocal_rank_baseline_script"]
        ))
        baseline_hits = {k: baseline_df[f"mlp_hits_at_{k}"].mean() for k in TOP_K}
        baseline_mrr = baseline_df["mlp_reciprocal_rank"].mean()

print(f"\nBaseline MLP classifier results file found: {baseline_available}")
if baseline_available:
    print(f"Split consistency check vs baseline MLP script: {split_consistent}")

rf_hardneg_hits = {k: float("nan") for k in TOP_K}
rf_hardneg_mrr = float("nan")
if RF_HARDNEG_RETRIEVAL_RESULTS.exists():
    rf_hardneg_df = pd.read_csv(RF_HARDNEG_RETRIEVAL_RESULTS)
    if len(rf_hardneg_df) == len(retrieval_df):
        rf_hardneg_hits = {k: rf_hardneg_df[f"rf_hits_at_{k}"].mean() for k in TOP_K}
        rf_hardneg_mrr = rf_hardneg_df["rf_reciprocal_rank"].mean()


# =====================================================
# FINAL SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("MLP CLASSIFIER, HARD NEGATIVES (1443 dataset) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Random seed              : {SEED}")
summary_lines.append(f"Positive gold-standard pairs retained: {len(gold)}")
summary_lines.append(f"Proteins with known family (hard-negative eligible): {len(family_map)}")
summary_lines.append("")
summary_lines.append(f"Negative mix: HARD_FRAC={HARD_FRAC} hard (safe cross-family, "
                      f"hardest-of-{HARD_CANDIDATES_PER_DRAW}), rest uniform-random from the FULL pool.")
summary_lines.append(f"  Train negatives: {len(train_negative_pairs)} "
                      f"({n_train_hard} hard / {n_train_easy} easy)")
summary_lines.append(f"  Test negatives : {len(test_negative_pairs)} "
                      f"({n_test_hard} hard / {n_test_easy} easy)")
summary_lines.append("")
summary_lines.append("Evidence-level sample weights used in TRAINING LOSS only, identical to "
                      "ml/mlp_baseline_1443.py:")
for bucket, weight in EVIDENCE_WEIGHTS.items():
    count = sum(1 for p in gold_pairs if p["evidence_bucket"] == bucket)
    summary_lines.append(f"  {bucket:<24}: {count:5d} pairs  (weight={weight})")
summary_lines.append("")
summary_lines.append("Split strategy: group-aware protein-level split, identical to "
                      "ml/mlp_baseline_1443.py.")
summary_lines.append(f"  Train positive pairs  : {len(train_positive_pairs)}")
summary_lines.append(f"  Test positive pairs   : {len(test_positive_pairs)}")
if split_consistent is not None:
    summary_lines.append(f"  Split consistency check vs baseline MLP: {split_consistent}")
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

header = (f"{'Metric':<10}{'Cosine (same test)':<20}{'MLP (baseline)':<20}"
          f"{'MLP (hard neg)':<20}{'RF (hard neg)':<20}")
summary_lines.append(header)
summary_lines.append("-" * len(header))
for k in TOP_K:
    summary_lines.append(
        f"{'Hits@' + str(k):<10}{cosine_test_hits[k]:<20.4f}"
        f"{baseline_hits[k]:<20.4f}{mlp_hits[k]:<20.4f}{rf_hardneg_hits[k]:<20.4f}"
    )
summary_lines.append(
    f"{'MRR':<10}{cosine_test_mrr:<20.4f}{baseline_mrr:<20.4f}{mlp_mrr:<20.4f}{rf_hardneg_mrr:<20.4f}"
)

if baseline_available:
    delta_mrr = mlp_mrr - baseline_mrr
    summary_lines.append(f"\nDelta vs baseline MLP classifier (uniform-random negatives): MRR {delta_mrr:+.4f}")
    verdict = "IMPROVED" if delta_mrr > 0 else ("WORSE" if delta_mrr < 0 else "UNCHANGED")
    summary_lines.append(f"Hard-negative mining {verdict} MLP classifier retrieval on this dataset.")
else:
    summary_lines.append(f"\nNOTE: {BASELINE_MLP_RESULTS} not found -- run ml/mlp_baseline_1443.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
