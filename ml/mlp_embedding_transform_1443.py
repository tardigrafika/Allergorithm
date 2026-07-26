"""
Pristup B: MLP transformacija embedding prostora 1443

embedding_A (1280) → MLP → predvidjeni embedding_B (1280)

Nema sample weightinga
Za razliku od MLP klasifikatora  ovde se svih 1.432 pozitivnih parova tretira jednako.

1.267 train parova umesto 241
oko 2.153 training primera umesto 410
bolji odnos broja parametara i primera nego kod 296-pair verzije
model je i dalje overparameterized, pa se radi sanity check za degenerate/identity resenje

Izlaz:
    output/mlp_embedding_transform_model_1443.pt
    output/mlp_embedding_transform_retrieval_results_1443.csv
    output/mlp_embedding_transform_summary_1443.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
RF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_retrieval_results_1443.csv")
MLP_CLF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/mlp_retrieval_results_1443.csv")
OLD_EMBED_RESULTS = Path("/home/lana/ALERGRAF/output/mlp_embedding_transform_retrieval_results.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODEL_OUTPUT = OUTPUT_DIR / "mlp_embedding_transform_model_1443.pt"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "mlp_embedding_transform_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mlp_embedding_transform_summary_1443.txt"


# =====================================================
# CONFIGURATION (identical to the 296-pair embedding-transform script)
# =====================================================

SEED = 42
TEST_FRACTION = 0.2
VAL_FRACTION = 0.15
TOP_K = [1, 5, 10, 20]

BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

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
# EVIDENCE-LEVEL FILTERING (exclude negative/contested rows only;
# no weighting in this script, see docstring)
# =====================================================

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()
print(f"\nExcluded negative/contested/risky rows: {len(excluded)}")
print(f"Positive gold-standard pairs retained : {len(gold)} (all equal-weight)")


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

print(f"Official names mapped : {len(name_to_id)}")


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
# MAP GOLD STANDARD PAIRS TO ALLERGEN IDS
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

    gold_pairs.append(
        {
            "pair_id": row["pair_id"],
            "id_1": id_1,
            "id_2": id_2,
            "name_1": name_1,
            "name_2": name_2,
            "family_1": row["family_1"],
            "family_2": row["family_2"],
        }
    )

print(f"Mapped gold pairs : {len(gold_pairs)}")
print(f"Missing/unmapped  : {missing_pairs}")


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
# STANDARDIZATION (fit on TRAIN-split protein embeddings only)
# =====================================================

train_protein_indices = sorted(id_to_index[pid] for pid in train_ids)
train_embeddings_subset = embedding_matrix[train_protein_indices]

feature_mean = train_embeddings_subset.mean(axis=0)
feature_std = train_embeddings_subset.std(axis=0)
feature_std[feature_std < 1e-8] = 1.0


def scale(x):
    return (x - feature_mean) / feature_std


def unscale(x):
    return x * feature_std + feature_mean


# =====================================================
# PAIR-LEVEL FIT/VALIDATION SPLIT (within TRAIN pairs)
# =====================================================

pair_rng = np.random.default_rng(SEED)
pair_order = pair_rng.permutation(len(train_positive_pairs))
n_val_pairs = round(VAL_FRACTION * len(train_positive_pairs))

val_pair_positions = set(pair_order[:n_val_pairs].tolist())
fit_pairs = [p for i, p in enumerate(train_positive_pairs) if i not in val_pair_positions]
val_pairs = [p for i, p in enumerate(train_positive_pairs) if i in val_pair_positions]

print(f"\nFit pairs (train-split): {len(fit_pairs)}  -> {2 * len(fit_pairs)} directed examples")
print(f"Val pairs (train-split): {len(val_pairs)}  -> {2 * len(val_pairs)} directed examples")


def build_directed_examples(pairs):
    inputs, targets = [], []
    for p in pairs:
        emb_a = embedding_matrix[id_to_index[p["id_1"]]]
        emb_b = embedding_matrix[id_to_index[p["id_2"]]]
        inputs.append(scale(emb_a))
        targets.append(scale(emb_b))
        inputs.append(scale(emb_b))
        targets.append(scale(emb_a))
    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


X_fit, Y_fit = build_directed_examples(fit_pairs)
X_val, Y_val = build_directed_examples(val_pairs)

print(f"Fit examples: X={X_fit.shape}, Y={Y_fit.shape}")
print(f"Val examples: X={X_val.shape}, Y={Y_val.shape}")


# =====================================================
# DEGENERATE-SOLUTION SANITY CHECK
# =====================================================

identity_val_mse = float(np.mean((X_val - Y_val) ** 2))
print(f"\nSanity check -- identity-mapping MSE on val set (predicted_B = A): {identity_val_mse:.4f}")


# =====================================================
# MODEL (identical architecture)
# =====================================================

class EmbeddingTransformMLP(nn.Module):
    def __init__(self, dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, dim),
        )

    def forward(self, x):
        return self.net(x)


embedding_dim = embedding_matrix.shape[1]
model = EmbeddingTransformMLP(embedding_dim)

n_params = sum(p.numel() for p in model.parameters())
print("\n==============================")
print("MODEL ARCHITECTURE")
print("==============================")
print(model)
print(f"\nTotal parameters: {n_params:,}  vs. fit examples: {X_fit.shape[0]} "
      f"(ratio: {n_params / X_fit.shape[0]:.0f} params per example)")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

X_fit_t = torch.from_numpy(X_fit)
Y_fit_t = torch.from_numpy(Y_fit)
X_val_t = torch.from_numpy(X_val)
Y_val_t = torch.from_numpy(Y_val)


# =====================================================
# TRAINING LOOP WITH EARLY STOPPING
# =====================================================

print("\n==============================")
print("TRAINING MLP (embedding transformation)")
print("==============================")
print(f"Max epochs: {MAX_EPOCHS}, patience: {PATIENCE}, batch size: {BATCH_SIZE}, "
      f"lr: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY}")

n_fit = X_fit_t.shape[0]
batch_rng = np.random.default_rng(SEED)

best_val_mse = np.inf
best_state = None
epochs_without_improvement = 0
epoch = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    permutation = batch_rng.permutation(n_fit)

    epoch_loss = 0.0
    for start in range(0, n_fit, BATCH_SIZE):
        batch_idx = permutation[start:start + BATCH_SIZE]
        xb = X_fit_t[batch_idx]
        yb = Y_fit_t[batch_idx]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_idx)

    epoch_loss /= n_fit

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_mse = criterion(val_pred, Y_val_t).item()

    improved = val_mse < best_val_mse
    if improved:
        best_val_mse = val_mse
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 20 == 0 or improved:
        marker = " *" if improved else ""
        print(f"Epoch {epoch:3d}: fit_mse={epoch_loss:.4f}  val_mse={val_mse:.4f}{marker}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no val_mse improvement for {PATIENCE} epochs).")
        break

model.load_state_dict(best_state)
model.eval()

print(f"\nBest validation MSE      : {best_val_mse:.4f}")
print(f"Identity-baseline val MSE: {identity_val_mse:.4f}")
beats_identity = best_val_mse < identity_val_mse
print("-> Model beats the trivial identity baseline." if beats_identity
      else "-> WARNING: model does NOT beat the trivial identity baseline.")


# =====================================================
# RETRIEVAL EVALUATION
# =====================================================

print("\n==============================")
print("RETRIEVAL EVALUATION (Hits@K / MRR)")
print("==============================")
print(f"Test-split gold pairs: {len(test_positive_pairs)}  "
      f"-> up to {2 * len(test_positive_pairs)} retrieval queries")

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
            scaled_input = torch.from_numpy(scale(query_vec).astype(np.float32)).unsqueeze(0)

            predicted_scaled = model(scaled_input).numpy()[0]
            predicted_raw = unscale(predicted_scaled)

            mlp_scores = cosine_similarity(predicted_raw.reshape(1, -1), embedding_matrix)[0]
            mlp_scores[query_index] = -np.inf

            mlp_ranked = np.argsort(mlp_scores)[::-1]
            mlp_rank = int(np.where(mlp_ranked == target_index)[0][0]) + 1
            mlp_reciprocal_rank = 1.0 / mlp_rank
            mlp_true_pair_similarity = mlp_scores[target_index]

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
                "predicted_embedding_cosine_to_true_target": mlp_true_pair_similarity,
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

emb_hits = {k: retrieval_df[f"mlp_hits_at_{k}"].mean() for k in TOP_K}
emb_mrr = retrieval_df["mlp_reciprocal_rank"].mean()
cosine_test_hits = {k: retrieval_df[f"cosine_hits_at_{k}"].mean() for k in TOP_K}
cosine_test_mrr = retrieval_df["cosine_reciprocal_rank"].mean()

print(f"Retrieval queries evaluated: {len(retrieval_df)}")
for k in TOP_K:
    print(f"Hits@{k:<2d}: EmbeddingTransform={emb_hits[k]:.4f}  Cosine={cosine_test_hits[k]:.4f}")
print(f"MRR    : EmbeddingTransform={emb_mrr:.4f}  Cosine={cosine_test_mrr:.4f}")


# =====================================================
# SAVE OUTPUTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nRetrieval results saved to: {RETRIEVAL_OUTPUT}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "embedding_dim": embedding_dim,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "architecture": "1280 -> Linear(256) -> ReLU -> Dropout(0.3) -> "
                         "Linear(128) -> ReLU -> Dropout(0.2) -> Linear(1280)",
        "seed": SEED,
    },
    MODEL_OUTPUT,
)
print(f"Model saved to: {MODEL_OUTPUT}")


# =====================================================
# CROSS-CHECK + LOAD RF / MLP CLASSIFIER SAVED RESULTS (same test set)
# =====================================================

def load_other_method(path, hits_prefix, mrr_col, method_label):
    if not path.exists():
        print(f"\n{method_label} results file NOT found: {path}")
        return {k: float("nan") for k in TOP_K}, float("nan"), None

    other_df = pd.read_csv(path)
    if len(other_df) != len(retrieval_df):
        print(f"\nWARNING: {method_label} file has {len(other_df)} queries, "
              f"this script has {len(retrieval_df)} -- skipping.")
        return {k: float("nan") for k in TOP_K}, float("nan"), None

    merged = retrieval_df.merge(
        other_df[["pair_id", "query_allergen", "cosine_reciprocal_rank"]],
        on=["pair_id", "query_allergen"], suffixes=("_this_script", "_other_script"),
    )
    consistent = bool(np.allclose(
        merged["cosine_reciprocal_rank_this_script"], merged["cosine_reciprocal_rank_other_script"]
    ))
    hits = {k: other_df[f"{hits_prefix}_hits_at_{k}"].mean() for k in TOP_K}
    mrr = other_df[mrr_col].mean()
    print(f"\n{method_label} results file found. Split consistency check: {consistent}")
    return hits, mrr, consistent


rf_hits, rf_mrr, rf_consistent = load_other_method(
    RF_RETRIEVAL_RESULTS, "rf", "rf_reciprocal_rank", "Random Forest (1443)"
)
mlp_clf_hits, mlp_clf_mrr, mlp_clf_consistent = load_other_method(
    MLP_CLF_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank", "MLP classifier (1443)"
)


# =====================================================
# COMPARISON WITH OLD 296-PAIR EMBEDDING TRANSFORM
# =====================================================

old_available = OLD_EMBED_RESULTS.exists()
old_emb_hits = {k: float("nan") for k in TOP_K}
old_emb_mrr = float("nan")
if old_available:
    old_df = pd.read_csv(OLD_EMBED_RESULTS)
    old_emb_hits = {k: old_df[f"mlp_hits_at_{k}"].mean() for k in TOP_K}
    old_emb_mrr = old_df["mlp_reciprocal_rank"].mean()


# =====================================================
# FINAL SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("MLP EMBEDDING TRANSFORMATION (1443 dataset, Approach B) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Random seed              : {SEED}")
summary_lines.append(f"Rows in gold file (1443) : {len(gold_raw)}")
summary_lines.append(f"Excluded (negative/contested/risky): {len(excluded)}")
summary_lines.append(f"Positive gold-standard pairs retained: {len(gold)} (all equal-weight, "
                      "no evidence-level weighting in this script -- see docstring)")
summary_lines.append("")
summary_lines.append("Split strategy: group-aware, protein-level split (same algorithm/seed "
                      "as the other 1443-dataset scripts).")
summary_lines.append(f"  Train proteins        : {len(train_ids)} ({len(train_ids)/len(all_ids):.1%})")
summary_lines.append(f"  Test proteins         : {len(test_ids)} ({len(test_ids)/len(all_ids):.1%})")
summary_lines.append(f"  Train positive pairs  : {len(train_positive_pairs)} "
                      f"(fit={len(fit_pairs)}, val={len(val_pairs)})")
summary_lines.append(f"  Test positive pairs   : {len(test_positive_pairs)}")
if rf_consistent is not None:
    summary_lines.append(f"  Split consistency check vs RF_1443    : {rf_consistent}")
if mlp_clf_consistent is not None:
    summary_lines.append(f"  Split consistency check vs MLP_1443   : {mlp_clf_consistent}")
summary_lines.append("")
summary_lines.append(f"Total parameters: {n_params:,}  (fit examples: {X_fit.shape[0]})")
summary_lines.append(f"Training stopped at epoch {epoch}")
summary_lines.append(f"Best validation MSE       : {best_val_mse:.4f}")
summary_lines.append(f"Identity-baseline val MSE : {identity_val_mse:.4f}")
summary_lines.append(
    "  -> model BEATS identity baseline (non-degenerate)" if beats_identity
    else "  -> WARNING: model does NOT beat identity baseline (possibly degenerate)"
)
summary_lines.append("")
summary_lines.append(f"Retrieval evaluation: {len(retrieval_df)} queries "
                      f"({len(test_positive_pairs)} test pairs x 2 directions)")
summary_lines.append("")

header = (f"{'Metric':<10}{'Cosine (same test)':<20}{'RF (1443)':<20}"
          f"{'MLP clf (1443)':<20}{'MLP embed. (1443)':<20}")
summary_lines.append(header)
summary_lines.append("-" * len(header))
for k in TOP_K:
    summary_lines.append(
        f"{'Hits@' + str(k):<10}{cosine_test_hits[k]:<20.4f}"
        f"{rf_hits[k]:<20.4f}{mlp_clf_hits[k]:<20.4f}{emb_hits[k]:<20.4f}"
    )
summary_lines.append(
    f"{'MRR':<10}{cosine_test_mrr:<20.4f}{rf_mrr:<20.4f}{mlp_clf_mrr:<20.4f}{emb_mrr:<20.4f}"
)

if old_available:
    summary_lines.append("")
    summary_lines.append("Comparison with the OLD 296-pair embedding-transform experiment "
                          "(each on its own held-out test split):")
    header2 = f"{'Metric':<10}{'Embed (296, old)':<20}{'Embed (1443, new)':<20}"
    summary_lines.append(header2)
    summary_lines.append("-" * len(header2))
    for k in TOP_K:
        summary_lines.append(f"{'Hits@' + str(k):<10}{old_emb_hits[k]:<20.4f}{emb_hits[k]:<20.4f}")
    summary_lines.append(f"{'MRR':<10}{old_emb_mrr:<20.4f}{emb_mrr:<20.4f}")
else:
    summary_lines.append(f"\nNOTE: {OLD_EMBED_RESULTS} not found -- run ml/mlp_embedding_transform.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
