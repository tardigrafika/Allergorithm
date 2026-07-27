"""
Pristup B v3: triplet loss + BEZBEDNI geometrijski hard negativi - 1443 dataset.

Ne menja mlp_embedding_transform_triplet_1443.py (random-negative verzija ostaje odvojena).
Hard negativi biraju se SAMO iz razlicitih, poznatih familija (nikad ista familija -
"missing negatives" rizik: nedokumentovan par nije isto sto i potvrdjen negativan par),
i od nekoliko takvih kandidata bira se najslicniji ("hardest of K", K=10).
Ako protein nema poznatu familiju, fallback na obican random negativ.

Rezultat: LOSIJE od random-negative verzije (MRR 0.030 -> 0.005) - trening
skup postaje previse uzak (samo 263 proteina sa poznatom familijom),
model ne generalizuje na pun pool od 1534 kandidata.

Izlaz:
    output/mlp_embedding_transform_triplet_hardneg_model_1443.pt
    output/mlp_embedding_transform_triplet_hardneg_retrieval_results_1443.csv
    output/mlp_embedding_transform_triplet_hardneg_summary_1443.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")

RF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_retrieval_results_1443.csv")
MLP_CLF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/mlp_retrieval_results_1443.csv")
MSE_EMBED_RETRIEVAL_RESULTS = Path(
    "/home/lana/ALERGRAF/output/mlp_embedding_transform_retrieval_results_1443.csv"
)
RANDOM_TRIPLET_RETRIEVAL_RESULTS = Path(
    "/home/lana/ALERGRAF/output/mlp_embedding_transform_triplet_retrieval_results_1443.csv"
)

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODEL_OUTPUT = OUTPUT_DIR / "mlp_embedding_transform_triplet_hardneg_model_1443.pt"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "mlp_embedding_transform_triplet_hardneg_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mlp_embedding_transform_triplet_hardneg_summary_1443.txt"


# =====================================================
# CONFIGURATION (identical to the random-negative triplet version)
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
MARGIN = 0.3

HARD_CANDIDATES_PER_DRAW = 10  # "hardest of K" -- same K as data/build_ml_dataset.py's default

np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================================================
# LOAD DATA (identical to the random-negative triplet version)
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
# EVIDENCE-LEVEL FILTERING (identical rule to the other *_1443 scripts)
# =====================================================

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()
print(f"\nExcluded negative/contested/risky rows: {len(excluded)}")
print(f"Positive gold-standard pairs retained : {len(gold)}")


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
# MAP GOLD STANDARD PAIRS TO ALLERGEN IDS (+ FAMILY MAP for hard negatives)
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

positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}

# allergen_id -> family, built the same way as data/build_ml_dataset.py's
# load_family_map: only proteins that appear in at least one gold pair get
# a known family (the other ~83% are ineligible for HARD negatives, see below)
family_map = {}
for p in gold_pairs:
    family_map.setdefault(p["id_1"], p["family_1"])
    family_map.setdefault(p["id_2"], p["family_2"])

print(f"Proteins with a known family label: {len(family_map)} "
      f"(eligible as hard-negative candidates)")


# =====================================================
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm/seed to every
# other *_1443 script -- reproduces the identical 1227/307 protein
# split and the identical 165 held-out test pairs)
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
# PAIR-LEVEL FIT/VALIDATION SPLIT (within TRAIN pairs -- identical)
# =====================================================

pair_rng = np.random.default_rng(SEED)
pair_order = pair_rng.permutation(len(train_positive_pairs))
n_val_pairs = round(VAL_FRACTION * len(train_positive_pairs))

val_pair_positions = set(pair_order[:n_val_pairs].tolist())
fit_pairs = [p for i, p in enumerate(train_positive_pairs) if i not in val_pair_positions]
val_pairs = [p for i, p in enumerate(train_positive_pairs) if i in val_pair_positions]

print(f"\nFit pairs (train-split): {len(fit_pairs)}  -> {2 * len(fit_pairs)} directed examples")
print(f"Val pairs (train-split): {len(val_pairs)}  -> {2 * len(val_pairs)} directed examples")


# =====================================================
# NEGATIVE SAMPLING FOR TRIPLETS -- SAFE + GEOMETRICALLY HARD
# (see module docstring for the full rationale; mirrors the fixed
# data/build_ml_dataset.py's sample_negatives logic, restructured to
# draw one negative per anchor)
# =====================================================

TRAIN_PROTEIN_POOL = sorted(train_ids)
TRAIN_FAMILY_BUCKETS = {}
for pid in TRAIN_PROTEIN_POOL:
    fam = family_map.get(pid)
    if fam is not None:
        TRAIN_FAMILY_BUCKETS.setdefault(fam, []).append(pid)
TRAIN_FAMILY_NAMES = list(TRAIN_FAMILY_BUCKETS.keys())

hard_negative_count = 0
fallback_random_count = 0


def cosine_sim_raw(id_a, id_b):
    ea = embedding_matrix[id_to_index[id_a]]
    eb = embedding_matrix[id_to_index[id_b]]
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12))


def sample_random_negative(anchor_id, positive_id, rng):
    """Fallback: identical rule to ml/mlp_embedding_transform_triplet_1443.py."""
    while True:
        candidate = rng.choice(TRAIN_PROTEIN_POOL)
        if candidate == anchor_id or candidate == positive_id:
            continue
        if tuple(sorted((anchor_id, candidate))) in positive_pair_set:
            continue
        return candidate


def sample_negative_target_hard(anchor_id, positive_id, rng):
    """
    Safe + geometrically hard negative for `anchor_id`:
      1. If anchor's family is unknown, fall back to uniform random
         (cannot verify cross-family safety without a known family).
      2. Otherwise draw HARD_CANDIDATES_PER_DRAW random proteins from a
         DIFFERENT known family, skip any that collide with a
         documented positive pair, and keep the one with the highest
         cosine similarity to the anchor's raw embedding.
    """
    global hard_negative_count, fallback_random_count

    anchor_family = family_map.get(anchor_id)
    other_families = [f for f in TRAIN_FAMILY_NAMES if f != anchor_family]

    if anchor_family is None or not other_families:
        fallback_random_count += 1
        return sample_random_negative(anchor_id, positive_id, rng)

    best_candidate, best_similarity = None, -2.0
    for _ in range(HARD_CANDIDATES_PER_DRAW):
        fam = other_families[rng.integers(0, len(other_families))]
        candidate = TRAIN_FAMILY_BUCKETS[fam][rng.integers(0, len(TRAIN_FAMILY_BUCKETS[fam]))]
        if candidate == anchor_id or candidate == positive_id:
            continue
        if tuple(sorted((anchor_id, candidate))) in positive_pair_set:
            continue
        similarity = cosine_sim_raw(anchor_id, candidate)
        if similarity > best_similarity:
            best_similarity = similarity
            best_candidate = candidate

    if best_candidate is None:
        fallback_random_count += 1
        return sample_random_negative(anchor_id, positive_id, rng)

    hard_negative_count += 1
    return best_candidate


def build_triplet_examples(pairs, seed):
    local_rng = np.random.default_rng(seed)
    anchors, positives, negatives = [], [], []
    for p in pairs:
        for a_id, b_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            neg_id = sample_negative_target_hard(a_id, b_id, local_rng)
            anchors.append(scale(embedding_matrix[id_to_index[a_id]]))
            positives.append(scale(embedding_matrix[id_to_index[b_id]]))
            negatives.append(scale(embedding_matrix[id_to_index[neg_id]]))
    return (
        np.array(anchors, dtype=np.float32),
        np.array(positives, dtype=np.float32),
        np.array(negatives, dtype=np.float32),
    )


A_fit, P_fit, N_fit = build_triplet_examples(fit_pairs, seed=SEED)
print(f"\nFit triplets: {A_fit.shape[0]}  "
      f"({hard_negative_count} hard_cross_family_geometric / {fallback_random_count} fallback_random "
      f"so far)")

hard_negative_count = 0
fallback_random_count = 0
A_val, P_val, N_val = build_triplet_examples(val_pairs, seed=SEED + 2)
print(f"Val triplets: {A_val.shape[0]}  "
      f"({hard_negative_count} hard_cross_family_geometric / {fallback_random_count} fallback_random)")


# =====================================================
# SANITY CHECK (identical framing to the random-negative triplet version)
# =====================================================

raw_cosine_to_target_val = np.array([
    1.0 - np.sum(a * p) / (np.linalg.norm(a) * np.linalg.norm(p) + 1e-12)
    for a, p in zip(A_val, P_val)
])
print(f"\nSanity check -- mean RAW cosine DISTANCE(A, true B) on val set "
      f"(pre-training baseline): {raw_cosine_to_target_val.mean():.4f}")


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
print("MODEL ARCHITECTURE (unchanged)")
print("==============================")
print(model)


# =====================================================
# LOSS: cosine-distance TripletMarginWithDistanceLoss (identical to the
# random-negative triplet version -- only the negatives differ)
# =====================================================

def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y)


criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=MARGIN)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"\nLoss: TripletMarginWithDistanceLoss(distance=1-cosine_similarity, margin={MARGIN})")

A_fit_t = torch.from_numpy(A_fit)
P_fit_t = torch.from_numpy(P_fit)
N_fit_t = torch.from_numpy(N_fit)
A_val_t = torch.from_numpy(A_val)
P_val_t = torch.from_numpy(P_val)
N_val_t = torch.from_numpy(N_val)


# =====================================================
# TRAINING LOOP WITH EARLY STOPPING
# =====================================================

print("\n==============================")
print("TRAINING MLP (triplet, hard negatives)")
print("==============================")
print(f"Max epochs: {MAX_EPOCHS}, patience: {PATIENCE}, batch size: {BATCH_SIZE}, "
      f"lr: {LEARNING_RATE}, weight_decay: {WEIGHT_DECAY}, margin: {MARGIN}")

n_fit = A_fit_t.shape[0]
batch_rng = np.random.default_rng(SEED)

best_val_loss = np.inf
best_state = None
epochs_without_improvement = 0
epoch = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    permutation = batch_rng.permutation(n_fit)

    epoch_loss = 0.0
    for start in range(0, n_fit, BATCH_SIZE):
        batch_idx = permutation[start:start + BATCH_SIZE]
        a_batch = A_fit_t[batch_idx]
        p_batch = P_fit_t[batch_idx]
        n_batch = N_fit_t[batch_idx]

        optimizer.zero_grad()
        predicted = model(a_batch)
        loss = criterion(predicted, p_batch, n_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_idx)

    epoch_loss /= n_fit

    model.eval()
    with torch.no_grad():
        val_predicted = model(A_val_t)
        val_loss = criterion(val_predicted, P_val_t, N_val_t).item()
        val_dist_pos = cosine_distance(val_predicted, P_val_t)
        val_dist_neg = cosine_distance(val_predicted, N_val_t)
        val_triplet_accuracy = (val_dist_pos < val_dist_neg).float().mean().item()

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 20 == 0 or improved:
        marker = " *" if improved else ""
        print(f"Epoch {epoch:3d}: fit_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_triplet_acc={val_triplet_accuracy:.4f}{marker}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no val_loss improvement for {PATIENCE} epochs).")
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    val_predicted = model(A_val_t)
    val_dist_pos_final = cosine_distance(val_predicted, P_val_t).mean().item()
    val_triplet_accuracy_final = (
        cosine_distance(val_predicted, P_val_t) < cosine_distance(val_predicted, N_val_t)
    ).float().mean().item()

print(f"\nBest validation triplet loss : {best_val_loss:.4f}")
print(f"Final val triplet accuracy   : {val_triplet_accuracy_final:.4f} "
      f"(compare to 0.9868 for the random-negative version -- a LOWER accuracy here "
      f"is expected and desired: it means the negatives are genuinely harder to "
      f"distinguish from the positive, forcing real learning)")


# =====================================================
# RETRIEVAL EVALUATION (identical protocol)
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

hardneg_hits = {k: retrieval_df[f"mlp_hits_at_{k}"].mean() for k in TOP_K}
hardneg_mrr = retrieval_df["mlp_reciprocal_rank"].mean()
cosine_test_hits = {k: retrieval_df[f"cosine_hits_at_{k}"].mean() for k in TOP_K}
cosine_test_mrr = retrieval_df["cosine_reciprocal_rank"].mean()

print(f"Retrieval queries evaluated: {len(retrieval_df)}")
for k in TOP_K:
    print(f"Hits@{k:<2d}: HardNegTriplet={hardneg_hits[k]:.4f}  Cosine={cosine_test_hits[k]:.4f}")
print(f"MRR    : HardNegTriplet={hardneg_mrr:.4f}  Cosine={cosine_test_mrr:.4f}")


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
        "loss": f"TripletMarginWithDistanceLoss(cosine_distance, margin={MARGIN}), hard negatives",
        "seed": SEED,
    },
    MODEL_OUTPUT,
)
print(f"Model saved to: {MODEL_OUTPUT}")


# =====================================================
# CROSS-CHECK + LOAD ALL OTHER METHODS' SAVED RESULTS (same test set)
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
mse_embed_hits, mse_embed_mrr, mse_embed_consistent = load_other_method(
    MSE_EMBED_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank", "MLP embed. transform, MSE (1443)"
)
random_triplet_hits, random_triplet_mrr, random_triplet_consistent = load_other_method(
    RANDOM_TRIPLET_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank",
    "MLP embed. transform, Triplet/random-neg (1443)"
)


# =====================================================
# FINAL COMPARISON TABLE + DISCUSSION
# =====================================================

add_lines = []


def add(line=""):
    add_lines.append(line)
    print(line)


add("\n" + "=" * 60)
add("MLP EMBEDDING TRANSFORMATION -- TRIPLET, HARD NEGATIVES (1443)")
add("=" * 60)
add(f"Random seed              : {SEED}")
add(f"Positive gold-standard pairs retained: {len(gold)}")
add(f"Proteins with known family (hard-negative eligible): {len(family_map)}")
add("")
add("Split strategy: IDENTICAL group-aware protein-level split as every other "
    "*_1443 script.")
add(f"  Train positive pairs  : {len(train_positive_pairs)} (fit={len(fit_pairs)}, val={len(val_pairs)})")
add(f"  Test positive pairs   : {len(test_positive_pairs)}")
if rf_consistent is not None:
    add(f"  Split consistency check vs RF_1443              : {rf_consistent}")
if mlp_clf_consistent is not None:
    add(f"  Split consistency check vs MLP_1443             : {mlp_clf_consistent}")
if mse_embed_consistent is not None:
    add(f"  Split consistency check vs MSE-embed_1443       : {mse_embed_consistent}")
if random_triplet_consistent is not None:
    add(f"  Split consistency check vs random-neg triplet   : {random_triplet_consistent}")
add("")
add(f"Negative-sampling rule: safe cross-family + hardest-of-{HARD_CANDIDATES_PER_DRAW} "
    f"geometric mining (falls back to uniform random when the anchor has no known family)")
add(f"Best validation triplet loss : {best_val_loss:.4f}")
add(f"Final val triplet accuracy   : {val_triplet_accuracy_final:.4f}")
add("")
add(f"Retrieval evaluation: {len(retrieval_df)} queries "
    f"({len(test_positive_pairs)} test pairs x 2 directions)")
add("")

header = f"{'Method':<38}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
add(header)
add("-" * len(header))


def fmt(label, hits, mrr):
    return f"{label:<38}{hits[1]:<10.4f}{hits[5]:<10.4f}{hits[10]:<10.4f}{hits[20]:<10.4f}{mrr:<10.4f}"


add(fmt("Cosine baseline", cosine_test_hits, cosine_test_mrr))
add(fmt("Random Forest", rf_hits, rf_mrr))
add(fmt("MLP classifier", mlp_clf_hits, mlp_clf_mrr))
add(fmt("MLP embed. transform (MSE)", mse_embed_hits, mse_embed_mrr))
add(fmt("MLP embed. transform (Triplet, random neg)", random_triplet_hits, random_triplet_mrr))
add(fmt("MLP embed. transform (Triplet, HARD neg)", hardneg_hits, hardneg_mrr))


# =====================================================
# DISCUSSION (dynamically derived from the numbers above)
# =====================================================

add("")
add("=" * 60)
add("DISCUSSION")
add("=" * 60)

random_valid = not np.isnan(random_triplet_mrr)
if random_valid:
    delta = hardneg_mrr - random_triplet_mrr
    add(f"Hard negatives vs random negatives (both triplet loss): "
        f"MRR {random_triplet_mrr:.4f} -> {hardneg_mrr:.4f}  ({delta:+.4f})")
    if delta > 0:
        add("Hard-negative mining IMPROVED retrieval over random-negative triplet training.")
    else:
        add("Hard-negative mining did NOT improve retrieval over random-negative triplet "
            "training on this run -- see val_triplet_accuracy above for whether the "
            "negatives were in fact harder (lower accuracy = harder).")

still_collapses = hardneg_hits[1] < 0.02 and hardneg_mrr < 0.05
add("")
add(f"Still far below cosine baseline: {still_collapses} "
    f"(cosine MRR={cosine_test_mrr:.4f} vs this model's MRR={hardneg_mrr:.4f})")

summary_text = "\n".join(add_lines)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
