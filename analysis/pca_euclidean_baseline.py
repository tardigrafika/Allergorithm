"""
Cilj: Proveriti da li PCA smanjenje embeddinga sa 1280 na 128 dimenzija, uz Euclidean distance, daje bolje rezultate od cosine similarity na originalnim embeddingima.

Metodologija:
Koriste se isti embeddingi, metadata i 296 gold-standard parova.
Proverava se da li su embeddingi normalizovani.
PCA se fituje samo na embeddingima, bez korišćenja gold-standard labela.
Embeddingi se smanjuju na 128 dimenzija i meri se objašnjena varijansa.
Računa se Euclidean distance između svih 1534 proteina.
Evaluacija se radi na 592 query-ja uz isti candidate pool kao cosine baseline.
Računaju se Hits@1/5/10/20 i MRR.

Izlazi:

output/pca_euclidean_retrieval_results.csv
output/pca_euclidean_summary.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_combined.csv")
COSINE_RESULTS = Path("/home/lana/ALERGRAF/output/hits_mrr_results.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
RETRIEVAL_OUTPUT = OUTPUT_DIR / "pca_euclidean_retrieval_results.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "pca_euclidean_summary.txt"


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
PCA_COMPONENTS = 128
TOP_K = [1, 5, 10, 20]


# =====================================================
# LOAD DATA
# (identical to analysis/hitsk_and_mrr.py, kept in sync on purpose so
#  the two retrieval benchmarks evaluate the exact same query set)
# =====================================================

import pickle  # noqa: E402  (kept near point of use, matches project convention)

print("\n==============================")
print("LOADING DATA")
print("==============================")

print("Loading embeddings...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Proteins in embeddings: {len(embeddings_dict)}")

print("Loading metadata...")
metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()
print(f"Metadata rows with embeddings: {len(metadata)}")

print("Loading gold standard...")
gold = pd.read_csv(GOLD)
print(f"Gold standard pairs: {len(gold)}")


# =====================================================
# NAME -> ALLERGEN_ID MAPPING (identical logic to hitsk_and_mrr.py)
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
print(f"Duplicate names skipped: {duplicate_names}")


# =====================================================
# EMBEDDING MATRIX (same protein order used throughout)
# =====================================================

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array(
    [embeddings_dict[allergen_id] for allergen_id in all_ids],
    dtype=np.float64,
)
original_dim = embedding_matrix.shape[1]
print(f"Embedding matrix shape: {embedding_matrix.shape}")


# =====================================================
# CHECK L2 NORMALIZATION (measured, not assumed)
# =====================================================

print("\n==============================")
print("EMBEDDING NORMALIZATION CHECK")
print("==============================")

embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
is_l2_normalized = bool(np.allclose(embedding_norms, 1.0, atol=1e-3))

print(f"L2 norm  mean: {embedding_norms.mean():.4f}")
print(f"L2 norm  std : {embedding_norms.std():.4f}")
print(f"L2 norm  min : {embedding_norms.min():.4f}")
print(f"L2 norm  max : {embedding_norms.max():.4f}")
print(f"Embeddings L2-normalized: {is_l2_normalized}")

if not is_l2_normalized:
    print(
        "NOTE: raw ESM-2 mean-pooled embeddings are NOT unit-norm. PCA is "
        "applied to the raw (mean-centered, unscaled) embeddings below, "
        "exactly mirroring how the cosine baseline also uses the raw "
        "embeddings (cosine similarity itself is scale-invariant, but "
        "Euclidean distance is NOT -- this is reported for transparency, "
        "not corrected, since normalizing was not part of the requested "
        "methodology and would make this a different experiment)."
    )


# =====================================================
# PCA (unsupervised -- fit on embeddings only, no gold-standard labels)
# =====================================================

print("\n==============================")
print("PCA DIMENSIONALITY REDUCTION")
print("==============================")

print(f"Original embedding dimension: {original_dim}")
print(f"PCA target dimension        : {PCA_COMPONENTS}")

# svd_solver="full" -> exact, deterministic SVD (no randomized-solver
# variance); random_state is set regardless for explicitness/reproducibility.
pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=SEED)
pca_matrix = pca.fit_transform(embedding_matrix)

print(f"PCA-reduced matrix shape: {pca_matrix.shape}")

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = float(explained_variance_ratio.sum())

print(f"Explained variance (component 1)  : {explained_variance_ratio[0]:.4f}")
print(f"Explained variance (component 128): {explained_variance_ratio[-1]:.6f}")
print(f"Cumulative explained variance ({PCA_COMPONENTS} components): "
      f"{cumulative_explained_variance:.4f} "
      f"({cumulative_explained_variance:.2%})")


# =====================================================
# PAIRWISE EUCLIDEAN DISTANCE IN PCA SPACE
# =====================================================

print("\n==============================")
print("PAIRWISE EUCLIDEAN DISTANCES")
print("==============================")

distance_matrix = euclidean_distances(pca_matrix)
print(f"Distance matrix shape: {distance_matrix.shape}")


# =====================================================
# MAP GOLD STANDARD PAIRS TO ALLERGEN IDS
# (identical control flow to analysis/hitsk_and_mrr.py)
# =====================================================

print("\n==============================")
print("RUNNING RETRIEVAL EVALUATION")
print("==============================")

results = []
missing_pairs = 0
evaluated_pairs = 0
evaluated_queries = 0

for _, row in gold.iterrows():
    pair_id = row["pair_id"]

    name_1 = str(row["allergen_id_1"]).strip()
    name_2 = str(row["allergen_id_2"]).strip()

    if name_1 not in name_to_id or name_2 not in name_to_id:
        missing_pairs += 1
        print(f"WARNING: Missing mapping for pair {pair_id}")
        continue

    allergen_1 = name_to_id[name_1]
    allergen_2 = name_to_id[name_2]

    if allergen_1 not in id_to_index or allergen_2 not in id_to_index:
        missing_pairs += 1
        print(f"WARNING: Mapped IDs not found in embedding matrix for pair {pair_id}")
        continue

    index_1 = id_to_index[allergen_1]
    index_2 = id_to_index[allergen_2]

    evaluated_pairs += 1

    directions = [
        {
            "query_name": name_1, "target_name": name_2,
            "query_id": allergen_1, "target_id": allergen_2,
            "query_index": index_1, "target_index": index_2,
            "family_query": row["family_1"], "family_target": row["family_2"],
        },
        {
            "query_name": name_2, "target_name": name_1,
            "query_id": allergen_2, "target_id": allergen_1,
            "query_index": index_2, "target_index": index_1,
            "family_query": row["family_2"], "family_target": row["family_1"],
        },
    ]

    for direction in directions:
        query_index = direction["query_index"]
        target_index = direction["target_index"]

        # ---- rank by ASCENDING Euclidean distance (nearest = best) ----
        distances = distance_matrix[query_index].copy()
        distances[query_index] = np.inf  # exclude self-match

        ranked_indices = np.argsort(distances)  # ascending
        positions = np.where(ranked_indices == target_index)[0]

        if len(positions) == 0:
            print(f"WARNING: Could not rank {direction['target_name']} "
                  f"for query {direction['query_name']}")
            continue

        rank = int(positions[0]) + 1
        true_distance = distance_matrix[query_index, target_index]
        reciprocal_rank = 1.0 / rank

        results.append({
            "pair_id": pair_id,
            "query_allergen": direction["query_name"],
            "target_allergen": direction["target_name"],
            "query_allergen_id": direction["query_id"],
            "target_allergen_id": direction["target_id"],
            "query_family": direction["family_query"],
            "target_family": direction["family_target"],
            "euclidean_distance": true_distance,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "hits_at_1": int(rank <= 1),
            "hits_at_5": int(rank <= 5),
            "hits_at_10": int(rank <= 10),
            "hits_at_20": int(rank <= 20),
        })

        evaluated_queries += 1

result_df = pd.DataFrame(results)

print(f"\nGold standard pairs      : {len(gold)}")
print(f"Valid pairs evaluated    : {evaluated_pairs}")
print(f"Missing pairs            : {missing_pairs}")
print(f"Retrieval queries evaluated: {evaluated_queries}")

if len(result_df) == 0:
    print("\nERROR: No valid pairs were found.")
    raise SystemExit(1)


# =====================================================
# METRICS
# =====================================================

pca_hits = {k: result_df[f"hits_at_{k}"].mean() for k in TOP_K}
pca_mrr = result_df["reciprocal_rank"].mean()

print("\n==============================")
print("PCA(128) + EUCLIDEAN RESULTS")
print("==============================")
for k in TOP_K:
    print(f"Hits@{k:<2d}: {pca_hits[k]:.4f}")
print(f"MRR    : {pca_mrr:.4f}")


# =====================================================
# SAVE DETAILED RESULTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nDetailed results saved to: {RETRIEVAL_OUTPUT}")


# =====================================================
# COMPARISON WITH EXISTING COSINE BASELINE
# =====================================================

cosine_available = COSINE_RESULTS.exists()
cosine_hits = {k: float("nan") for k in TOP_K}
cosine_mrr = float("nan")

if cosine_available:
    cosine_df = pd.read_csv(COSINE_RESULTS)
    cosine_hits = {k: cosine_df[f"hits_at_{k}"].mean() for k in TOP_K}
    cosine_mrr = cosine_df["reciprocal_rank"].mean()


# =====================================================
# FINAL SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("PCA(128) + EUCLIDEAN DISTANCE BASELINE - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Random seed                 : {SEED}")
summary_lines.append(f"Original embedding dimension: {original_dim}")
summary_lines.append(f"PCA dimension                : {PCA_COMPONENTS}")
summary_lines.append(f"Embeddings L2-normalized     : {is_l2_normalized} "
                      f"(mean norm={embedding_norms.mean():.4f}, std={embedding_norms.std():.4f})")
summary_lines.append(f"Explained variance ({PCA_COMPONENTS} components): "
                      f"{cumulative_explained_variance:.4f} ({cumulative_explained_variance:.2%})")
summary_lines.append(f"Gold-standard pairs           : {len(gold)}")
summary_lines.append(f"Valid mapped pairs             : {evaluated_pairs}")
summary_lines.append(f"Retrieval queries evaluated    : {evaluated_queries}")
summary_lines.append("")
summary_lines.append(f"Hits@1  : {pca_hits[1]:.4f}")
summary_lines.append(f"Hits@5  : {pca_hits[5]:.4f}")
summary_lines.append(f"Hits@10 : {pca_hits[10]:.4f}")
summary_lines.append(f"Hits@20 : {pca_hits[20]:.4f}")
summary_lines.append(f"MRR     : {pca_mrr:.4f}")
summary_lines.append("")
summary_lines.append(
    "PCA was fit only on the 1534 x 1280 embedding matrix, with no "
    "gold-standard cross-reactivity labels used at any point (unsupervised "
    "dimensionality reduction)."
)
summary_lines.append("")

header = f"{'Method':<24}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
summary_lines.append(
    f"{'Cosine similarity':<24}"
    f"{cosine_hits[1]:<10.4f}{cosine_hits[5]:<10.4f}{cosine_hits[10]:<10.4f}"
    f"{cosine_hits[20]:<10.4f}{cosine_mrr:<10.4f}"
)
summary_lines.append(
    f"{'PCA(128)+Euclidean':<24}"
    f"{pca_hits[1]:<10.4f}{pca_hits[5]:<10.4f}{pca_hits[10]:<10.4f}"
    f"{pca_hits[20]:<10.4f}{pca_mrr:<10.4f}"
)

if not cosine_available:
    summary_lines.append(f"\nNOTE: {COSINE_RESULTS} not found -- cosine baseline columns are NaN.")
else:
    summary_lines.append("")
    summary_lines.append("Delta (PCA+Euclidean minus Cosine similarity), same 592-query benchmark:")
    for k in TOP_K:
        delta = pca_hits[k] - cosine_hits[k]
        summary_lines.append(f"  Hits@{k:<3d}: {delta:+.4f}  ({delta:+.1%} absolute)")
    delta_mrr = pca_mrr - cosine_mrr
    summary_lines.append(f"  MRR    : {delta_mrr:+.4f}")

    better_count = sum(1 for k in TOP_K if pca_hits[k] > cosine_hits[k]) + int(pca_mrr > cosine_mrr)
    total_metrics = len(TOP_K) + 1
    if better_count == total_metrics:
        verdict = "PCA(128) + Euclidean distance is better than cosine similarity on ALL metrics."
    elif better_count == 0:
        verdict = "PCA(128) + Euclidean distance is worse than cosine similarity on ALL metrics."
    else:
        verdict = (
            f"PCA(128) + Euclidean distance is better on {better_count}/{total_metrics} "
            f"metrics and worse on {total_metrics - better_count}/{total_metrics} -- mixed result."
        )
    summary_lines.append("")
    summary_lines.append(f"VERDICT: {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
