"""
PCA(128) + Euclidean distance retrieval baseline na prosirenom gold standardu (1.443 redova).

* PCA se trenira samo na 1.534 × 1.280 embedding matrici, bez labela
* Embeddinge smanjuje na tacno 128 komponenti
* Rangiranje se radi pomocu Euclidean distance
* Self-match je iskljucen

Izlaz:
    output/pca_euclidean_retrieval_results_1443.csv
    output/pca_euclidean_summary_1443.txt
"""

import pickle
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
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
COSINE_RESULTS_1443 = Path("/home/lana/ALERGRAF/output/hits_mrr_results_1443.csv")
PCA_RESULTS_OLD = Path("/home/lana/ALERGRAF/output/pca_euclidean_retrieval_results.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
RETRIEVAL_OUTPUT = OUTPUT_DIR / "pca_euclidean_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "pca_euclidean_summary_1443.txt"

SEED = 42
PCA_COMPONENTS = 128
TOP_K = [1, 5, 10, 20]


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
print(f"Rows in file: {len(gold_raw)}")


# =====================================================
# EVIDENCE-LEVEL FILTERING (identical rule to hitsk_and_mrr_1443.py)
# =====================================================

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()
print(f"Excluded negative/contested/risky rows: {len(excluded)}")
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
# EMBEDDING MATRIX
# =====================================================

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array(
    [embeddings_dict[allergen_id] for allergen_id in all_ids],
    dtype=np.float64,
)
original_dim = embedding_matrix.shape[1]
print(f"Embedding matrix shape: {embedding_matrix.shape}")

embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
is_l2_normalized = bool(np.allclose(embedding_norms, 1.0, atol=1e-3))
print(f"Embeddings L2-normalized: {is_l2_normalized} (mean norm={embedding_norms.mean():.4f})")


# =====================================================
# PCA (unsupervised, same embedding matrix/config as the 296-pair version --
# only the evaluation query set changes)
# =====================================================

print("\n==============================")
print("PCA DIMENSIONALITY REDUCTION")
print("==============================")

pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full", random_state=SEED)
pca_matrix = pca.fit_transform(embedding_matrix)
cumulative_explained_variance = float(pca.explained_variance_ratio_.sum())

print(f"PCA-reduced matrix shape: {pca_matrix.shape}")
print(f"Cumulative explained variance ({PCA_COMPONENTS} components): "
      f"{cumulative_explained_variance:.4f} ({cumulative_explained_variance:.2%})")

distance_matrix = euclidean_distances(pca_matrix)


# =====================================================
# RETRIEVAL EVALUATION (all 1432 positive pairs, both directions)
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
        continue

    allergen_1 = name_to_id[name_1]
    allergen_2 = name_to_id[name_2]

    if allergen_1 not in id_to_index or allergen_2 not in id_to_index:
        missing_pairs += 1
        continue

    index_1 = id_to_index[allergen_1]
    index_2 = id_to_index[allergen_2]
    evaluated_pairs += 1

    directions = [
        {"query_name": name_1, "target_name": name_2, "query_id": allergen_1,
         "target_id": allergen_2, "query_index": index_1, "target_index": index_2,
         "family_query": row["family_1"], "family_target": row["family_2"]},
        {"query_name": name_2, "target_name": name_1, "query_id": allergen_2,
         "target_id": allergen_1, "query_index": index_2, "target_index": index_1,
         "family_query": row["family_2"], "family_target": row["family_1"]},
    ]

    for direction in directions:
        query_index = direction["query_index"]
        target_index = direction["target_index"]

        distances = distance_matrix[query_index].copy()
        distances[query_index] = np.inf

        ranked_indices = np.argsort(distances)
        positions = np.where(ranked_indices == target_index)[0]
        if len(positions) == 0:
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
print(f"\nPositive pairs (post-filter)  : {len(gold)}")
print(f"Retrieval queries evaluated   : {evaluated_queries}")

if len(result_df) == 0:
    print("\nERROR: No valid pairs were found.")
    raise SystemExit(1)


# =====================================================
# METRICS
# =====================================================

pca_hits = {k: result_df[f"hits_at_{k}"].mean() for k in TOP_K}
pca_mrr = result_df["reciprocal_rank"].mean()

print("\n==============================")
print("PCA(128) + EUCLIDEAN RESULTS (1443 dataset)")
print("==============================")
for k in TOP_K:
    print(f"Hits@{k:<2d}: {pca_hits[k]:.4f}")
print(f"MRR    : {pca_mrr:.4f}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nDetailed results saved to: {RETRIEVAL_OUTPUT}")


# =====================================================
# COMPARISON: cosine (1443), old cosine (296), old PCA (296)
# =====================================================

def load_hits_mrr(path):
    if not path.exists():
        return {k: float("nan") for k in TOP_K}, float("nan"), False
    df = pd.read_csv(path)
    return {k: df[f"hits_at_{k}"].mean() for k in TOP_K}, df["reciprocal_rank"].mean(), True


cosine_1443_hits, cosine_1443_mrr, cosine_1443_available = load_hits_mrr(COSINE_RESULTS_1443)
pca_old_hits, pca_old_mrr, pca_old_available = load_hits_mrr(PCA_RESULTS_OLD)


# =====================================================
# SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("PCA(128) + EUCLIDEAN BASELINE (1443 dataset) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Rows in output/cross_reactive_1443.csv: {len(gold_raw)}")
summary_lines.append(f"Excluded (negative/contested/risky)    : {len(excluded)}")
summary_lines.append(f"Positive gold-standard pairs retained  : {len(gold)}")
summary_lines.append(f"Original embedding dimension: {original_dim}")
summary_lines.append(f"PCA dimension                : {PCA_COMPONENTS}")
summary_lines.append(f"Embeddings L2-normalized     : {is_l2_normalized}")
summary_lines.append(f"Cumulative explained variance ({PCA_COMPONENTS} components): "
                      f"{cumulative_explained_variance:.4f} ({cumulative_explained_variance:.2%})")
summary_lines.append(f"Retrieval queries evaluated  : {evaluated_queries}")
summary_lines.append("")
summary_lines.append(f"Hits@1  : {pca_hits[1]:.4f}")
summary_lines.append(f"Hits@5  : {pca_hits[5]:.4f}")
summary_lines.append(f"Hits@10 : {pca_hits[10]:.4f}")
summary_lines.append(f"Hits@20 : {pca_hits[20]:.4f}")
summary_lines.append(f"MRR     : {pca_mrr:.4f}")
summary_lines.append("")

header = f"{'Method':<26}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
summary_lines.append(header)
summary_lines.append("-" * len(header))

rows_to_print = [
    ("PCA (296, old)", pca_old_hits, pca_old_mrr, pca_old_available),
    ("Cosine (1432, new)", cosine_1443_hits, cosine_1443_mrr, cosine_1443_available),
    ("PCA (1432, new)", pca_hits, pca_mrr, True),
]
for label, hits_dict, mrr_val, available in rows_to_print:
    if available:
        summary_lines.append(
            f"{label:<26}{hits_dict[1]:<10.4f}{hits_dict[5]:<10.4f}{hits_dict[10]:<10.4f}"
            f"{hits_dict[20]:<10.4f}{mrr_val:<10.4f}"
        )
    else:
        summary_lines.append(f"{label:<26}(results file not found)")

if pca_old_available:
    summary_lines.append("")
    summary_lines.append("Delta (PCA 1432-pair minus PCA 296-pair):")
    for k in TOP_K:
        summary_lines.append(f"  Hits@{k:<3d}: {pca_hits[k] - pca_old_hits[k]:+.4f}")
    summary_lines.append(f"  MRR    : {pca_mrr - pca_old_mrr:+.4f}")

if cosine_1443_available:
    summary_lines.append("")
    summary_lines.append("Delta (PCA minus Cosine), both on the 1432-pair dataset:")
    for k in TOP_K:
        summary_lines.append(f"  Hits@{k:<3d}: {pca_hits[k] - cosine_1443_hits[k]:+.4f}")
    summary_lines.append(f"  MRR    : {pca_mrr - cosine_1443_mrr:+.4f}")
    better_count = sum(1 for k in TOP_K if pca_hits[k] > cosine_1443_hits[k]) + int(pca_mrr > cosine_1443_mrr)
    total_metrics = len(TOP_K) + 1
    if better_count == 0:
        verdict = "PCA(128)+Euclidean is worse than cosine similarity on ALL metrics (1432-pair dataset)."
    elif better_count == total_metrics:
        verdict = "PCA(128)+Euclidean is better than cosine similarity on ALL metrics (1432-pair dataset)."
    else:
        verdict = (f"PCA(128)+Euclidean is better on {better_count}/{total_metrics} metrics "
                   f"(1432-pair dataset) -- mixed result.")
    summary_lines.append(f"\nVERDICT: {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
