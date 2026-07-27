"""
Cosine retrieval benchmark sa MAX pooling embeddinzima (Eksperiment 1) - 1443 dataset.

Ne menja hitsk_and_mrr_1443.py (mean pooling ostaje odvojen, reproducibilan).
Isti protokol, samo drugi embeddinzi (embeddings_maxpool.pkl, generisano na VM-u).
Cisto poredjenje bez treninga - ako max pooling ne pobedi mean pooling ovde,
nema smisla trenirati RF/MLP na njemu.

Izlaz:
    output/hits_mrr_results_maxpool_1443.csv
    output/hits_mrr_summary_maxpool_1443.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings_maxpool.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings_maxpool.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")

MEANPOOL_RESULTS = Path("/home/lana/ALERGRAF/output/hits_mrr_results_1443.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
RETRIEVAL_OUTPUT = OUTPUT_DIR / "hits_mrr_results_maxpool_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "hits_mrr_summary_maxpool_1443.txt"

TOP_K = [1, 5, 10, 20]


# =====================================================
# LOAD EMBEDDINGS + METADATA
# =====================================================

print("\n==============================")
print("LOADING DATA (max-pooled embeddings)")
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

print(f"\nOfficial names mapped : {len(name_to_id)}")


# =====================================================
# EMBEDDING MATRIX + COSINE SIMILARITY MATRIX
# =====================================================

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array(
    [embeddings_dict[allergen_id] for allergen_id in all_ids],
    dtype=np.float64,
)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
print(f"L2 norm  mean: {embedding_norms.mean():.4f}  std: {embedding_norms.std():.4f}")

similarity_matrix = cosine_similarity(embedding_matrix)


# =====================================================
# GOLD STANDARD EVALUATION (identical protocol to hitsk_and_mrr_1443.py)
# =====================================================

print("\n==============================")
print("RUNNING HITS@K / MRR (max pooling)")
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

        similarities = similarity_matrix[query_index].copy()
        similarities[query_index] = -np.inf

        ranked_indices = np.argsort(similarities)[::-1]
        positions = np.where(ranked_indices == target_index)[0]
        if len(positions) == 0:
            continue

        rank = int(positions[0]) + 1
        true_similarity = similarity_matrix[query_index, target_index]
        reciprocal_rank = 1.0 / rank

        results.append({
            "pair_id": pair_id,
            "query_allergen": direction["query_name"],
            "target_allergen": direction["target_name"],
            "query_allergen_id": direction["query_id"],
            "target_allergen_id": direction["target_id"],
            "query_family": direction["family_query"],
            "target_family": direction["family_target"],
            "cosine_similarity": true_similarity,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "hits_at_1": int(rank <= 1),
            "hits_at_5": int(rank <= 5),
            "hits_at_10": int(rank <= 10),
            "hits_at_20": int(rank <= 20),
        })
        evaluated_queries += 1

result_df = pd.DataFrame(results)

print(f"\nGold standard pairs (post-filter): {len(gold)}")
print(f"Valid pairs evaluated             : {evaluated_pairs}")
print(f"Missing pairs                     : {missing_pairs}")
print(f"Retrieval queries evaluated       : {evaluated_queries}")

if len(result_df) == 0:
    print("\nERROR: No valid pairs were found.")
    raise SystemExit(1)


# =====================================================
# METRICS
# =====================================================

hits = {k: result_df[f"hits_at_{k}"].mean() for k in TOP_K}
mrr = result_df["reciprocal_rank"].mean()

print("\n==============================")
print("MAX-POOLING COSINE RESULTS")
print("==============================")
for k in TOP_K:
    print(f"Hits@{k:<2d}: {hits[k]:.4f}")
print(f"MRR    : {mrr:.4f}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nDetailed results saved to: {RETRIEVAL_OUTPUT}")


# =====================================================
# COMPARISON WITH MEAN-POOLING COSINE BASELINE
# =====================================================

meanpool_available = MEANPOOL_RESULTS.exists()
meanpool_hits = {k: float("nan") for k in TOP_K}
meanpool_mrr = float("nan")

if meanpool_available:
    meanpool_df = pd.read_csv(MEANPOOL_RESULTS)
    meanpool_hits = {k: meanpool_df[f"hits_at_{k}"].mean() for k in TOP_K}
    meanpool_mrr = meanpool_df["reciprocal_rank"].mean()


# =====================================================
# SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("COSINE SIMILARITY, MAX POOLING (1443 dataset) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Excluded (negative/contested/risky): {len(excluded)}")
summary_lines.append(f"Positive gold-standard pairs retained: {len(gold)}")
summary_lines.append(f"Retrieval queries evaluated: {evaluated_queries}")
summary_lines.append(f"Embedding L2 norm mean/std: {embedding_norms.mean():.4f} / {embedding_norms.std():.4f}")
summary_lines.append("")
summary_lines.append(f"Hits@1  : {hits[1]:.4f}")
summary_lines.append(f"Hits@5  : {hits[5]:.4f}")
summary_lines.append(f"Hits@10 : {hits[10]:.4f}")
summary_lines.append(f"Hits@20 : {hits[20]:.4f}")
summary_lines.append(f"MRR     : {mrr:.4f}")
summary_lines.append("")

header = f"{'Method':<26}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
summary_lines.append(
    f"{'Cosine (mean pooling)':<26}{meanpool_hits[1]:<10.4f}{meanpool_hits[5]:<10.4f}"
    f"{meanpool_hits[10]:<10.4f}{meanpool_hits[20]:<10.4f}{meanpool_mrr:<10.4f}"
)
summary_lines.append(
    f"{'Cosine (max pooling)':<26}{hits[1]:<10.4f}{hits[5]:<10.4f}"
    f"{hits[10]:<10.4f}{hits[20]:<10.4f}{mrr:<10.4f}"
)

if meanpool_available:
    summary_lines.append("")
    summary_lines.append("Delta (max pooling minus mean pooling):")
    for k in TOP_K:
        summary_lines.append(f"  Hits@{k:<3d}: {hits[k] - meanpool_hits[k]:+.4f}")
    summary_lines.append(f"  MRR    : {mrr - meanpool_mrr:+.4f}")

    better_count = sum(1 for k in TOP_K if hits[k] > meanpool_hits[k]) + int(mrr > meanpool_mrr)
    total_metrics = len(TOP_K) + 1
    if better_count == total_metrics:
        verdict = "Max pooling is BETTER than mean pooling on ALL metrics."
    elif better_count == 0:
        verdict = "Max pooling is WORSE than mean pooling on ALL metrics."
    else:
        verdict = (f"Max pooling is better on {better_count}/{total_metrics} metrics "
                   f"-- mixed result.")
    summary_lines.append(f"\nVERDICT: {verdict}")
    summary_lines.append(
        "\n(This is a purely unsupervised, zero-training comparison. If max pooling "
        "does not clearly beat mean pooling here, retraining RF/MLP on top of it is "
        "unlikely to be worth the effort either -- see the pooling-strategy discussion "
        "this experiment grew out of.)"
    )
else:
    summary_lines.append(f"\nNOTE: {MEANPOOL_RESULTS} not found -- run analysis/hitsk_and_mrr_1443.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
