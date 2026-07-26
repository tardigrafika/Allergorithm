"""
HITS@K i MRR benchmark za ESM-2 embeddinge na prosirenom gold standardu 1443

Koristi isti protokol kao originalni skript: bez treniranja, svaki gold pair se proverava u oba smera prema celom poolu od 1.534 proteina bez self-matcha.


*11 je negativno ili osporeno i izbaceno je iz pozitivnog gold standarda

Confirmed / Strong: 244
Suspected: 76
Inferred (family-level): 1.112

Ulaz:
    embeddings/embeddings.pkl, embeddings/embeddings.parquet
    output/cross_reactive_1443.csv

Izlaz:
    output/hits_mrr_results_1443.csv
    output/hits_mrr_summary_1443.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
RETRIEVAL_OUTPUT = OUTPUT_DIR / "hits_mrr_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "hits_mrr_summary_1443.txt"

OLD_RESULTS = OUTPUT_DIR / "hits_mrr_results.csv"  # original 296-pair benchmark, for comparison

TOP_K = [1, 5, 10, 20]


# =====================================================
# LOAD EMBEDDINGS + METADATA
# =====================================================

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

print("Loading extended gold standard (1443)...")
gold_raw = pd.read_csv(GOLD)
print(f"Rows in file: {len(gold_raw)}")


# =====================================================
# EXCLUDE NEGATIVE / CONTESTED ROWS (data quality step, see docstring)
# =====================================================

print("\n==============================")
print("EVIDENCE-LEVEL FILTERING")
print("==============================")

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()

print(f"Rows excluded as negative/contested/risky: {len(excluded)}")
for _, row in excluded.iterrows():
    print(f"  {row['pair_id']}: {row['allergen_id_1']} <-> {row['allergen_id_2']} "
          f"-- \"{row['evidence_level']}\"")
print(f"\nPositive gold-standard pairs retained: {len(gold)}")

evidence_bucket_counts = {"Confirmed/Strong": 0, "Suspected": 0, "Inferred/family-level": 0, "UNMAPPED": 0}
for v in gold["evidence_level"]:
    if v.startswith("Confirmed") or v.startswith("Strong evidence"):
        evidence_bucket_counts["Confirmed/Strong"] += 1
    elif v.startswith("Suspected"):
        evidence_bucket_counts["Suspected"] += 1
    elif v.startswith("Inferred"):
        evidence_bucket_counts["Inferred/family-level"] += 1
    else:
        evidence_bucket_counts["UNMAPPED"] += 1

print("\nEvidence-level breakdown of retained positives (informational only -- "
      "cosine similarity does not use these weights):")
for bucket, count in evidence_bucket_counts.items():
    print(f"  {bucket:<24}: {count}")


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
print(f"Duplicate names skipped: {duplicate_names}")


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

similarity_matrix = cosine_similarity(embedding_matrix)


# =====================================================
# GOLD STANDARD EVALUATION 
# =====================================================

print("\n==============================")
print("RUNNING HITS@K / MRR")
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
print("COSINE (1443 dataset) RESULTS")
print("==============================")
for k in TOP_K:
    print(f"Hits@{k:<2d}: {hits[k]:.4f}")
print(f"MRR    : {mrr:.4f}")


# =====================================================
# SAVE
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nDetailed results saved to: {RETRIEVAL_OUTPUT}")


# =====================================================
# COMPARISON WITH ORIGINAL 296 BENCHMARK
# =====================================================

old_available = OLD_RESULTS.exists()
old_hits = {k: float("nan") for k in TOP_K}
old_mrr = float("nan")

if old_available:
    old_df = pd.read_csv(OLD_RESULTS)
    old_hits = {k: old_df[f"hits_at_{k}"].mean() for k in TOP_K}
    old_mrr = old_df["reciprocal_rank"].mean()


# =====================================================
# SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("COSINE SIMILARITY BASELINE (1443 dataset) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Rows in output/cross_reactive_1443.csv : {len(gold_raw)}")
summary_lines.append(f"Excluded (negative/contested/risky)     : {len(excluded)}")
summary_lines.append(f"Positive gold-standard pairs retained   : {len(gold)}")
for bucket, count in evidence_bucket_counts.items():
    summary_lines.append(f"  {bucket:<24}: {count}")
summary_lines.append(f"Retrieval queries evaluated              : {evaluated_queries}")
summary_lines.append("")
summary_lines.append(f"Hits@1  : {hits[1]:.4f}")
summary_lines.append(f"Hits@5  : {hits[5]:.4f}")
summary_lines.append(f"Hits@10 : {hits[10]:.4f}")
summary_lines.append(f"Hits@20 : {hits[20]:.4f}")
summary_lines.append(f"MRR     : {mrr:.4f}")
summary_lines.append("")

header = f"{'Metric':<10}{'Cosine (296, old)':<20}{'Cosine (1432, new)':<20}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
for k in TOP_K:
    summary_lines.append(f"{'Hits@' + str(k):<10}{old_hits[k]:<20.4f}{hits[k]:<20.4f}")
summary_lines.append(f"{'MRR':<10}{old_mrr:<20.4f}{mrr:<20.4f}")

if old_available:
    summary_lines.append("")
    summary_lines.append("Delta (1432-pair minus 296-pair), full-dataset benchmarks:")
    for k in TOP_K:
        delta = hits[k] - old_hits[k]
        summary_lines.append(f"  Hits@{k:<3d}: {delta:+.4f}")
    summary_lines.append(f"  MRR    : {mrr - old_mrr:+.4f}")
else:
    summary_lines.append(f"\nNOTE: {OLD_RESULTS} not found -- run analysis/hitsk_and_mrr.py first for the old column.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
