"""
Cosine retrieval sa residue-level top-k slicnoscu (Eksperiment 2, korak 2) - 1443 dataset.

Pokrece se na VM-u (GPU), posle generate_residue_embeddings.py (korak 1).
Za svaki par proteina racuna La x Lb matricu slicnosti po amino-kiselini,
pa top-k prosek (k=5,15,30) kao skor - testira da li lokalni region
(epitope) daje bolji signal od whole-protein pooling-a.

QUERY_LIMIT ogranicava broj query-ja zbog cene racunanja (None = pun run).

Rezultat: nema razlike od mean/max pooling-a (sve unutar suma) - problem
nije agregacija preko sekvence, verovatno je potreban 3D/strukturni signal.

Ulaz:
    /content/residue_embeddings.pkl.gz
    /content/residue_embeddings_metadata.parquet
    /content/cross_reactive_1443.csv
    /content/hits_mrr_results_1443.csv          (mean-pooling cosine result)
    /content/hits_mrr_results_maxpool_1443.csv  (max-pooling cosine result)

Izlaz:
    /content/hits_mrr_results_residue_topk_1443.csv
    /content/hits_mrr_summary_residue_topk_1443.txt
"""

import gzip
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =====================================================
# PATHS
# =====================================================

RESIDUE_EMBEDDINGS = Path("/content/residue_embeddings.pkl.gz")
METADATA = Path("/content/residue_embeddings_metadata.parquet")
GOLD = Path("/content/cross_reactive_1443.csv")

MEANPOOL_RESULTS = Path("/content/hits_mrr_results_1443.csv")
MAXPOOL_RESULTS = Path("/content/hits_mrr_results_maxpool_1443.csv")

RETRIEVAL_OUTPUT = Path("/content/hits_mrr_results_residue_topk_1443.csv")
SUMMARY_OUTPUT = Path("/content/hits_mrr_summary_residue_topk_1443.txt")


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
TOP_K_SIMILARITY_VALUES = [5, 15, 30]   # residue-pair aggregation sizes (sensitivity check)
TOP_K = [1, 5, 10, 20]                   # retrieval Hits@K, same as every other *_1443 script

QUERY_LIMIT = 200   # set to None for the full 2864-query benchmark once you've timed a subsample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# LOAD DATA
# =====================================================

print("\n==============================")
print("LOADING DATA")
print("==============================")
print(f"Device: {device}")

print("Loading residue-level embeddings (this can take a while)...")
with gzip.open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)
print(f"Proteins with residue embeddings: {len(residue_embeddings)}")

metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(residue_embeddings.keys())].copy()
print(f"Metadata rows with embeddings: {len(metadata)}")

gold_raw = pd.read_csv(GOLD)
print(f"Rows in gold file: {len(gold_raw)}")


# =====================================================
# EVIDENCE-LEVEL FILTERING (identical rule to every other *_1443 script)
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
for _, row in metadata.iterrows():
    official_name = str(row["official_name"]).strip()
    if official_name and official_name.lower() != "nan" and official_name not in name_to_id:
        name_to_id[official_name] = row["allergen_id"]

print(f"Official names mapped : {len(name_to_id)}")


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
    id_1, id_2 = name_to_id[name_1], name_to_id[name_2]
    if id_1 not in residue_embeddings or id_2 not in residue_embeddings or id_1 == id_2:
        missing_pairs += 1
        continue
    gold_pairs.append({
        "pair_id": row["pair_id"], "id_1": id_1, "id_2": id_2,
        "name_1": name_1, "name_2": name_2,
        "family_1": row["family_1"], "family_2": row["family_2"],
    })

print(f"Mapped gold pairs : {len(gold_pairs)}")
print(f"Missing/unmapped  : {missing_pairs}")


# =====================================================
# BUILD QUERY LIST (both directions, optionally subsampled)
# =====================================================

all_queries = []
for p in gold_pairs:
    all_queries.append((p["pair_id"], p["id_1"], p["id_2"], p["name_1"], p["name_2"],
                         p["family_1"], p["family_2"]))
    all_queries.append((p["pair_id"], p["id_2"], p["id_1"], p["name_2"], p["name_1"],
                         p["family_2"], p["family_1"]))

print(f"\nTotal possible queries: {len(all_queries)}")

if QUERY_LIMIT is not None and QUERY_LIMIT < len(all_queries):
    rng = np.random.default_rng(SEED)
    subsample_idx = rng.choice(len(all_queries), size=QUERY_LIMIT, replace=False)
    queries_to_run = [all_queries[i] for i in sorted(subsample_idx)]
    print(f"QUERY_LIMIT={QUERY_LIMIT} -- evaluating a fixed-seed random subsample. "
          f"Set QUERY_LIMIT = None for the full benchmark once timing looks OK.")
else:
    queries_to_run = all_queries
    print("Evaluating the FULL query set (QUERY_LIMIT = None).")


# =====================================================
# MOVE (L2-NORMALIZED) RESIDUE MATRICES TO DEVICE
# =====================================================

print("\n==============================")
print("PRELOADING RESIDUE MATRICES ONTO DEVICE")
print("==============================")

all_ids = list(residue_embeddings.keys())
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

residue_tensors = {}
for allergen_id, matrix in residue_embeddings.items():
    t = torch.from_numpy(matrix).to(device=device, dtype=torch.float32)
    t = t / (t.norm(dim=1, keepdim=True) + 1e-12)  # L2-normalize each residue row once
    residue_tensors[allergen_id] = t

print(f"Preloaded {len(residue_tensors)} proteins onto {device}.")


# =====================================================
# TOP-K RESIDUE SIMILARITY
# =====================================================

def topk_residue_similarity(query_id, candidate_id, k_values):
    """Returns {k: mean of the top-k values in the La x Lb cosine
    similarity matrix} for every k in k_values, computed together from
    one sorted pass."""
    q = residue_tensors[query_id]       # (Lq, 1280), already L2-normalized
    c = residue_tensors[candidate_id]   # (Lc, 1280), already L2-normalized

    sim_matrix = q @ c.T                # (Lq, Lc) cosine similarities
    flat_sorted, _ = torch.sort(sim_matrix.flatten(), descending=True)

    return {k: flat_sorted[:k].mean().item() for k in k_values}


# =====================================================
# RETRIEVAL EVALUATION
# =====================================================

print("\n==============================")
print("RUNNING RESIDUE TOP-K RETRIEVAL")
print("==============================")
print(f"Queries to evaluate: {len(queries_to_run)}  x  {len(all_ids) - 1} candidates each")
print(f"K values (residue-pair aggregation): {TOP_K_SIMILARITY_VALUES}")

results = []
start_time = time.time()

for q_num, (pair_id, query_id, target_id, query_name, target_name, family_q, family_t) in enumerate(queries_to_run, 1):
    target_index = id_to_index[target_id]

    # score every OTHER candidate for each K simultaneously
    scores_per_k = {k: np.full(len(all_ids), -np.inf, dtype=np.float64) for k in TOP_K_SIMILARITY_VALUES}

    for candidate_id in all_ids:
        if candidate_id == query_id:
            continue
        candidate_index = id_to_index[candidate_id]
        per_k_scores = topk_residue_similarity(query_id, candidate_id, TOP_K_SIMILARITY_VALUES)
        for k, score in per_k_scores.items():
            scores_per_k[k][candidate_index] = score

    row = {
        "pair_id": pair_id, "query_allergen": query_name, "target_allergen": target_name,
        "query_allergen_id": query_id, "target_allergen_id": target_id,
        "query_family": family_q, "target_family": family_t,
    }

    for k in TOP_K_SIMILARITY_VALUES:
        ranked = np.argsort(scores_per_k[k])[::-1]
        rank = int(np.where(ranked == target_index)[0][0]) + 1
        row[f"top{k}_rank"] = rank
        row[f"top{k}_reciprocal_rank"] = 1.0 / rank
        for hk in TOP_K:
            row[f"top{k}_hits_at_{hk}"] = int(rank <= hk)

    results.append(row)

    if q_num % 20 == 0 or q_num == len(queries_to_run):
        elapsed = time.time() - start_time
        rate = q_num / elapsed
        remaining = (len(queries_to_run) - q_num) / rate if rate > 0 else float("inf")
        print(f"  {q_num}/{len(queries_to_run)} queries  "
              f"({rate:.3f} q/s, ~{remaining/60:.1f} min remaining)")

result_df = pd.DataFrame(results)
print(f"\nTotal time: {(time.time() - start_time)/60:.1f} minutes "
      f"for {len(queries_to_run)} queries")

result_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"Detailed results saved to: {RETRIEVAL_OUTPUT}")


# =====================================================
# METRICS PER K
# =====================================================

print("\n==============================")
print("RESULTS PER TOP-K")
print("==============================")

metrics_per_k = {}
for k in TOP_K_SIMILARITY_VALUES:
    hits = {hk: result_df[f"top{k}_hits_at_{hk}"].mean() for hk in TOP_K}
    mrr = result_df[f"top{k}_reciprocal_rank"].mean()
    metrics_per_k[k] = (hits, mrr)
    print(f"\ntop-{k} residue similarity:")
    for hk in TOP_K:
        print(f"  Hits@{hk:<2d}: {hits[hk]:.4f}")
    print(f"  MRR    : {mrr:.4f}")


# =====================================================
# OPTIONAL COMPARISON (only if the other results files were uploaded)
# =====================================================

def load_hits_mrr(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {k: df[f"hits_at_{k}"].mean() for k in TOP_K}, df["reciprocal_rank"].mean()


meanpool = load_hits_mrr(MEANPOOL_RESULTS)
maxpool = load_hits_mrr(MAXPOOL_RESULTS)


# =====================================================
# SUMMARY
# =====================================================

lines = []


def add(line=""):
    lines.append(line)
    print(line)


add("\n" + "=" * 60)
add("RESIDUE TOP-K SIMILARITY (1443 dataset) - SUMMARY")
add("=" * 60)
add(f"Query subsample: {len(queries_to_run)} / {len(all_queries)} possible queries "
    f"(QUERY_LIMIT={QUERY_LIMIT})")
add(f"Device used: {device}")
add("")

header = f"{'Method':<28}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
add(header)
add("-" * len(header))
if meanpool:
    add(f"{'Cosine (mean pooling, full)':<28}{meanpool[0][1]:<10.4f}{meanpool[0][5]:<10.4f}"
        f"{meanpool[0][10]:<10.4f}{meanpool[0][20]:<10.4f}{meanpool[1]:<10.4f}")
if maxpool:
    add(f"{'Cosine (max pooling, full)':<28}{maxpool[0][1]:<10.4f}{maxpool[0][5]:<10.4f}"
        f"{maxpool[0][10]:<10.4f}{maxpool[0][20]:<10.4f}{maxpool[1]:<10.4f}")
for k in TOP_K_SIMILARITY_VALUES:
    hits, mrr = metrics_per_k[k]
    add(f"{'Residue top-' + str(k) + ' (subsample)':<28}{hits[1]:<10.4f}{hits[5]:<10.4f}"
        f"{hits[10]:<10.4f}{hits[20]:<10.4f}{mrr:<10.4f}")

if not meanpool and not maxpool:
    add("\nNOTE: mean/max pooling comparison files not found in /content/ -- "
        "download this script's CSV output and compare locally instead.")
if QUERY_LIMIT is not None:
    add(f"\nCAUTION: this ran on a {len(queries_to_run)}-query SUBSAMPLE, not the full "
        f"{len(all_queries)}-query benchmark -- treat as a directional signal, not a "
        f"final number, until re-run with QUERY_LIMIT = None.")

summary_text = "\n".join(lines)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDownload both output files back to your local output/ folder.")
print("Done.")
