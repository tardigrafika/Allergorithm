"""
Dodaje Ankh cosine similarity kao NOVI nezavisan glas u RRF fuziju
(cosine_ESM + BLAST + FoldseekTM + cosine_Ankh) - 1548 dataset.

Ankh je T5-stil (encoder-decoder) protein language model, arhitekturno
razlicit od ESM-2 (encoder-only) - test da li drugaciji PLM nosi genuinski
nezavisan signal, ili je (kao k-mer/Pfam) redundantan sa onim sto vec imamo.

Izlaz:
    output/rank_fusion_ankh_1548_summary.txt
    output/rank_fusion_ankh_1548_per_query.csv
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
ANKH_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings_ankh.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_fusion_ankh_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rank_fusion_ankh_1548_per_query.csv"

RRF_K = 60
TOP_K = [1, 5, 10, 20]


# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
with open(ANKH_EMBEDDINGS, "rb") as f:
    ankh_dict = pickle.load(f)
metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {aid: i for i, aid in enumerate(all_ids)}
n_candidates = len(all_ids)
embedding_matrix = np.array([embeddings_dict[aid] for aid in all_ids], dtype=np.float64)
cosine_matrix = cosine_similarity(embedding_matrix)

n_missing_ankh = sum(1 for aid in all_ids if aid not in ankh_dict)
print(f"Proteins missing an Ankh embedding: {n_missing_ankh}/{n_candidates}")
ankh_dim = len(next(iter(ankh_dict.values())))
ankh_matrix_raw = np.array(
    [ankh_dict.get(aid, np.zeros(ankh_dim)) for aid in all_ids], dtype=np.float64
)
ankh_cosine_matrix = cosine_similarity(ankh_matrix_raw)
has_ankh = np.array([aid in ankh_dict for aid in all_ids])

perm = np.array([blast_id_to_index.get(aid, -1) for aid in all_ids])
valid = perm >= 0
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

print("Building dense Foldseek TM-score matrix...")
foldseek_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in id_to_index and b in id_to_index:
        i, j = id_to_index[a], id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})
print(f"Gold pairs: {len(gold_pairs)}")


# =====================================================
# RANKING HELPERS
# =====================================================

def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def rank_of(ranks, target_index):
    return int(ranks[target_index])


# =====================================================
# MAIN LOOP
# =====================================================

print("\nScoring all queries...")
start = time.time()
records = []

for qi, p in enumerate(gold_pairs):
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)

        rrf3_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)

        ankh_contrib = np.zeros(n_candidates, dtype=np.float64)
        query_has_ankh = has_ankh[qidx]
        if query_has_ankh:
            ankh_ranks = ranks_from_scores(ankh_cosine_matrix[qidx], qidx)
            ankh_contrib = 1.0 / (RRF_K + ankh_ranks)

        rrf_ankh_score = rrf3_score + ankh_contrib

        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)
        rrf_ankh_ranks = ranks_from_scores(rrf_ankh_score, qidx)
        ankh_only_ranks = ranks_from_scores(ankh_cosine_matrix[qidx], qidx)

        records.append({
            "pair_id": p["pair_id"],
            "query_has_ankh": bool(query_has_ankh),
            "cosine_esm_rank": rank_of(cos_ranks, tidx),
            "ankh_rank": rank_of(ankh_only_ranks, tidx),
            "blast_rank": rank_of(blast_ranks, tidx),
            "foldseektm_rank": rank_of(fs_ranks, tidx),
            "rrf3_rank": rank_of(rrf3_ranks, tidx),
            "rrf_ankh_rank": rank_of(rrf_ankh_ranks, tidx),
        })

    if (qi + 1) % 200 == 0 or (qi + 1) == len(gold_pairs):
        elapsed = time.time() - start
        print(f"  {qi+1}/{len(gold_pairs)} pairs ({elapsed/60:.1f} min elapsed)", flush=True)

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}")


# =====================================================
# AGGREGATE
# =====================================================

ankh_mrr = (1.0 / df["ankh_rank"]).mean()
cos_esm_mrr = (1.0 / df["cosine_esm_rank"]).mean()
rrf3_mrr = (1.0 / df["rrf3_rank"]).mean()
rrf_ankh_mrr = (1.0 / df["rrf_ankh_rank"]).mean()

summary_lines = [
    "=" * 70,
    f"RRF: does a second PLM (Ankh) help as a voter? ({len(df)} queries, 1548 dataset)",
    "=" * 70,
    f"Query has Ankh embedding: {df['query_has_ankh'].sum()}/{len(df)}",
    "",
    f"Ankh cosine ALONE (individual signal)   MRR = {ankh_mrr:.4f}",
    f"ESM cosine ALONE (for reference)        MRR = {cos_esm_mrr:.4f}",
    "",
    f"RRF-3 (cosine_ESM+BLAST+FoldseekTM)     MRR = {rrf3_mrr:.4f}",
    f"RRF-3 + cosine_Ankh                     MRR = {rrf_ankh_mrr:.4f}",
    f"Delta: {rrf_ankh_mrr - rrf3_mrr:+.4f}",
    "",
]
for k in TOP_K:
    h3 = (df["rrf3_rank"] <= k).mean()
    ha = (df["rrf_ankh_rank"] <= k).mean()
    summary_lines.append(f"Hits@{k}: RRF-3={h3:.4f}  RRF-3+Ankh={ha:.4f}")

rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["rrf_ankh_rank"], weights=w) - np.average(1.0 / sub["rrf3_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
summary_lines.append("")
summary_lines.append(f"Bootstrap 95% CI (RRF-3+Ankh - RRF-3): [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]")
summary_lines.append(f"Fraction of bootstrap resamples favoring RRF-3+Ankh: {(deltas>0).mean():.3f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
