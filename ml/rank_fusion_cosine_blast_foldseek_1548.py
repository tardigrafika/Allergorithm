"""
Rank fusion (RRF + Borda count): cosine + BLAST + Foldseek TM-score kao TRI
NEZAVISNA glasa, kombinovana bez ucenja (bez RF/MLP) - 1548 dataset.

Zasto ovo, ne RF ensemble (koji smo vec probali): rank fusion metode (RRF,
Borda count) ne uce tezine iz podataka - samo kombinuju vec postojece
rangove po fiksnom pravilu. To ih cini otpornijim na overfitting na mali
dataset (kao sto predlog navodi), i - kljucno - NE TREBA im train/test split
uopste (nema parametara koji se uce), pa mozemo evaluirati SVE 1537 poznatih
parova odjednom, bez LOCO fold-ova. Vise statisticke snage nego bilo koji
raniji test u sesiji.

Prethodni pokusaj slicnog ensembl-a (ensemble_cosine_rf_1443.py, RANO u
sesiji, na jednom naivnom split-u): RRF je popravio cosine (0.181->0.186)
ali nije prestigao cist RF (0.198). Nikad nije prosao kroz rigorozan
LOCO/micro protokol - ovo je ta provera, sa sva tri signala koja sad imamo
(cosine, BLAST, Foldseek TM - poslednji nije ni postojao tada).

RRF: score(c) = sum_s 1/(K + rank_s(c)), K=60 (standardna konstanta iz
literature, ne fitovana na nasim podacima).
Borda: score(c) = -sum_s rank_s(c) (nizi zbir rangova = bolje).

Izlaz:
    output/rank_fusion_1548_summary.txt
    output/rank_fusion_1548_per_query.csv
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_fusion_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rank_fusion_1548_per_query.csv"

RRF_K = 60
TOP_K = [1, 5, 10, 20]


# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_score_matrix = blast_data["score_matrix"]  # BLAST score, not identity -- established as the informative one
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

# BLAST matrix re-aligned to all_ids ordering (vectorized permutation, not a python double loop)
missing_from_blast = [aid for aid in all_ids if aid not in blast_id_to_index]
if missing_from_blast:
    print(f"WARNING: {len(missing_from_blast)} proteins missing from BLAST matrix -- will get score 0.0")
perm = np.array([blast_id_to_index.get(aid, -1) for aid in all_ids])
valid = perm >= 0
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

# dense Foldseek TM-score matrix aligned to all_ids (0.0 fallback for undetected pairs)
print("Building dense Foldseek TM-score matrix (0.0 fallback for undetected pairs)...")
foldseek_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue  # self-pair (frozenset collapsed to 1 element) -- not needed, diagonal unused in ranking
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
    """Vraca 1-indeksirane rangove (1=najbolji) za sve kandidate, self iskljucen."""
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]  # descending
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def rank_of(ranks, target_index):
    return int(ranks[target_index])


# =====================================================
# MAIN LOOP -- no training, no folds needed (nothing is fit to data)
# =====================================================

print("\nScoring all queries (cosine / BLAST / FoldseekTM individually, + RRF + Borda fusion)...")
start = time.time()
records = []

for qi, p in enumerate(gold_pairs):
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)

        rrf_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
        borda_score = -(cos_ranks + blast_ranks + fs_ranks)

        rrf_ranks = ranks_from_scores(rrf_score, qidx)
        borda_ranks = ranks_from_scores(borda_score, qidx)

        records.append({
            "pair_id": p["pair_id"],
            "cosine_rank": rank_of(cos_ranks, tidx),
            "blast_rank": rank_of(blast_ranks, tidx),
            "foldseektm_rank": rank_of(fs_ranks, tidx),
            "rrf_rank": rank_of(rrf_ranks, tidx),
            "borda_rank": rank_of(borda_ranks, tidx),
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

methods = ["cosine", "blast", "foldseektm", "rrf", "borda"]
summary_lines = [
    "=" * 70,
    f"Rank fusion (RRF + Borda): cosine vs BLAST vs FoldseekTM vs fusion ({len(df)} queries, 1548 dataset)",
    "=" * 70,
    "No training / no folds needed -- fusion has no learned parameters, full pair set used directly.",
    "",
]

for m in methods:
    rr = 1.0 / df[f"{m}_rank"]
    mrr = rr.mean()
    hits = {k: (df[f"{m}_rank"] <= k).mean() for k in TOP_K}
    hits_str = "  ".join(f"Hits@{k}={hits[k]:.4f}" for k in TOP_K)
    summary_lines.append(f"{m:12s} MRR={mrr:.4f}  {hits_str}")

summary_lines.append("")
best_individual = max(["cosine", "blast", "foldseektm"], key=lambda m: (1.0 / df[f"{m}_rank"]).mean())
best_individual_mrr = (1.0 / df[f"{best_individual}_rank"]).mean()
rrf_mrr = (1.0 / df["rrf_rank"]).mean()
borda_mrr = (1.0 / df["borda_rank"]).mean()
summary_lines.append(f"Best individual signal: {best_individual} (MRR={best_individual_mrr:.4f})")
summary_lines.append(f"RRF fusion delta vs best individual: {rrf_mrr - best_individual_mrr:+.4f}")
summary_lines.append(f"Borda fusion delta vs best individual: {borda_mrr - best_individual_mrr:+.4f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
