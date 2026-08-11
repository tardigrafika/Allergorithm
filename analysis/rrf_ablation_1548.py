"""
RRF ablation: svih 7 kombinacija {cosine, BLAST, FoldseekTM} na sva tri
evidence tier-a (A/B/C). Da li je RRF-3 (sva tri) stvarno najbolja
kombinacija, ili neka manja podgrupa pobedjuje na nekom tier-u?

Racuna se iz istih matrica kao rank_fusion_cosine_blast_foldseek_1548.py
(cosine, BLAST, FoldseekTM), samo se sad kombinuju SVE 7 podgrupa signala
po upitu, ne samo puna trojka.

Izlaz:
    output/rrf_ablation_1548_summary.txt
"""

import pickle
import time
from itertools import combinations
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
SUMMARY_OUTPUT = OUTPUT_DIR / "rrf_ablation_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rrf_ablation_1548_per_query.csv"

RRF_K = 60

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
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
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"], "evidence_level": row["evidence_level"]})
print(f"Gold pairs: {len(gold_pairs)}")


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def rank_of(ranks, target_index):
    return int(ranks[target_index])


SIGNALS = ["cosine", "blast", "foldseek"]
SUBSETS = []
for r in range(1, 4):
    SUBSETS.extend(combinations(SIGNALS, r))

print(f"\nScoring all queries ({len(SUBSETS)} combinations per query)...")
start = time.time()
records = []

for qi, p in enumerate(gold_pairs):
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)
        rank_by_signal = {"cosine": cos_ranks, "blast": blast_ranks, "foldseek": fs_ranks}

        row = {"pair_id": p["pair_id"], "evidence_level": p["evidence_level"]}
        for subset in SUBSETS:
            combined_score = sum(1.0 / (RRF_K + rank_by_signal[s]) for s in subset)
            combined_ranks = ranks_from_scores(combined_score, qidx)
            row["+".join(subset)] = rank_of(combined_ranks, tidx)
        records.append(row)

    if (qi + 1) % 300 == 0 or (qi + 1) == len(gold_pairs):
        elapsed = time.time() - start
        print(f"  {qi+1}/{len(gold_pairs)} pairs ({elapsed/60:.1f} min elapsed)", flush=True)

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}")


# =====================================================
# AGGREGATE BY TIER
# =====================================================

is_confirmed_strong = df["evidence_level"].str.startswith(("Confirmed", "Strong evidence"), na=False)
is_suspected = df["evidence_level"].str.startswith("Suspected", na=False)

tiers = [
    ("A: Confirmed+Strong only", df[is_confirmed_strong]),
    ("B: A + Suspected", df[is_confirmed_strong | is_suspected]),
    ("C: full dataset", df),
]

subset_labels = ["+".join(s) for s in SUBSETS]
summary_lines = ["=" * 70, "RRF ablation: all 7 signal combinations, by evidence tier", "=" * 70, ""]

for tier_name, sub in tiers:
    summary_lines.append(f"--- {tier_name} (n={len(sub)} queries) ---")
    mrrs = {}
    for label in subset_labels:
        mrr = (1.0 / sub[label]).mean()
        mrrs[label] = mrr
    for label in sorted(mrrs, key=mrrs.get, reverse=True):
        marker = "  <-- RRF-3 (all three)" if label == "cosine+blast+foldseek" else ""
        summary_lines.append(f"  {label:25s} MRR={mrrs[label]:.4f}{marker}")
    best = max(mrrs, key=mrrs.get)
    summary_lines.append(f"  Best on this tier: {best} ({mrrs[best]:.4f})")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
