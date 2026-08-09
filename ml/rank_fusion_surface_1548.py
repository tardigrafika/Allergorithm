"""
Dodaje surface-residue top-K similarity kao PETI nezavisan glas u RRF fuziju
(cosine + BLAST + FoldseekTM + kmer + surface-residue-topK) - 1548 dataset.

Motivacija: ciljan test (residue_topk_nsltp_profilin_1548.py) je pokazao da
povrsinsko filtriranje ZNACAJNO pomaze BAS na nsLTP/Profilin parovima (delta
+0.0034, 95% CI iskljucuje nulu) - familijama koje su dominirale worst-50
error analizu. RRF princip nagradjuje bas ovakve "slabe u proseku, ali jake
na tacno onim mestima gde ostali signali kolabiraju" glasove.

Coverage: surface-residue signal je racunljiv samo za 1016/1534 proteina
(imaju AlphaFold strukturu + uskladjenu duzinu embeddinga). Za upite/kandidate
bez validnih podataka, ovaj glas jednostavno NE UCESTVUJE (doprinos 0 u RRF
sumi) - ne kaznjava ih, samo ne glasa za njih (isti princip kao FoldseekTM
0.0 fallback ranije).

Poredi RRF-3 (kontrola, bez ovog glasa) sa RRF-3+surface na ISTIM upitima.

Izlaz:
    output/rank_fusion_surface_1548_summary.txt
    output/rank_fusion_surface_1548_per_query.csv
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
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
SURFACE_MASKS = Path("/home/lana/ALERGRAF/output/surface_residue_masks_1548.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_fusion_surface_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rank_fusion_surface_1548_per_query.csv"

RRF_K = 60
TOP_K = [1, 5, 10, 20]
RESIDUE_TOPK = 15


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

missing_from_blast = [aid for aid in all_ids if aid not in blast_id_to_index]
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

print("Loading residue embeddings + surface masks...")
with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_emb = pickle.load(f)
with open(SURFACE_MASKS, "rb") as f:
    surface_masks = pickle.load(f)

surface_pool_ids = sorted(aid for aid in surface_masks if aid in embeddings_dict and aid in residue_emb)
surface_pool_set = set(surface_pool_ids)
print(f"Surface-residue signal coverage: {len(surface_pool_ids)}/{n_candidates} proteins")

surface_residue_norm = {}
for aid in surface_pool_ids:
    mat = residue_emb[aid].astype(np.float32)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    mask = surface_masks[aid]
    surf = norm[mask]
    if surf.shape[0] == 0:
        surf = norm
    surface_residue_norm[aid] = surf
del residue_emb, surface_masks

surface_pool_indices = np.array([id_to_index[aid] for aid in surface_pool_ids])
surface_pos_within_pool = {aid: i for i, aid in enumerate(surface_pool_ids)}


def topk_score(mat_a, mat_b, k=RESIDUE_TOPK):
    sim = mat_a @ mat_b.T
    flat = sim.reshape(-1)
    k = min(k, flat.shape[0])
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


TARGET_FAMILIES = {"nsLTP", "Profilin"}  # familije gde je surface-residue glas VALIDIRAN (residue_topk_nsltp_profilin_1548.py)

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    fam1 = str(row.get("family_1", "")).strip()
    fam2 = str(row.get("family_2", "")).strip()
    gold_pairs.append({
        "id_1": id1, "id_2": id2, "pair_id": row["pair_id"],
        "fam_1": fam1, "fam_2": fam2,
    })
print(f"Gold pairs: {len(gold_pairs)}")
n_target_family_pairs = sum(1 for p in gold_pairs if p["fam_1"] in TARGET_FAMILIES or p["fam_2"] in TARGET_FAMILIES)
print(f"Pairs involving a target family ({TARGET_FAMILIES}): {n_target_family_pairs}")


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
last_print = start
records = []
n_done = 0

for p in gold_pairs:
    directions = [
        (p["id_1"], p["id_2"], p["fam_1"]),
        (p["id_2"], p["id_1"], p["fam_2"]),
    ]
    for query_id, target_id, query_family in directions:
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)

        rrf3_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)

        # surface-residue voter: ONLY counted for queries from a validated target family
        # (nsLTP/Profilin) -- contributes 0 otherwise, even if data is technically available
        gate_open = query_family in TARGET_FAMILIES and query_id in surface_pool_set
        surface_contrib = np.zeros(n_candidates, dtype=np.float64)
        if gate_open:
            q_surf = surface_residue_norm[query_id]
            local_scores = np.array([
                topk_score(q_surf, surface_residue_norm[cid]) if cid != query_id else -np.inf
                for cid in surface_pool_ids
            ])
            local_ranks = ranks_from_scores(local_scores, surface_pos_within_pool[query_id])
            surface_contrib[surface_pool_indices] = 1.0 / (RRF_K + local_ranks)
            surface_contrib[qidx] = 0.0

        rrf3s_score = rrf3_score + surface_contrib

        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)
        rrf3s_ranks = ranks_from_scores(rrf3s_score, qidx)

        records.append({
            "pair_id": p["pair_id"],
            "query_family": query_family,
            "gate_open": gate_open,
            "cosine_rank": rank_of(cos_ranks, tidx),
            "blast_rank": rank_of(blast_ranks, tidx),
            "foldseektm_rank": rank_of(fs_ranks, tidx),
            "rrf3_rank": rank_of(rrf3_ranks, tidx),
            "rrf3_surface_rank": rank_of(rrf3s_ranks, tidx),
        })
        n_done += 1
        now = time.time()
        if n_done % 50 == 0 or (now - last_print) >= 30:
            elapsed = now - start
            rate = n_done / elapsed
            remaining = (len(gold_pairs) * 2 - n_done) / rate if rate > 0 else float("inf")
            print(f"  {n_done}/{len(gold_pairs)*2} queries ({rate:.2f} q/s, "
                  f"{elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)", flush=True)
            last_print = now

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}")


# =====================================================
# AGGREGATE
# =====================================================

rrf3_mrr = (1.0 / df["rrf3_rank"]).mean()
rrf3s_mrr = (1.0 / df["rrf3_surface_rank"]).mean()

summary_lines = [
    "=" * 70,
    f"RRF: does surface-residue top-K as a 5th voter help? ({len(df)} queries, 1548 dataset)",
    "=" * 70,
    f"Surface-residue voter coverage: {df['gate_open'].sum()}/{len(df)} queries have data "
    f"(no-vote/0-contribution fallback for the rest)",
    "",
    f"RRF-3 (cosine+BLAST+FoldseekTM)          MRR = {rrf3_mrr:.4f}",
    f"RRF-3 + surface-residue-topK              MRR = {rrf3s_mrr:.4f}",
    f"Delta: {rrf3s_mrr - rrf3_mrr:+.4f}",
    "",
]

for k in TOP_K:
    h3 = (df["rrf3_rank"] <= k).mean()
    h3s = (df["rrf3_surface_rank"] <= k).mean()
    summary_lines.append(f"Hits@{k}: RRF-3={h3:.4f}  RRF-3+surface={h3s:.4f}")

# bootstrap by pair_id
rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["rrf3_surface_rank"], weights=w) - np.average(1.0 / sub["rrf3_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
summary_lines.append("")
summary_lines.append(f"Bootstrap 95% CI (RRF-3+surface - RRF-3): [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]")
summary_lines.append(f"Fraction of bootstrap resamples favoring RRF-3+surface: {(deltas>0).mean():.3f}")

# same, but restricted to queries that actually had surface data (where the voter could act at all)
sub_has = df[df["gate_open"]]
pair_ids_has = sub_has["pair_id"].unique()
deltas_has = []
for _ in range(2000):
    sampled = rng.choice(pair_ids_has, size=len(pair_ids_has), replace=True)
    counts = pd.Series(sampled).value_counts()
    s2 = sub_has.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = s2["w"].to_numpy()
    d = np.average(1.0 / s2["rrf3_surface_rank"], weights=w) - np.average(1.0 / s2["rrf3_rank"], weights=w)
    deltas_has.append(d)
deltas_has = np.array(deltas_has)
summary_lines.append("")
summary_lines.append(f"Restricted to queries WITH surface data ({len(sub_has)} queries):")
summary_lines.append(f"  Bootstrap 95% CI: [{np.percentile(deltas_has,2.5):+.4f}, {np.percentile(deltas_has,97.5):+.4f}]")
summary_lines.append(f"  Fraction favoring RRF-3+surface: {(deltas_has>0).mean():.3f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
