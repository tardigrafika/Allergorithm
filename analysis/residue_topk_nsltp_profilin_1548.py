"""
Da li residue-level top-K slicnost (sve rezidue vs samo povrsinske) pomaze
SPECIFICNO na nsLTP/Profilin parovima - familijama koje su dominirale
worst-50 error analizu (70% + 20% najgorih RRF upita).

Ranije testirano U PROSEKU preko celog dataset-a (analysis/loco_surface_residue_topk_1548.py)
i nije pomoglo (delta +0.001, u sumu). Hipoteza ovde: taj efekat je mogao
biti razvodnjen ostatkom dataset-a gde globalna slicnost vec radi dobro -
nsLTP/Profilin su POZNATI panalergeni (deljen fold/funkcija, NISKA
sekvencijalna konzervacija) gde bi lokalni/motiv signal trebalo da ima
najvise sanse da pomogne, ako uopste igde pomaze.

Nema treninga, nema LOCO fold-ova potrebno (cisto poredjenje slicnosti,
isto kao rank fusion skripta) - direktna evaluacija na svim nsLTP/Profilin
parovima.

Izlaz:
    output/residue_topk_nsltp_profilin_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
SURFACE_MASKS = Path("/home/lana/ALERGRAF/output/surface_residue_masks_1548.pkl")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "residue_topk_nsltp_profilin_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "residue_topk_nsltp_profilin_1548_per_query.csv"

TOP_K = 15
TARGET_FAMILIES = {"nsLTP", "Profilin"}


# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    pooled_emb = pickle.load(f)
with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_emb = pickle.load(f)
with open(SURFACE_MASKS, "rb") as f:
    surface_masks = pickle.load(f)

metadata = pd.read_parquet(METADATA)
name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

pool_ids = sorted(aid for aid in surface_masks if aid in pooled_emb and aid in residue_emb)
print(f"Candidate/query pool (valid surface mask + embeddings): {len(pool_ids)}")
pool_set = set(pool_ids)
id_to_pos = {aid: i for i, aid in enumerate(pool_ids)}

pooled_matrix = np.array([pooled_emb[aid] for aid in pool_ids], dtype=np.float64)
pooled_cosine_matrix = cosine_similarity(pooled_matrix)
del pooled_emb, pooled_matrix

residue_norm = {}
surface_residue_norm = {}
for aid in pool_ids:
    mat = residue_emb[aid].astype(np.float32)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    residue_norm[aid] = norm
    mask = surface_masks[aid]
    surf = norm[mask]
    if surf.shape[0] == 0:
        surf = norm
    surface_residue_norm[aid] = surf
del residue_emb, surface_masks


def topk_score(mat_a, mat_b, k=TOP_K):
    sim = mat_a @ mat_b.T
    flat = sim.reshape(-1)
    k = min(k, flat.shape[0])
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


# =====================================================
# GOLD PAIRS -- filtered to nsLTP/Profilin-involving, both in pool
# =====================================================

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()
fam_mask = gold["family_1"].isin(TARGET_FAMILIES) | gold["family_2"].isin(TARGET_FAMILIES)
gold = gold.loc[fam_mask].copy()

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in pool_set or id2 not in pool_set:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})
print(f"nsLTP/Profilin gold pairs usable within pool: {len(gold_pairs)}")


# =====================================================
# MAIN LOOP -- no training, direct evaluation
# =====================================================

print("\nScoring nsLTP/Profilin queries...")
start = time.time()
records = []

for p in gold_pairs:
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        q_pos = id_to_pos[query_id]
        t_pos = id_to_pos[target_id]

        pooled_scores = pooled_cosine_matrix[q_pos].copy()
        pooled_scores[q_pos] = -np.inf
        pooled_rank = int(np.argsort(pooled_scores)[::-1].tolist().index(t_pos)) + 1

        q_all = residue_norm[query_id]
        q_surf = surface_residue_norm[query_id]
        all_scores = np.full(len(pool_ids), -np.inf)
        surf_scores = np.full(len(pool_ids), -np.inf)
        for cand_id in pool_ids:
            if cand_id == query_id:
                continue
            cpos = id_to_pos[cand_id]
            all_scores[cpos] = topk_score(q_all, residue_norm[cand_id])
            surf_scores[cpos] = topk_score(q_surf, surface_residue_norm[cand_id])

        all_rank = int(np.argsort(all_scores)[::-1].tolist().index(t_pos)) + 1
        surf_rank = int(np.argsort(surf_scores)[::-1].tolist().index(t_pos)) + 1

        records.append({
            "pair_id": p["pair_id"], "query_id": query_id, "target_id": target_id,
            "pooled_rank": pooled_rank, "all_res_rank": all_rank, "surf_res_rank": surf_rank,
        })

    if len(records) % 100 == 0:
        print(f"  {len(records)} queries done ({(time.time()-start)/60:.1f} min elapsed)", flush=True)

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)

pooled_mrr = (1.0 / df["pooled_rank"]).mean()
all_res_mrr = (1.0 / df["all_res_rank"]).mean()
surf_res_mrr = (1.0 / df["surf_res_rank"]).mean()

# bootstrap by pair_id
rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["surf_res_rank"], weights=w) - np.average(1.0 / sub["all_res_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)

summary_lines = [
    "=" * 70,
    f"Residue-level top-K similarity on nsLTP/Profilin subset only ({len(df)} queries)",
    "=" * 70,
    "",
    f"pooled cosine MRR   : {pooled_mrr:.4f}",
    f"all-residue topK MRR: {all_res_mrr:.4f}",
    f"surface-only topK MRR: {surf_res_mrr:.4f}",
    "",
    f"Delta (surface - all-residues): {surf_res_mrr - all_res_mrr:+.4f}",
    f"Bootstrap 95% CI: [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]",
    f"Fraction of bootstrap resamples favoring surface-only: {(deltas>0).mean():.3f}",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
