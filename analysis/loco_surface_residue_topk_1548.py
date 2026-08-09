"""
LOCO: da li ogranicavanje residue-level top-K slicnosti na POVRSINSKE
(solvent-exposed) rezidue pomaze u odnosu na poredjenje SVIH rezidua
("Eksperiment 2", koji nije pomogao - MRR ~0.105-0.106, identicno pooling-u)
- 1548 dataset.

Hipoteza: cross-reactivity zavisi od IgE-dostupnih (povrsinskih) epitopa, ne
od zakopanih rezidua. Eksperiment 2 je mozda propao jer je porede indiskriminatno
SVE rezidue (ukljucujuci biolski irelevantne zakopane), razvodnjavajuci signal.

Candidate/query pool je ogranicen na 1016 proteina koji imaju i AlphaFold
strukturu i validnu (duzinski uskladjenu) povrsinsku masku
(output/surface_residue_masks_1548.pkl) - potrebno za posten A/B poredjenje
(isti kandidati za sve tri metode ispod).

Tri metode, ISTI upiti/kandidati/foldovi:
  - mean-pool cosine (baseline, isti kao svuda u projektu)
  - top-K residue similarity, SVE rezidue (= Eksperiment 2 protokol)
  - top-K residue similarity, SAMO povrsinske rezidue (nova hipoteza)

K=15 (Eksperiment 2 je pokazao da K=5/15/30 daju identican rezultat, nema
potrebe da se sve tri ponovo testiraju).

LOCO (leave-one-component-out) na komponentama RESTRIKOVANIM na ovaj pool -
nema treninga modela ovde (cisto poredjenje slicnosti), pa je LOCO jeftin
(nema PU bagging troska koji je bio problem ranije).

Izlaz:
    output/loco_surface_residue_topk_1548_per_fold.csv
    output/loco_surface_residue_topk_1548_summary.txt
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
PER_FOLD_OUTPUT = OUTPUT_DIR / "loco_surface_residue_topk_1548_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "loco_surface_residue_topk_1548_summary.txt"

TOP_K = 15
RETRIEVAL_TOP_K = [1, 5, 10, 20]


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

# candidate pool = proteins with a valid surface mask (needed for fair A/B; also has pooled+residue emb)
pool_ids = sorted(aid for aid in surface_masks if aid in pooled_emb and aid in residue_emb)
print(f"Candidate/query pool (valid surface mask + embeddings): {len(pool_ids)}")

pool_set = set(pool_ids)
id_to_pos = {aid: i for i, aid in enumerate(pool_ids)}

pooled_matrix = np.array([pooled_emb[aid] for aid in pool_ids], dtype=np.float64)
pooled_cosine_matrix = cosine_similarity(pooled_matrix)
del pooled_emb, pooled_matrix  # free ~15MB, not needed after this point

# pre-normalize residue matrices once (both full and surface-only views)
residue_norm = {}
surface_residue_norm = {}
for aid in pool_ids:
    mat = residue_emb[aid].astype(np.float32)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    residue_norm[aid] = norm
    mask = surface_masks[aid]
    surf = norm[mask]
    if surf.shape[0] == 0:  # degenerate: no residue passed threshold, fall back to all
        surf = norm
    surface_residue_norm[aid] = surf

del residue_emb, surface_masks  # free ~1.9GB, not needed after this point (real culprit of the last run's slowdown)

lengths = np.array([residue_norm[aid].shape[0] for aid in pool_ids])
surf_lengths = np.array([surface_residue_norm[aid].shape[0] for aid in pool_ids])
print(f"Residue lengths -- mean={lengths.mean():.0f}, median={np.median(lengths):.0f}, max={lengths.max()}")
print(f"Surface residue lengths -- mean={surf_lengths.mean():.0f}, median={np.median(surf_lengths):.0f}, max={surf_lengths.max()}")


def topk_score(mat_a, mat_b, k=TOP_K):
    sim = mat_a @ mat_b.T
    flat = sim.reshape(-1)
    k = min(k, flat.shape[0])
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


# =====================================================
# GOLD PAIRS + LOCO FOLDS (restricted to pool)
# =====================================================

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in pool_set or id2 not in pool_set:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})

print(f"Gold pairs usable within pool: {len(gold_pairs)}")

parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for p in gold_pairs:
    union(p["id_1"], p["id_2"])

components = {}
for pid in parent:
    components.setdefault(find(pid), set()).add(pid)
component_list = list(components.values())
K_FOLDS = len(component_list)
print(f"Connected components within pool (= LOCO folds): {K_FOLDS}")


# =====================================================
# MAIN LOCO LOOP (no training -- pure similarity scoring)
# =====================================================

pooled_rr, all_res_rr, surf_res_rr = [], [], []
start = time.time()
last_print = start
n_queries = 0
n_queries_expected = len(gold_pairs) * 2
print(f"Expected total queries: {n_queries_expected}", flush=True)

for fold_idx, test_component in enumerate(component_list):
    test_pairs = [p for p in gold_pairs if p["id_1"] in test_component and p["id_2"] in test_component]

    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            q_pos = id_to_pos[query_id]
            t_pos = id_to_pos[target_id]

            pooled_scores = pooled_cosine_matrix[q_pos].copy()
            pooled_scores[q_pos] = -np.inf
            pooled_rank = int(np.argsort(pooled_scores)[::-1].tolist().index(t_pos)) + 1
            pooled_rr.append(1.0 / pooled_rank)

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
            all_res_rr.append(1.0 / all_rank)
            surf_res_rr.append(1.0 / surf_rank)

            n_queries += 1
            now = time.time()
            if n_queries % 5 == 0 or (now - last_print) >= 20:
                elapsed = now - start
                rate = n_queries / elapsed if elapsed > 0 else 0
                remaining = (n_queries_expected - n_queries) / rate if rate > 0 else float("inf")
                print(f"  query {n_queries}/~{n_queries_expected} (fold {fold_idx+1}/{K_FOLDS}, "
                      f"q_len={q_all.shape[0]}) -- {rate:.2f} q/s, {elapsed/60:.1f} min elapsed, "
                      f"~{remaining/60:.1f} min remaining", flush=True)
                last_print = now

total_elapsed = time.time() - start
print(f"\nDone: {n_queries} queries in {total_elapsed/60:.1f} min")

pooled_mrr = float(np.mean(pooled_rr))
all_res_mrr = float(np.mean(all_res_rr))
surf_res_mrr = float(np.mean(surf_res_rr))

summary_lines = [
    "=" * 70,
    f"LOCO surface-only vs all-residue top-K similarity ({K_FOLDS} folds, 1548 dataset)",
    "=" * 70,
    f"Pool size: {len(pool_ids)} proteins, K={TOP_K}",
    f"Queries: {n_queries}",
    "",
    f"mean-pool cosine MRR      : {pooled_mrr:.4f}",
    f"top-K, ALL residues MRR   : {all_res_mrr:.4f}   (= Eksperiment 2 protokol)",
    f"top-K, SURFACE-only MRR   : {surf_res_mrr:.4f}   (nova hipoteza)",
    "",
    f"Delta (surface - all-residues): {surf_res_mrr - all_res_mrr:+.4f}",
    f"Delta (surface - pooled cosine): {surf_res_mrr - pooled_mrr:+.4f}",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

per_fold_df = pd.DataFrame({
    "pooled_rr": pooled_rr, "all_res_rr": all_res_rr, "surf_res_rr": surf_res_rr,
})
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {PER_FOLD_OUTPUT}, {SUMMARY_OUTPUT}")
