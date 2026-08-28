"""
PRAVA verzija BepiPred testa (posle prethodnog pilota koji je imao manu:
sirove cosine vrednosti na malom, familijski-homogenom skupu od 41 proteina
-- anisotropy efekat cini null baseline vestacki visokim tamo).

Ovde: ISTA metodologija kao analysis/residue_topk_nsltp_profilin_1548.py
(surface-residue test) -- rang u PUNOM candidate pool-u, top-K (K=15)
rezidue-par max-similarity, bootstrap CI po pair_id -- samo je "surface"
maska (AlphaFold SASA >=25%) zamenjena "epitope" maskom (BepiPred-3.0
predikcija, default prag 0.1512, isti prag koji alat sam preporucuje --
ne biramo proizvoljan prag naknadno).

BepiPred ne zavisi od strukture, pa pokriva CEO pool (1523 proteina), ne
samo 1016 koliko je surface-residue imao zbog nedostajucih AlphaFold
struktura -- sira, kompletnija verzija tog ranijeg (delimicno pozitivnog)
testa.

Ulaz (nakon sto se BepiPred pokrene na VM nad output/bepipred_full_1548.fasta
i donese nazad):
    output/bepipred_full_1548_raw_output.csv

Izlaz:
    output/bepipred_topk_nsltp_profilin_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BEPIPRED_RAW = Path("/home/lana/ALERGRAF/output/bepipred_full_1548_raw_output.csv")
EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "bepipred_topk_nsltp_profilin_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "bepipred_topk_nsltp_profilin_1548_per_query.csv"

TOP_K = 15
TARGET_FAMILIES = {"nsLTP", "Profilin"}
BEPIPRED_THRESHOLD = 0.1512  # BepiPred-3.0's own default classification threshold

if not BEPIPRED_RAW.exists():
    raise SystemExit(f"Nedostaje {BEPIPRED_RAW} -- pokreni BepiPred-3.0 na VM nad "
                      f"output/bepipred_full_1548.fasta i donesi raw_output.csv nazad.")

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    pooled_emb = pickle.load(f)
with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_emb = pickle.load(f)

bp = pd.read_csv(BEPIPRED_RAW)
id_col = [c for c in bp.columns if "acc" in c.lower() or c.lower() == "id"][0]
score_col = [c for c in bp.columns if "score" in c.lower() and "linear" not in c.lower()][0]
print(f"  koristim id_col={id_col!r}, score_col={score_col!r}")

epitope_scores = {}
for aid, group in bp.groupby(id_col):
    epitope_scores[aid] = group[score_col].to_numpy(dtype=np.float64)

metadata = pd.read_parquet(METADATA)
name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

pool_ids = sorted(aid for aid in epitope_scores if aid in pooled_emb and aid in residue_emb
                   and residue_emb[aid].shape[0] == len(epitope_scores[aid]))
print(f"Candidate/query pool (BepiPred + embeddings, matched duzina): {len(pool_ids)}")
pool_set = set(pool_ids)
id_to_pos = {aid: i for i, aid in enumerate(pool_ids)}

pooled_matrix = np.array([pooled_emb[aid] for aid in pool_ids], dtype=np.float64)
pooled_cosine_matrix = cosine_similarity(pooled_matrix)
del pooled_emb, pooled_matrix

residue_norm = {}
epitope_residue_norm = {}
n_no_epitope = 0
for aid in pool_ids:
    mat = residue_emb[aid].astype(np.float32)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    residue_norm[aid] = norm
    mask = epitope_scores[aid] >= BEPIPRED_THRESHOLD
    epi = norm[mask]
    if epi.shape[0] == 0:
        epi = norm
        n_no_epitope += 1
    epitope_residue_norm[aid] = epi
del residue_emb
print(f"Proteini bez ijedne rezidue iznad BepiPred praga (koriste sve rezidue): {n_no_epitope}")


def topk_score(mat_a, mat_b, k=TOP_K):
    sim = mat_a @ mat_b.T
    flat = sim.reshape(-1)
    k = min(k, flat.shape[0])
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


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
        q_epi = epitope_residue_norm[query_id]
        all_scores = np.full(len(pool_ids), -np.inf)
        epi_scores = np.full(len(pool_ids), -np.inf)
        for cand_id in pool_ids:
            if cand_id == query_id:
                continue
            cpos = id_to_pos[cand_id]
            all_scores[cpos] = topk_score(q_all, residue_norm[cand_id])
            epi_scores[cpos] = topk_score(q_epi, epitope_residue_norm[cand_id])

        all_rank = int(np.argsort(all_scores)[::-1].tolist().index(t_pos)) + 1
        epi_rank = int(np.argsort(epi_scores)[::-1].tolist().index(t_pos)) + 1

        records.append({
            "pair_id": p["pair_id"], "query_id": query_id, "target_id": target_id,
            "pooled_rank": pooled_rank, "all_res_rank": all_rank, "epitope_res_rank": epi_rank,
        })

    if len(records) % 100 == 0:
        print(f"  {len(records)} queries done ({(time.time()-start)/60:.1f} min elapsed)", flush=True)

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)

pooled_mrr = (1.0 / df["pooled_rank"]).mean()
all_res_mrr = (1.0 / df["all_res_rank"]).mean()
epi_res_mrr = (1.0 / df["epitope_res_rank"]).mean()

rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["epitope_res_rank"], weights=w) - np.average(1.0 / sub["all_res_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
sig = (ci_lo > 0) or (ci_hi < 0)

summary_lines = [
    "=" * 70,
    f"BepiPred epitope-residue top-K similarity on nsLTP/Profilin subset ({len(df)} queries)",
    "=" * 70,
    "",
    f"pooled cosine MRR    : {pooled_mrr:.4f}",
    f"all-residue topK MRR : {all_res_mrr:.4f}",
    f"epitope-only topK MRR: {epi_res_mrr:.4f}",
    "",
    f"Delta (epitope - all-residues): {epi_res_mrr - all_res_mrr:+.4f}",
    f"Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  -- {'ZNACAJNO' if sig else 'nije znacajno'}",
    f"Fraction of bootstrap resamples favoring epitope-only: {(deltas>0).mean():.3f}",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
