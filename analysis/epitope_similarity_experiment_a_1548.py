"""
Eksperiment A: da li slicnost STVARNIH (IEDB) epitope regiona nosi vise
signala od slicnosti CELOG proteina, na parovima gde OBA proteina imaju
eksperimentalno mapirane epitope - 1548 dataset.

Jeftin falsifikacioni test PRE ulaganja u BepiPred (predikciju epitopa za
proteine bez poznatih): ako epitope pristup ne pokaze signal ni ovde, gde
imamo prave podatke, nema smisla trositi vreme na predikciju.

Epitope maska po proteinu: unija svih (start, end) opsega iz
output/iedb_epitopes_1548.csv (IEDB pozicije), poravnata sa residue-level
ESM embeddinzima. VAZNO: IEDB pozicije su relativne na UniProt sekvencu, koja
se moze razlikovati od nase FASTA sekvence (npr. signal peptid) - proveravaju
se granice (end <= duzina embeddinga), preskace se protein ako ne prolazi.

Poredi na 101 par (202 upita, oba proteina imaju epitope podatke):
  - whole-protein cosine (postojeci baseline)
  - epitope-region top-K similarity (samo epitope pozicije)

Izlaz:
    output/epitope_similarity_experiment_a_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
IEDB_EPITOPES = Path("/home/lana/ALERGRAF/output/iedb_epitopes_1548.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "epitope_similarity_experiment_a_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "epitope_similarity_experiment_a_1548_per_query.csv"

TOP_K = 15

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    pooled_emb = pickle.load(f)
with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_emb = pickle.load(f)
metadata = pd.read_parquet(METADATA)
name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {aid: i for i, aid in enumerate(all_ids)}
pooled_matrix = np.array([pooled_emb[aid] for aid in all_ids], dtype=np.float64)
pooled_cosine_matrix = cosine_similarity(pooled_matrix)

# =====================================================
# BUILD EPITOPE MASKS (with bounds checking against embedding length)
# =====================================================

epi_df = pd.read_csv(IEDB_EPITOPES)
epitope_masks = {}
n_out_of_bounds = 0
for row in epi_df.itertuples(index=False):
    if row.n_positive_records == 0 or pd.isna(row.epitope_ranges) or not row.epitope_ranges:
        continue
    aid = row.allergen_id
    if aid not in residue_emb:
        continue
    L = residue_emb[aid].shape[0]
    mask = np.zeros(L, dtype=bool)
    valid = True
    for part in row.epitope_ranges.split(";"):
        s, e = part.split("-")
        s, e = int(s), int(e)
        if s < 1 or e > L or s > e:
            valid = False
            break
        mask[s - 1:e] = True  # IEDB positions are 1-indexed
    if not valid or mask.sum() == 0:
        n_out_of_bounds += 1
        continue
    epitope_masks[aid] = mask

print(f"Proteins with a usable epitope mask: {len(epitope_masks)} "
      f"({n_out_of_bounds} skipped due to position/length mismatch)")

# pre-normalize epitope-region residue matrices
epitope_residue_norm = {}
for aid, mask in epitope_masks.items():
    mat = residue_emb[aid].astype(np.float32)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    epitope_residue_norm[aid] = norm[mask]


def topk_score(mat_a, mat_b, k=TOP_K):
    sim = mat_a @ mat_b.T
    flat = sim.reshape(-1)
    k = min(k, flat.shape[0])
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


# =====================================================
# GOLD PAIRS -- both endpoints need a usable epitope mask
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
    if id1 == id2 or id1 not in epitope_masks or id2 not in epitope_masks:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"],
                        "family_1": row["family_1"], "family_2": row["family_2"]})
print(f"Gold pairs with BOTH endpoints having a usable epitope mask: {len(gold_pairs)}")


# =====================================================
# MAIN LOOP -- no training, direct evaluation against full candidate pool
# =====================================================

# candidate pool for retrieval ranking: all proteins that have a residue
# embedding (whole-protein cosine always available); epitope-similarity
# candidates are restricted to those WITH an epitope mask (fair, like all
# earlier "no-vote" scoping -- can't rank against epitope data we don't have)
epitope_pool_ids = sorted(epitope_masks.keys())
epitope_id_to_pos = {aid: i for i, aid in enumerate(epitope_pool_ids)}

print("\nScoring...")
start = time.time()
records = []

for p in gold_pairs:
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        # whole-protein cosine, ranked against FULL candidate pool
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]
        pooled_scores = pooled_cosine_matrix[qidx].copy()
        pooled_scores[qidx] = -np.inf
        pooled_rank = int(np.argsort(pooled_scores)[::-1].tolist().index(tidx)) + 1

        # epitope-region similarity, ranked ONLY against the epitope-mask pool
        q_epi = epitope_residue_norm[query_id]
        epi_scores = np.full(len(epitope_pool_ids), -np.inf)
        for cand_id in epitope_pool_ids:
            if cand_id == query_id:
                continue
            epi_scores[epitope_id_to_pos[cand_id]] = topk_score(q_epi, epitope_residue_norm[cand_id])
        epi_rank = int(np.argsort(epi_scores)[::-1].tolist().index(epitope_id_to_pos[target_id])) + 1

        # whole-protein cosine, ranked against the SAME restricted pool (fair comparison)
        pooled_scores_restricted = np.array([
            pooled_cosine_matrix[qidx, id_to_index[cid]] if cid != query_id else -np.inf
            for cid in epitope_pool_ids
        ])
        pooled_rank_restricted = int(np.argsort(pooled_scores_restricted)[::-1].tolist().index(
            epitope_id_to_pos[target_id])) + 1

        records.append({
            "pair_id": p["pair_id"], "family_1": p["family_1"], "family_2": p["family_2"],
            "pool_size": len(epitope_pool_ids),
            "pooled_rank_full": pooled_rank,
            "pooled_rank_restricted": pooled_rank_restricted,
            "epitope_rank": epi_rank,
        })

    if len(records) % 40 == 0:
        print(f"  {len(records)} queries done ({(time.time()-start)/60:.1f} min elapsed)", flush=True)

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)

pooled_full_mrr = (1.0 / df["pooled_rank_full"]).mean()
pooled_restricted_mrr = (1.0 / df["pooled_rank_restricted"]).mean()
epitope_mrr = (1.0 / df["epitope_rank"]).mean()

rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["epitope_rank"], weights=w) - np.average(1.0 / sub["pooled_rank_restricted"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)

summary_lines = [
    "=" * 70,
    f"Eksperiment A: epitope-region vs whole-protein similarity ({len(df)} upita, "
    f"{len(gold_pairs)} parova, {len(epitope_pool_ids)}-protein pool)",
    "=" * 70,
    "",
    f"whole-protein cosine, PUN pool (1534)         MRR = {pooled_full_mrr:.4f}  (referenca, nije direktno uporedivo)",
    f"whole-protein cosine, SAMO epitope-pool ({len(epitope_pool_ids)})  MRR = {pooled_restricted_mrr:.4f}  (posten baseline)",
    f"epitope-region top-K, SAMO epitope-pool        MRR = {epitope_mrr:.4f}",
    "",
    f"Delta (epitope - whole-protein, ISTI pool): {epitope_mrr - pooled_restricted_mrr:+.4f}",
    f"Bootstrap 95% CI: [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]",
    f"Fraction of bootstrap resamples favoring epitope similarity: {(deltas>0).mean():.3f}",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
