"""
RRF_K osetljivost: cela sesija koristi K=60 (standardna IR konstanta iz
Cormack et al. 2009) bez provere da li neka druga vrednost bolje odgovara
BAS ovom datasetu. Brz, jeftin test na postojecim podacima.

Metod: mreza K vrednosti na PUNOM datasetu (isto kao rank_fusion_cosine_
blast_foldseek_1548.py -- RRF nema naucene parametre, pa se uobicajeno
racuna na celom skupu). Da bi izbor K bio posten (ne overfit na isti skup
na kom se meri), dodatno se radi split-half provera: da li K koji je
najbolji na PRVOJ polovini parova ostaje dobar i na DRUGOJ.

Izlaz:
    output/rrf_k_sensitivity_1548_summary.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/rrf_k_sensitivity_1548_summary.txt")

K_GRID = [5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
SEED = 42

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
negative_mask = gold_raw["evidence_level"].str.contains("negative|Contested|Risky|NO cross", case=False, na=False)
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
valid_idx = np.where(perm >= 0)[0]
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

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


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


print("Precomputing per-query cosine/blast/foldseek ranks (K-independent)...")
records = []
for p in gold_pairs:
    for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx, tidx = id_to_index[qid], id_to_index[tid]
        cr = ranks_from_scores(cosine_matrix[qidx], qidx)
        br = ranks_from_scores(blast_matrix[qidx], qidx)
        fr = ranks_from_scores(foldseek_matrix[qidx], qidx)
        records.append({"pair_id": p["pair_id"], "cos_rank": cr[tidx], "blast_rank": br[tidx],
                         "fs_rank": fr[tidx],
                         "_cr": cr, "_br": br, "_fr": fr, "qidx": qidx, "tidx": tidx})

# store the full rank arrays temporarily for RRF recombination per K (kept in memory, n=3074 queries x 1534 -- fine)
df_meta = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_") and k not in ("qidx", "tidx")} for r in records])


def mrr_for_k(k):
    rr = []
    for r in records:
        combined = 1.0 / (k + r["_cr"]) + 1.0 / (k + r["_br"]) + 1.0 / (k + r["_fr"])
        combined[r["qidx"]] = -np.inf
        order = np.argsort(combined)[::-1]
        ranks = np.empty(len(combined), dtype=np.int64)
        ranks[order] = np.arange(1, len(combined) + 1)
        rr.append(1.0 / ranks[r["tidx"]])
    return float(np.mean(rr))


print(f"\nEvaluacija preko K mreze: {K_GRID}")
results = {}
for k in K_GRID:
    mrr = mrr_for_k(k)
    results[k] = mrr
    print(f"  K={k:4d}  MRR={mrr:.4f}")

best_k = max(results, key=results.get)

# split-half provera: da li najbolji K na 1. polovini parova ostaje dobar na 2.
rng = np.random.default_rng(SEED)
pair_ids = list({r["pair_id"] for r in records})
rng.shuffle(pair_ids)
half = len(pair_ids) // 2
half_a, half_b = set(pair_ids[:half]), set(pair_ids[half:])
records_a = [r for r in records if r["pair_id"] in half_a]
records_b = [r for r in records if r["pair_id"] in half_b]


def mrr_for_k_subset(k, subset):
    rr = []
    for r in subset:
        combined = 1.0 / (k + r["_cr"]) + 1.0 / (k + r["_br"]) + 1.0 / (k + r["_fr"])
        combined[r["qidx"]] = -np.inf
        order = np.argsort(combined)[::-1]
        ranks = np.empty(len(combined), dtype=np.int64)
        ranks[order] = np.arange(1, len(combined) + 1)
        rr.append(1.0 / ranks[r["tidx"]])
    return float(np.mean(rr))


results_a = {k: mrr_for_k_subset(k, records_a) for k in K_GRID}
results_b = {k: mrr_for_k_subset(k, records_b) for k in K_GRID}
best_k_a = max(results_a, key=results_a.get)

summary_lines = ["=" * 70, "RRF_K osetljivost (K=60 je bio proizvoljna standardna konstanta)", "=" * 70, "",
                  "Puni dataset:"]
for k in K_GRID:
    marker = "  <-- K=60 (ustanovljena konstanta)" if k == 60 else ("  <-- najbolji" if k == best_k else "")
    summary_lines.append(f"  K={k:4d}  MRR={results[k]:.4f}{marker}")
summary_lines.append("")
summary_lines.append(f"Najbolji K na punom datasetu: {best_k} (MRR={results[best_k]:.4f}), "
                      f"K=60 MRR={results[60]:.4f}, delta={results[best_k]-results[60]:+.4f}")
summary_lines.append("")
summary_lines.append("Split-half provera (da li se najbolji K prenosi na drugu polovinu):")
summary_lines.append(f"  Najbolji K na polovini A: {best_k_a} (MRR_A={results_a[best_k_a]:.4f})")
summary_lines.append(f"  Taj isti K na polovini B: MRR_B={results_b[best_k_a]:.4f}  "
                      f"(K=60 na B: MRR_B={results_b[60]:.4f})")
transfers = results_b[best_k_a] > results_b[60]
summary_lines.append(f"  Prenosi se na B bolje od K=60? {'DA' if transfers else 'NE -- verovatno sum, ostati na K=60'}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
