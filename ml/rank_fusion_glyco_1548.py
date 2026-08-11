"""
Dodaje N-glikozilacioni profil kao NOVI nezavisan glas u RRF fuziju
(cosine + BLAST + FoldseekTM + glyco) - 1548 dataset.

Motivacija: glycan-posredovana cross-reaktivnost je dokumentovan fenomen u
alergologiji (glikani mogu da sakriju ili stvore epitope). N-vezana
glikozilacija se detektuje cisto sekvencijalno - sequon N-X-[S/T], X != Pro
- nema potrebe za ML/eksternim alatom (NetNGlyc i slicni su neuronski
prediktori, ali sam sequon je poznato pravilo, ne treba nam predikcija).

Feature: gustina sequon-a po proteinu (broj/duzina*100). Slicnost dva
proteina = -|razlika gustina| (blize gustine = veca slicnost). Grubo (ne
uzima u obzir POZICIJU glikozilacije relativno na epitope), ali jednostavno,
interpretabilno, i ne zahteva nikakve nove podatke (FASTA sekvence vec imamo).

Izlaz:
    output/rank_fusion_glyco_1548_summary.txt
    output/rank_fusion_glyco_1548_per_query.csv
"""

import pickle
import re
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
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_fusion_glyco_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rank_fusion_glyco_1548_per_query.csv"

RRF_K = 60
TOP_K = [1, 5, 10, 20]

NGLYC_SEQUON = re.compile(r"N[^P][ST]")


def nglyc_density(seq):
    seq = str(seq).upper()
    if len(seq) == 0:
        return 0.0
    n_sites = len(NGLYC_SEQUON.findall(seq))
    return 100.0 * n_sites / len(seq)


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

clean = pd.read_csv(CLEAN_ALLERGENS)
clean = clean[clean["fasta_sequence"].notna() & (clean["fasta_sequence"] != "")]
id_to_seq = dict(zip(clean["allergen_id"], clean["fasta_sequence"]))

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

print("Computing N-glycosylation sequon densities...")
densities = np.array([nglyc_density(id_to_seq.get(aid, "")) for aid in all_ids])
n_missing_seq = sum(1 for aid in all_ids if aid not in id_to_seq)
print(f"Proteins missing a FASTA sequence: {n_missing_seq}/{n_candidates}")
print(f"N-glyc sequon density -- mean={densities.mean():.3f}, max={densities.max():.3f}, "
      f"n_zero_sites={int((densities == 0).sum())}")

glyco_matrix = -np.abs(densities[:, None] - densities[None, :])  # higher (closer to 0) = more similar

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

        glyco_ranks = ranks_from_scores(glyco_matrix[qidx], qidx)
        glyco_contrib = 1.0 / (RRF_K + glyco_ranks)

        rrf_glyco_score = rrf3_score + glyco_contrib

        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)
        rrf_glyco_ranks = ranks_from_scores(rrf_glyco_score, qidx)
        glyco_only_ranks = ranks_from_scores(glyco_matrix[qidx], qidx)

        records.append({
            "pair_id": p["pair_id"],
            "glyco_only_rank": rank_of(glyco_only_ranks, tidx),
            "cosine_rank": rank_of(cos_ranks, tidx),
            "blast_rank": rank_of(blast_ranks, tidx),
            "foldseektm_rank": rank_of(fs_ranks, tidx),
            "rrf3_rank": rank_of(rrf3_ranks, tidx),
            "rrf_glyco_rank": rank_of(rrf_glyco_ranks, tidx),
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

glyco_only_mrr = (1.0 / df["glyco_only_rank"]).mean()
rrf3_mrr = (1.0 / df["rrf3_rank"]).mean()
rrf_glyco_mrr = (1.0 / df["rrf_glyco_rank"]).mean()

summary_lines = [
    "=" * 70,
    f"RRF: does N-glycosylation density help as a voter? ({len(df)} queries, 1548 dataset)",
    "=" * 70,
    "",
    f"Glyco density similarity ALONE (individual signal) MRR = {glyco_only_mrr:.4f}",
    "",
    f"RRF-3 (cosine+BLAST+FoldseekTM) MRR = {rrf3_mrr:.4f}",
    f"RRF-3 + glyco density           MRR = {rrf_glyco_mrr:.4f}",
    f"Delta: {rrf_glyco_mrr - rrf3_mrr:+.4f}",
    "",
]
for k in TOP_K:
    h3 = (df["rrf3_rank"] <= k).mean()
    hg = (df["rrf_glyco_rank"] <= k).mean()
    summary_lines.append(f"Hits@{k}: RRF-3={h3:.4f}  RRF-3+glyco={hg:.4f}")

rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["rrf_glyco_rank"], weights=w) - np.average(1.0 / sub["rrf3_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
summary_lines.append("")
summary_lines.append(f"Bootstrap 95% CI (RRF-3+glyco - RRF-3): [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]")
summary_lines.append(f"Fraction of bootstrap resamples favoring RRF-3+glyco: {(deltas>0).mean():.3f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
