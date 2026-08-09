"""
Rank fusion (RRF): dodaje k-mer (tripeptide) slicnost kao CETVRTI nezavisan
glas uz cosine + BLAST + Foldseek TM-score - 1548 dataset.

Zasto ovo, a ne kao RF feature (probano ranije, nije pomoglo preko BLAST-a):
RRF princip je vec dokazan da radi kad se kombinuju NEZAVISNI signali bez
ucenja (prethodni test: RRF-3 pobedjuje cosine, 95% CI iskljucuje nulu).
K-mer je jos jedan ESM-nezavisan, BLAST-nezavisan signal (tripeptide
frekvencijski profil, ne poravnanje) - vredi probati kao dodatni glas u
istom, vec validiranom mehanizmu.

Racuna RRF-3 (cosine+BLAST+FoldseekTM, isto kao ranije - kontrola) i RRF-4
(+ kmer) na ISTIM upitima, da vidimo da li dodatni glas pomera MRR.

Izlaz:
    output/rank_fusion_kmer_1548_summary.txt
    output/rank_fusion_kmer_1548_per_query.csv
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
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_fusion_kmer_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rank_fusion_kmer_1548_per_query.csv"

RRF_K = 60
TOP_K = [1, 5, 10, 20]
KMER_K = 3
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


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

# BLAST matrix re-aligned to all_ids ordering
missing_from_blast = [aid for aid in all_ids if aid not in blast_id_to_index]
if missing_from_blast:
    print(f"WARNING: {len(missing_from_blast)} proteins missing from BLAST matrix -- will get score 0.0")
perm = np.array([blast_id_to_index.get(aid, -1) for aid in all_ids])
valid = perm >= 0
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

# dense Foldseek TM-score matrix aligned to all_ids (0.0 fallback)
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

# k-mer (tripeptide) frequency vectors + cosine similarity
print(f"Building k-mer (k={KMER_K}) similarity matrix...")
kmer_vocab = {}
for a in AMINO_ACIDS:
    for b in AMINO_ACIDS:
        for c in AMINO_ACIDS:
            kmer_vocab[a + b + c] = len(kmer_vocab)


def kmer_frequency_vector(sequence):
    vec = np.zeros(len(kmer_vocab), dtype=np.float32)
    seq = sequence.upper()
    n_kmers = 0
    for i in range(len(seq) - KMER_K + 1):
        idx = kmer_vocab.get(seq[i:i + KMER_K])
        if idx is not None:
            vec[idx] += 1.0
            n_kmers += 1
    if n_kmers > 0:
        vec /= n_kmers
    return vec


kmer_matrix_raw = np.zeros((n_candidates, len(kmer_vocab)), dtype=np.float32)
missing_seq = 0
for i, aid in enumerate(all_ids):
    seq = id_to_seq.get(aid, "")
    if not seq:
        missing_seq += 1
    kmer_matrix_raw[i] = kmer_frequency_vector(seq)
if missing_seq:
    print(f"WARNING: {missing_seq} proteins missing a FASTA sequence -- all-zero k-mer vector")
kmer_matrix = cosine_similarity(kmer_matrix_raw)

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

print("\nScoring all queries (cosine / BLAST / FoldseekTM / kmer + RRF-3 + RRF-4)...")
start = time.time()
records = []

for qi, p in enumerate(gold_pairs):
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)
        kmer_ranks = ranks_from_scores(kmer_matrix[qidx], qidx)

        rrf3_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
        rrf4_score = rrf3_score + 1.0 / (RRF_K + kmer_ranks)

        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)
        rrf4_ranks = ranks_from_scores(rrf4_score, qidx)

        records.append({
            "pair_id": p["pair_id"],
            "cosine_rank": rank_of(cos_ranks, tidx),
            "blast_rank": rank_of(blast_ranks, tidx),
            "foldseektm_rank": rank_of(fs_ranks, tidx),
            "kmer_rank": rank_of(kmer_ranks, tidx),
            "rrf3_rank": rank_of(rrf3_ranks, tidx),
            "rrf4_rank": rank_of(rrf4_ranks, tidx),
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

methods = ["cosine", "blast", "foldseektm", "kmer", "rrf3", "rrf4"]
summary_lines = [
    "=" * 70,
    f"Rank fusion: does k-mer as a 4th independent voter help RRF? ({len(df)} queries, 1548 dataset)",
    "=" * 70,
    "",
]
for m in methods:
    rr = 1.0 / df[f"{m}_rank"]
    mrr = rr.mean()
    hits = {k: (df[f"{m}_rank"] <= k).mean() for k in TOP_K}
    hits_str = "  ".join(f"Hits@{k}={hits[k]:.4f}" for k in TOP_K)
    summary_lines.append(f"{m:12s} MRR={mrr:.4f}  {hits_str}")

rrf3_mrr = (1.0 / df["rrf3_rank"]).mean()
rrf4_mrr = (1.0 / df["rrf4_rank"]).mean()
summary_lines.append("")
summary_lines.append(f"Delta (RRF-4 - RRF-3): {rrf4_mrr - rrf3_mrr:+.4f}")

# bootstrap by pair_id
rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["rrf4_rank"], weights=w) - np.average(1.0 / sub["rrf3_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
summary_lines.append(f"Bootstrap 95% CI (RRF-4 - RRF-3): [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]")
summary_lines.append(f"Fraction of bootstrap resamples favoring RRF-4: {(deltas>0).mean():.3f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
