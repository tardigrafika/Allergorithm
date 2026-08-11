"""
Evaluacija LoRA fine-tuned ESM-2 embeddinga naspram frozen baseline-a, i u
RRF-3 fuziji.

Leakage disciplina: samo UPITI su ograniceni na held-out test_ids
protein-level split (identican Union-Find + seed=42 split kao u
ml/lora_finetune_esm_1548.py) -- model tokom treninga nikad nije video FASTA
sekvence test proteina. Candidate POOL ostaje CEO dataset (svi ~1534
proteina), isto kao svuda drugde u projektu (rank_fusion, rrf_ablation) --
ogranicavanje pool-a na samo 311 test proteina bi vestacki smanjilo zadatak
(vec dokazano u kfold_restricted_universe_1548.py: manji pool -> naduvan MRR
koji nije realna dobit) i ucinilo brojeve neuporedivim sa RRF-3 baseline-om
(0.1294 na punom pool-u). Koriscenje train-protein embeddinga KAO KANDIDATA
nije curenje -- njihovi labeli se nigde ne koriste kao ground truth u ovoj
evaluaciji, samo kao distraktori, isto kao za svaki drugi signal u sesiji.

Dva testa:
  1) frozen cosine vs LoRA cosine (samo taj signal), paired bootstrap po
     pair_id -- da li je VM-ov sirovi delta (+0.0020 MRR) stvarno signal.
  2) RRF-3 sa frozen cosine vs RRF-3 sa LoRA cosine (BLAST i FoldseekTM
     nepromenjeni) -- da li LoRA embedding poboljsava fuziju.

Izlaz:
    output/lora_vs_frozen_evaluation_1548_summary.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
FROZEN_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
LORA_EMBEDDINGS = Path("/home/lana/ALERGRAF/output/embeddings_lora_1548.pkl")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/lora_vs_frozen_evaluation_1548_summary.txt")

SEED = 42
TEST_FRACTION = 0.2
RRF_K = 60
TOP_K = [1, 5, 10, 20]
N_BOOTSTRAP = 2000

# =====================================================
# REPRODUCE THE EXACT SPLIT FROM ml/lora_finetune_esm_1548.py
# =====================================================

allergens = pd.read_csv(CLEAN_ALLERGENS)
allergens = allergens[allergens["fasta_sequence"].notna() & (allergens["fasta_sequence"] != "")].copy()
id_to_seq = dict(zip(allergens["allergen_id"], allergens["fasta_sequence"]))
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id
all_ids = sorted(id_to_seq.keys())

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

gold_pairs = []
for row in gold.itertuples(index=False):
    n1, n2 = str(row.allergen_id_1).strip(), str(row.allergen_id_2).strip()
    id1, id2 = name_to_id.get(n1), name_to_id.get(n2)
    if id1 is None or id2 is None or id1 == id2:
        continue
    if id1 not in id_to_seq or id2 not in id_to_seq:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row.pair_id, "evidence_level": row.evidence_level})

print(f"Mapped gold pairs: {len(gold_pairs)}")

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

rng = np.random.default_rng(SEED)
order = rng.permutation(len(component_list))
gold_protein_count = sum(len(c) for c in component_list)
target_test = round(TEST_FRACTION * gold_protein_count)

train_ids, test_ids = set(), set()
running_test = 0
for idx in order:
    c = component_list[idx]
    if running_test < target_test:
        test_ids |= c
        running_test += len(c)
    else:
        train_ids |= c

free_proteins = [pid for pid in all_ids if pid not in train_ids and pid not in test_ids]
free_proteins = sorted(free_proteins)
rng.shuffle(free_proteins)
n_free_test = round(TEST_FRACTION * len(free_proteins))
test_ids |= set(free_proteins[:n_free_test])
train_ids |= set(free_proteins[n_free_test:])

test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
print(f"Reproduced split -- Train proteins: {len(train_ids)}, Test proteins: {len(test_ids)}")
print(f"Held-out test positive pairs: {len(test_positive_pairs)}")

# =====================================================
# LOAD EMBEDDINGS + BLAST + FOLDSEEK, RESTRICT POOL TO test_ids
# =====================================================

with open(FROZEN_EMBEDDINGS, "rb") as f:
    frozen_dict = pickle.load(f)
with open(LORA_EMBEDDINGS, "rb") as f:
    lora_dict = pickle.load(f)

with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)

pool = sorted(frozen_dict.keys() & lora_dict.keys())
print(f"Full candidate pool (all proteins with both frozen and LoRA embeddings): {len(pool)}")
id_to_index = {aid: i for i, aid in enumerate(pool)}
n_pool = len(pool)

frozen_matrix = np.array([frozen_dict[aid] for aid in pool], dtype=np.float64)
lora_matrix = np.array([lora_dict[aid] for aid in pool], dtype=np.float64)
cosine_frozen = cosine_similarity(frozen_matrix)
cosine_lora = cosine_similarity(lora_matrix)

perm = np.array([blast_id_to_index.get(aid, -1) for aid in pool])
valid = perm >= 0
blast_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

foldseek_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in id_to_index and b in id_to_index:
        i, j = id_to_index[a], id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score

# pairs usable within this candidate pool (queries still restricted to held-out test_ids pairs)
test_positive_pairs = [p for p in test_positive_pairs if p["id_1"] in id_to_index and p["id_2"] in id_to_index]
print(f"Held-out query pairs usable within full pool: {len(test_positive_pairs)}")


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order_ = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order_] = np.arange(1, len(s) + 1)
    return ranks


records = []
for p in test_positive_pairs:
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx, tidx = id_to_index[query_id], id_to_index[target_id]

        frozen_ranks = ranks_from_scores(cosine_frozen[qidx], qidx)
        lora_ranks = ranks_from_scores(cosine_lora[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)

        rrf_frozen_scores = 1.0 / (RRF_K + frozen_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
        rrf_lora_scores = 1.0 / (RRF_K + lora_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
        rrf_frozen_ranks = ranks_from_scores(rrf_frozen_scores, qidx)
        rrf_lora_ranks = ranks_from_scores(rrf_lora_scores, qidx)

        records.append({
            "pair_id": p["pair_id"],
            "evidence_level": p["evidence_level"],
            "frozen_rank": int(frozen_ranks[tidx]),
            "lora_rank": int(lora_ranks[tidx]),
            "blast_rank": int(blast_ranks[tidx]),
            "foldseektm_rank": int(fs_ranks[tidx]),
            "rrf_frozen_rank": int(rrf_frozen_ranks[tidx]),
            "rrf_lora_rank": int(rrf_lora_ranks[tidx]),
        })

df = pd.DataFrame(records)
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/lora_vs_frozen_evaluation_1548_per_query.csv")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}")

# =====================================================
# AGGREGATE MRR / Hits@K
# =====================================================

methods = [("frozen_rank", "Frozen ESM cosine"), ("lora_rank", "LoRA cosine"),
           ("blast_rank", "BLAST"), ("foldseektm_rank", "FoldseekTM"),
           ("rrf_frozen_rank", "RRF-3 (frozen cosine)"), ("rrf_lora_rank", "RRF-3 (LoRA cosine)")]

summary_lines = ["=" * 70, "LoRA vs frozen ESM-2, held-out test_ids split (n={} queries, {} pairs)"
                 .format(len(df), df["pair_id"].nunique()), "=" * 70, ""]

for rank_col, label in methods:
    mrr = (1.0 / df[rank_col]).mean()
    hits = "  ".join(f"Hits@{k}={(df[rank_col] <= k).mean():.4f}" for k in TOP_K)
    summary_lines.append(f"  {label:24s} MRR={mrr:.4f}  {hits}")
summary_lines.append("")

# =====================================================
# PAIRED BOOTSTRAP, by pair_id
# =====================================================

rng2 = np.random.default_rng(SEED)
pair_ids = df["pair_id"].unique()


def bootstrap_delta(df, col_better, col_baseline, label):
    deltas = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng2.choice(pair_ids, size=len(pair_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = (np.average(1.0 / resampled[col_better], weights=w)
             - np.average(1.0 / resampled[col_baseline], weights=w))
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    frac_better = (deltas > 0).mean()
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    line = (f"  {label}: mean delta = {deltas.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
            f"bolji u {frac_better:.1%} resample-ova -- {verdict}")
    return line


summary_lines.append("--- Paired bootstrap (2000 resamples by pair_id) ---")
summary_lines.append(bootstrap_delta(df, "lora_rank", "frozen_rank", "LoRA cosine vs frozen cosine"))
summary_lines.append(bootstrap_delta(df, "rrf_lora_rank", "rrf_frozen_rank", "RRF-3(LoRA) vs RRF-3(frozen)"))
summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
