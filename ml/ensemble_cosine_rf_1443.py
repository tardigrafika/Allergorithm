"""
Ensemble cosine similarity + Random Forest - 1443 dataset.

Ne trenira nista novo - ucitava vec istrenirani RF (random_forest_model_1443.joblib)
i kombinuje njegov skor sa cosine similarity-jem u retrieval fazi.
Dve strategije: normalizovan prosek (50/50) i Reciprocal Rank Fusion (RRF).

Rezultat: ensemble popravlja cosine, ali ne prestize cist RF - RF vec
sadrzi cosine kao jedan od svojih feature-a, pa blend ne donosi nezavisan signal.

Izlaz:
    output/ensemble_cosine_rf_retrieval_results_1443.csv
    output/ensemble_cosine_rf_summary_1443.txt
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
RF_MODEL = Path("/home/lana/ALERGRAF/output/random_forest_model_1443.joblib")

RF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_retrieval_results_1443.csv")
MLP_CLF_RETRIEVAL_RESULTS = Path("/home/lana/ALERGRAF/output/mlp_retrieval_results_1443.csv")
MSE_EMBED_RETRIEVAL_RESULTS = Path(
    "/home/lana/ALERGRAF/output/mlp_embedding_transform_retrieval_results_1443.csv"
)
RANDOM_TRIPLET_RETRIEVAL_RESULTS = Path(
    "/home/lana/ALERGRAF/output/mlp_embedding_transform_triplet_retrieval_results_1443.csv"
)
HARDNEG_TRIPLET_RETRIEVAL_RESULTS = Path(
    "/home/lana/ALERGRAF/output/mlp_embedding_transform_triplet_hardneg_retrieval_results_1443.csv"
)

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
RETRIEVAL_OUTPUT = OUTPUT_DIR / "ensemble_cosine_rf_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "ensemble_cosine_rf_summary_1443.txt"


# =====================================================
# CONFIGURATION (split params identical to ml/random_forest_baseline_1443.py)
# =====================================================

SEED = 42
TEST_FRACTION = 0.2
TOP_K = [1, 5, 10, 20]
RRF_K = 60  # conventional Reciprocal Rank Fusion constant


# =====================================================
# LOAD DATA (identical to ml/random_forest_baseline_1443.py)
# =====================================================

print("\n==============================")
print("LOADING DATA")
print("==============================")

with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Proteins in embeddings: {len(embeddings_dict)}")

metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()
print(f"Metadata rows with embeddings: {len(metadata)}")

gold_raw = pd.read_csv(GOLD)
print(f"Rows in gold file: {len(gold_raw)}")


# =====================================================
# EVIDENCE-LEVEL FILTERING (identical rule to the other *_1443 scripts)
# =====================================================

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()
print(f"\nExcluded negative/contested/risky rows: {len(excluded)}")
print(f"Positive gold-standard pairs retained : {len(gold)}")


# =====================================================
# NAME -> ALLERGEN_ID MAPPING
# =====================================================

name_to_id = {}
duplicate_names = 0
for _, row in metadata.iterrows():
    official_name = str(row["official_name"]).strip()
    if official_name == "" or official_name.lower() == "nan":
        continue
    if official_name in name_to_id:
        duplicate_names += 1
        continue
    name_to_id[official_name] = row["allergen_id"]

print(f"Official names mapped : {len(name_to_id)}")


# =====================================================
# EMBEDDING MATRIX + COSINE MATRIX
# =====================================================

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array(
    [embeddings_dict[allergen_id] for allergen_id in all_ids],
    dtype=np.float64,
)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

cosine_similarity_matrix = cosine_similarity(embedding_matrix)


# =====================================================
# MAP GOLD STANDARD PAIRS TO ALLERGEN IDS
# =====================================================

gold_pairs = []
missing_pairs = 0

for _, row in gold.iterrows():
    name_1 = str(row["allergen_id_1"]).strip()
    name_2 = str(row["allergen_id_2"]).strip()

    if name_1 not in name_to_id or name_2 not in name_to_id:
        missing_pairs += 1
        continue

    id_1 = name_to_id[name_1]
    id_2 = name_to_id[name_2]

    if id_1 not in id_to_index or id_2 not in id_to_index or id_1 == id_2:
        missing_pairs += 1
        continue

    gold_pairs.append(
        {
            "pair_id": row["pair_id"],
            "id_1": id_1,
            "id_2": id_2,
            "name_1": name_1,
            "name_2": name_2,
            "family_1": row["family_1"],
            "family_2": row["family_2"],
        }
    )

print(f"Mapped gold pairs : {len(gold_pairs)}")
print(f"Missing/unmapped  : {missing_pairs}")


# =====================================================
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm/seed to every
# other *_1443 script -- reproduces the identical 165 held-out test pairs)
# =====================================================

print("\n==============================")
print("PROTEIN-LEVEL SPLIT (leakage prevention)")
print("==============================")

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
for protein_id in parent:
    root = find(protein_id)
    components.setdefault(root, set()).add(protein_id)

component_list = list(components.values())

rng = np.random.default_rng(SEED)
order = rng.permutation(len(component_list))
gold_protein_count = sum(len(c) for c in component_list)
target_component_test = round(TEST_FRACTION * gold_protein_count)

train_ids, test_ids = set(), set()
running_test = 0
for idx in order:
    component = component_list[idx]
    if running_test < target_component_test:
        test_ids |= component
        running_test += len(component)
    else:
        train_ids |= component

free_proteins = [pid for pid in all_ids if pid not in train_ids and pid not in test_ids]
free_proteins = list(free_proteins)
rng.shuffle(free_proteins)
n_free_test = round(TEST_FRACTION * len(free_proteins))
test_ids |= set(free_proteins[:n_free_test])
train_ids |= set(free_proteins[n_free_test:])

assert train_ids.isdisjoint(test_ids), "protein split is not disjoint (bug)"
assert len(train_ids) + len(test_ids) == len(all_ids), "split does not cover all proteins"

test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
print(f"Test proteins       : {len(test_ids)}")
print(f"Test positive pairs : {len(test_positive_pairs)}")


# =====================================================
# LOAD THE ALREADY-TRAINED RANDOM FOREST (not retrained here)
# =====================================================

print("\n==============================")
print("LOADING TRAINED RANDOM FOREST")
print("==============================")

if not RF_MODEL.exists():
    raise SystemExit(
        f"ERROR: {RF_MODEL} not found. Run ml/random_forest_baseline_1443.py first "
        f"to train and save the RF model this ensemble depends on."
    )

rf = joblib.load(RF_MODEL)
print(f"Loaded: {RF_MODEL}")


def pairwise_features(emb_a, emb_b):
    """Identical to ml/random_forest_baseline_1443.py -- must match exactly,
    since this is the feature format the loaded RF model was trained on."""
    emb_a = np.atleast_2d(emb_a)
    emb_b = np.atleast_2d(emb_b)
    abs_diff = np.abs(emb_a - emb_b)
    dot = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    cosine = dot / (norm_a * norm_b + 1e-12)
    return np.hstack([abs_diff, cosine.reshape(-1, 1)])


# =====================================================
# RETRIEVAL EVALUATION: cosine, RF, and two ensembles, all on the
# SAME 330 queries / same 1534-protein candidate pool
# =====================================================

print("\n==============================")
print("RETRIEVAL EVALUATION (Hits@K / MRR)")
print("==============================")
print(f"Test-split gold pairs: {len(test_positive_pairs)}  "
      f"-> up to {2 * len(test_positive_pairs)} retrieval queries")

retrieval_results = []

for p in test_positive_pairs:
    directions = [
        (p["id_1"], p["id_2"], p["name_1"], p["name_2"], p["family_1"], p["family_2"]),
        (p["id_2"], p["id_1"], p["name_2"], p["name_1"], p["family_2"], p["family_1"]),
    ]

    for query_id, target_id, query_name, target_name, family_q, family_t in directions:
        query_index = id_to_index[query_id]
        target_index = id_to_index[target_id]

        # ---- cosine scores over the full candidate pool ----
        cos_scores = cosine_similarity_matrix[query_index].copy()
        cos_scores[query_index] = -np.inf

        # ---- RF probability scores over the full candidate pool ----
        query_vec = embedding_matrix[query_index]
        query_batch = np.tile(query_vec, (len(all_ids), 1))
        X_candidates = pairwise_features(query_batch, embedding_matrix)
        rf_scores = rf.predict_proba(X_candidates)[:, 1]
        rf_scores[query_index] = -np.inf

        # ---- individual rankings (for reference / RRF ranks) ----
        cos_ranked = np.argsort(cos_scores)[::-1]
        rf_ranked = np.argsort(rf_scores)[::-1]
        cos_rank_of = {idx: r + 1 for r, idx in enumerate(cos_ranked)}
        rf_rank_of = {idx: r + 1 for r, idx in enumerate(rf_ranked)}

        # ---- Strategy 1: min-max normalize per query, then 50/50 average ----
        valid_mask = np.ones(len(all_ids), dtype=bool)
        valid_mask[query_index] = False

        cos_valid = cos_scores[valid_mask]
        rf_valid = rf_scores[valid_mask]
        cos_norm_all = (cos_scores - cos_valid.min()) / (cos_valid.max() - cos_valid.min() + 1e-12)
        rf_norm_all = (rf_scores - rf_valid.min()) / (rf_valid.max() - rf_valid.min() + 1e-12)
        avg_scores = 0.5 * cos_norm_all + 0.5 * rf_norm_all
        avg_scores[query_index] = -np.inf

        avg_ranked = np.argsort(avg_scores)[::-1]
        avg_rank = int(np.where(avg_ranked == target_index)[0][0]) + 1

        # ---- Strategy 2: Reciprocal Rank Fusion ----
        rrf_scores = np.array([
            1.0 / (RRF_K + cos_rank_of[i]) + 1.0 / (RRF_K + rf_rank_of[i])
            for i in range(len(all_ids))
        ])
        rrf_scores[query_index] = -np.inf
        rrf_ranked = np.argsort(rrf_scores)[::-1]
        rrf_rank = int(np.where(rrf_ranked == target_index)[0][0]) + 1

        # ---- reference: standalone cosine / RF ranks (for the consistency check) ----
        cos_rank = cos_rank_of[target_index]
        rf_rank = rf_rank_of[target_index]

        retrieval_results.append({
            "pair_id": p["pair_id"],
            "query_allergen": query_name,
            "target_allergen": target_name,
            "query_allergen_id": query_id,
            "target_allergen_id": target_id,
            "query_family": family_q,
            "target_family": family_t,
            "cosine_rank": cos_rank,
            "cosine_reciprocal_rank": 1.0 / cos_rank,
            "cosine_hits_at_1": int(cos_rank <= 1),
            "cosine_hits_at_5": int(cos_rank <= 5),
            "cosine_hits_at_10": int(cos_rank <= 10),
            "cosine_hits_at_20": int(cos_rank <= 20),
            "rf_rank": rf_rank,
            "rf_reciprocal_rank": 1.0 / rf_rank,
            "rf_hits_at_1": int(rf_rank <= 1),
            "rf_hits_at_5": int(rf_rank <= 5),
            "rf_hits_at_10": int(rf_rank <= 10),
            "rf_hits_at_20": int(rf_rank <= 20),
            "ensemble_avg_rank": avg_rank,
            "ensemble_avg_reciprocal_rank": 1.0 / avg_rank,
            "ensemble_avg_hits_at_1": int(avg_rank <= 1),
            "ensemble_avg_hits_at_5": int(avg_rank <= 5),
            "ensemble_avg_hits_at_10": int(avg_rank <= 10),
            "ensemble_avg_hits_at_20": int(avg_rank <= 20),
            "ensemble_rrf_rank": rrf_rank,
            "ensemble_rrf_reciprocal_rank": 1.0 / rrf_rank,
            "ensemble_rrf_hits_at_1": int(rrf_rank <= 1),
            "ensemble_rrf_hits_at_5": int(rrf_rank <= 5),
            "ensemble_rrf_hits_at_10": int(rrf_rank <= 10),
            "ensemble_rrf_hits_at_20": int(rrf_rank <= 20),
        })

retrieval_df = pd.DataFrame(retrieval_results)


def agg(prefix, hits_prefix, mrr_col):
    hits = {k: retrieval_df[f"{hits_prefix}_hits_at_{k}"].mean() for k in TOP_K}
    mrr = retrieval_df[mrr_col].mean()
    return hits, mrr


cosine_hits, cosine_mrr = agg("cosine", "cosine", "cosine_reciprocal_rank")
rf_hits, rf_mrr = agg("rf", "rf", "rf_reciprocal_rank")
avg_hits, avg_mrr = agg("ensemble_avg", "ensemble_avg", "ensemble_avg_reciprocal_rank")
rrf_hits, rrf_mrr = agg("ensemble_rrf", "ensemble_rrf", "ensemble_rrf_reciprocal_rank")

print(f"Retrieval queries evaluated: {len(retrieval_df)}")
print(f"{'Metric':<10}{'Cosine':<12}{'RF':<12}{'Ens(avg)':<12}{'Ens(RRF)':<12}")
for k in TOP_K:
    print(f"Hits@{k:<5d}{cosine_hits[k]:<12.4f}{rf_hits[k]:<12.4f}{avg_hits[k]:<12.4f}{rrf_hits[k]:<12.4f}")
print(f"{'MRR':<10}{cosine_mrr:<12.4f}{rf_mrr:<12.4f}{avg_mrr:<12.4f}{rrf_mrr:<12.4f}")


# =====================================================
# SAVE OUTPUTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nRetrieval results saved to: {RETRIEVAL_OUTPUT}")


# =====================================================
# CONSISTENCY CHECK against RF's own saved retrieval file (confirms
# this script reconstructed the identical test split / rankings)
# =====================================================

rf_file_consistent = None
if RF_RETRIEVAL_RESULTS.exists():
    rf_saved = pd.read_csv(RF_RETRIEVAL_RESULTS)
    if len(rf_saved) == len(retrieval_df):
        merged = retrieval_df.merge(
            rf_saved[["pair_id", "query_allergen", "rf_reciprocal_rank", "cosine_reciprocal_rank"]],
            on=["pair_id", "query_allergen"], suffixes=("_this_script", "_rf_script"),
        )
        rf_file_consistent = bool(
            np.allclose(merged["rf_reciprocal_rank_this_script"], merged["rf_reciprocal_rank_rf_script"])
            and np.allclose(merged["cosine_reciprocal_rank_this_script"], merged["cosine_reciprocal_rank_rf_script"])
        )
    print(f"Split/ranking consistency check vs ml/random_forest_baseline_1443.py: {rf_file_consistent}")


# =====================================================
# LOAD OTHER METHODS' SAVED RESULTS FOR THE FULL COMPARISON TABLE
# =====================================================

def load_other_method(path, hits_prefix, mrr_col, method_label):
    if not path.exists():
        print(f"{method_label} results file NOT found: {path}")
        return {k: float("nan") for k in TOP_K}, float("nan")
    other_df = pd.read_csv(path)
    if len(other_df) != len(retrieval_df):
        print(f"WARNING: {method_label} file has {len(other_df)} queries, "
              f"this script has {len(retrieval_df)} -- skipping.")
        return {k: float("nan") for k in TOP_K}, float("nan")
    hits = {k: other_df[f"{hits_prefix}_hits_at_{k}"].mean() for k in TOP_K}
    mrr = other_df[mrr_col].mean()
    return hits, mrr


mlp_clf_hits, mlp_clf_mrr = load_other_method(
    MLP_CLF_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank", "MLP classifier"
)
mse_embed_hits, mse_embed_mrr = load_other_method(
    MSE_EMBED_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank", "MLP embed. (MSE)"
)
random_triplet_hits, random_triplet_mrr = load_other_method(
    RANDOM_TRIPLET_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank", "MLP embed. (Triplet, random neg)"
)
hardneg_triplet_hits, hardneg_triplet_mrr = load_other_method(
    HARDNEG_TRIPLET_RETRIEVAL_RESULTS, "mlp", "mlp_reciprocal_rank", "MLP embed. (Triplet, hard neg)"
)


# =====================================================
# FINAL SUMMARY
# =====================================================

lines = []


def add(line=""):
    lines.append(line)
    print(line)


add("\n" + "=" * 60)
add("ENSEMBLE: COSINE + RANDOM FOREST (1443 dataset) - SUMMARY")
add("=" * 60)
add(f"Random seed              : {SEED}")
add(f"Positive gold-standard pairs retained: {len(gold)}")
add(f"Test positive pairs      : {len(test_positive_pairs)}")
add(f"Retrieval queries        : {len(retrieval_df)}")
if rf_file_consistent is not None:
    add(f"Split/ranking consistency check vs RF script: {rf_file_consistent}")
add("")
add(f"RRF constant K = {RRF_K} (standard default, not tuned)")
add("Score-average strategy: per-query min-max normalization, 50/50 blend (not tuned)")
add("")

header = f"{'Method':<38}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
add(header)
add("-" * len(header))


def fmt(label, hits, mrr):
    return f"{label:<38}{hits[1]:<10.4f}{hits[5]:<10.4f}{hits[10]:<10.4f}{hits[20]:<10.4f}{mrr:<10.4f}"


add(fmt("Cosine baseline", cosine_hits, cosine_mrr))
add(fmt("Random Forest", rf_hits, rf_mrr))
add(fmt("MLP classifier", mlp_clf_hits, mlp_clf_mrr))
add(fmt("MLP embed. (MSE)", mse_embed_hits, mse_embed_mrr))
add(fmt("MLP embed. (Triplet, random neg)", random_triplet_hits, random_triplet_mrr))
add(fmt("MLP embed. (Triplet, hard neg)", hardneg_triplet_hits, hardneg_triplet_mrr))
add(fmt("Ensemble: Cosine+RF (avg, normalized)", avg_hits, avg_mrr))
add(fmt("Ensemble: Cosine+RF (RRF)", rrf_hits, rrf_mrr))

add("")
best_overall = max(
    [
        ("Cosine baseline", cosine_mrr), ("Random Forest", rf_mrr),
        ("MLP classifier", mlp_clf_mrr), ("MLP embed. (MSE)", mse_embed_mrr),
        ("MLP embed. (Triplet, random neg)", random_triplet_mrr),
        ("MLP embed. (Triplet, hard neg)", hardneg_triplet_mrr),
        ("Ensemble (avg)", avg_mrr), ("Ensemble (RRF)", rrf_mrr),
    ],
    key=lambda kv: kv[1] if not np.isnan(kv[1]) else -1,
)
add(f"Best method by MRR: {best_overall[0]} (MRR={best_overall[1]:.4f})")

ensemble_beats_both = (avg_mrr > cosine_mrr and avg_mrr > rf_mrr) or (rrf_mrr > cosine_mrr and rrf_mrr > rf_mrr)
add(f"\nDoes at least one ensemble beat BOTH cosine and RF individually? {ensemble_beats_both}")
if avg_mrr > cosine_mrr and avg_mrr > rf_mrr:
    add(f"  -> Score-average ensemble: MRR {avg_mrr:.4f} > cosine {cosine_mrr:.4f} and > RF {rf_mrr:.4f}")
if rrf_mrr > cosine_mrr and rrf_mrr > rf_mrr:
    add(f"  -> RRF ensemble: MRR {rrf_mrr:.4f} > cosine {cosine_mrr:.4f} and > RF {rf_mrr:.4f}")

summary_text = "\n".join(lines)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
