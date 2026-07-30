"""
Random Forest + BLAST + TM-score (strukturno poravnanje) - 1443 dataset.

Ne menja random_forest_blast_1443.py (RF+BLAST bez TM-score ostaje odvojena,
reproducibilna verzija -- trenutno najbolji cist/ne-cirkularan rezultat, MRR 0.2028).

Feature vektor: 1280 (abs_diff) + cosine + blast_identity + blast_score + tm_score
= 1284 dimenzije.

tm_score: TM-align strukturno poravnanje (AlphaFold DB strukture, 1240/1534
proteina sa dostupnom strukturom preko UniProt ID-ja). Za proteine bez
strukture (nema UniProt ID ili AlphaFold 404), tm_score = 0.0 (nema
strukturnog dokaza, ne izbacuje se par -- ESM/BLAST i dalje daju signal).

Racunato unapred (data/fetch_alphafold_structures.py +
data/compute_tmscore_matrix_1443.py, paralelizovano preko 8 jezgara jer je
TM-align znatno sporiji od sequence poravnanja: ~27 poravnanja/sek po jezgru).

Rizik na koji treba paziti (isti kao same_family): TM-score meri CELU
strukturu, ne epitope region -- moguce je da je delimicno redundantan sa
ESM-om (oba hvataju fold-nivo slicnost), pa ocekivanje treba biti umereno
u odnosu na BLAST koji je dao jasan, nezavisan dobitak.

Izlaz:
    output/random_forest_blast_tmscore_model_1443.joblib
    output/random_forest_blast_tmscore_retrieval_results_1443.csv
    output/random_forest_blast_tmscore_summary_1443.txt
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
TMSCORE_MATRIX = Path("/home/lana/ALERGRAF/output/tmscore_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODEL_OUTPUT = OUTPUT_DIR / "random_forest_blast_tmscore_model_1443.joblib"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "random_forest_blast_tmscore_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "random_forest_blast_tmscore_summary_1443.txt"

BASELINE_RF_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_retrieval_results_1443.csv")
BLAST_RF_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_blast_retrieval_results_1443.csv")


# =====================================================
# CONFIGURATION (identical to ml/random_forest_baseline_1443.py)
# =====================================================

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)


# =====================================================
# LOAD DATA
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

print("Loading precomputed BLAST identity matrix...")
with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_identity_matrix = blast_data["identity_matrix"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {allergen_id: i for i, allergen_id in enumerate(blast_ids)}
print(f"BLAST matrix shape: {blast_identity_matrix.shape}")

print("Loading precomputed TM-score matrix...")
with open(TMSCORE_MATRIX, "rb") as f:
    tm_data = pickle.load(f)
tm_ids = tm_data["ids"]
tm_score_matrix = tm_data["tm_score_matrix"]
tm_id_to_index = {allergen_id: i for i, allergen_id in enumerate(tm_ids)}
print(f"TM-score matrix shape: {tm_score_matrix.shape} "
      f"({len(tm_id_to_index)}/{len(embeddings_dict)} proteins have a structure)")

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
print(f"Positive gold-standard pairs retained    : {len(gold)}")


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

print(f"\nOfficial names mapped : {len(name_to_id)}")


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

positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}


# =====================================================
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm to the baseline script)
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
print(f"Connected components in gold-standard graph: {len(component_list)}")

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

print(f"Train proteins: {len(train_ids)}  ({len(train_ids)/len(all_ids):.1%})")
print(f"Test proteins : {len(test_ids)}  ({len(test_ids)/len(all_ids):.1%})")

train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
cross_split_pairs = len(gold_pairs) - len(train_positive_pairs) - len(test_positive_pairs)

print(f"Train positive pairs: {len(train_positive_pairs)}")
print(f"Test positive pairs : {len(test_positive_pairs)}")
assert cross_split_pairs == 0, "a gold pair spans both splits (leakage, should be impossible)"


# =====================================================
# NEGATIVE SAMPLING (identical function to the baseline script)
# =====================================================

def sample_negative_pairs(protein_pool, n_needed, seed):
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    negatives = set()
    max_attempts = n_needed * 50 + 2000
    attempts = 0
    while len(negatives) < n_needed and attempts < max_attempts:
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1
        if pair in positive_pair_set or pair in negatives:
            continue
        negatives.add(pair)
    if len(negatives) < n_needed:
        print(f"WARNING: only sampled {len(negatives)}/{n_needed} negatives")
    return sorted(negatives)


n_train_neg = len(train_positive_pairs) * NEG_PER_POS
n_test_neg = len(test_positive_pairs) * NEG_PER_POS

train_negative_pairs = sample_negative_pairs(train_ids, n_train_neg, seed=SEED)
test_negative_pairs = sample_negative_pairs(test_ids, n_test_neg, seed=SEED + 1)

print(f"\nTrain negative pairs sampled: {len(train_negative_pairs)} (target ratio {NEG_PER_POS}:1)")
print(f"Test negative pairs sampled : {len(test_negative_pairs)} (target ratio {NEG_PER_POS}:1)")


# =====================================================
# FEATURE CONSTRUCTION -- EXTENDED with BLAST identity + alignment score
# + TM-score (everything else identical to
# ml/random_forest_baseline_1443.py's pairwise_features)
# =====================================================

def tm_score_lookup(id_a, id_b):
    """0.0 (no structural evidence) if either protein has no AlphaFold
    structure -- see module docstring."""
    if id_a not in tm_id_to_index or id_b not in tm_id_to_index:
        return 0.0
    return float(tm_score_matrix[tm_id_to_index[id_a], tm_id_to_index[id_b]])


def pairwise_features(emb_a, emb_b, ids_a, ids_b):
    emb_a = np.atleast_2d(emb_a)
    emb_b = np.atleast_2d(emb_b)
    abs_diff = np.abs(emb_a - emb_b)
    dot = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    cosine = dot / (norm_a * norm_b + 1e-12)

    blast_id = np.array([blast_identity_matrix[blast_id_to_index[a], blast_id_to_index[b]]
                          for a, b in zip(ids_a, ids_b)])
    blast_sc = np.array([blast_score_matrix[blast_id_to_index[a], blast_id_to_index[b]]
                          for a, b in zip(ids_a, ids_b)])
    tm_sc = np.array([tm_score_lookup(a, b) for a, b in zip(ids_a, ids_b)])

    return np.hstack([
        abs_diff, cosine.reshape(-1, 1), blast_id.reshape(-1, 1), blast_sc.reshape(-1, 1),
        tm_sc.reshape(-1, 1),
    ])


def pairwise_features_batch_same_query(query_emb, query_id, candidate_embs, candidate_ids):
    """Fast path for retrieval: one fixed query against many candidates."""
    query_batch = np.tile(query_emb, (len(candidate_ids), 1))
    query_ids_batch = [query_id] * len(candidate_ids)
    return pairwise_features(query_batch, candidate_embs, query_ids_batch, candidate_ids)


def build_feature_matrix(positive_pairs, negative_pairs):
    rows_a, rows_b, ids_a, ids_b, labels = [], [], [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        ids_a.append(p["id_1"])
        ids_b.append(p["id_2"])
        labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        ids_a.append(a)
        ids_b.append(b)
        labels.append(0)
    X = pairwise_features(np.array(rows_a), np.array(rows_b), ids_a, ids_b)
    y = np.array(labels)
    return X, y


print("\n==============================")
print("BUILDING FEATURE MATRICES (1280 abs_diff + cosine + blast_id + blast_score + tm_score = 1284 dims)")
print("==============================")

X_train, y_train = build_feature_matrix(train_positive_pairs, train_negative_pairs)
X_test, y_test = build_feature_matrix(test_positive_pairs, test_negative_pairs)

print(f"Train: X={X_train.shape}, positives={int(y_train.sum())}, negatives={int((y_train == 0).sum())}")
print(f"Test : X={X_test.shape}, positives={int(y_test.sum())}, negatives={int((y_test == 0).sum())}")


# =====================================================
# TRAIN RANDOM FOREST
# =====================================================

print("\n==============================")
print("TRAINING RANDOM FOREST (with BLAST + TM-score features)")
print("==============================")
print(f"Hyperparameters: {RF_PARAMS}")

rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X_train, y_train)
print("Training complete.")

# feature importance sanity check: where do the 3 new columns rank?
importances = rf.feature_importances_
feature_names_tail = ["cosine", "blast_identity", "blast_score", "tm_score"]
print("\nFeature importances (non-abs_diff features):")
for name, imp in zip(feature_names_tail, importances[-4:]):
    rank = int((importances > imp).sum()) + 1
    print(f"  {name:<16}: {imp:.5f}  (rank {rank}/{len(importances)})")
print(f"  mean importance across all {len(importances)} features: {importances.mean():.5f}")


# =====================================================
# A) CLASSIFICATION METRICS
# =====================================================

print("\n==============================")
print("CLASSIFICATION METRICS (test split)")
print("==============================")

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

clf_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "pr_auc": average_precision_score(y_test, y_proba),
}
conf_matrix = confusion_matrix(y_test, y_pred)

for name, value in clf_metrics.items():
    print(f"{name:10s}: {value:.4f}")
print("\nConfusion matrix (rows=true, cols=predicted, order=[0,1]):")
print(conf_matrix)


# =====================================================
# B) RETRIEVAL EVALUATION
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

        query_vec = embedding_matrix[query_index]
        X_candidates = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)
        rf_scores = rf.predict_proba(X_candidates)[:, 1]
        rf_scores[query_index] = -np.inf

        rf_ranked = np.argsort(rf_scores)[::-1]
        rf_rank = int(np.where(rf_ranked == target_index)[0][0]) + 1
        rf_reciprocal_rank = 1.0 / rf_rank
        rf_true_pair_probability = rf_scores[target_index]

        cos_scores = cosine_similarity_matrix[query_index].copy()
        cos_scores[query_index] = -np.inf
        cos_ranked = np.argsort(cos_scores)[::-1]
        cos_rank = int(np.where(cos_ranked == target_index)[0][0]) + 1
        cos_reciprocal_rank = 1.0 / cos_rank

        retrieval_results.append({
            "pair_id": p["pair_id"],
            "query_allergen": query_name,
            "target_allergen": target_name,
            "query_allergen_id": query_id,
            "target_allergen_id": target_id,
            "query_family": family_q,
            "target_family": family_t,
            "rf_probability": rf_true_pair_probability,
            "rf_rank": rf_rank,
            "rf_reciprocal_rank": rf_reciprocal_rank,
            "rf_hits_at_1": int(rf_rank <= 1),
            "rf_hits_at_5": int(rf_rank <= 5),
            "rf_hits_at_10": int(rf_rank <= 10),
            "rf_hits_at_20": int(rf_rank <= 20),
            "cosine_rank": cos_rank,
            "cosine_reciprocal_rank": cos_reciprocal_rank,
            "cosine_hits_at_1": int(cos_rank <= 1),
            "cosine_hits_at_5": int(cos_rank <= 5),
            "cosine_hits_at_10": int(cos_rank <= 10),
            "cosine_hits_at_20": int(cos_rank <= 20),
        })

retrieval_df = pd.DataFrame(retrieval_results)

rf_hits = {k: retrieval_df[f"rf_hits_at_{k}"].mean() for k in TOP_K}
rf_mrr = retrieval_df["rf_reciprocal_rank"].mean()
cosine_test_hits = {k: retrieval_df[f"cosine_hits_at_{k}"].mean() for k in TOP_K}
cosine_test_mrr = retrieval_df["cosine_reciprocal_rank"].mean()

print(f"Retrieval queries evaluated: {len(retrieval_df)}")
print(f"{'Metric':<10}{'Cosine (same test)':<20}{'RF+BLAST+TM':<20}")
for k in TOP_K:
    print(f"Hits@{k:<5d}{cosine_test_hits[k]:<20.4f}{rf_hits[k]:<20.4f}")
print(f"{'MRR':<10}{cosine_test_mrr:<20.4f}{rf_mrr:<20.4f}")


# =====================================================
# SAVE OUTPUTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nRetrieval results saved to: {RETRIEVAL_OUTPUT}")

joblib.dump(rf, MODEL_OUTPUT)
print(f"Model saved to: {MODEL_OUTPUT}")


# =====================================================
# COMPARISON WITH BASELINE (ESM-only) RF AND RF+BLAST
# =====================================================

def load_other(path, prefix):
    if not path.exists():
        return {k: float("nan") for k in TOP_K}, float("nan"), None
    df = pd.read_csv(path)
    if len(df) != len(retrieval_df):
        return {k: float("nan") for k in TOP_K}, float("nan"), None
    merged = retrieval_df.merge(
        df[["pair_id", "query_allergen", "cosine_reciprocal_rank"]],
        on=["pair_id", "query_allergen"], suffixes=("_this", "_other"),
    )
    consistent = bool(np.allclose(merged["cosine_reciprocal_rank_this"], merged["cosine_reciprocal_rank_other"]))
    hits = {k: df[f"{prefix}_hits_at_{k}"].mean() for k in TOP_K}
    mrr = df[f"{prefix}_reciprocal_rank"].mean()
    return hits, mrr, consistent


baseline_hits, baseline_mrr, baseline_consistent = load_other(BASELINE_RF_RESULTS, "rf")
blast_hits, blast_mrr, blast_consistent = load_other(BLAST_RF_RESULTS, "rf")

print(f"\nSplit consistency vs baseline RF: {baseline_consistent}")
print(f"Split consistency vs RF+BLAST   : {blast_consistent}")


# =====================================================
# FINAL SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("RANDOM FOREST + BLAST + TM-SCORE (1443 dataset) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Random seed              : {SEED}")
summary_lines.append(f"Positive gold-standard pairs retained: {len(gold)}")
summary_lines.append(f"Feature vector: 1280 (abs_diff) + cosine + blast_identity + blast_score + "
                      f"tm_score = {X_train.shape[1]} dims")
summary_lines.append(f"Proteins with AlphaFold structure: {len(tm_id_to_index)}/{len(embeddings_dict)}")
summary_lines.append("Feature importances (tail features):")
for name, imp in zip(feature_names_tail, importances[-4:]):
    rank = int((importances > imp).sum()) + 1
    summary_lines.append(f"  {name:<16}: {imp:.5f} (rank {rank}/{len(importances)})")
summary_lines.append("")
summary_lines.append("Split strategy: group-aware protein-level split, identical to "
                      "ml/random_forest_baseline_1443.py.")
summary_lines.append(f"  Train positive pairs  : {len(train_positive_pairs)}")
summary_lines.append(f"  Test positive pairs   : {len(test_positive_pairs)}")
summary_lines.append(f"  Split consistency check vs baseline RF: {baseline_consistent}")
summary_lines.append(f"  Split consistency check vs RF+BLAST   : {blast_consistent}")
summary_lines.append("")
summary_lines.append("Classification metrics (test split, positives vs sampled negatives):")
for name, value in clf_metrics.items():
    summary_lines.append(f"  {name:10s}: {value:.4f}")
summary_lines.append(f"  confusion matrix [ [TN FP] [FN TP] ]: {conf_matrix.tolist()}")
summary_lines.append("")
summary_lines.append(f"Retrieval evaluation: {len(retrieval_df)} queries "
                      f"({len(test_positive_pairs)} test pairs x 2 directions)")
summary_lines.append("")

header = f"{'Metric':<10}{'Cosine':<12}{'RF (ESM)':<12}{'RF+BLAST':<12}{'RF+BLAST+TM':<12}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
for k in TOP_K:
    summary_lines.append(
        f"{'Hits@' + str(k):<10}{cosine_test_hits[k]:<12.4f}"
        f"{baseline_hits[k]:<12.4f}{blast_hits[k]:<12.4f}{rf_hits[k]:<12.4f}"
    )
summary_lines.append(
    f"{'MRR':<10}{cosine_test_mrr:<12.4f}{baseline_mrr:<12.4f}{blast_mrr:<12.4f}{rf_mrr:<12.4f}"
)

if not np.isnan(blast_mrr):
    delta_mrr = rf_mrr - blast_mrr
    summary_lines.append(f"\nDelta vs RF+BLAST (adding TM-score): MRR {delta_mrr:+.4f}")
    verdict = "IMPROVED" if delta_mrr > 0 else ("WORSE" if delta_mrr < 0 else "UNCHANGED")
    summary_lines.append(f"Adding TM-score {verdict} Random Forest retrieval on this dataset.")
else:
    summary_lines.append(f"\nNOTE: {BLAST_RF_RESULTS} not found -- run ml/random_forest_blast_1443.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
