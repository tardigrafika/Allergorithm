"""
Random Forest + BLAST + same_family flag + k-mer sastav sekvence - 1443 dataset.

Ne menja random_forest_blast_1443.py (ostaje odvojena, reproducibilna verzija).
Isti split, isti negative sampling, isti RF hiperparametri - feature vektor
se siri sa 1283 na 1285 dimenzija:
  1280 (abs_diff) + cosine + blast_identity + blast_score + same_family + kmer_similarity

same_family: 1 ako oba proteina imaju poznatu (istu) familiju iz gold standarda
(family_map, isti izvor kao za hard-negative bezbednost), inace 0 (razlicita ili
nepoznata familija - metadata "protein_family" kolona je prazna za sve proteine,
pa se koristi jedini dostupni izvor familije).

kmer_similarity: cosine slicnost izmedju k=3 (tripeptid) frekvencijskih vektora
sekvenci - klasican, ESM-nezavisan feature.

Izlaz:
    output/random_forest_features2_model_1443.joblib
    output/random_forest_features2_retrieval_results_1443.csv
    output/random_forest_features2_summary_1443.txt
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
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODEL_OUTPUT = OUTPUT_DIR / "random_forest_features2_model_1443.joblib"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "random_forest_features2_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "random_forest_features2_summary_1443.txt"

BASELINE_RF_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_retrieval_results_1443.csv")
BLAST_RF_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_blast_retrieval_results_1443.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]
KMER_K = 3

RF_PARAMS = dict(
    n_estimators=300, max_depth=12, min_samples_leaf=3,
    class_weight="balanced", random_state=SEED, n_jobs=-1,
)

# =====================================================
# LOAD DATA
# =====================================================

print("\n==============================")
print("LOADING DATA")
print("==============================")

with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)

metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_identity_matrix = blast_data["identity_matrix"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {allergen_id: i for i, allergen_id in enumerate(blast_ids)}

clean = pd.read_csv(CLEAN_ALLERGENS)
clean = clean[clean["fasta_sequence"].notna() & (clean["fasta_sequence"] != "")]
id_to_seq = dict(zip(clean["allergen_id"], clean["fasta_sequence"]))

gold_raw = pd.read_csv(GOLD)
print(f"Rows in gold file: {len(gold_raw)}")

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_raw.loc[negative_mask]
gold = gold_raw.loc[~negative_mask].copy()
print(f"Excluded negative/contested/risky rows: {len(excluded)}")
print(f"Positive gold-standard pairs retained    : {len(gold)}")

name_to_id = {}
for _, row in metadata.iterrows():
    official_name = str(row["official_name"]).strip()
    if official_name and official_name.lower() != "nan" and official_name not in name_to_id:
        name_to_id[official_name] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array([embeddings_dict[a] for a in all_ids], dtype=np.float64)
cosine_similarity_matrix = cosine_similarity(embedding_matrix)
print(f"Embedding matrix shape: {embedding_matrix.shape}")


# =====================================================
# GOLD PAIRS + FAMILY MAP (same source as hard-negative safety logic)
# =====================================================

gold_pairs = []
missing_pairs = 0
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        missing_pairs += 1
        continue
    id_1, id_2 = name_to_id[n1], name_to_id[n2]
    if id_1 not in id_to_index or id_2 not in id_to_index or id_1 == id_2:
        missing_pairs += 1
        continue
    gold_pairs.append({
        "pair_id": row["pair_id"], "id_1": id_1, "id_2": id_2,
        "name_1": n1, "name_2": n2,
        "family_1": row["family_1"], "family_2": row["family_2"],
    })
print(f"Mapped gold pairs : {len(gold_pairs)}")
print(f"Missing/unmapped  : {missing_pairs}")

positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}

family_map = {}
for p in gold_pairs:
    family_map.setdefault(p["id_1"], p["family_1"])
    family_map.setdefault(p["id_2"], p["family_2"])
print(f"Proteins with a known family label: {len(family_map)}")


# =====================================================
# K-MER (k=3) FREQUENCY MATRIX + FULL COSINE SIMILARITY MATRIX
# (computed once for all proteins, cheap -- like the ESM cosine matrix)
# =====================================================

print("\n==============================")
print("K-MER FEATURES")
print("==============================")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
kmer_vocab = {}
for a in AMINO_ACIDS:
    for b in AMINO_ACIDS:
        for c in AMINO_ACIDS:
            kmer_vocab[a + b + c] = len(kmer_vocab)
print(f"k={KMER_K} vocabulary size: {len(kmer_vocab)}")


def kmer_frequency_vector(sequence):
    vec = np.zeros(len(kmer_vocab), dtype=np.float32)
    seq = sequence.upper()
    n_kmers = 0
    for i in range(len(seq) - KMER_K + 1):
        kmer = seq[i:i + KMER_K]
        idx = kmer_vocab.get(kmer)
        if idx is not None:
            vec[idx] += 1.0
            n_kmers += 1
    if n_kmers > 0:
        vec /= n_kmers
    return vec


kmer_matrix = np.zeros((len(all_ids), len(kmer_vocab)), dtype=np.float32)
for i, allergen_id in enumerate(all_ids):
    seq = id_to_seq.get(allergen_id, "")
    kmer_matrix[i] = kmer_frequency_vector(seq)

kmer_similarity_matrix = cosine_similarity(kmer_matrix)
print(f"K-mer similarity matrix shape: {kmer_similarity_matrix.shape}")


# =====================================================
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm to other *_1443 scripts)
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
    components.setdefault(find(protein_id), set()).add(protein_id)
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

assert train_ids.isdisjoint(test_ids)
assert len(train_ids) + len(test_ids) == len(all_ids)

train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
assert len(gold_pairs) - len(train_positive_pairs) - len(test_positive_pairs) == 0

print(f"Train proteins: {len(train_ids)}  Test proteins: {len(test_ids)}")
print(f"Train positive pairs: {len(train_positive_pairs)}  Test positive pairs: {len(test_positive_pairs)}")


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
    return sorted(negatives)


n_train_neg = len(train_positive_pairs) * NEG_PER_POS
n_test_neg = len(test_positive_pairs) * NEG_PER_POS
train_negative_pairs = sample_negative_pairs(train_ids, n_train_neg, seed=SEED)
test_negative_pairs = sample_negative_pairs(test_ids, n_test_neg, seed=SEED + 1)
print(f"\nTrain negatives: {len(train_negative_pairs)}  Test negatives: {len(test_negative_pairs)}")


# =====================================================
# FEATURE CONSTRUCTION -- 1280 abs_diff + cosine + blast_identity +
# blast_score + same_family + kmer_similarity = 1285 dims
# =====================================================

def same_family_flag(id_a, id_b):
    fa, fb = family_map.get(id_a), family_map.get(id_b)
    return 1.0 if (fa is not None and fb is not None and fa == fb) else 0.0


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
    same_fam = np.array([same_family_flag(a, b) for a, b in zip(ids_a, ids_b)])
    kmer_sim = np.array([kmer_similarity_matrix[id_to_index[a], id_to_index[b]]
                          for a, b in zip(ids_a, ids_b)])

    return np.hstack([
        abs_diff, cosine.reshape(-1, 1), blast_id.reshape(-1, 1), blast_sc.reshape(-1, 1),
        same_fam.reshape(-1, 1), kmer_sim.reshape(-1, 1),
    ])


def pairwise_features_batch_same_query(query_emb, query_id, candidate_embs, candidate_ids):
    query_batch = np.tile(query_emb, (len(candidate_ids), 1))
    query_ids_batch = [query_id] * len(candidate_ids)
    return pairwise_features(query_batch, candidate_embs, query_ids_batch, candidate_ids)


def build_feature_matrix(positive_pairs, negative_pairs):
    rows_a, rows_b, ids_a, ids_b, labels = [], [], [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        ids_a.append(p["id_1"]); ids_b.append(p["id_2"]); labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        ids_a.append(a); ids_b.append(b); labels.append(0)
    X = pairwise_features(np.array(rows_a), np.array(rows_b), ids_a, ids_b)
    y = np.array(labels)
    return X, y


print("\n==============================")
print("BUILDING FEATURE MATRICES (1285 dims)")
print("==============================")

X_train, y_train = build_feature_matrix(train_positive_pairs, train_negative_pairs)
X_test, y_test = build_feature_matrix(test_positive_pairs, test_negative_pairs)
print(f"Train: X={X_train.shape}  Test: X={X_test.shape}")

n_same_family_train = int(X_train[:, -2].sum())
n_same_family_test = int(X_test[:, -2].sum())
print(f"same_family=1 count -- train: {n_same_family_train}/{len(y_train)}, "
      f"test: {n_same_family_test}/{len(y_test)}")


# =====================================================
# TRAIN RANDOM FOREST
# =====================================================

print("\n==============================")
print("TRAINING RANDOM FOREST (BLAST + same_family + kmer)")
print("==============================")

rf = RandomForestClassifier(**RF_PARAMS)
rf.fit(X_train, y_train)
print("Training complete.")

importances = rf.feature_importances_
feature_names_tail = ["cosine", "blast_identity", "blast_score", "same_family", "kmer_similarity"]
print("\nFeature importances (non-abs_diff features):")
for name, imp in zip(feature_names_tail, importances[-5:]):
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
print(conf_matrix)


# =====================================================
# B) RETRIEVAL EVALUATION
# =====================================================

print("\n==============================")
print("RETRIEVAL EVALUATION (Hits@K / MRR)")
print("==============================")

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

        cos_scores = cosine_similarity_matrix[query_index].copy()
        cos_scores[query_index] = -np.inf
        cos_ranked = np.argsort(cos_scores)[::-1]
        cos_rank = int(np.where(cos_ranked == target_index)[0][0]) + 1
        cos_reciprocal_rank = 1.0 / cos_rank

        retrieval_results.append({
            "pair_id": p["pair_id"], "query_allergen": query_name, "target_allergen": target_name,
            "query_allergen_id": query_id, "target_allergen_id": target_id,
            "query_family": family_q, "target_family": family_t,
            "rf_rank": rf_rank, "rf_reciprocal_rank": rf_reciprocal_rank,
            "rf_hits_at_1": int(rf_rank <= 1), "rf_hits_at_5": int(rf_rank <= 5),
            "rf_hits_at_10": int(rf_rank <= 10), "rf_hits_at_20": int(rf_rank <= 20),
            "cosine_rank": cos_rank, "cosine_reciprocal_rank": cos_reciprocal_rank,
            "cosine_hits_at_1": int(cos_rank <= 1), "cosine_hits_at_5": int(cos_rank <= 5),
            "cosine_hits_at_10": int(cos_rank <= 10), "cosine_hits_at_20": int(cos_rank <= 20),
        })

retrieval_df = pd.DataFrame(retrieval_results)
rf_hits = {k: retrieval_df[f"rf_hits_at_{k}"].mean() for k in TOP_K}
rf_mrr = retrieval_df["rf_reciprocal_rank"].mean()
cosine_test_hits = {k: retrieval_df[f"cosine_hits_at_{k}"].mean() for k in TOP_K}
cosine_test_mrr = retrieval_df["cosine_reciprocal_rank"].mean()

print(f"Retrieval queries evaluated: {len(retrieval_df)}")
for k in TOP_K:
    print(f"Hits@{k:<2d}: Cosine={cosine_test_hits[k]:.4f}  RF+feat2={rf_hits[k]:.4f}")
print(f"MRR    : Cosine={cosine_test_mrr:.4f}  RF+feat2={rf_mrr:.4f}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUTPUT, index=False)
joblib.dump(rf, MODEL_OUTPUT)
print(f"\nSaved: {RETRIEVAL_OUTPUT}")
print(f"Saved: {MODEL_OUTPUT}")


# =====================================================
# COMPARISON WITH BASELINE RF AND RF+BLAST
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
# SUMMARY
# =====================================================

lines = []


def add(s=""):
    lines.append(s)
    print(s)


add("\n" + "=" * 60)
add("RANDOM FOREST + BLAST + SAME_FAMILY + KMER (1443) - SUMMARY")
add("=" * 60)
add(f"Feature vector: 1280 abs_diff + cosine + blast_id + blast_score + "
    f"same_family + kmer_sim = {X_train.shape[1]} dims")
add(f"same_family=1: train {n_same_family_train}/{len(y_train)}, test {n_same_family_test}/{len(y_test)}")
add("Feature importances (tail features):")
for name, imp in zip(feature_names_tail, importances[-5:]):
    rank = int((importances > imp).sum()) + 1
    add(f"  {name:<16}: {imp:.5f} (rank {rank}/{len(importances)})")
add(f"Split consistency vs baseline RF: {baseline_consistent}")
add(f"Split consistency vs RF+BLAST   : {blast_consistent}")
add("")
add("Classification metrics (test split):")
for name, value in clf_metrics.items():
    add(f"  {name:10s}: {value:.4f}")
add("")

header = f"{'Metric':<10}{'Cosine':<12}{'RF (ESM)':<12}{'RF+BLAST':<12}{'RF+feat2':<12}"
add(header)
add("-" * len(header))
for k in TOP_K:
    add(f"{'Hits@'+str(k):<10}{cosine_test_hits[k]:<12.4f}{baseline_hits[k]:<12.4f}"
        f"{blast_hits[k]:<12.4f}{rf_hits[k]:<12.4f}")
add(f"{'MRR':<10}{cosine_test_mrr:<12.4f}{baseline_mrr:<12.4f}{blast_mrr:<12.4f}{rf_mrr:<12.4f}")

if not np.isnan(blast_mrr):
    delta = rf_mrr - blast_mrr
    add(f"\nDelta vs RF+BLAST: MRR {delta:+.4f} "
        f"({'IMPROVED' if delta > 0 else 'WORSE' if delta < 0 else 'UNCHANGED'})")

with open(SUMMARY_OUTPUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("Done.")
