"""
PU (Positive-Unlabeled) bagging na RF+BLAST modelu - 1443 dataset.

Isti split, iste BLAST feature-e i isti feature vektor (1283 dim) kao
ml/random_forest_blast_1443.py - jedina razlika je KAKO se koristi "negativni"
skup za trening.

Problem koji ovo resava: svaki dosadasnji negative-sampling pristup (i easy
random i safe cross-family hard mining) pravi JEDAN fiksni izbor negativa i
tretira ga kao da je sigurno tacan. Ali odsustvo dokumentovane cross-reaktivnosti
NIJE isto sto i dokazana odsutnost - neki od tih "negativa" su verovatno
nedokumentovani pravi pozitivi (missing negatives problem).

PU bagging (Mordelet & Vert, 2014) resava ovo bez potrebe da se bilo koji
par proglasi definitivno negativnim:
  - trenira se B nezavisnih RF modela
  - svaki bag vidi SVE pozitive + SVOJ SOPSTVENI random uzorak iz unlabeled
    skupa (razlicit seed po bagu), tretiran kao negativan SAMO za taj bag
  - finalni skor = prosek verovatnoca preko svih B modela

Ako je neki "negativ" zapravo nedokumentovan pravi pozitiv, to ce ga jedan
konkretan bag pogresno oznaciti - ali on ce se pojavljivati kao negativ u
razlicitom nasumicnom podskupu u svakom bagu, pa greska jednog bag-a ne
dominira finalni (usrednjeni) skor. Bonus: razilazenje predikcija izmedju
bagova (std) je besplatna mera neizvesnosti po paru.

Izlaz:
    output/random_forest_pu_bagging_models_1443.joblib   (lista od N_BAGS modela)
    output/random_forest_pu_bagging_retrieval_results_1443.csv
    output/random_forest_pu_bagging_summary_1443.txt
"""

import pickle
import time
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

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
MODELS_OUTPUT = OUTPUT_DIR / "random_forest_pu_bagging_models_1443.joblib"
RETRIEVAL_OUTPUT = OUTPUT_DIR / "random_forest_pu_bagging_retrieval_results_1443.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "random_forest_pu_bagging_summary_1443.txt"

BLAST_RESULTS = Path("/home/lana/ALERGRAF/output/random_forest_blast_retrieval_results_1443.csv")
BLAST_SUMMARY = Path("/home/lana/ALERGRAF/output/random_forest_blast_summary_1443.txt")


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10  # isto kao random_forest_blast_1443.py, ali sad se izvlaci NEZAVISNO po bagu
TOP_K = [1, 5, 10, 20]

N_BAGS = 20
# manje stabala po bagu nego solo RF (300) - diverzitet sad dolazi i od
# razlicitog negativnog uzorka po bagu, ne samo od bootstrap-a unutar stabala.
# ukupno N_BAGS * 100 = 2000 stabala preko celog ansambla.
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
)


# =====================================================
# LOAD DATA (identical to random_forest_blast_1443.py)
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
# GROUP-AWARE PROTEIN-LEVEL SPLIT (identical algorithm to random_forest_blast_1443.py)
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
# UNLABELED SAMPLING (per-bag, different seed each time)
# =====================================================

def sample_unlabeled_pairs(protein_pool, n_needed, seed):
    """Isto kao negative sampling ranije, ali se poziva jednom PO BAGU sa
    drugim seed-om - to je sustina PU bagging-a (svaki bag vidi drugaciji
    nasumicni podskup unlabeled prostora tretiran kao 'negativan')."""
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    unlabeled = set()
    max_attempts = n_needed * 50 + 2000
    attempts = 0
    while len(unlabeled) < n_needed and attempts < max_attempts:
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1
        if pair in positive_pair_set or pair in unlabeled:
            continue
        unlabeled.add(pair)
    if len(unlabeled) < n_needed:
        print(f"WARNING: only sampled {len(unlabeled)}/{n_needed} unlabeled pairs")
    return sorted(unlabeled)


# fiksni test negativni uzorak (isti princip kao ranije) - test metrika mora
# da bude uporediva izmedju skripti, PU bagging menja samo TRENING stranu
n_test_neg = len(test_positive_pairs) * NEG_PER_POS
test_negative_pairs = sample_unlabeled_pairs(test_ids, n_test_neg, seed=SEED + 1)
print(f"\nTest negative pairs sampled (fixed, for comparability): {len(test_negative_pairs)}")


# =====================================================
# FEATURE CONSTRUCTION (identical to random_forest_blast_1443.py)
# =====================================================

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

    return np.hstack([abs_diff, cosine.reshape(-1, 1), blast_id.reshape(-1, 1), blast_sc.reshape(-1, 1)])


def pairwise_features_batch_same_query(query_emb, query_id, candidate_embs, candidate_ids):
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


X_test, y_test = build_feature_matrix(test_positive_pairs, test_negative_pairs)
print(f"\nTest: X={X_test.shape}, positives={int(y_test.sum())}, negatives={int((y_test == 0).sum())}")


# =====================================================
# TRAIN PU BAGGING ENSEMBLE
# =====================================================

print("\n==============================")
print(f"TRAINING PU BAGGING ENSEMBLE ({N_BAGS} bags)")
print("==============================")
print(f"Per-bag RF hyperparameters: {RF_PARAMS}")

n_train_neg = len(train_positive_pairs) * NEG_PER_POS
bag_models = []
bag_train_sizes = []

start = time.time()
for bag_idx in range(N_BAGS):
    bag_seed = SEED + 1000 + bag_idx  # razlicit seed po bagu -> razlicit unlabeled uzorak
    bag_negatives = sample_unlabeled_pairs(train_ids, n_train_neg, seed=bag_seed)
    X_bag, y_bag = build_feature_matrix(train_positive_pairs, bag_negatives)
    bag_train_sizes.append(len(y_bag))

    rf = RandomForestClassifier(random_state=bag_seed, **RF_PARAMS)
    rf.fit(X_bag, y_bag)
    bag_models.append(rf)
    print(f"  bag {bag_idx + 1}/{N_BAGS} trained on {len(y_bag)} rows "
          f"({int(y_bag.sum())} pos / {int((y_bag == 0).sum())} unlabeled-as-neg)")

elapsed_train = time.time() - start
print(f"\nAll {N_BAGS} bags trained in {elapsed_train/60:.1f} min")


def bagged_predict_proba(X):
    """Prosek P(y=1) preko svih bagova. Vraca i std (mera neizvesnosti)."""
    probs = np.stack([m.predict_proba(X)[:, 1] for m in bag_models], axis=0)
    return probs.mean(axis=0), probs.std(axis=0)


# feature importance: prosek preko bagova, gde rangiraju BLAST kolone
mean_importances = np.mean([m.feature_importances_ for m in bag_models], axis=0)
blast_identity_rank = int((mean_importances > mean_importances[-2]).sum()) + 1
blast_score_rank = int((mean_importances > mean_importances[-1]).sum()) + 1
print(f"\nBLAST identity feature importance (avg over bags): {mean_importances[-2]:.5f}  "
      f"(rank {blast_identity_rank}/{len(mean_importances)})")
print(f"BLAST score feature importance   (avg over bags): {mean_importances[-1]:.5f}  "
      f"(rank {blast_score_rank}/{len(mean_importances)})")


# =====================================================
# A) CLASSIFICATION METRICS
# =====================================================

print("\n==============================")
print("CLASSIFICATION METRICS (test split, bagged average)")
print("==============================")

y_proba, y_proba_std = bagged_predict_proba(X_test)
y_pred = (y_proba >= 0.5).astype(int)

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
print(f"\nMean inter-bag std of P(y=1) on test set: {y_proba_std.mean():.4f} "
      f"(vece = model manje siguran/bagovi se vise ne slazu)")


# =====================================================
# B) RETRIEVAL EVALUATION
# =====================================================

print("\n==============================")
print("RETRIEVAL EVALUATION (Hits@K / MRR)")
print("==============================")
print(f"Test-split gold pairs: {len(test_positive_pairs)}  "
      f"-> up to {2 * len(test_positive_pairs)} retrieval queries")

retrieval_results = []
retrieval_start = time.time()

for qi, p in enumerate(test_positive_pairs):
    directions = [
        (p["id_1"], p["id_2"], p["name_1"], p["name_2"], p["family_1"], p["family_2"]),
        (p["id_2"], p["id_1"], p["name_2"], p["name_1"], p["family_2"], p["family_1"]),
    ]

    for query_id, target_id, query_name, target_name, family_q, family_t in directions:
        query_index = id_to_index[query_id]
        target_index = id_to_index[target_id]

        query_vec = embedding_matrix[query_index]
        # feature matrix racunata JEDNOM, deljena preko svih bagova (bagovi se
        # razlikuju samo po treniranom modelu, ne po feature-ima)
        X_candidates = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)
        pu_scores, pu_scores_std = bagged_predict_proba(X_candidates)
        pu_scores[query_index] = -np.inf

        pu_ranked = np.argsort(pu_scores)[::-1]
        pu_rank = int(np.where(pu_ranked == target_index)[0][0]) + 1
        pu_reciprocal_rank = 1.0 / pu_rank
        pu_true_pair_probability = pu_scores[target_index]
        pu_true_pair_std = pu_scores_std[target_index]

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
            "pu_probability": pu_true_pair_probability,
            "pu_probability_std": pu_true_pair_std,
            "pu_rank": pu_rank,
            "pu_reciprocal_rank": pu_reciprocal_rank,
            "pu_hits_at_1": int(pu_rank <= 1),
            "pu_hits_at_5": int(pu_rank <= 5),
            "pu_hits_at_10": int(pu_rank <= 10),
            "pu_hits_at_20": int(pu_rank <= 20),
            "cosine_rank": cos_rank,
            "cosine_reciprocal_rank": cos_reciprocal_rank,
            "cosine_hits_at_1": int(cos_rank <= 1),
            "cosine_hits_at_5": int(cos_rank <= 5),
            "cosine_hits_at_10": int(cos_rank <= 10),
            "cosine_hits_at_20": int(cos_rank <= 20),
        })

    if (qi + 1) % 50 == 0 or (qi + 1) == len(test_positive_pairs):
        elapsed = time.time() - retrieval_start
        print(f"  {qi + 1}/{len(test_positive_pairs)} pairs processed ({elapsed/60:.1f} min elapsed)")

retrieval_df = pd.DataFrame(retrieval_results)

pu_hits = {k: retrieval_df[f"pu_hits_at_{k}"].mean() for k in TOP_K}
pu_mrr = retrieval_df["pu_reciprocal_rank"].mean()
cosine_test_hits = {k: retrieval_df[f"cosine_hits_at_{k}"].mean() for k in TOP_K}
cosine_test_mrr = retrieval_df["cosine_reciprocal_rank"].mean()

print(f"\nRetrieval queries evaluated: {len(retrieval_df)}")
print(f"{'Metric':<10}{'Cosine (same test)':<20}{'PU bagging':<20}")
for k in TOP_K:
    print(f"Hits@{k:<5d}{cosine_test_hits[k]:<20.4f}{pu_hits[k]:<20.4f}")
print(f"{'MRR':<10}{cosine_test_mrr:<20.4f}{pu_mrr:<20.4f}")


# =====================================================
# SAVE OUTPUTS
# =====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retrieval_df.to_csv(RETRIEVAL_OUTPUT, index=False)
print(f"\nRetrieval results saved to: {RETRIEVAL_OUTPUT}")

joblib.dump(bag_models, MODELS_OUTPUT)
print(f"Models (list of {N_BAGS}) saved to: {MODELS_OUTPUT}")


# =====================================================
# COMPARISON WITH RF+BLAST (single-draw negative sampling)
# =====================================================

blast_available = BLAST_RESULTS.exists()
blast_hits = {k: float("nan") for k in TOP_K}
blast_mrr = float("nan")
split_consistent = None

if blast_available:
    blast_df = pd.read_csv(BLAST_RESULTS)
    if len(blast_df) == len(retrieval_df):
        merged = retrieval_df.merge(
            blast_df[["pair_id", "query_allergen", "cosine_reciprocal_rank"]],
            on=["pair_id", "query_allergen"], suffixes=("_this_script", "_blast_script"),
        )
        split_consistent = bool(np.allclose(
            merged["cosine_reciprocal_rank_this_script"], merged["cosine_reciprocal_rank_blast_script"]
        ))
        blast_hits = {k: blast_df[f"rf_hits_at_{k}"].mean() for k in TOP_K}
        blast_mrr = blast_df["rf_reciprocal_rank"].mean()

print(f"\nRF+BLAST (single-draw) results file found: {blast_available}")
if blast_available:
    print(f"Split consistency check vs RF+BLAST script: {split_consistent}")


# =====================================================
# FINAL SUMMARY
# =====================================================

summary_lines = []
summary_lines.append("=" * 60)
summary_lines.append("PU BAGGING RF + BLAST (1443 dataset) - SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Random seed (base)       : {SEED}")
summary_lines.append(f"Bags                     : {N_BAGS}")
summary_lines.append(f"Per-bag n_estimators     : {RF_PARAMS['n_estimators']} "
                      f"(total ensemble = {N_BAGS * RF_PARAMS['n_estimators']} trees)")
summary_lines.append(f"Per-bag negatives        : {n_train_neg} (fresh random draw per bag, "
                      f"different seed each time)")
summary_lines.append(f"Positive gold-standard pairs retained: {len(gold)}")
summary_lines.append(f"Feature vector: 1280 (abs_diff) + cosine + blast_identity + blast_score = "
                      f"{X_test.shape[1]} dims")
summary_lines.append(f"BLAST identity feature importance (avg over bags): {mean_importances[-2]:.5f} "
                      f"(rank {blast_identity_rank}/{len(mean_importances)})")
summary_lines.append(f"BLAST score feature importance   (avg over bags): {mean_importances[-1]:.5f} "
                      f"(rank {blast_score_rank}/{len(mean_importances)})")
summary_lines.append("")
summary_lines.append("Split strategy: group-aware protein-level split, identical to "
                      "ml/random_forest_blast_1443.py.")
summary_lines.append(f"  Train positive pairs  : {len(train_positive_pairs)}")
summary_lines.append(f"  Test positive pairs   : {len(test_positive_pairs)}")
if split_consistent is not None:
    summary_lines.append(f"  Split consistency check vs RF+BLAST: {split_consistent}")
summary_lines.append("")
summary_lines.append("Classification metrics (test split, bagged average probability):")
for name, value in clf_metrics.items():
    summary_lines.append(f"  {name:10s}: {value:.4f}")
summary_lines.append(f"  confusion matrix [ [TN FP] [FN TP] ]: {conf_matrix.tolist()}")
summary_lines.append(f"  mean inter-bag std of P(y=1) on test set: {y_proba_std.mean():.4f}")
summary_lines.append("")
summary_lines.append(f"Retrieval evaluation: {len(retrieval_df)} queries "
                      f"({len(test_positive_pairs)} test pairs x 2 directions)")
summary_lines.append("")

header = f"{'Metric':<10}{'Cosine (same test)':<20}{'RF+BLAST (single)':<20}{'PU bagging':<20}"
summary_lines.append(header)
summary_lines.append("-" * len(header))
for k in TOP_K:
    summary_lines.append(
        f"{'Hits@' + str(k):<10}{cosine_test_hits[k]:<20.4f}"
        f"{blast_hits[k]:<20.4f}{pu_hits[k]:<20.4f}"
    )
summary_lines.append(
    f"{'MRR':<10}{cosine_test_mrr:<20.4f}{blast_mrr:<20.4f}{pu_mrr:<20.4f}"
)

if blast_available:
    delta_mrr = pu_mrr - blast_mrr
    summary_lines.append(f"\nDelta vs RF+BLAST (single-draw negatives): MRR {delta_mrr:+.4f}")
    verdict = "IMPROVED" if delta_mrr > 0 else ("WORSE" if delta_mrr < 0 else "UNCHANGED")
    summary_lines.append(f"PU bagging {verdict} retrieval vs single-draw RF+BLAST on this dataset.")
else:
    summary_lines.append(f"\nNOTE: {BLAST_RESULTS} not found -- run ml/random_forest_blast_1443.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
