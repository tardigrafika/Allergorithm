"""
K-fold unakrsna validacija: Cosine vs RF+BLAST vs PU bagging - 1443 dataset.

Do sad je svako poredjenje (cosine MRR 0.181, RF+BLAST 0.203, PU bagging 0.211)
bilo na JEDNOM train/test protein-level split-u. Gold standard je mali (1432
para)
pa razlika od par hiljaditih MRR moze biti sum konkretnog split-a, ne
stvaran signal
Ovaj skript to proverava.

Isti feature-i i isti hiperparametri kao vec validirani skriptovi:
  - RF+BLAST: identicna konfiguracija kao ml/random_forest_blast_1443.py
    (1 model, 300 stabala, jedan izvuceni negativni uzorak)
  - PU bagging: identicna konfiguracija kao ml/random_forest_pu_bagging_1443.py
    (20 bagova, 100 stabala svaki, svez nasumican negativni uzorak po bagu)

Ponavlja se na K=5 nezavisnih, protein-level (group-aware) foldova 
istiUnion-Find leakage-prevention princip kao svuda (nijedan gold par ne sme da
predje granicu folda)

K=5 umesto 10: gold graf ima samo 29 povezanih
komponenti sa >=1 pozitivnim parom  sa 10 foldova bi u proseku bilo ~2-3 komponente po foldu

Za svaki fold: taj fold je test, preostala 4 su train. Retrieval evaluacija
je identicnog protokola kao ranije (upit protiv PUNOG skupa od ~1534 proteina,
ne samo protiv fold-a).

Krajnji rezultat: mean +/- std MRR/Hits@K preko K foldova po metodi, plus
koliko od K foldova PU bagging pobedi RF+BLAST (paired, fold-po-fold poredjenje).

Izlaz:
    output/kfold_comparison_1443_per_fold.csv
    output/kfold_comparison_1443_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS / CONFIG
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PER_FOLD_OUTPUT = OUTPUT_DIR / "kfold_comparison_1443_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "kfold_comparison_1443_summary.txt"

SEED = 42
K_FOLDS = 5
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]

RF_BLAST_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
)

N_BAGS = 20
PU_BAG_PARAMS = dict(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
)


# =====================================================
# LOAD DATA (identical to random_forest_blast_1443.py / random_forest_pu_bagging_1443.py)
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

gold_raw = pd.read_csv(GOLD)

negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()
print(f"Positive gold-standard pairs retained: {len(gold)}")

name_to_id = {}
for _, row in metadata.iterrows():
    official_name = str(row["official_name"]).strip()
    if official_name == "" or official_name.lower() == "nan":
        continue
    if official_name not in name_to_id:
        name_to_id[official_name] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {allergen_id: i for i, allergen_id in enumerate(all_ids)}

embedding_matrix = np.array(
    [embeddings_dict[allergen_id] for allergen_id in all_ids],
    dtype=np.float64,
)
cosine_similarity_matrix = cosine_similarity(embedding_matrix)
print(f"Embedding matrix shape: {embedding_matrix.shape}")

gold_pairs = []
for _, row in gold.iterrows():
    name_1 = str(row["allergen_id_1"]).strip()
    name_2 = str(row["allergen_id_2"]).strip()
    if name_1 not in name_to_id or name_2 not in name_to_id:
        continue
    id_1, id_2 = name_to_id[name_1], name_to_id[name_2]
    if id_1 not in id_to_index or id_2 not in id_to_index or id_1 == id_2:
        continue
    gold_pairs.append({
        "pair_id": row["pair_id"], "id_1": id_1, "id_2": id_2,
        "name_1": name_1, "name_2": name_2,
        "family_1": row["family_1"], "family_2": row["family_2"],
    })

print(f"Mapped gold pairs: {len(gold_pairs)}")
positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}


# =====================================================
# CONNECTED COMPONENTS (identical Union-Find as other *_1443 scripts)
# =====================================================

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

free_proteins = [pid for pid in all_ids if pid not in parent]
print(f"Proteins with no known positive pair (free): {len(free_proteins)}")


# =====================================================
# K-FOLD GROUP-AWARE SPLIT (greedy deficit bin-packing, same idea as
# data/build_ml_dataset.py's train/val/test packer, generalized to K bins)
# =====================================================

def bin_pack_k_folds(items, k, rng):
    items = list(items)
    rng.shuffle(items)
    items.sort(key=len, reverse=True)
    total = sum(len(c) for c in items)
    target = total / k
    bins = [[] for _ in range(k)]
    filled = [0] * k
    for c in items:
        deficit = [target - filled[i] for i in range(k)]
        choice = int(np.argmax(deficit))
        bins[choice].append(c)
        filled[choice] += len(c)
    return bins, filled


split_rng = np.random.default_rng(SEED)
component_bins, component_filled = bin_pack_k_folds(component_list, K_FOLDS, split_rng)

shuffled_free = list(free_proteins)
split_rng.shuffle(shuffled_free)
free_bins = [shuffled_free[i::K_FOLDS] for i in range(K_FOLDS)]

fold_protein_sets = []
for k in range(K_FOLDS):
    fold_set = set()
    for c in component_bins[k]:
        fold_set |= c
    fold_set |= set(free_bins[k])
    fold_protein_sets.append(fold_set)

print("\nFold sizes (proteins):")
for k in range(K_FOLDS):
    n_components_in_fold = len(component_bins[k])
    n_pos_in_fold = sum(
        1 for p in gold_pairs if p["id_1"] in fold_protein_sets[k] and p["id_2"] in fold_protein_sets[k]
    )
    print(f"  fold {k}: {len(fold_protein_sets[k])} proteins, "
          f"{n_components_in_fold} informative components, {n_pos_in_fold} positive pairs")

# sanity: every gold pair's two endpoints must land in the same fold (they're
# always in the same connected component, and components are never split
# across folds) -- verify explicitly, not just assumed
protein_to_fold = {}
for k in range(K_FOLDS):
    for pid in fold_protein_sets[k]:
        protein_to_fold[pid] = k
for p in gold_pairs:
    assert protein_to_fold[p["id_1"]] == protein_to_fold[p["id_2"]], \
        "a gold pair spans two folds (leakage, should be impossible)"
print("\nLeakage check passed: no gold pair spans two folds.")


# =====================================================
# SHARED FEATURE / SAMPLING HELPERS (identical logic to the two prior scripts)
# =====================================================

def sample_unlabeled_pairs(protein_pool, n_needed, seed):
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
    return sorted(unlabeled)


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
        ids_a.append(p["id_1"]); ids_b.append(p["id_2"]); labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        ids_a.append(a); ids_b.append(b); labels.append(0)
    X = pairwise_features(np.array(rows_a), np.array(rows_b), ids_a, ids_b)
    return X, np.array(labels)


def bagged_predict_proba(models, X):
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return probs.mean(axis=0)


# =====================================================
# MAIN K-FOLD LOOP
# =====================================================

print("\n==============================")
print(f"RUNNING {K_FOLDS}-FOLD COMPARISON")
print("==============================")

per_fold_rows = []
overall_start = time.time()

for fold_idx in range(K_FOLDS):
    fold_start = time.time()
    test_ids = fold_protein_sets[fold_idx]
    train_ids = set()
    for other in range(K_FOLDS):
        if other != fold_idx:
            train_ids |= fold_protein_sets[other]

    train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
    test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]

    print(f"\n--- FOLD {fold_idx} ---")
    print(f"Train proteins: {len(train_ids)}  Test proteins: {len(test_ids)}")
    print(f"Train positive pairs: {len(train_positive_pairs)}  Test positive pairs: {len(test_positive_pairs)}")

    # --- negative sampling ---
    n_train_neg = len(train_positive_pairs) * NEG_PER_POS
    train_negatives_single = sample_unlabeled_pairs(train_ids, n_train_neg, seed=SEED + fold_idx)

    # --- train RF+BLAST (single model, identical config to random_forest_blast_1443.py) ---
    X_train_rf, y_train_rf = build_feature_matrix(train_positive_pairs, train_negatives_single)
    rf_blast = RandomForestClassifier(random_state=SEED + fold_idx, **RF_BLAST_PARAMS)
    rf_blast.fit(X_train_rf, y_train_rf)

    # --- train PU bagging (20 bags, identical config to random_forest_pu_bagging_1443.py) ---
    pu_models = []
    for bag_idx in range(N_BAGS):
        bag_seed = SEED + 1000 + fold_idx * 100 + bag_idx
        bag_negatives = sample_unlabeled_pairs(train_ids, n_train_neg, seed=bag_seed)
        X_bag, y_bag = build_feature_matrix(train_positive_pairs, bag_negatives)
        rf_bag = RandomForestClassifier(random_state=bag_seed, **PU_BAG_PARAMS)
        rf_bag.fit(X_bag, y_bag)
        pu_models.append(rf_bag)

    train_elapsed = time.time() - fold_start
    print(f"Training done in {train_elapsed/60:.1f} min (1 RF+BLAST model + {N_BAGS} PU bags)")

    # --- retrieval evaluation: cosine vs RF+BLAST vs PU bagging, same queries ---
    retrieval_start = time.time()
    cos_rr, rf_rr, pu_rr = [], [], []
    cos_hits = {k: [] for k in TOP_K}
    rf_hits = {k: [] for k in TOP_K}
    pu_hits = {k: [] for k in TOP_K}

    for p in test_positive_pairs:
        directions = [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]
        for query_id, target_id in directions:
            query_index = id_to_index[query_id]
            target_index = id_to_index[target_id]
            query_vec = embedding_matrix[query_index]

            X_candidates = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)

            rf_scores = rf_blast.predict_proba(X_candidates)[:, 1]
            rf_scores[query_index] = -np.inf
            rf_rank = int(np.where(np.argsort(rf_scores)[::-1] == target_index)[0][0]) + 1

            pu_scores = bagged_predict_proba(pu_models, X_candidates)
            pu_scores[query_index] = -np.inf
            pu_rank = int(np.where(np.argsort(pu_scores)[::-1] == target_index)[0][0]) + 1

            cos_scores = cosine_similarity_matrix[query_index].copy()
            cos_scores[query_index] = -np.inf
            cos_rank = int(np.where(np.argsort(cos_scores)[::-1] == target_index)[0][0]) + 1

            cos_rr.append(1.0 / cos_rank)
            rf_rr.append(1.0 / rf_rank)
            pu_rr.append(1.0 / pu_rank)
            for k in TOP_K:
                cos_hits[k].append(int(cos_rank <= k))
                rf_hits[k].append(int(rf_rank <= k))
                pu_hits[k].append(int(pu_rank <= k))

    retrieval_elapsed = time.time() - retrieval_start
    n_queries = len(cos_rr)
    print(f"Retrieval done in {retrieval_elapsed/60:.1f} min ({n_queries} queries)")

    fold_row = {
        "fold": fold_idx,
        "n_train_positive_pairs": len(train_positive_pairs),
        "n_test_positive_pairs": len(test_positive_pairs),
        "n_queries": n_queries,
        "cosine_mrr": float(np.mean(cos_rr)),
        "rf_blast_mrr": float(np.mean(rf_rr)),
        "pu_bagging_mrr": float(np.mean(pu_rr)),
    }
    for k in TOP_K:
        fold_row[f"cosine_hits_at_{k}"] = float(np.mean(cos_hits[k]))
        fold_row[f"rf_blast_hits_at_{k}"] = float(np.mean(rf_hits[k]))
        fold_row[f"pu_bagging_hits_at_{k}"] = float(np.mean(pu_hits[k]))
    per_fold_rows.append(fold_row)

    print(f"Fold {fold_idx} MRR -> cosine={fold_row['cosine_mrr']:.4f}  "
          f"RF+BLAST={fold_row['rf_blast_mrr']:.4f}  PU bagging={fold_row['pu_bagging_mrr']:.4f}  "
          f"(fold total: {(time.time()-fold_start)/60:.1f} min)")

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} folds done in {total_elapsed/60:.1f} min")


# =====================================================
# AGGREGATE + SAVE
# =====================================================

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)
print(f"\nPer-fold results saved to: {PER_FOLD_OUTPUT}")

methods = [("cosine", "cosine"), ("rf_blast", "RF+BLAST"), ("pu_bagging", "PU bagging")]

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append(f"{K_FOLDS}-FOLD COMPARISON: Cosine vs RF+BLAST vs PU bagging (1443 dataset)")
summary_lines.append("=" * 70)
summary_lines.append(f"Random seed (base): {SEED}")
summary_lines.append(f"Total runtime: {total_elapsed/60:.1f} min")
summary_lines.append("")
summary_lines.append("Per-fold protein/positive-pair counts:")
for row in per_fold_rows:
    summary_lines.append(f"  fold {row['fold']}: train_pos={row['n_train_positive_pairs']:4d}  "
                          f"test_pos={row['n_test_positive_pairs']:4d}  queries={row['n_queries']:4d}")
summary_lines.append("")
summary_lines.append("MRR mean +/- std across folds:")
for key, label in methods:
    vals = per_fold_df[f"{key}_mrr"].to_numpy()
    summary_lines.append(f"  {label:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}   (per-fold: "
                          + ", ".join(f"{v:.4f}" for v in vals) + ")")
summary_lines.append("")
summary_lines.append("Hits@K mean +/- std across folds:")
for k in TOP_K:
    summary_lines.append(f"  Hits@{k}:")
    for key, label in methods:
        vals = per_fold_df[f"{key}_hits_at_{k}"].to_numpy()
        summary_lines.append(f"    {label:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

pu_vals = per_fold_df["pu_bagging_mrr"].to_numpy()
rf_vals = per_fold_df["rf_blast_mrr"].to_numpy()
cos_vals = per_fold_df["cosine_mrr"].to_numpy()
pu_wins_vs_rf = int((pu_vals > rf_vals).sum())
pu_wins_vs_cos = int((pu_vals > cos_vals).sum())
rf_wins_vs_cos = int((rf_vals > cos_vals).sum())

summary_lines.append("")
summary_lines.append("Paired fold-by-fold win counts (out of {} folds):".format(K_FOLDS))
summary_lines.append(f"  PU bagging beats RF+BLAST on MRR: {pu_wins_vs_rf}/{K_FOLDS} folds")
summary_lines.append(f"  PU bagging beats cosine on MRR   : {pu_wins_vs_cos}/{K_FOLDS} folds")
summary_lines.append(f"  RF+BLAST beats cosine on MRR     : {rf_wins_vs_cos}/{K_FOLDS} folds")

mean_delta = float((pu_vals - rf_vals).mean())
std_delta = float((pu_vals - rf_vals).std(ddof=1))
summary_lines.append("")
summary_lines.append(f"Mean per-fold delta (PU bagging - RF+BLAST) MRR: {mean_delta:+.4f} +/- {std_delta:.4f}")
if pu_wins_vs_rf >= K_FOLDS - 1 and mean_delta > 0:
    verdict = "PU bagging's improvement over RF+BLAST looks STABLE across folds."
elif pu_wins_vs_rf <= 1 and mean_delta < 0:
    verdict = "PU bagging does NOT look better than RF+BLAST once split-noise is accounted for."
else:
    verdict = "Mixed: PU bagging wins on some folds but not consistently -- treat the single-split result with caution."
summary_lines.append(verdict)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
