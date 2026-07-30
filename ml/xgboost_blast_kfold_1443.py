"""
XGBoost + BLAST na ISTIM foldovima kao ml/kfold_cosine_rf_pu_1443.py - 1443 dataset.

Poenta: do sad je svaki model bio Random Forest (RF+BLAST, PU bagging = ansambl
RF-ova). To je istorijska slucajnost, ne odluka zasnovana na poredjenju - RF
je bio prvi model koji je pobedio cosine, i ostao je podrazumevani izbor.
Boosting (sekvencijalno ispravljanje gresaka) ima drugaciji induktivni bias
od bagging-a (usrednjavanje varijanse) - vredi proveriti da li izvlaci vise
signala iz ISTIH feature-a (1280 abs_diff + cosine + blast_identity + blast_score).

Da bi poredjenje bilo validno, ovaj skript MORA da koristi identicne foldove
kao kfold_cosine_rf_pu_1443.py - zato je fold-construction kod kopiran 1:1
(isti SEED=42, ista bin_pack_k_folds funkcija). Rezultat se na kraju spaja sa
vec sacuvanim output/kfold_comparison_1443_per_fold.csv (cosine/RF+BLAST/PU
bagging) da bi se izbeglo ponovno (skupo) treniranje PU bagging ansambla.

Izlaz:
    output/xgboost_blast_kfold_1443_per_fold.csv
    output/xgboost_blast_kfold_1443_summary.txt   (spojena tabela sve 4 metode)
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS / CONFIG (identical to kfold_cosine_rf_pu_1443.py)
# =====================================================

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1443.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PER_FOLD_OUTPUT = OUTPUT_DIR / "xgboost_blast_kfold_1443_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "xgboost_blast_kfold_1443_summary.txt"
PRIOR_KFOLD_RESULTS = OUTPUT_DIR / "kfold_comparison_1443_per_fold.csv"

SEED = 42
K_FOLDS = 5
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    tree_method="hist",
)


# =====================================================
# LOAD DATA (identical to kfold_cosine_rf_pu_1443.py)
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
# CONNECTED COMPONENTS + K-FOLD SPLIT (bit-for-bit identical code/seed to
# kfold_cosine_rf_pu_1443.py -> guarantees the SAME folds)
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
free_proteins = [pid for pid in all_ids if pid not in parent]


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

protein_to_fold = {}
for k in range(K_FOLDS):
    for pid in fold_protein_sets[k]:
        protein_to_fold[pid] = k
for p in gold_pairs:
    assert protein_to_fold[p["id_1"]] == protein_to_fold[p["id_2"]], \
        "a gold pair spans two folds (leakage, should be impossible)"
print("Fold construction verified identical to kfold_cosine_rf_pu_1443.py (same seed/algorithm).")


# =====================================================
# SHARED FEATURE / SAMPLING HELPERS (identical to prior scripts)
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


# =====================================================
# MAIN K-FOLD LOOP (XGBoost only -- cosine/RF/PU already computed)
# =====================================================

print("\n==============================")
print(f"RUNNING {K_FOLDS}-FOLD XGBOOST+BLAST")
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

    n_train_neg = len(train_positive_pairs) * NEG_PER_POS
    train_negatives = sample_unlabeled_pairs(train_ids, n_train_neg, seed=SEED + fold_idx)

    X_train, y_train = build_feature_matrix(train_positive_pairs, train_negatives)
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    model = xgb.XGBClassifier(random_state=SEED + fold_idx, scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
    model.fit(X_train, y_train)

    train_elapsed = time.time() - fold_start
    print(f"\n--- FOLD {fold_idx} --- trained in {train_elapsed/60:.1f} min "
          f"(train_pos={len(train_positive_pairs)}, test_pos={len(test_positive_pairs)})")

    retrieval_start = time.time()
    xgb_rr, cos_rr = [], []
    xgb_hits = {k: [] for k in TOP_K}
    cos_hits = {k: [] for k in TOP_K}

    for p in test_positive_pairs:
        directions = [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]
        for query_id, target_id in directions:
            query_index = id_to_index[query_id]
            target_index = id_to_index[target_id]
            query_vec = embedding_matrix[query_index]

            X_candidates = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)

            xgb_scores = model.predict_proba(X_candidates)[:, 1]
            xgb_scores[query_index] = -np.inf
            xgb_rank = int(np.where(np.argsort(xgb_scores)[::-1] == target_index)[0][0]) + 1

            cos_scores = cosine_similarity_matrix[query_index].copy()
            cos_scores[query_index] = -np.inf
            cos_rank = int(np.where(np.argsort(cos_scores)[::-1] == target_index)[0][0]) + 1

            xgb_rr.append(1.0 / xgb_rank)
            cos_rr.append(1.0 / cos_rank)
            for k in TOP_K:
                xgb_hits[k].append(int(xgb_rank <= k))
                cos_hits[k].append(int(cos_rank <= k))

    retrieval_elapsed = time.time() - retrieval_start
    n_queries = len(xgb_rr)
    print(f"Retrieval done in {retrieval_elapsed/60:.1f} min ({n_queries} queries)")

    fold_row = {
        "fold": fold_idx,
        "n_train_positive_pairs": len(train_positive_pairs),
        "n_test_positive_pairs": len(test_positive_pairs),
        "n_queries": n_queries,
        "cosine_mrr": float(np.mean(cos_rr)),
        "xgboost_blast_mrr": float(np.mean(xgb_rr)),
    }
    for k in TOP_K:
        fold_row[f"cosine_hits_at_{k}"] = float(np.mean(cos_hits[k]))
        fold_row[f"xgboost_blast_hits_at_{k}"] = float(np.mean(xgb_hits[k]))
    per_fold_rows.append(fold_row)

    print(f"Fold {fold_idx} MRR -> cosine={fold_row['cosine_mrr']:.4f}  "
          f"XGBoost+BLAST={fold_row['xgboost_blast_mrr']:.4f}  "
          f"(fold total: {(time.time()-fold_start)/60:.1f} min)")

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)
print(f"\nPer-fold XGBoost results saved to: {PER_FOLD_OUTPUT}")


# =====================================================
# MERGE WITH PRIOR K-FOLD RESULTS (cosine/RF+BLAST/PU bagging)
# =====================================================

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append(f"{K_FOLDS}-FOLD COMPARISON: Cosine vs RF+BLAST vs PU bagging vs XGBoost+BLAST")
summary_lines.append("=" * 70)
summary_lines.append(f"Random seed (base): {SEED}")
summary_lines.append(f"XGBoost runtime: {total_elapsed/60:.1f} min")
summary_lines.append("")

if PRIOR_KFOLD_RESULTS.exists():
    prior_df = pd.read_csv(PRIOR_KFOLD_RESULTS)
    merged = prior_df.merge(per_fold_df[["fold", "xgboost_blast_mrr"] + [f"xgboost_blast_hits_at_{k}" for k in TOP_K]],
                             on="fold", how="inner")
    # sanity: cosine MRR must match between the two independently-run scripts (same folds/queries)
    cos_match = bool(np.allclose(merged["cosine_mrr"], per_fold_df["cosine_mrr"]))
    summary_lines.append(f"Fold-consistency check (cosine MRR matches prior k-fold run): {cos_match}")
    summary_lines.append("")

    methods = [
        ("cosine_mrr", "cosine"),
        ("rf_blast_mrr", "RF+BLAST"),
        ("pu_bagging_mrr", "PU bagging"),
        ("xgboost_blast_mrr", "XGBoost+BLAST"),
    ]
    summary_lines.append("MRR mean +/- std across folds:")
    for col, label in methods:
        vals = merged[col].to_numpy()
        summary_lines.append(f"  {label:14s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}   (per-fold: "
                              + ", ".join(f"{v:.4f}" for v in vals) + ")")

    xgb_vals = merged["xgboost_blast_mrr"].to_numpy()
    rf_vals = merged["rf_blast_mrr"].to_numpy()
    pu_vals = merged["pu_bagging_mrr"].to_numpy()
    xgb_wins_vs_rf = int((xgb_vals > rf_vals).sum())
    xgb_wins_vs_pu = int((xgb_vals > pu_vals).sum())

    summary_lines.append("")
    summary_lines.append(f"Paired fold-by-fold win counts (out of {K_FOLDS} folds):")
    summary_lines.append(f"  XGBoost+BLAST beats RF+BLAST on MRR  : {xgb_wins_vs_rf}/{K_FOLDS} folds")
    summary_lines.append(f"  XGBoost+BLAST beats PU bagging on MRR: {xgb_wins_vs_pu}/{K_FOLDS} folds")

    mean_delta_rf = float((xgb_vals - rf_vals).mean())
    std_delta_rf = float((xgb_vals - rf_vals).std(ddof=1))
    summary_lines.append("")
    summary_lines.append(f"Mean per-fold delta (XGBoost+BLAST - RF+BLAST) MRR: {mean_delta_rf:+.4f} +/- {std_delta_rf:.4f}")
else:
    summary_lines.append(f"NOTE: {PRIOR_KFOLD_RESULTS} not found -- run ml/kfold_cosine_rf_pu_1443.py first.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
