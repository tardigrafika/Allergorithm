"""
K=5 fold (ne LOCO): da li trening SAMO na proteinima sa poznatom
cross-reaktivnoscu pomaze, i koliko "laksi zadatak" sam po sebi naduva
metriku - 1548 dataset.

Laksa verzija ml/loco_restricted_universe_1548.py (ta je bila prespora -
44 folda x 2 trening uslova x 20 PU bagova x 5 eval uslova = predugo, ubijena
posle skoro 2h bez vidljivog napretka). Ovde:
  - K=5 fold umesto LOCO (44) -- brz, vec testiran protokol
  - single-draw RF+BLAST umesto PU bagging (20 bagova) -- PU bagging je kroz
    celu sesiju statisticki nerazluciv od single RF+BLAST, pa je ovo
    razumno pojednostavljenje za brz signal

Dizajn (korisnicin, A/B):
  - kontrola:    trening na PUNOM svetu (1534), eval na PUNOM pool-u
  - Eksperiment A: trening SAMO na 348 povezanih proteina, eval na PUNOM pool-u
  - Eksperiment B: isti model kao A, eval SAMO na 348 povezanih proteina
  - cosine na oba pool-a: "nulta linija" efekta lakseg zadatka bez treninga

Izlaz:
    output/kfold_restricted_universe_1548_per_fold.csv
    output/kfold_restricted_universe_1548_summary.txt
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
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PER_FOLD_OUTPUT = OUTPUT_DIR / "kfold_restricted_universe_1548_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "kfold_restricted_universe_1548_summary.txt"

SEED = 42
K_FOLDS = 5
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
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
# CONNECTED COMPONENTS + K=5 FOLD SPLIT (bin-pack components only -- free
# proteins never appear in a positive pair, so their fold assignment doesn't
# matter for defining test queries; they're just always-available candidates)
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
connected_universe = set(parent.keys())
connected_universe_sorted = sorted(connected_universe)

print(f"Connected components: {len(component_list)}")
print(f"Proteins in connected universe: {len(connected_universe)}")
print(f"Free proteins (no known pair): {len(free_proteins)}")


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
fold_test_proteins = []
for k in range(K_FOLDS):
    fold_set = set()
    for c in component_bins[k]:
        fold_set |= c
    fold_test_proteins.append(fold_set)

print("\nFold sizes (connected proteins only; free proteins are always in train):")
for k in range(K_FOLDS):
    n_pos = sum(1 for p in gold_pairs if p["id_1"] in fold_test_proteins[k] and p["id_2"] in fold_test_proteins[k])
    print(f"  fold {k}: {len(fold_test_proteins[k])} proteins, {len(component_bins[k])} components, "
          f"{n_pos} positive pairs")


# =====================================================
# SHARED FEATURE / SAMPLING HELPERS
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


def rank_of_target(scores, candidate_pos, query_id, target_id):
    scores = scores.copy()
    query_pos = candidate_pos.get(query_id)
    if query_pos is not None:
        scores[query_pos] = -np.inf
    target_pos = candidate_pos[target_id]
    ranked = np.argsort(scores)[::-1]
    return int(np.where(ranked == target_pos)[0][0]) + 1


# restricted candidate pool -- fixed across all folds, O(1) position lookup
restricted_indices = np.array([id_to_index[i] for i in connected_universe_sorted])
restricted_candidate_embs = embedding_matrix[restricted_indices]
restricted_pos = {aid: i for i, aid in enumerate(connected_universe_sorted)}


# =====================================================
# MAIN K-FOLD LOOP
# =====================================================

print("\n==============================")
print(f"RUNNING {K_FOLDS}-FOLD -- full-world vs restricted-world training")
print("==============================")

per_fold_rows = []
overall_start = time.time()
pooled = {k: [] for k in ["cos_full", "cos_restricted", "full_full", "restricted_full", "restricted_restricted"]}

for fold_idx in range(K_FOLDS):
    fold_start = time.time()
    test_ids = fold_test_proteins[fold_idx]

    train_ids_full = set(all_ids) - test_ids
    train_ids_restricted = connected_universe - test_ids

    train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids_full and p["id_2"] in train_ids_full]
    test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
    n_train_neg = len(train_positive_pairs) * NEG_PER_POS

    train_neg_full = sample_unlabeled_pairs(train_ids_full, n_train_neg, seed=SEED + fold_idx)
    train_neg_restricted = sample_unlabeled_pairs(train_ids_restricted, n_train_neg, seed=SEED + 500 + fold_idx)

    X_full, y_full = build_feature_matrix(train_positive_pairs, train_neg_full)
    model_full = RandomForestClassifier(random_state=SEED + fold_idx, **RF_PARAMS)
    model_full.fit(X_full, y_full)

    X_restricted, y_restricted = build_feature_matrix(train_positive_pairs, train_neg_restricted)
    model_restricted = RandomForestClassifier(random_state=SEED + 500 + fold_idx, **RF_PARAMS)
    model_restricted.fit(X_restricted, y_restricted)

    print(f"  [fold {fold_idx}] train_pos={len(train_positive_pairs)} n_train_neg={n_train_neg} "
          f"restricted_pool_size={len(train_ids_restricted)} -- training done in "
          f"{(time.time()-fold_start)/60:.1f} min", flush=True)

    fold_scores = {k: [] for k in pooled}
    retrieval_start = time.time()

    for p in test_positive_pairs:
        directions = [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]
        for query_id, target_id in directions:
            query_index = id_to_index[query_id]
            query_vec = embedding_matrix[query_index]

            X_full_cand = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)

            cos_scores_full = cosine_similarity_matrix[query_index].copy()
            rank = rank_of_target(cos_scores_full, id_to_index, query_id, target_id)
            fold_scores["cos_full"].append(1.0 / rank)

            scores = model_full.predict_proba(X_full_cand)[:, 1]
            rank = rank_of_target(scores, id_to_index, query_id, target_id)
            fold_scores["full_full"].append(1.0 / rank)

            scores = model_restricted.predict_proba(X_full_cand)[:, 1]
            rank = rank_of_target(scores, id_to_index, query_id, target_id)
            fold_scores["restricted_full"].append(1.0 / rank)

            X_restr_cand = pairwise_features_batch_same_query(
                query_vec, query_id, restricted_candidate_embs, connected_universe_sorted
            )

            cos_scores_restricted = cosine_similarity_matrix[query_index][restricted_indices].copy()
            rank = rank_of_target(cos_scores_restricted, restricted_pos, query_id, target_id)
            fold_scores["cos_restricted"].append(1.0 / rank)

            scores = model_restricted.predict_proba(X_restr_cand)[:, 1]
            rank = rank_of_target(scores, restricted_pos, query_id, target_id)
            fold_scores["restricted_restricted"].append(1.0 / rank)

    for k in pooled:
        pooled[k].extend(fold_scores[k])

    fold_row = {"fold": fold_idx, "n_test_positive_pairs": len(test_positive_pairs), "n_queries": len(fold_scores["cos_full"])}
    for k in pooled:
        fold_row[f"{k}_mrr"] = float(np.mean(fold_scores[k])) if fold_scores[k] else float("nan")
    per_fold_rows.append(fold_row)

    elapsed = time.time() - overall_start
    print(f"  [fold {fold_idx}] retrieval done in {(time.time()-retrieval_start)/60:.1f} min "
          f"({fold_row['n_queries']} queries) -- cos_full={fold_row['cos_full_mrr']:.3f} "
          f"full_full={fold_row['full_full_mrr']:.3f} restr_full={fold_row['restricted_full_mrr']:.3f} "
          f"restr_restr={fold_row['restricted_restricted_mrr']:.3f} ({elapsed/60:.1f} min total elapsed)", flush=True)

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)
print(f"Per-fold results saved to: {PER_FOLD_OUTPUT}")


# =====================================================
# AGGREGATE + SAVE (micro / query-weighted -- established as the reliable number)
# =====================================================

micro = {k: float(np.mean(v)) for k, v in pooled.items()}

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append(f"K={K_FOLDS} fold: full-world vs restricted-world training (1548 dataset)")
summary_lines.append("=" * 70)
summary_lines.append(f"Random seed (base): {SEED}")
summary_lines.append(f"Total runtime: {total_elapsed/60:.1f} min")
summary_lines.append(f"Connected universe: {len(connected_universe)} proteins, {len(component_list)} components")
summary_lines.append("")
summary_lines.append("MICRO (query-weighted) MRR:")
summary_lines.append(f"  cosine,       full pool (1534)       : {micro['cos_full']:.4f}")
summary_lines.append(f"  cosine,       restricted pool (348)  : {micro['cos_restricted']:.4f}   "
                      f"<- 'laksi zadatak' bazna linija, BEZ treninga")
summary_lines.append(f"  RF (full-world train), full pool     : {micro['full_full']:.4f}   <- kontrola")
summary_lines.append(f"  RF (restricted train), full pool     : {micro['restricted_full']:.4f}   <- Eksperiment A")
summary_lines.append(f"  RF (restricted train), restricted pool: {micro['restricted_restricted']:.4f}   <- Eksperiment B")
summary_lines.append("")

delta_easy_task_only = micro["cos_restricted"] - micro["cos_full"]
delta_A = micro["restricted_full"] - micro["full_full"]
delta_B_extra = micro["restricted_restricted"] - micro["restricted_full"]

summary_lines.append(f"Efekat 'sam po sebi lakseg zadatka' (cosine, bez treninga): {delta_easy_task_only:+.4f}")
summary_lines.append(f"Eksperiment A (restricted train vs full train, ISTI eval): {delta_A:+.4f}")
summary_lines.append(f"Eksperiment B dodatni efekat (uzi eval pool, ISTI model) : {delta_B_extra:+.4f}")
summary_lines.append("")
if delta_A > 0.005:
    verdict = ("A pokazuje poboljsanje na ISTOM (teskom) evaluacionom zadatku -- "
               "slobodni proteini VEROVATNO unose sum u trening negative. Vredi validirati sa PU bagging-om/LOCO.")
elif delta_A < -0.005:
    verdict = ("A je LOSIJI od kontrole na istom zadatku -- restrikcija univerzuma "
               "za trening negative je stetna.")
else:
    verdict = ("A ~= kontrola -- restrikcija univerzuma za trening negative ne menja mnogo. "
               "Ako se B ipak popravio, to je iskljucivo efekat lakseg evaluacionog zadatka, ne pravi dobitak.")
summary_lines.append(verdict)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
