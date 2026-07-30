"""
Leave-one-component-out (LOCO): Cosine vs RF+BLAST - 1512 dataset.

Zasto: 5-fold poredjenje (ml/kfold_cosine_rf_pu_1512.py) je pokazalo da
standardna greska proseka MRR-a preko K=5 foldova (std/sqrt(K) ~ 0.038/2.24
~= 0.017) NADMASUJE razlike izmedju metoda koje pokusavamo da detektujemo
(~0.005-0.01) -- nemoguce je bilo sta zakljuciti sa tolikom greskom.

LOCO to resava BEZ ijednog novog kurirranog para: umesto K=5 foldova (svaki
~4-8 komponenti), koristi SVIH 36 trenutnih povezanih komponenti kao 36
odvojenih foldova (jedna komponenta = test, ostatak = train). Standardna
greska pada na std/sqrt(36) ~= std/6 -- ~2.7x precizniji rezultat sa ISTIM
podacima, samo temeljitijom evaluacijom.

PU bagging je namerno izbacen iz ovog run-a: 5-fold rezultat je vec pokazao
da je statisticki nerazluciv od RF+BLAST (delta manja od std), pa bi 20x
ponavljanje po foldu (sad 36 foldova) odužilo run sa ~20-25 min na >2h bez
ikakve nove informacije.

Izvestava DVA proseka:
  - "macro" (unweighted mean-of-fold-means preko 36 foldova) -- ovo je broj
    relevantan za std/sqrt(K) racun gore
  - "micro" (query-weighted, sve upite spoji pa uzme jedan MRR) -- korisno
    jer neke komponente imaju samo 1-2 para (2-4 upita), pa im "macro"
    prosek daje isti uticaj kao komponenti sa desetinama parova

Izlaz:
    output/loco_1512_per_fold.csv
    output/loco_1512_summary.txt
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
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1512.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PER_FOLD_OUTPUT = OUTPUT_DIR / "loco_1512_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "loco_1512_summary.txt"

SEED = 42
NEG_PER_POS = 10
TOP_K = [1, 5, 10, 20]

RF_BLAST_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    n_jobs=-1,
)


# =====================================================
# LOAD DATA (identical to kfold_cosine_rf_pu_1512.py)
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
# CONNECTED COMPONENTS (each one becomes its own fold)
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
K_FOLDS = len(component_list)
print(f"Connected components (= number of LOCO folds): {K_FOLDS}")
print(f"Free proteins (no known positive pair, added to every train pool): {len(free_proteins)}")


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
# MAIN LOCO LOOP
# =====================================================

print("\n==============================")
print(f"RUNNING LOCO ({K_FOLDS} folds, 1 per component)")
print("==============================")

per_fold_rows = []
overall_start = time.time()
all_cos_rr, all_rf_rr = [], []

for fold_idx, held_out in enumerate(component_list):
    fold_start = time.time()
    test_ids = held_out
    train_ids = set(free_proteins)
    for j, c in enumerate(component_list):
        if j != fold_idx:
            train_ids |= c

    train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
    test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]

    n_train_neg = len(train_positive_pairs) * NEG_PER_POS
    train_negatives = sample_unlabeled_pairs(train_ids, n_train_neg, seed=SEED + fold_idx)

    X_train, y_train = build_feature_matrix(train_positive_pairs, train_negatives)
    rf_blast = RandomForestClassifier(random_state=SEED + fold_idx, **RF_BLAST_PARAMS)
    rf_blast.fit(X_train, y_train)

    cos_rr, rf_rr = [], []
    cos_hits = {k: [] for k in TOP_K}
    rf_hits = {k: [] for k in TOP_K}

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

            cos_scores = cosine_similarity_matrix[query_index].copy()
            cos_scores[query_index] = -np.inf
            cos_rank = int(np.where(np.argsort(cos_scores)[::-1] == target_index)[0][0]) + 1

            cos_rr.append(1.0 / cos_rank)
            rf_rr.append(1.0 / rf_rank)
            for k in TOP_K:
                cos_hits[k].append(int(cos_rank <= k))
                rf_hits[k].append(int(rf_rank <= k))

    all_cos_rr.extend(cos_rr)
    all_rf_rr.extend(rf_rr)

    fold_row = {
        "fold": fold_idx,
        "component_size": len(held_out),
        "n_test_positive_pairs": len(test_positive_pairs),
        "n_queries": len(cos_rr),
        "cosine_mrr": float(np.mean(cos_rr)) if cos_rr else float("nan"),
        "rf_blast_mrr": float(np.mean(rf_rr)) if rf_rr else float("nan"),
    }
    for k in TOP_K:
        fold_row[f"cosine_hits_at_{k}"] = float(np.mean(cos_hits[k])) if cos_hits[k] else float("nan")
        fold_row[f"rf_blast_hits_at_{k}"] = float(np.mean(rf_hits[k])) if rf_hits[k] else float("nan")
    per_fold_rows.append(fold_row)

    if (fold_idx + 1) % 5 == 0 or (fold_idx + 1) == K_FOLDS:
        elapsed = time.time() - overall_start
        print(f"  fold {fold_idx + 1}/{K_FOLDS} done (size={len(held_out)}, queries={len(cos_rr)}) "
              f"-- cosine={fold_row['cosine_mrr']:.4f} rf_blast={fold_row['rf_blast_mrr']:.4f} "
              f"({elapsed/60:.1f} min elapsed)")

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)
print(f"Per-fold results saved to: {PER_FOLD_OUTPUT}")


# =====================================================
# AGGREGATE + SAVE
# =====================================================

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
rf_macro = per_fold_df["rf_blast_mrr"].to_numpy()

cos_macro_mean, cos_macro_std = cos_macro.mean(), cos_macro.std(ddof=1)
rf_macro_mean, rf_macro_std = rf_macro.mean(), rf_macro.std(ddof=1)
cos_macro_se = cos_macro_std / np.sqrt(K_FOLDS)
rf_macro_se = rf_macro_std / np.sqrt(K_FOLDS)

cos_micro = float(np.mean(all_cos_rr))
rf_micro = float(np.mean(all_rf_rr))

paired_delta = rf_macro - cos_macro
wins = int((paired_delta > 0).sum())

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append(f"LEAVE-ONE-COMPONENT-OUT ({K_FOLDS} folds): Cosine vs RF+BLAST (1512 dataset)")
summary_lines.append("=" * 70)
summary_lines.append(f"Random seed (base): {SEED}")
summary_lines.append(f"Total runtime: {total_elapsed/60:.1f} min")
summary_lines.append("")
summary_lines.append("MACRO (unweighted mean across the 36 component-folds -- each component "
                      "counts equally regardless of size):")
summary_lines.append(f"  cosine     MRR: {cos_macro_mean:.4f} +/- {cos_macro_std:.4f}  "
                      f"(standard error of mean: {cos_macro_se:.4f})")
summary_lines.append(f"  RF+BLAST   MRR: {rf_macro_mean:.4f} +/- {rf_macro_std:.4f}  "
                      f"(standard error of mean: {rf_macro_se:.4f})")
summary_lines.append("")
summary_lines.append("MICRO (query-weighted -- all queries pooled into one MRR, big components "
                      "count more, matches how a single 80/20 split would have counted them):")
summary_lines.append(f"  cosine     MRR: {cos_micro:.4f}")
summary_lines.append(f"  RF+BLAST   MRR: {rf_micro:.4f}")
summary_lines.append("")
summary_lines.append(f"Paired per-component wins (RF+BLAST > cosine): {wins}/{K_FOLDS} components")
mean_delta = float(paired_delta.mean())
se_delta = float(paired_delta.std(ddof=1) / np.sqrt(K_FOLDS))
summary_lines.append(f"Mean paired delta (RF+BLAST - cosine): {mean_delta:+.4f} (SE {se_delta:.4f})")
if abs(mean_delta) > 2 * se_delta:
    verdict = "Delta is more than 2 standard errors from zero -- looks like a REAL, statistically distinguishable effect."
else:
    verdict = "Delta is within ~2 standard errors of zero -- still not statistically distinguishable from noise."
summary_lines.append(verdict)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
