"""
LOCO: Cosine vs RF+BLAST vs RF+BLAST+FoldseekTM - 1548 dataset.

Foldseek (van Kempen et al. 2023) je zamenio spori tmtools/TM-align pristup
koji je satima blokirao lokalno i na VM (procena i do 16 dana zbog VM
core-count problema). Foldseek all-vs-all na svih 1240 AlphaFold struktura:
43.7 SEKUNDI (output/foldseek_tmscore_1548.tsv -> lookup dict izgradjen
pomocu data/build_foldseek_tmscore_lookup.py).

VAZNO metodolosko upozorenje pre citanja rezultata: RF+BLAST je vec testiran
kroz LOCO/micro na ovom istom dataset-u i NIJE se pokazao bolji od cosine-a
(micro MRR 0.1153 vs 0.1163, statisticki nerazluciv) - na Confirmed+Strong
podskupu je bio CAK GORI (0.2071 vs 0.2210). Foldseek TM-score je GENUINSKI
drugaciji tip signala (3D strukturna slicnost, ne sekvenca/embedding) pa
vredi testirati nezavisno od tog neuspeha, ali ne treba a priori ocekivati
da ce "samo dodavanje jos jednog feature-a" pomoci - to smo vec vidali da
ne radi (kmer feature preko BLAST-a, same_family cirkularan itd).

LOCO (leave-one-component-out) preko svih povezanih komponenti 1548 dataset-a,
isti protokol kao svuda: standardna greska pada sa brojem foldova, micro
(query-weighted) MRR je najpouzdaniji broj za zakljucak.

Izlaz:
    output/loco_rf_blast_foldseektm_1548_per_fold.csv
    output/loco_rf_blast_foldseektm_1548_summary.txt
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
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PER_FOLD_OUTPUT = OUTPUT_DIR / "loco_rf_blast_foldseektm_1548_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "loco_rf_blast_foldseektm_1548_summary.txt"

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

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)
print(f"Foldseek TM-score lookup: {len(foldseek_lookup)} unordered pairs")


def foldseek_tm(id_a, id_b):
    return foldseek_lookup.get(frozenset((id_a, id_b)), 0.0)  # 0.0 = nema detektovane strukturne slicnosti

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
    foldseek_tm_arr = np.array([foldseek_tm(a, b) for a, b in zip(ids_a, ids_b)])
    return np.hstack([abs_diff, cosine.reshape(-1, 1), blast_id.reshape(-1, 1),
                       blast_sc.reshape(-1, 1), foldseek_tm_arr.reshape(-1, 1)])


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
all_cos_rr, all_rf_rr, all_rf_fs_rr = [], [], []

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

    X_train_full, y_train = build_feature_matrix(train_positive_pairs, train_negatives)
    X_train_blast = X_train_full[:, :-1]  # drop foldseek_tm column -> RF+BLAST only, same as before

    rf_blast = RandomForestClassifier(random_state=SEED + fold_idx, **RF_BLAST_PARAMS)
    rf_blast.fit(X_train_blast, y_train)

    rf_fs = RandomForestClassifier(random_state=SEED + 500 + fold_idx, **RF_BLAST_PARAMS)
    rf_fs.fit(X_train_full, y_train)

    cos_rr, rf_rr, rf_fs_rr = [], [], []
    cos_hits = {k: [] for k in TOP_K}
    rf_hits = {k: [] for k in TOP_K}
    rf_fs_hits = {k: [] for k in TOP_K}

    for p in test_positive_pairs:
        directions = [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]
        for query_id, target_id in directions:
            query_index = id_to_index[query_id]
            target_index = id_to_index[target_id]
            query_vec = embedding_matrix[query_index]

            X_candidates_full = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)
            X_candidates_blast = X_candidates_full[:, :-1]

            rf_scores = rf_blast.predict_proba(X_candidates_blast)[:, 1]
            rf_scores[query_index] = -np.inf
            rf_rank = int(np.where(np.argsort(rf_scores)[::-1] == target_index)[0][0]) + 1

            rf_fs_scores = rf_fs.predict_proba(X_candidates_full)[:, 1]
            rf_fs_scores[query_index] = -np.inf
            rf_fs_rank = int(np.where(np.argsort(rf_fs_scores)[::-1] == target_index)[0][0]) + 1

            cos_scores = cosine_similarity_matrix[query_index].copy()
            cos_scores[query_index] = -np.inf
            cos_rank = int(np.where(np.argsort(cos_scores)[::-1] == target_index)[0][0]) + 1

            cos_rr.append(1.0 / cos_rank)
            rf_rr.append(1.0 / rf_rank)
            rf_fs_rr.append(1.0 / rf_fs_rank)
            for k in TOP_K:
                cos_hits[k].append(int(cos_rank <= k))
                rf_hits[k].append(int(rf_rank <= k))
                rf_fs_hits[k].append(int(rf_fs_rank <= k))

    all_cos_rr.extend(cos_rr)
    all_rf_rr.extend(rf_rr)
    all_rf_fs_rr.extend(rf_fs_rr)

    fold_row = {
        "fold": fold_idx,
        "component_size": len(held_out),
        "n_test_positive_pairs": len(test_positive_pairs),
        "n_queries": len(cos_rr),
        "cosine_mrr": float(np.mean(cos_rr)) if cos_rr else float("nan"),
        "rf_blast_mrr": float(np.mean(rf_rr)) if rf_rr else float("nan"),
        "rf_blast_foldseektm_mrr": float(np.mean(rf_fs_rr)) if rf_fs_rr else float("nan"),
    }
    for k in TOP_K:
        fold_row[f"cosine_hits_at_{k}"] = float(np.mean(cos_hits[k])) if cos_hits[k] else float("nan")
        fold_row[f"rf_blast_hits_at_{k}"] = float(np.mean(rf_hits[k])) if rf_hits[k] else float("nan")
        fold_row[f"rf_blast_foldseektm_hits_at_{k}"] = float(np.mean(rf_fs_hits[k])) if rf_fs_hits[k] else float("nan")
    per_fold_rows.append(fold_row)

    if True:
        elapsed = time.time() - overall_start
        print(f"  fold {fold_idx + 1}/{K_FOLDS} done (size={len(held_out)}, queries={len(cos_rr)}) "
              f"-- cosine={fold_row['cosine_mrr']:.4f} rf_blast={fold_row['rf_blast_mrr']:.4f} "
              f"rf_blast_foldseektm={fold_row['rf_blast_foldseektm_mrr']:.4f} "
              f"({elapsed/60:.1f} min elapsed)", flush=True)

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
rf_fs_macro = per_fold_df["rf_blast_foldseektm_mrr"].to_numpy()

cos_macro_mean, cos_macro_std = cos_macro.mean(), cos_macro.std(ddof=1)
rf_macro_mean, rf_macro_std = rf_macro.mean(), rf_macro.std(ddof=1)
rf_fs_macro_mean, rf_fs_macro_std = rf_fs_macro.mean(), rf_fs_macro.std(ddof=1)
cos_macro_se = cos_macro_std / np.sqrt(K_FOLDS)
rf_macro_se = rf_macro_std / np.sqrt(K_FOLDS)
rf_fs_macro_se = rf_fs_macro_std / np.sqrt(K_FOLDS)

cos_micro = float(np.mean(all_cos_rr))
rf_micro = float(np.mean(all_rf_rr))
rf_fs_micro = float(np.mean(all_rf_fs_rr))

paired_delta_rf = rf_macro - cos_macro
paired_delta_fs = rf_fs_macro - rf_macro  # foldseek TM feature's OWN contribution, holding BLAST fixed
wins_rf = int((paired_delta_rf > 0).sum())
wins_fs = int((paired_delta_fs > 0).sum())

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append(f"LOCO ({K_FOLDS} folds): Cosine vs RF+BLAST vs RF+BLAST+FoldseekTM (1548 dataset)")
summary_lines.append("=" * 70)
summary_lines.append(f"Random seed (base): {SEED}")
summary_lines.append(f"Total runtime: {total_elapsed/60:.1f} min")
summary_lines.append("")
summary_lines.append(f"MACRO (unweighted mean across the {K_FOLDS} component-folds):")
summary_lines.append(f"  cosine              MRR: {cos_macro_mean:.4f} +/- {cos_macro_std:.4f}  (SE: {cos_macro_se:.4f})")
summary_lines.append(f"  RF+BLAST            MRR: {rf_macro_mean:.4f} +/- {rf_macro_std:.4f}  (SE: {rf_macro_se:.4f})")
summary_lines.append(f"  RF+BLAST+FoldseekTM MRR: {rf_fs_macro_mean:.4f} +/- {rf_fs_macro_std:.4f}  (SE: {rf_fs_macro_se:.4f})")
summary_lines.append("")
summary_lines.append("MICRO (query-weighted -- most reliable number given small-N noise established earlier):")
summary_lines.append(f"  cosine              MRR: {cos_micro:.4f}")
summary_lines.append(f"  RF+BLAST            MRR: {rf_micro:.4f}")
summary_lines.append(f"  RF+BLAST+FoldseekTM MRR: {rf_fs_micro:.4f}")
summary_lines.append("")
summary_lines.append(f"Paired per-component wins (RF+BLAST > cosine): {wins_rf}/{K_FOLDS}")
summary_lines.append(f"Paired per-component wins (RF+BLAST+FoldseekTM > RF+BLAST): {wins_fs}/{K_FOLDS}")
mean_delta_fs = float(paired_delta_fs.mean())
se_delta_fs = float(paired_delta_fs.std(ddof=1) / np.sqrt(K_FOLDS))
summary_lines.append(f"Mean paired delta (RF+BLAST+FoldseekTM - RF+BLAST): {mean_delta_fs:+.4f} (SE {se_delta_fs:.4f})")
if abs(mean_delta_fs) > 2 * se_delta_fs:
    verdict = "FoldseekTM delta is more than 2 SE from zero -- looks like a REAL, statistically distinguishable effect."
else:
    verdict = "FoldseekTM delta is within ~2 SE of zero -- not statistically distinguishable from noise."
summary_lines.append(verdict)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
print("\nDone.")
