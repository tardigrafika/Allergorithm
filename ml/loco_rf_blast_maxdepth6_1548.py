"""
LOCO: Cosine vs RF+BLAST (max_depth=12, ustanovljen baseline) vs RF+BLAST
(max_depth=6, kandidat) - 1548 dataset.

Zasto: analysis/rf_hyperparam_sensitivity_1548.py (jedan 80/20 split,
bootstrap CI) je pokazao da je max_depth=6 jedina od 9 testiranih
hiperparametar-konfiguracija koja daje ZNACAJAN pozitivan delta naspram
cosine-a (+0.0127, CI[+0.0017,+0.0239]). Ali to je rezultat na JEDNOM
split-u, sa 9 konfiguracija testiranih uporedo (multiple-comparisons rizik) -
isti metodoloski problem koji je vec ranije naveo projekat da pređe sa
k-fold na LOCO (videti ml/loco_cosine_rf_blast_1512.py: "standardna greska
proseka MRR-a NADMASUJE razlike koje pokusavamo da detektujemo"). Ovaj
skript to proverava rigoroznije, istim LOCO protokolom kao svuda.

Established referenca (ml/loco_rf_blast_foldseektm_1548.py, isti dataset/
protokol/RF hiperparametri OSIM max_depth): RF+BLAST (depth=12) micro MRR
0.1249 vs cosine 0.1209 - vec ranije nadjeno kao statisticki nerazluciva.

Izlaz:
    output/loco_rf_blast_maxdepth6_1548_per_fold.csv
    output/loco_rf_blast_maxdepth6_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PER_FOLD_OUTPUT = OUTPUT_DIR / "loco_rf_blast_maxdepth6_1548_per_fold.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "loco_rf_blast_maxdepth6_1548_summary.txt"

SEED = 42
NEG_PER_POS = 10

RF_PARAMS_D12 = dict(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", n_jobs=-1)
RF_PARAMS_D6 = dict(n_estimators=300, max_depth=6, min_samples_leaf=3, class_weight="balanced", n_jobs=-1)

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
embedding_matrix = np.array([embeddings_dict[allergen_id] for allergen_id in all_ids], dtype=np.float64)
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
    gold_pairs.append({"pair_id": row["pair_id"], "id_1": id_1, "id_2": id_2})

print(f"Mapped gold pairs: {len(gold_pairs)}")
positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}

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
free_proteins = [pid for pid in all_ids if pid not in parent]
K_FOLDS = len(component_list)
print(f"Connected components (= number of LOCO folds): {K_FOLDS}")
print(f"Free proteins (added to every train pool): {len(free_proteins)}")


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


print("\n==============================")
print(f"RUNNING LOCO ({K_FOLDS} folds, 1 per component)")
print("==============================")

per_fold_rows = []
overall_start = time.time()
all_cos_rr, all_d12_rr, all_d6_rr = [], [], []

for fold_idx, held_out in enumerate(component_list):
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

    rf_d12 = RandomForestClassifier(random_state=SEED + fold_idx, **RF_PARAMS_D12)
    rf_d12.fit(X_train, y_train)
    rf_d6 = RandomForestClassifier(random_state=SEED + 500 + fold_idx, **RF_PARAMS_D6)
    rf_d6.fit(X_train, y_train)

    cos_rr, d12_rr, d6_rr = [], [], []
    for p in test_positive_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            query_index = id_to_index[query_id]
            target_index = id_to_index[target_id]
            query_vec = embedding_matrix[query_index]
            X_candidates = pairwise_features_batch_same_query(query_vec, query_id, embedding_matrix, all_ids)

            d12_scores = rf_d12.predict_proba(X_candidates)[:, 1]
            d12_scores[query_index] = -np.inf
            d12_rank = int(np.where(np.argsort(d12_scores)[::-1] == target_index)[0][0]) + 1

            d6_scores = rf_d6.predict_proba(X_candidates)[:, 1]
            d6_scores[query_index] = -np.inf
            d6_rank = int(np.where(np.argsort(d6_scores)[::-1] == target_index)[0][0]) + 1

            cos_scores = cosine_similarity_matrix[query_index].copy()
            cos_scores[query_index] = -np.inf
            cos_rank = int(np.where(np.argsort(cos_scores)[::-1] == target_index)[0][0]) + 1

            cos_rr.append(1.0 / cos_rank)
            d12_rr.append(1.0 / d12_rank)
            d6_rr.append(1.0 / d6_rank)

    all_cos_rr.extend(cos_rr)
    all_d12_rr.extend(d12_rr)
    all_d6_rr.extend(d6_rr)

    fold_row = {
        "fold": fold_idx, "component_size": len(held_out), "n_queries": len(cos_rr),
        "cosine_mrr": float(np.mean(cos_rr)) if cos_rr else float("nan"),
        "rf_blast_d12_mrr": float(np.mean(d12_rr)) if d12_rr else float("nan"),
        "rf_blast_d6_mrr": float(np.mean(d6_rr)) if d6_rr else float("nan"),
    }
    per_fold_rows.append(fold_row)
    elapsed = time.time() - overall_start
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(held_out)}, queries={len(cos_rr)}) "
          f"-- cosine={fold_row['cosine_mrr']:.4f} d12={fold_row['rf_blast_d12_mrr']:.4f} "
          f"d6={fold_row['rf_blast_d6_mrr']:.4f} ({elapsed/60:.1f} min)", flush=True)

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
d12_macro = per_fold_df["rf_blast_d12_mrr"].to_numpy()
d6_macro = per_fold_df["rf_blast_d6_mrr"].to_numpy()

cos_micro = float(np.mean(all_cos_rr))
d12_micro = float(np.mean(all_d12_rr))
d6_micro = float(np.mean(all_d6_rr))

delta_d12_vs_cos = d12_macro - cos_macro
delta_d6_vs_cos = d6_macro - cos_macro
delta_d6_vs_d12 = d6_macro - d12_macro

se_d12_vs_cos = float(delta_d12_vs_cos.std(ddof=1) / np.sqrt(K_FOLDS))
se_d6_vs_cos = float(delta_d6_vs_cos.std(ddof=1) / np.sqrt(K_FOLDS))
se_d6_vs_d12 = float(delta_d6_vs_d12.std(ddof=1) / np.sqrt(K_FOLDS))

summary_lines = [
    "=" * 70,
    f"LOCO ({K_FOLDS} folds): Cosine vs RF+BLAST depth=12 vs RF+BLAST depth=6 (1548)",
    "=" * 70,
    f"Total runtime: {total_elapsed/60:.1f} min", "",
    "MACRO (unweighted mean across component-folds):",
    f"  cosine          MRR: {cos_macro.mean():.4f} +/- {cos_macro.std(ddof=1):.4f}",
    f"  RF+BLAST d=12   MRR: {d12_macro.mean():.4f} +/- {d12_macro.std(ddof=1):.4f}",
    f"  RF+BLAST d=6    MRR: {d6_macro.mean():.4f} +/- {d6_macro.std(ddof=1):.4f}", "",
    "MICRO (query-weighted -- najpouzdaniji broj):",
    f"  cosine          MRR: {cos_micro:.4f}",
    f"  RF+BLAST d=12   MRR: {d12_micro:.4f}",
    f"  RF+BLAST d=6    MRR: {d6_micro:.4f}", "",
    f"Paired delta d=12 vs cosine: {delta_d12_vs_cos.mean():+.4f} (SE {se_d12_vs_cos:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_d12_vs_cos.mean()) > 2*se_d12_vs_cos else '- unutar suma'}",
    f"Paired delta d=6  vs cosine: {delta_d6_vs_cos.mean():+.4f} (SE {se_d6_vs_cos:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_d6_vs_cos.mean()) > 2*se_d6_vs_cos else '- unutar suma'}",
    f"Paired delta d=6  vs d=12:   {delta_d6_vs_d12.mean():+.4f} (SE {se_d6_vs_d12:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_d6_vs_d12.mean()) > 2*se_d6_vs_d12 else '- unutar suma'}",
    f"Paired wins (d=6 > cosine): {int((delta_d6_vs_cos > 0).sum())}/{K_FOLDS}",
    f"Paired wins (d=6 > d=12):   {int((delta_d6_vs_d12 > 0).sum())}/{K_FOLDS}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
