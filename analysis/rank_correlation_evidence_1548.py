"""
Rank-correlation: da li se cosine/RF+BLAST skor slaze sa JACINOM dokaza
(Confirmed > Strong > Suspected > Inferred), umesto binarnog Hits@K/MRR
- 1548 dataset.

Zasto: sve dosadasnje retrieval metrike (MRR, Hits@K) tretiraju SVAKI poznat
pozitivan par kao "tacan odgovor" bez obzira na evidence tier - Confirmed i
Inferred su ravnopravni za MRR svrhe. Ovo odbacuje informaciju koju vec
imamo (evidence_level po paru). Rank-correlation je drugacije pitanje:
da li VISI skor prati JACI dokaz? To iskoriscava graded label umesto binarne.

Bonus: statisticka snaga je mnogo veca ovde nego kod MRR-a (~1537 PAROVA,
ne 44 KOMPONENTE) - iako parovi nisu potpuno nezavisni (mnogi dele familiju),
n=1537 je mnogo bolja osnova za korelaciju nego n=44 za MRR prosek.

RF+BLAST skor je OUT-OF-FOLD (LOCO): za svaki od 44 foldova, model se trenira
na ostalih 43 komponenti, pa skorira SAMO pozitivne parove iz held-out
komponente - isti leakage-safe protokol kao svuda u sesiji, ne in-sample skor
(koji bi bio trivijalno/lazno visok za trenirani model).

Evidence -> weight mapping: identican EVIDENCE_WEIGHTS iz data/build_ml_dataset.py
(uvezen direktno, ne kopiran, da ne dodje do razmimoilazenja).

Izlaz:
    output/rank_correlation_evidence_1548_pairs.csv (score/weight po paru)
    output/rank_correlation_evidence_1548_summary.txt
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path("/home/lana/ALERGRAF/data")))
from build_ml_dataset import EVIDENCE_WEIGHTS, DEFAULT_POS_WEIGHT  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
PAIRS_OUTPUT = OUTPUT_DIR / "rank_correlation_evidence_1548_pairs.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_correlation_evidence_1548_summary.txt"

SEED = 42
NEG_PER_POS = 10

RF_PARAMS = dict(
    n_estimators=300, max_depth=12, min_samples_leaf=3,
    class_weight="balanced", n_jobs=-1,
)


# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")
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
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {aid: i for i, aid in enumerate(all_ids)}
embedding_matrix = np.array([embeddings_dict[aid] for aid in all_ids], dtype=np.float64)
cosine_similarity_matrix = cosine_similarity(embedding_matrix)

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    level = str(row["evidence_level"]).strip()
    weight = EVIDENCE_WEIGHTS.get(level, DEFAULT_POS_WEIGHT)
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"],
                        "evidence_level": level, "weight": weight})

print(f"Mapped gold pairs: {len(gold_pairs)}")
positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}


# =====================================================
# CONNECTED COMPONENTS (LOCO folds, identical to prior scripts)
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
for pid in parent:
    components.setdefault(find(pid), set()).add(pid)
component_list = list(components.values())
free_proteins = [pid for pid in all_ids if pid not in parent]
K_FOLDS = len(component_list)
print(f"Connected components (= LOCO folds): {K_FOLDS}")


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
# MAIN LOCO LOOP -- out-of-fold scoring of every positive pair
# =====================================================

print(f"\nRunning LOCO ({K_FOLDS} folds) to get out-of-fold scores for every positive pair...")
start = time.time()
records = []

for fold_idx, held_out in enumerate(component_list):
    test_ids = held_out
    train_ids = set(free_proteins)
    for j, c in enumerate(component_list):
        if j != fold_idx:
            train_ids |= c

    train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
    test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
    if not test_positive_pairs:
        continue

    n_train_neg = len(train_positive_pairs) * NEG_PER_POS
    train_negatives = sample_unlabeled_pairs(train_ids, n_train_neg, seed=SEED + fold_idx)

    X_train, y_train = build_feature_matrix(train_positive_pairs, train_negatives)
    rf = RandomForestClassifier(random_state=SEED + fold_idx, **RF_PARAMS)
    rf.fit(X_train, y_train)

    ids_a = [p["id_1"] for p in test_positive_pairs]
    ids_b = [p["id_2"] for p in test_positive_pairs]
    emb_a = np.array([embedding_matrix[id_to_index[a]] for a in ids_a])
    emb_b = np.array([embedding_matrix[id_to_index[b]] for b in ids_b])
    X_test = pairwise_features(emb_a, emb_b, ids_a, ids_b)
    rf_scores = rf.predict_proba(X_test)[:, 1]

    for p, rf_score in zip(test_positive_pairs, rf_scores):
        cos_score = cosine_similarity_matrix[id_to_index[p["id_1"]], id_to_index[p["id_2"]]]
        records.append({
            "pair_id": p["pair_id"], "evidence_level": p["evidence_level"], "weight": p["weight"],
            "cosine_score": float(cos_score), "rf_blast_score": float(rf_score),
        })

    if (fold_idx + 1) % 10 == 0 or (fold_idx + 1) == K_FOLDS:
        print(f"  fold {fold_idx+1}/{K_FOLDS}, {len(records)} pairs scored so far "
              f"({(time.time()-start)/60:.1f} min elapsed)", flush=True)

print(f"\nDone: {len(records)} out-of-fold-scored pairs in {(time.time()-start)/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PAIRS_OUTPUT, index=False)
print(f"Saved: {PAIRS_OUTPUT}")


# =====================================================
# RANK CORRELATION
# =====================================================

cos_rho, cos_p = spearmanr(df["weight"], df["cosine_score"])
rf_rho, rf_p = spearmanr(df["weight"], df["rf_blast_score"])

summary_lines = [
    "=" * 70,
    f"Rank correlation: score vs evidence-strength weight ({len(df)} pairs, {K_FOLDS} LOCO folds)",
    "=" * 70,
    "",
    "Weight per evidence tier: EVIDENCE_WEIGHTS from data/build_ml_dataset.py "
    "(Confirmed=1.0 .. Inferred=0.4, default 0.5)",
    "",
    f"Mean weight by tier:",
]
tier_summary = df.groupby("evidence_level").agg(weight=("weight", "first"), n=("weight", "size")).sort_values("weight", ascending=False)
for level, row in tier_summary.iterrows():
    summary_lines.append(f"  {level[:60]:60s} weight={row['weight']:.2f}  n={int(row['n'])}")

summary_lines += [
    "",
    f"Spearman rho (cosine_score vs weight)   : {cos_rho:+.4f}  (p={cos_p:.2e})",
    f"Spearman rho (rf_blast_score vs weight) : {rf_rho:+.4f}  (p={rf_p:.2e})",
    "",
    f"Delta (RF+BLAST rho - cosine rho): {rf_rho - cos_rho:+.4f}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
