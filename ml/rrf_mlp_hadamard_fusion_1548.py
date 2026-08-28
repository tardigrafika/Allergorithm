"""
LOCO (44 folds): RRF-3 (cosine+BLAST+FoldseekTM, produkcioni signal) vs
RRF-4-MLP (isto + MLP(hadamard) kao 4. NEZAVISAN rank-based signal) -- 1548.

OBRAZLOZENJE (ne "Frankenstein" spajanje koda -- isti kriterijum kao svaki
raniji RRF dodatak u ovoj sesiji):
  RRF-3 kombinuje TRI ODVOJENA izvora dokaza: cosine (embedding slicnost),
  BLAST (egzaktno poravnanje sekvence), FoldseekTM (3D strukturna slicnost).
  MLP(hadamard) NE koristi nijedan od ta tri direktno -- to je naucena,
  nelinearna transformacija sirovog ESM embeddinga (elementwise produkt kroz
  skriveni sloj), NIJE isto sto i cosine (koji je fiksna, netrenirana
  formula). LOCO (ml/loco_mlp_hadamard_1548.py) je pokazao da MLP(hadamard)
  micro MRR TACNO izjednacuje cosine (0.1209=0.1209) preko svih 44 komponenti
  -- dosledno drugaciji profil gresaka od RRF-a (real-world test:
  test/evaluate_test_cases_mlp_hadamard.py, bolji Wilcoxon p ali losije hard-
  subset razdvajanje od RRF-a) -- ovo je isti "genuinski nezavisan signal"
  kriterijum koji je vec potvrdjen za graph-propagation dodatak (RRF-3->RRF-4,
  bootstrap CI [+0.0037,+0.0123], znacajno). NE testiramo ovo zato sto
  "mozemo da spojimo dva modela", nego zato sto oba modela zadovoljavaju
  isti, vec vazeci kriterijum za probu fuzije.

MLP(hadamard) se trenira ISPOCETKA u svakom LOCO foldu (samo na train_pairs
tog folda, isti protokol kao loco_mlp_hadamard_1548.py) -- nema curenja iz
held-out komponente.

Izlaz:
    output/rrf_mlp_hadamard_fusion_1548_per_fold.csv
    output/rrf_mlp_hadamard_fusion_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.features import load_blast_matrices  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
PER_FOLD_OUTPUT = Path("/home/lana/ALERGRAF/output/rrf_mlp_hadamard_fusion_1548_per_fold.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/rrf_mlp_hadamard_fusion_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
RRF_K = 20  # ml/rrf_k_sensitivity_1548.py nalaz

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
blast = load_blast_matrices(BLAST_MATRIX)

import pickle  # noqa: E402
with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)
foldseek_matrix = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in dataset.id_to_index and b in dataset.id_to_index:
        i, j = dataset.id_to_index[a], dataset.id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score

# vektorizovano popunjavanje BLAST matrice poravnate sa dataset.all_ids redosledom
perm = np.array([blast["id_to_index"].get(aid, -1) for aid in dataset.all_ids])
valid_idx = np.where(perm >= 0)[0]
blast_score_matrix_full = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
blast_score_matrix_full[np.ix_(valid_idx, valid_idx)] = blast["score_matrix"][np.ix_(perm[valid_idx], perm[valid_idx])]


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def sample_negatives(protein_pool, n_needed, seed, positive_pair_set):
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


folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}")

per_fold_rows = []
all_cos_rr, all_rrf3_rr, all_rrf4mlp_rr = [], [], []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_ids = {pid for p in train_pairs for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = len(train_pairs) * NEG_PER_POS
    train_negatives = sample_negatives(train_ids, n_train_neg, SEED + fold_idx, dataset.positive_pair_set)

    mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED + fold_idx)
    mlp.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    cos_rr, rrf3_rr, rrf4mlp_rr = [], [], []
    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            cos_rank = ranks_from_scores(cosine_matrix[qi], qi)
            blast_rank = ranks_from_scores(blast_score_matrix_full[qi], qi)
            fs_rank = ranks_from_scores(foldseek_matrix[qi], qi)
            mlp_scores = mlp.score_all(query_id)
            mlp_rank = ranks_from_scores(mlp_scores, qi)

            rrf3 = 1.0 / (RRF_K + cos_rank) + 1.0 / (RRF_K + blast_rank) + 1.0 / (RRF_K + fs_rank)
            rrf4mlp = rrf3 + 1.0 / (RRF_K + mlp_rank)

            cos_final_rank = int(cos_rank[ti])  # cos_rank[ti] je vec rang cilja (ranks_from_scores vraca rang po indeksu)
            rrf3_order = np.argsort(rrf3)[::-1]
            rrf3_final_rank = int(np.where(rrf3_order == ti)[0][0]) + 1
            rrf4mlp_order = np.argsort(rrf4mlp)[::-1]
            rrf4mlp_final_rank = int(np.where(rrf4mlp_order == ti)[0][0]) + 1

            cos_rr.append(1.0 / cos_final_rank)
            rrf3_rr.append(1.0 / rrf3_final_rank)
            rrf4mlp_rr.append(1.0 / rrf4mlp_final_rank)

    all_cos_rr.extend(cos_rr)
    all_rrf3_rr.extend(rrf3_rr)
    all_rrf4mlp_rr.extend(rrf4mlp_rr)

    per_fold_rows.append({
        "fold": fold_idx, "component_size": len(test_ids), "n_queries": len(cos_rr),
        "cosine_mrr": float(np.mean(cos_rr)),
        "rrf3_mrr": float(np.mean(rrf3_rr)),
        "rrf4mlp_mrr": float(np.mean(rrf4mlp_rr)),
    })
    elapsed = time.time() - overall_start
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(cos_rr)}) -- "
          f"cosine={per_fold_rows[-1]['cosine_mrr']:.4f} rrf3={per_fold_rows[-1]['rrf3_mrr']:.4f} "
          f"rrf4mlp={per_fold_rows[-1]['rrf4mlp_mrr']:.4f} ({elapsed/60:.1f} min)", flush=True)

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
rrf3_macro = per_fold_df["rrf3_mrr"].to_numpy()
rrf4mlp_macro = per_fold_df["rrf4mlp_mrr"].to_numpy()

delta_rrf3 = rrf3_macro - cos_macro
delta_rrf4mlp_vs_cos = rrf4mlp_macro - cos_macro
delta_rrf4mlp_vs_rrf3 = rrf4mlp_macro - rrf3_macro

se_rrf3 = float(delta_rrf3.std(ddof=1) / np.sqrt(K_FOLDS))
se_rrf4mlp_cos = float(delta_rrf4mlp_vs_cos.std(ddof=1) / np.sqrt(K_FOLDS))
se_rrf4mlp_rrf3 = float(delta_rrf4mlp_vs_rrf3.std(ddof=1) / np.sqrt(K_FOLDS))

summary_lines = [
    "=" * 70, f"LOCO ({K_FOLDS} folds): Cosine vs RRF-3 vs RRF-4-MLP(hadamard) (1548)", "=" * 70,
    f"Total runtime: {total_elapsed/60:.1f} min", "",
    "MACRO (unweighted mean across component-folds):",
    f"  cosine       MRR: {cos_macro.mean():.4f} +/- {cos_macro.std(ddof=1):.4f}",
    f"  RRF-3        MRR: {rrf3_macro.mean():.4f} +/- {rrf3_macro.std(ddof=1):.4f}",
    f"  RRF-4-MLP    MRR: {rrf4mlp_macro.mean():.4f} +/- {rrf4mlp_macro.std(ddof=1):.4f}", "",
    "MICRO (query-weighted, pooled -- najpouzdaniji broj):",
    f"  cosine       MRR: {np.mean(all_cos_rr):.4f}",
    f"  RRF-3        MRR: {np.mean(all_rrf3_rr):.4f}",
    f"  RRF-4-MLP    MRR: {np.mean(all_rrf4mlp_rr):.4f}", "",
    f"Paired delta RRF-3 vs cosine:           {delta_rrf3.mean():+.4f} (SE {se_rrf3:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_rrf3.mean()) > 2*se_rrf3 else '- unutar suma'}",
    f"Paired delta RRF-4-MLP vs cosine:       {delta_rrf4mlp_vs_cos.mean():+.4f} (SE {se_rrf4mlp_cos:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(delta_rrf4mlp_vs_cos.mean()) > 2*se_rrf4mlp_cos else '- unutar suma'}",
    f"Paired delta RRF-4-MLP vs RRF-3:        {delta_rrf4mlp_vs_rrf3.mean():+.4f} (SE {se_rrf4mlp_rrf3:.4f}) "
    f"{'>2SE - REALAN EFEKAT (MLP dodaje nezavisnu vrednost)' if abs(delta_rrf4mlp_vs_rrf3.mean()) > 2*se_rrf4mlp_rrf3 else '- unutar suma (MLP ne dodaje merljivo)'}",
    f"Paired wins (RRF-4-MLP > RRF-3): {int((delta_rrf4mlp_vs_rrf3 > 0).sum())}/{K_FOLDS}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
