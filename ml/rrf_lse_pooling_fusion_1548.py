"""
LOCO (40 folds): RRF-3 (cosine+BLAST+FoldseekTM, produkcioni signal) vs
RRF-4-LSE (isto + MI/LSE-pooling kao 4. NEZAVISAN rank-based signal) -- 1548.

OBRAZLOZENJE (isti kriterijum kao svaki raniji RRF dodatak ove sesije,
vidi ml/rrf_mlp_hadamard_fusion_1548.py za MLP presedan): LSE-pooling
(analysis/mi_lse_pooling_1548.py, LOCO-potvrdjeno u analysis/mi_lse_loco_1548.py)
je NAUCENA agregacija preko LOKALNIH (sliding-window) slicnosti, strukturno
razlicit signal od cosine-a (koji je whole-protein mean-pool). Znacajno
popravlja nsLTP (+0.0218) i Profilin (+0.0334) pod punom LOCO validacijom --
zadovoljava isti "genuinski nezavisan signal" prag kao MLP(hadamard) i
graph-propagation pre njega. NE testiramo protiv RRF-4 (graph-propagation
nije LOCO-kompatibilan na nivou komponente -- potpuno izdvojena komponenta
nema vidljive susede) nego protiv RRF-3, identicno kao MLP fuzija.

LSE(tau/scale/bias) se trenira ISPOCETKA u svakom LOCO foldu (samo na
train_pairs tog folda) -- nema curenja iz held-out komponente. Window
vektori i vektorizovana padded-gather eval infrastruktura se grade JEDNOM
(ne zavise od folda), samo naucen tau/scale/bias se menja po foldu.

Izlaz:
    output/rrf_lse_pooling_fusion_1548_per_fold.csv
    output/rrf_lse_pooling_fusion_1548_summary.txt
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.features import load_blast_matrices  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
PER_FOLD_OUTPUT = Path("/home/lana/ALERGRAF/output/rrf_lse_pooling_fusion_1548_per_fold.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/rrf_lse_pooling_fusion_1548_summary.txt")

SEED = 42
RRF_K = 20
WINDOW = 20
STRIDE = 5
N_NEGATIVES_TRAIN = 3000

print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
blast = load_blast_matrices(BLAST_MATRIX)

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

perm = np.array([blast["id_to_index"].get(aid, -1) for aid in dataset.all_ids])
valid_idx = np.where(perm >= 0)[0]
blast_score_matrix_full = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
blast_score_matrix_full[np.ix_(valid_idx, valid_idx)] = blast["score_matrix"][np.ix_(perm[valid_idx], perm[valid_idx])]

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

print("Racunam sliding-window embeddinge...")
window_vecs_per_protein = {}
for aid in dataset.all_ids:
    res_emb = residue_embeddings.get(aid)
    if res_emb is None or len(res_emb) == 0:
        continue
    L = res_emb.shape[0]
    if L <= WINDOW:
        w = res_emb.mean(axis=0, keepdims=True)
    else:
        w = np.array([res_emb[s:s + WINDOW].mean(axis=0) for s in range(0, L - WINDOW + 1, STRIDE)])
    window_vecs_per_protein[aid] = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
print(f"Gotovo za {len(window_vecs_per_protein)}/{len(dataset.all_ids)} proteina.")


def pair_similarity_matrix(id_a, id_b):
    wa, wb = window_vecs_per_protein.get(id_a), window_vecs_per_protein.get(id_b)
    if wa is None or wb is None:
        return None
    return wa @ wb.T


# vektorizovana eval infrastruktura, gradi se JEDNOM
all_ids_with_windows = [aid for aid in dataset.all_ids if aid in window_vecs_per_protein]
window_blocks = [window_vecs_per_protein[aid] for aid in all_ids_with_windows]
global_windows_n = np.vstack(window_blocks)

n_proteins = len(dataset.all_ids)
protein_indices = [dataset.id_to_index[aid] for aid in all_ids_with_windows]
counts = [len(b) for b in window_blocks]
max_windows = max(counts)

padded_row_idx = np.full((n_proteins, max_windows), -1, dtype=np.int64)
row_cursor = 0
for pidx, cnt in zip(protein_indices, counts):
    padded_row_idx[pidx, :cnt] = np.arange(row_cursor, row_cursor + cnt)
    row_cursor += cnt
valid_mask = padded_row_idx >= 0
safe_idx = np.where(valid_mask, padded_row_idx, 0)
n_valid_per_protein = valid_mask.sum(axis=1)


def lse_scores_for_query(query_id, tau):
    qw = window_vecs_per_protein.get(query_id)
    if qw is None:
        return None
    n_qw = qw.shape[0]
    sim = (qw @ global_windows_n.T).astype(np.float32)
    gathered = sim[:, safe_idx]
    mask_3d = valid_mask[None, :, :]
    with np.errstate(over="ignore", invalid="ignore"):
        exp_vals = np.where(mask_3d, np.exp(gathered / tau), 0.0)
        sum_exp = exp_vals.sum(axis=(0, 2))
        total_count = n_qw * np.maximum(n_valid_per_protein, 1)
        mean_exp = sum_exp / total_count
        scores = np.where(n_valid_per_protein > 0, tau * np.log(mean_exp + 1e-300), -np.inf)
    return scores


def train_lse(train_pairs, seed):
    train_neg_ids = sample_negative_pairs(dataset.all_ids, N_NEGATIVES_TRAIN, seed, dataset.positive_pair_set)
    train_sims, train_labels = [], []
    for p in train_pairs:
        S = pair_similarity_matrix(p["id_1"], p["id_2"])
        if S is not None:
            train_sims.append(S.flatten())
            train_labels.append(1.0)
    for a, b in train_neg_ids:
        S = pair_similarity_matrix(a, b)
        if S is not None:
            train_sims.append(S.flatten())
            train_labels.append(0.0)

    # Vektorizovano: padding do zajednicke duzine JEDNOM (ne po epohi) --
    # isti padded-gather princip kao eval infrastruktura, izbegava 300x
    # Python-level petlju preko ~N primera (bio je glavni usko grlo, ~4.3
    # min/fold i za najmanje foldove, dominantno trening a ne eval).
    max_len = max(len(s) for s in train_sims)
    n_examples = len(train_sims)
    padded = np.full((n_examples, max_len), -np.inf, dtype=np.float32)
    for i, s in enumerate(train_sims):
        padded[i, :len(s)] = s
    padded_t = torch.tensor(padded, dtype=torch.float32)
    valid_t = torch.isfinite(padded_t)
    n_valid_t = valid_t.sum(dim=1).float()
    padded_safe = torch.where(valid_t, padded_t, torch.tensor(0.0))  # -inf bi dao nan u gradijentu

    log_tau = torch.nn.Parameter(torch.tensor(0.0))
    scale = torch.nn.Parameter(torch.tensor(5.0))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    labels_t = torch.tensor(train_labels, dtype=torch.float32)
    optimizer = torch.optim.Adam([log_tau, scale, bias], lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    NEG_INF = torch.tensor(-1e9)
    for epoch in range(300):
        optimizer.zero_grad()
        tau = torch.exp(log_tau) + 1e-3
        scaled = torch.where(valid_t, padded_safe / tau, NEG_INF)
        pooled = tau * (torch.logsumexp(scaled, dim=1) - torch.log(n_valid_t))
        logits = scale * pooled + bias
        loss = loss_fn(logits, labels_t)
        loss.backward()
        optimizer.step()
    return float(torch.exp(log_tau).item() + 1e-3)


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}")

per_fold_rows = []
all_cos_rr, all_rrf3_rr, all_rrf4lse_rr = [], [], []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    fitted_tau = train_lse(train_pairs, SEED + fold_idx)

    cos_rr, rrf3_rr, rrf4lse_rr = [], [], []
    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            cos_rank = ranks_from_scores(cosine_matrix[qi], qi)
            blast_rank = ranks_from_scores(blast_score_matrix_full[qi], qi)
            fs_rank = ranks_from_scores(foldseek_matrix[qi], qi)
            lse_scores = lse_scores_for_query(query_id, fitted_tau)
            if lse_scores is None:
                continue
            lse_rank = ranks_from_scores(lse_scores, qi)

            rrf3 = 1.0 / (RRF_K + cos_rank) + 1.0 / (RRF_K + blast_rank) + 1.0 / (RRF_K + fs_rank)
            rrf4lse = rrf3 + 1.0 / (RRF_K + lse_rank)

            cos_final_rank = int(cos_rank[ti])
            rrf3_order = np.argsort(rrf3)[::-1]
            rrf3_final_rank = int(np.where(rrf3_order == ti)[0][0]) + 1
            rrf4lse_order = np.argsort(rrf4lse)[::-1]
            rrf4lse_final_rank = int(np.where(rrf4lse_order == ti)[0][0]) + 1

            cos_rr.append(1.0 / cos_final_rank)
            rrf3_rr.append(1.0 / rrf3_final_rank)
            rrf4lse_rr.append(1.0 / rrf4lse_final_rank)

    all_cos_rr.extend(cos_rr)
    all_rrf3_rr.extend(rrf3_rr)
    all_rrf4lse_rr.extend(rrf4lse_rr)

    per_fold_rows.append({
        "fold": fold_idx, "component_size": len(test_ids), "n_queries": len(cos_rr), "fitted_tau": fitted_tau,
        "cosine_mrr": float(np.mean(cos_rr)) if cos_rr else float("nan"),
        "rrf3_mrr": float(np.mean(rrf3_rr)) if rrf3_rr else float("nan"),
        "rrf4lse_mrr": float(np.mean(rrf4lse_rr)) if rrf4lse_rr else float("nan"),
    })
    elapsed = time.time() - overall_start
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(cos_rr)}, tau={fitted_tau:.3f}) -- "
          f"cosine={per_fold_rows[-1]['cosine_mrr']:.4f} rrf3={per_fold_rows[-1]['rrf3_mrr']:.4f} "
          f"rrf4lse={per_fold_rows[-1]['rrf4lse_mrr']:.4f} ({elapsed/60:.1f} min)", flush=True)

total_elapsed = time.time() - overall_start
print(f"\nAll {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min")

per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(PER_FOLD_OUTPUT, index=False)

cos_macro = per_fold_df["cosine_mrr"].to_numpy()
rrf3_macro = per_fold_df["rrf3_mrr"].to_numpy()
rrf4lse_macro = per_fold_df["rrf4lse_mrr"].to_numpy()

delta_rrf3 = rrf3_macro - cos_macro
delta_rrf4lse_vs_cos = rrf4lse_macro - cos_macro
delta_rrf4lse_vs_rrf3 = rrf4lse_macro - rrf3_macro

se_rrf3 = float(np.nanstd(delta_rrf3, ddof=1) / np.sqrt(K_FOLDS))
se_rrf4lse_cos = float(np.nanstd(delta_rrf4lse_vs_cos, ddof=1) / np.sqrt(K_FOLDS))
se_rrf4lse_rrf3 = float(np.nanstd(delta_rrf4lse_vs_rrf3, ddof=1) / np.sqrt(K_FOLDS))

summary_lines = [
    "=" * 70, f"LOCO ({K_FOLDS} folds): Cosine vs RRF-3 vs RRF-4-LSE(pooling) (1548)", "=" * 70,
    f"Total runtime: {total_elapsed/60:.1f} min", "",
    "MACRO (unweighted mean across component-folds):",
    f"  cosine       MRR: {np.nanmean(cos_macro):.4f} +/- {np.nanstd(cos_macro, ddof=1):.4f}",
    f"  RRF-3        MRR: {np.nanmean(rrf3_macro):.4f} +/- {np.nanstd(rrf3_macro, ddof=1):.4f}",
    f"  RRF-4-LSE    MRR: {np.nanmean(rrf4lse_macro):.4f} +/- {np.nanstd(rrf4lse_macro, ddof=1):.4f}", "",
    "MICRO (query-weighted, pooled -- najpouzdaniji broj):",
    f"  cosine       MRR: {np.mean(all_cos_rr):.4f}",
    f"  RRF-3        MRR: {np.mean(all_rrf3_rr):.4f}",
    f"  RRF-4-LSE    MRR: {np.mean(all_rrf4lse_rr):.4f}", "",
    f"Paired delta RRF-3 vs cosine:           {np.nanmean(delta_rrf3):+.4f} (SE {se_rrf3:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(np.nanmean(delta_rrf3)) > 2*se_rrf3 else '- unutar suma'}",
    f"Paired delta RRF-4-LSE vs cosine:       {np.nanmean(delta_rrf4lse_vs_cos):+.4f} (SE {se_rrf4lse_cos:.4f}) "
    f"{'>2SE - REALAN EFEKAT' if abs(np.nanmean(delta_rrf4lse_vs_cos)) > 2*se_rrf4lse_cos else '- unutar suma'}",
    f"Paired delta RRF-4-LSE vs RRF-3:        {np.nanmean(delta_rrf4lse_vs_rrf3):+.4f} (SE {se_rrf4lse_rrf3:.4f}) "
    f"{'>2SE - REALAN EFEKAT (LSE dodaje nezavisnu vrednost)' if abs(np.nanmean(delta_rrf4lse_vs_rrf3)) > 2*se_rrf4lse_rrf3 else '- unutar suma (LSE ne dodaje merljivo)'}",
    f"Paired wins (RRF-4-LSE > RRF-3): {int((delta_rrf4lse_vs_rrf3 > 0).sum())}/{K_FOLDS}",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved to: {SUMMARY_OUTPUT}")
