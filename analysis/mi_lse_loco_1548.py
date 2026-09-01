"""
Prava LOCO validacija za MI/hypergraph LSE-pooling (mi_lse_pooling_1548.py).

Razlika od pilot verzije: pilot je izdvojio SVE TRI ciljne familije (nsLTP/
Profilin/PR-10) IZ TRENINGA ISTOVREMENO -- jedan kombinovan test. Ovde:
3 ODVOJENA fold-a, svaki izdvaja TACNO JEDNU familiju (druge dve OSTAJU u
treningu). Ovo je moguce i smisleno jer je ranije utvrdjeno (bridge_protein_
analysis_1548.py) da je svaka od tri familije TACNO JEDNA connected
komponenta u gold grafu -- standardna "leave-one-connected-component-out"
LOCO disciplina ove sesije, primenjena na sve tri komponente odvojeno,
umesto simulirane kombinovanim izdvajanjem.

Motivacija: pilot rezultat (nsLTP +0.0219, Profilin +0.0336, oba znacajna
po bootstrap CI; PR-10 +0.0 nije) mogao bi zavisiti od toga sto model NIJE
video nijednu od tri "tesku" familije tokom treninga -- ovaj test proverava
da li rezultat ostaje kad model VIDI ostale dve tesko familije u treningu
(realisticnije, i pravi LOCO protokol koji projekat inace koristi).

Izlaz:
    output/mi_lse_loco_1548_summary.txt
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mi_lse_loco_1548_summary.txt")

WINDOW = 20
STRIDE = 5
TARGET_FAMILIES = ["nsLTP", "Profilin", "PR-10"]
SEED = 42
N_NEGATIVES_TRAIN = 3000

print("Loading dataset + residue embeddings...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

print("Racunam sliding-window embeddinge za sve proteine...")
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

print(f"Prozori izracunati za {len(window_vecs_per_protein)}/{len(dataset.all_ids)} proteina.")


def pair_similarity_matrix(id_a, id_b):
    wa, wb = window_vecs_per_protein.get(id_a), window_vecs_per_protein.get(id_b)
    if wa is None or wb is None:
        return None
    return wa @ wb.T


# -------------------------------------------------------
# Vektorizovana eval infrastruktura (ista za sve foldove -- ne menja se
# sa treningom, samo primenjeni tau se menja po foldu)
# -------------------------------------------------------
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


def train_lse(train_positive_pairs, seed):
    train_neg_ids = sample_negative_pairs(dataset.all_ids, N_NEGATIVES_TRAIN, seed, dataset.positive_pair_set)
    train_sims, train_labels = [], []
    for p in train_positive_pairs:
        S = pair_similarity_matrix(p["id_1"], p["id_2"])
        if S is not None:
            train_sims.append(S.flatten())
            train_labels.append(1.0)
    for a, b in train_neg_ids:
        S = pair_similarity_matrix(a, b)
        if S is not None:
            train_sims.append(S.flatten())
            train_labels.append(0.0)

    log_tau = torch.nn.Parameter(torch.tensor(0.0))
    scale = torch.nn.Parameter(torch.tensor(5.0))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    labels_t = torch.tensor(train_labels, dtype=torch.float32)
    sims_list = [torch.tensor(s, dtype=torch.float32) for s in train_sims]
    optimizer = torch.optim.Adam([log_tau, scale, bias], lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(300):
        optimizer.zero_grad()
        tau = torch.exp(log_tau) + 1e-3
        pooled = torch.stack([tau * torch.logsumexp(s / tau, dim=0) - tau * np.log(len(s)) for s in sims_list])
        logits = scale * pooled + bias
        loss = loss_fn(logits, labels_t)
        loss.backward()
        optimizer.step()

    return float(torch.exp(log_tau).item() + 1e-3), len(train_positive_pairs), len(train_neg_ids)


# -------------------------------------------------------
# 3 LOCO fold-a: svaki izdvaja TACNO JEDNU ciljnu familiju
# -------------------------------------------------------
all_results = []
fold_summaries = []

for held_out_fam in TARGET_FAMILIES:
    print(f"\n{'='*60}\nFOLD: izdvajam {held_out_fam} (ostale ciljne familije OSTAJU u treningu)\n{'='*60}")
    train_positive_pairs = [p for p in dataset.gold_pairs if p.get("family_1") != held_out_fam]
    eval_pairs = [p for p in dataset.gold_pairs if p.get("family_1") == held_out_fam]
    print(f"Trening pozitivnih: {len(train_positive_pairs)}, eval parova ({held_out_fam}): {len(eval_pairs)}")

    fitted_tau, n_train_pos, n_train_neg = train_lse(train_positive_pairs, SEED)
    print(f"Fitovan tau={fitted_tau:.4f}")

    fold_results = []
    for p in eval_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            cos_scores = cosine_matrix[qi].copy()
            cos_scores[qi] = -np.inf
            cos_rank = int(np.argsort(np.argsort(-cos_scores))[ti]) + 1

            lse_scores = lse_scores_for_query(query_id, fitted_tau)
            if lse_scores is None:
                continue
            lse_scores[qi] = -np.inf
            lse_rank = int(np.argsort(np.argsort(-lse_scores))[ti]) + 1

            fold_results.append({
                "pair_id": p["pair_id"], "family": held_out_fam,
                "query": query_id, "target": target_id,
                "cosine_rank": cos_rank, "lse_rank": lse_rank, "fitted_tau": fitted_tau,
            })

    fold_df = pd.DataFrame(fold_results)
    fold_df["cosine_rr"] = 1.0 / fold_df["cosine_rank"]
    fold_df["lse_rr"] = 1.0 / fold_df["lse_rank"]
    cos_mrr, lse_mrr = fold_df["cosine_rr"].mean(), fold_df["lse_rr"].mean()
    print(f"{held_out_fam}: cosine MRR={cos_mrr:.4f}  LSE MRR={lse_mrr:.4f}  delta={lse_mrr-cos_mrr:+.4f}")
    fold_summaries.append((held_out_fam, cos_mrr, lse_mrr, fitted_tau, len(fold_df)))
    all_results.append(fold_df)

results_df = pd.concat(all_results, ignore_index=True)

# -------------------------------------------------------
# Bootstrap CI po fold-u (ista metodologija kao mi_lse_bootstrap_ci_1548.py)
# -------------------------------------------------------
rng = np.random.default_rng(SEED)
N_BOOTSTRAP = 2000
summary_lines = ["=" * 80, "MI/hypergraph LSE-pooling -- PRAVA LOCO validacija (3 odvojena fold-a, "
                  "svaki izdvaja jednu familiju)", "=" * 80, ""]

for held_out_fam, cos_mrr, lse_mrr, fitted_tau, n in fold_summaries:
    sub = results_df[results_df["family"] == held_out_fam]
    pair_ids = sub["pair_id"].unique()
    deltas = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        counts_s = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts_s.rename("w"), left_on="pair_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["lse_rr"], weights=w) - np.average(resampled["cosine_rr"], weights=w)
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"{held_out_fam} (n={n} upita, fitovan tau={fitted_tau:.4f}): "
                          f"cosine MRR={cos_mrr:.4f}  LSE MRR={lse_mrr:.4f}  delta={lse_mrr-cos_mrr:+.4f}  "
                          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")

overall_cos, overall_lse = results_df["cosine_rr"].mean(), results_df["lse_rr"].mean()
summary_lines.append(f"\nUKUPNO (sva 3 folda spojena): cosine MRR={overall_cos:.4f}  LSE MRR={overall_lse:.4f}  "
                      f"delta={overall_lse - overall_cos:+.4f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
results_df.to_csv("/home/lana/ALERGRAF/output/mi_lse_loco_1548_per_query.csv", index=False)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
