"""
Multiple-instance pooling sa NAUCENOM agregacijom: umesto fiksnog mean-a
(cosine baseline, null) ili fiksnog MAX-a (sliding_window_esm_1548.py, null),
uci JEDAN parametar temperature tau za Log-Sum-Exp ("smooth-max") pooling
preko matrice slicnosti svih parova prozora dva proteina:

    LSE_tau(S) = tau * log( mean_i( exp(S_i / tau) ) )

Ovo je poznata MIL (multiple-instance learning) agregacija koja GLATKO
interpolira: tau -> 0  daje MAX (nas null sliding-window rezultat),
             tau -> inf daje MEAN (nas null cosine baseline).
Hipoteza: postoji tacka izmedju koja je bolja od obe krajnosti -- ako ne,
zatvaramo citav prostor izmedju mean/max jeftino (1 parametar).

Trening: logisticka regresija sa JEDNIM feature-om (LSE_tau skor), fituje
se tau + scale + bias zajedno gradient descent-om (PyTorch autograd).
Pozitivni parovi: SVI gold parovi OSIM nsLTP/Profilin/PR-10 (izdvojeni radi
fer generalizacionog testa, ne cirkularno). Negativni: nasumicno iz celog
poola (ISTA bezbedna metoda kao svuda u projektu -- ne in-family hard-neg,
ta ideja je odbacena kao biolski nepouzdana).

Evaluacija: MRR na nsLTP/Profilin/PR-10 (ISTI podskup kao sliding_window_
esm_1548.py i sliding_window_top3_1548.py, za direktno poredjenje), fitovan
tau primenjen preko ISTE vektorizovane padded-gather infrastrukture.

Izlaz:
    output/mi_lse_pooling_1548_summary.txt
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mi_lse_pooling_1548_summary.txt")

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
    return wa @ wb.T  # (n_a, n_b), vec normalizovano


# -------------------------------------------------------
# Trening skup: pozitivni (BEZ ciljnih familija) + nasumicni negativi
# -------------------------------------------------------
train_positive_pairs = [p for p in dataset.gold_pairs if p.get("family_1") not in TARGET_FAMILIES]
held_out_positive_pairs = [p for p in dataset.gold_pairs if p.get("family_1") in TARGET_FAMILIES]
print(f"\nTrening pozitivnih parova (bez ciljnih familija): {len(train_positive_pairs)}")
print(f"Izdvojenih (nsLTP/Profilin/PR-10) parova za eval: {len(held_out_positive_pairs)}")

train_neg_ids = sample_negative_pairs(dataset.all_ids, N_NEGATIVES_TRAIN, SEED, dataset.positive_pair_set)
print(f"Trening negativnih parova (nasumicno, ceo pool): {len(train_neg_ids)}")

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

print(f"Ukupno trening primera (posle filtriranja nedostajucih prozora): {len(train_sims)}")

# -------------------------------------------------------
# Fit: tau (temperatura LSE poolinga) + scale + bias, gradient descent
# -------------------------------------------------------
log_tau = torch.nn.Parameter(torch.tensor(0.0))  # tau = exp(log_tau), pocinje na tau=1
scale = torch.nn.Parameter(torch.tensor(5.0))
bias = torch.nn.Parameter(torch.tensor(0.0))

labels_t = torch.tensor(train_labels, dtype=torch.float32)
sims_list = [torch.tensor(s, dtype=torch.float32) for s in train_sims]

optimizer = torch.optim.Adam([log_tau, scale, bias], lr=0.05)
loss_fn = torch.nn.BCEWithLogitsLoss()

print("\nTreniram tau + scale + bias (LSE pooling, logisticka regresija)...")
for epoch in range(300):
    optimizer.zero_grad()
    tau = torch.exp(log_tau) + 1e-3
    pooled = torch.stack([tau * torch.logsumexp(s / tau, dim=0) - tau * np.log(len(s)) for s in sims_list])
    logits = scale * pooled + bias
    loss = loss_fn(logits, labels_t)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == 299:
        print(f"  epoch {epoch}: loss={loss.item():.4f}  tau={torch.exp(log_tau).item():.4f}  "
              f"scale={scale.item():.3f}  bias={bias.item():.3f}")

fitted_tau = float(torch.exp(log_tau).item() + 1e-3)
print(f"\nFitovan tau={fitted_tau:.4f} (tau->0 = MAX, tau->inf = MEAN)")

# -------------------------------------------------------
# Vektorizovana LSE-pooled evaluacija (isti padded-gather trik kao top3 skripta)
# -------------------------------------------------------
print("\nGradim globalnu matricu svih prozora + padded indeks-masku za evaluaciju...")
all_ids_with_windows = [aid for aid in dataset.all_ids if aid in window_vecs_per_protein]
window_blocks = [window_vecs_per_protein[aid] for aid in all_ids_with_windows]
global_windows_n = np.vstack(window_blocks)  # vec normalizovano

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
    """Mora se poklapati sa treningom: LSE preko CELE (n_qw x n_cand_windows)
    matrice parova prozora (flatten, ne max-reduce po jednoj osi prvo)."""
    qw = window_vecs_per_protein.get(query_id)
    if qw is None:
        return None
    n_qw = qw.shape[0]
    sim = (qw @ global_windows_n.T).astype(np.float32)      # (n_qw, total_windows)

    gathered = sim[:, safe_idx]                               # (n_qw, n_proteins, max_windows)
    mask_3d = valid_mask[None, :, :]

    with np.errstate(over="ignore", invalid="ignore"):
        exp_vals = np.where(mask_3d, np.exp(gathered / tau), 0.0)
        sum_exp = exp_vals.sum(axis=(0, 2))                     # (n_proteins,)
        total_count = n_qw * np.maximum(n_valid_per_protein, 1)
        mean_exp = sum_exp / total_count
        scores = np.where(n_valid_per_protein > 0, tau * np.log(mean_exp + 1e-300), -np.inf)
    return scores


target_pairs = held_out_positive_pairs
print(f"\nGold parova u {TARGET_FAMILIES} (held-out): {len(target_pairs)}")

results = []
for i, p in enumerate(target_pairs, 1):
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

        results.append({
            "pair_id": p["pair_id"], "family": p.get("family_1"),
            "query": query_id, "target": target_id,
            "cosine_rank": cos_rank, "lse_rank": lse_rank,
        })
    if i % 50 == 0:
        print(f"  {i}/{len(target_pairs)} parova obradjeno", flush=True)

results_df = pd.DataFrame(results)
results_df["cosine_rr"] = 1.0 / results_df["cosine_rank"]
results_df["lse_rr"] = 1.0 / results_df["lse_rank"]

summary_lines = ["=" * 80, "MI/hypergraph LSE-pooling (naucen tau) vs whole-protein cosine "
                  "(nsLTP/Profilin/PR-10, held-out iz treninga)", "=" * 80, "",
                  f"Window={WINDOW}, Stride={STRIDE}, Fitovan tau={fitted_tau:.4f}",
                  f"Trening: {len(train_positive_pairs)} pozitivnih (bez ciljnih familija) + "
                  f"{len(train_neg_ids)} nasumicnih negativa", f"Ukupno upita: {len(results_df)}", ""]
for fam in TARGET_FAMILIES:
    sub = results_df[results_df["family"] == fam]
    if len(sub) == 0:
        continue
    cos_mrr, lse_mrr = sub["cosine_rr"].mean(), sub["lse_rr"].mean()
    delta = lse_mrr - cos_mrr
    summary_lines.append(f"{fam} (n={len(sub)}): cosine MRR={cos_mrr:.4f}  LSE-pooling MRR={lse_mrr:.4f}  "
                          f"delta={delta:+.4f}")

overall_cos, overall_lse = results_df["cosine_rr"].mean(), results_df["lse_rr"].mean()
summary_lines.append(f"\nUKUPNO: cosine MRR={overall_cos:.4f}  LSE-pooling MRR={overall_lse:.4f}  "
                      f"delta={overall_lse - overall_cos:+.4f}")
n_improved = (results_df["lse_rr"] > results_df["cosine_rr"]).sum()
summary_lines.append(f"Broj upita gde LSE-pooling POBOLJŠAVA rang: {n_improved}/{len(results_df)} "
                      f"({n_improved/len(results_df)*100:.1f}%)")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
results_df.to_csv("/home/lana/ALERGRAF/output/mi_lse_pooling_1548_per_query.csv", index=False)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
