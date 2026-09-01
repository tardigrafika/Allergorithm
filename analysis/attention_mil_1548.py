"""
Attention-MIL: eskalacija MI/LSE-pooling rezultata (mi_lse_pooling_1548.py,
mi_lse_loco_1548.py -- LOCO-potvrdjeno poboljsanje na nsLTP/Profilin, PR-10
namerno izostavljen posle zasebne dijagnoze koja pokazuje da PR-10 nema
diskriminativan lokalni signal ni za jedan mehanizam agregacije).

LSE-pooling koristi FIKSNU (linearnu) formu tezinske funkcije preko parova
prozora: w_ij = softmax_ij(s_ij / tau), jedan skalarni parametar. Ovde:
zamenjujemo tu fiksnu formu NAUCENOM nelinearnom funkcijom f_theta(s_ij)
(mala MLP, 1->8->1) primenjenom na svaku slicnost pojedinacno, pa softmax
preko SVIH parova prozora daje attention tezine -- model moze nauciti
proizvoljniji oblik reagovanja (npr. "ignorisi sve ispod praga X, fokusiraj
se samo na par najboljih poklapanja") umesto samo glatke LSE interpolacije.

    attention_ij = softmax_ij( f_theta(s_ij) )
    score = sum_ij attention_ij * s_ij

Ovo je striktna generalizacija LSE-poolinga (LSE = specijalni slucaj sa
f_theta(s) = s/tau, fiksna linearna forma).

Trening: SAMO na nsLTP/Profilin -- 2 odvojena LOCO fold-a (svaki izdvaja
JEDNU familiju, druga ostaje u treningu + sve ostale familije), ista
disciplina kao mi_lse_loco_1548.py. PR-10 NIJE cilj ovog eksperimenta
(vec dijagnostikovan kao strukturno bez lokalnog signala) ali OSTAJE u
trening poolu (nije izbacen iz dataseta, samo nije evaluacioni cilj).

Izlaz:
    output/attention_mil_1548_summary.txt
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/attention_mil_1548_summary.txt")

WINDOW = 20
STRIDE = 5
TARGET_FAMILIES = ["nsLTP", "Profilin"]  # PR-10 namerno izostavljen (vidi dijagnozu)
SEED = 42
N_NEGATIVES_TRAIN = 3000
HIDDEN = 8

print("Loading dataset + residue embeddings...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

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


class AttentionScorer(nn.Module):
    """f_theta: 1 -> HIDDEN -> 1, primenjen elementwise na svaku slicnost."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, s_flat):
        # s_flat: (n_pairs,) -> attention logits (n_pairs,)
        return self.net(s_flat.unsqueeze(-1)).squeeze(-1)


def attention_pool(s_flat, scorer):
    logits = scorer(s_flat)
    weights = torch.softmax(logits, dim=0)
    return (weights * s_flat).sum()


# -------------------------------------------------------
# Eval infrastruktura (ista padded-gather logika, ali sada primenjuje
# NAUCENU f_theta umesto fiksnog exp(s/tau) -- radi se u torch da moze
# direktno da koristi trenirani AttentionScorer)
# -------------------------------------------------------
all_ids_with_windows = [aid for aid in dataset.all_ids if aid in window_vecs_per_protein]
window_blocks = [window_vecs_per_protein[aid] for aid in all_ids_with_windows]
global_windows_n = torch.tensor(np.vstack(window_blocks), dtype=torch.float32)

n_proteins = len(dataset.all_ids)
protein_indices = [dataset.id_to_index[aid] for aid in all_ids_with_windows]
counts = [len(b) for b in window_blocks]
max_windows = max(counts)

padded_row_idx = np.full((n_proteins, max_windows), -1, dtype=np.int64)
row_cursor = 0
for pidx, cnt in zip(protein_indices, counts):
    padded_row_idx[pidx, :cnt] = np.arange(row_cursor, row_cursor + cnt)
    row_cursor += cnt
valid_mask_np = padded_row_idx >= 0
safe_idx = torch.tensor(np.where(valid_mask_np, padded_row_idx, 0), dtype=torch.long)
valid_mask = torch.tensor(valid_mask_np)


@torch.no_grad()
def attention_scores_for_query(query_id, scorer):
    """Za svakog kandidata: softmax-attention preko SVIH (n_qw x n_cand_windows)
    parova prozora, primenjujuci naucenu f_theta -- mora se poklapati sa
    treningom (isti model, ista formula)."""
    qw = window_vecs_per_protein.get(query_id)
    if qw is None:
        return None
    qw_t = torch.tensor(qw, dtype=torch.float32)
    sim = qw_t @ global_windows_n.T  # (n_qw, total_windows)

    n_qw = sim.shape[0]
    gathered = sim[:, safe_idx]  # (n_qw, n_proteins, max_windows)
    mask_3d = valid_mask.unsqueeze(0).expand(n_qw, -1, -1)

    logits = scorer(gathered.reshape(-1)).reshape(gathered.shape)
    logits = logits.masked_fill(~mask_3d, float("-inf"))
    # softmax preko (n_qw, max_windows) zajedno, PO PROTEINU (dim 0 i 2 spojeno)
    flat_logits = logits.permute(1, 0, 2).reshape(n_proteins, -1)  # (n_proteins, n_qw*max_windows)
    flat_sims = gathered.permute(1, 0, 2).reshape(n_proteins, -1)
    weights = torch.softmax(flat_logits, dim=1)
    weights = torch.nan_to_num(weights, nan=0.0)
    scores = (weights * flat_sims).sum(dim=1)
    n_valid_per_protein = valid_mask.sum(dim=1)
    scores = torch.where(n_valid_per_protein > 0, scores, torch.tensor(float("-inf")))
    return scores.numpy()


def train_attention_mil(train_positive_pairs, seed):
    train_neg_ids = sample_negative_pairs(dataset.all_ids, N_NEGATIVES_TRAIN, seed, dataset.positive_pair_set)
    train_sims, train_labels = [], []
    for p in train_positive_pairs:
        S = pair_similarity_matrix(p["id_1"], p["id_2"])
        if S is not None:
            train_sims.append(torch.tensor(S.flatten(), dtype=torch.float32))
            train_labels.append(1.0)
    for a, b in train_neg_ids:
        S = pair_similarity_matrix(a, b)
        if S is not None:
            train_sims.append(torch.tensor(S.flatten(), dtype=torch.float32))
            train_labels.append(0.0)

    scorer = AttentionScorer()
    scale = torch.nn.Parameter(torch.tensor(5.0))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    labels_t = torch.tensor(train_labels, dtype=torch.float32)
    optimizer = torch.optim.Adam(list(scorer.parameters()) + [scale, bias], lr=0.02)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"  Trening primera: {len(train_sims)} ({sum(train_labels):.0f} pozitivnih)")
    for epoch in range(300):
        optimizer.zero_grad()
        pooled = torch.stack([attention_pool(s, scorer) for s in train_sims])
        logits = scale * pooled + bias
        loss = loss_fn(logits, labels_t)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == 299:
            print(f"    epoch {epoch}: loss={loss.item():.4f}  scale={scale.item():.3f}  bias={bias.item():.3f}")

    return scorer, len(train_positive_pairs), len(train_neg_ids)


# -------------------------------------------------------
# 2 LOCO fold-a (nsLTP, Profilin) -- ista disciplina kao mi_lse_loco_1548.py
# -------------------------------------------------------
all_results = []
fold_summaries = []

for held_out_fam in TARGET_FAMILIES:
    print(f"\n{'='*60}\nFOLD: izdvajam {held_out_fam}\n{'='*60}")
    train_positive_pairs = [p for p in dataset.gold_pairs if p.get("family_1") != held_out_fam]
    eval_pairs = [p for p in dataset.gold_pairs if p.get("family_1") == held_out_fam]
    print(f"Trening pozitivnih: {len(train_positive_pairs)}, eval parova ({held_out_fam}): {len(eval_pairs)}")

    scorer, n_train_pos, n_train_neg = train_attention_mil(train_positive_pairs, SEED)

    fold_results = []
    for i, p in enumerate(eval_pairs, 1):
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            cos_scores = cosine_matrix[qi].copy()
            cos_scores[qi] = -np.inf
            cos_rank = int(np.argsort(np.argsort(-cos_scores))[ti]) + 1

            att_scores = attention_scores_for_query(query_id, scorer)
            if att_scores is None:
                continue
            att_scores[qi] = -np.inf
            att_rank = int(np.argsort(np.argsort(-att_scores))[ti]) + 1

            fold_results.append({
                "pair_id": p["pair_id"], "family": held_out_fam,
                "query": query_id, "target": target_id,
                "cosine_rank": cos_rank, "att_rank": att_rank,
            })
        if i % 50 == 0:
            print(f"  {i}/{len(eval_pairs)} parova obradjeno", flush=True)

    fold_df = pd.DataFrame(fold_results)
    fold_df["cosine_rr"] = 1.0 / fold_df["cosine_rank"]
    fold_df["att_rr"] = 1.0 / fold_df["att_rank"]
    cos_mrr, att_mrr = fold_df["cosine_rr"].mean(), fold_df["att_rr"].mean()
    print(f"{held_out_fam}: cosine MRR={cos_mrr:.4f}  Attention-MIL MRR={att_mrr:.4f}  delta={att_mrr-cos_mrr:+.4f}")
    fold_summaries.append((held_out_fam, cos_mrr, att_mrr, len(fold_df)))
    all_results.append(fold_df)

results_df = pd.concat(all_results, ignore_index=True)

# -------------------------------------------------------
# Bootstrap CI po fold-u
# -------------------------------------------------------
rng = np.random.default_rng(SEED)
N_BOOTSTRAP = 2000
summary_lines = ["=" * 80, "Attention-MIL (naucena f_theta preko parova prozora) vs cosine "
                  "-- LOCO, nsLTP+Profilin", "=" * 80, ""]

for held_out_fam, cos_mrr, att_mrr, n in fold_summaries:
    sub = results_df[results_df["family"] == held_out_fam]
    pair_ids = sub["pair_id"].unique()
    deltas = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        counts_s = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts_s.rename("w"), left_on="pair_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["att_rr"], weights=w) - np.average(resampled["cosine_rr"], weights=w)
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"{held_out_fam} (n={n} upita): cosine MRR={cos_mrr:.4f}  Attention-MIL MRR={att_mrr:.4f}  "
                          f"delta={att_mrr-cos_mrr:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")

# poredjenje sa LSE-pooling rezultatom (mi_lse_loco_1548_summary.txt)
summary_lines.append("\nZa poredjenje, LSE-pooling (mi_lse_loco_1548.py, jedan globalni tau):")
summary_lines.append("  nsLTP delta=+0.0218 CI[+0.0116,+0.0329]; Profilin delta=+0.0334 CI[+0.0183,+0.0487]")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
results_df.to_csv("/home/lana/ALERGRAF/output/attention_mil_1548_per_query.csv", index=False)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
