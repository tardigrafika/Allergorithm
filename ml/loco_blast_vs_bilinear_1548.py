"""
OuterProductBilinear (low-rank, mentorov predlog) SAM vs BLAST SAM -- puna
LOCO (40 folda) validacija na GOLD datasetu, ista disciplina kao
loco_blast_vs_mlp_hadamard_only_1548.py: cist trening (training_eligible_pairs,
bez preostalih Inferred), bootstrap CI na OBA nivoa (po paru i po izvoru).

Izlaz:
    output/loco_blast_vs_bilinear_1548_per_query.csv
    output/loco_blast_vs_bilinear_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.features import load_blast_matrices  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_blast_vs_bilinear_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_blast_vs_bilinear_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
N_BOOTSTRAP = 2000
RANK = 64
HIDDEN = 32
EMBED_DIM = 1280


class OuterProductBilinear(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, rank=RANK, hidden=HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, rank, bias=False)
        self.mlp = nn.Sequential(nn.Linear(rank * rank, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def pair_features(self, u, v):
        a, b = self.proj(u), self.proj(v)
        sym = (torch.einsum("bi,bj->bij", a, b) + torch.einsum("bi,bj->bij", b, a)).reshape(a.shape[0], -1)
        return sym

    def forward(self, u, v):
        return self.mlp(self.pair_features(u, v)).squeeze(-1)

    @torch.no_grad()
    def score_against_all(self, query_vec, candidate_matrix):
        a = self.proj(query_vec.unsqueeze(0))
        b_all = self.proj(candidate_matrix)
        n = b_all.shape[0]
        a_exp = a.expand(n, -1)
        sym = (torch.einsum("bi,bj->bij", a_exp, b_all) + torch.einsum("bi,bj->bij", b_all, a_exp)).reshape(n, -1)
        return torch.sigmoid(self.mlp(sym).squeeze(-1)).numpy()


def train_bilinear(train_pairs, train_negatives, embedding_matrix_std, id_to_index, seed):
    torch.manual_seed(seed)

    def get_vec(aid):
        return embedding_matrix_std[id_to_index[aid]]

    train_u, train_v, train_y = [], [], []
    for p in train_pairs:
        train_u.append(get_vec(p["id_1"])); train_v.append(get_vec(p["id_2"])); train_y.append(1.0)
    for a, b in train_negatives:
        train_u.append(get_vec(a)); train_v.append(get_vec(b)); train_y.append(0.0)
    train_u, train_v = torch.stack(train_u), torch.stack(train_v)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    model = OuterProductBilinear()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(train_y)
    n_val = max(int(0.15 * n), 5)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    best_val_loss, best_state, no_improve = float("inf"), None, 0
    BATCH = 64
    for epoch in range(300):
        model.train()
        epoch_perm = tr_idx[torch.randperm(len(tr_idx))]
        for start in range(0, len(epoch_perm), BATCH):
            idx = epoch_perm[start:start + BATCH]
            optimizer.zero_grad()
            loss = loss_fn(model(train_u[idx], train_v[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(train_u[val_idx], train_v[val_idx]), train_y[val_idx]).item()
        if val_loss < best_val_loss:
            best_val_loss, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= 20:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
blast = load_blast_matrices(BLAST_MATRIX)

perm_b = np.array([blast["id_to_index"].get(aid, -1) for aid in dataset.all_ids])
valid_idx = np.where(perm_b >= 0)[0]
blast_score_matrix_full = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
blast_score_matrix_full[np.ix_(valid_idx, valid_idx)] = blast["score_matrix"][np.ix_(perm_b[valid_idx], perm_b[valid_idx])]

embedding_matrix_t = torch.tensor(dataset.embedding_matrix, dtype=torch.float32)
mean = embedding_matrix_t.mean(dim=0, keepdim=True)
std = embedding_matrix_t.std(dim=0, keepdim=True) + 1e-8
embedding_matrix_std = (embedding_matrix_t - mean) / std


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}", flush=True)

records = []
overall_start = time.time()

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_pairs_clean = training_eligible_pairs(train_pairs)
    train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
    train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED + fold_idx, dataset.positive_pair_set)

    model = None
    if len(train_pairs_clean) >= 5:
        model = train_bilinear(train_pairs_clean, train_negatives, embedding_matrix_std, dataset.id_to_index,
                                 SEED + fold_idx)

    for p in test_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]

            blast_rank = ranks_from_scores(blast_score_matrix_full[qi], qi)
            blast_final_rank = int(blast_rank[ti])

            if model is not None:
                bilinear_scores = model.score_against_all(embedding_matrix_std[qi], embedding_matrix_std)
                bilinear_rank_arr = ranks_from_scores(bilinear_scores, qi)
                bilinear_final_rank = int(bilinear_rank_arr[ti])
            else:
                bilinear_final_rank = None

            records.append({
                "fold": fold_idx, "pair_id": p["pair_id"],
                "blast_rank": blast_final_rank, "bilinear_rank": bilinear_final_rank,
                "blast_rr": 1.0 / blast_final_rank,
                "bilinear_rr": (1.0 / bilinear_final_rank) if bilinear_final_rank is not None else None,
            })

    elapsed = time.time() - overall_start
    fold_df_tmp = pd.DataFrame([r for r in records if r["fold"] == fold_idx])
    bl_str = f"{fold_df_tmp['bilinear_rr'].mean():.4f}" if fold_df_tmp["bilinear_rr"].notna().any() else "N/A"
    print(f"  fold {fold_idx + 1}/{K_FOLDS} (size={len(test_ids)}, queries={len(fold_df_tmp)}, "
          f"clean_train={len(train_pairs_clean)}) -- blast={fold_df_tmp['blast_rr'].mean():.4f} "
          f"bilinear={bl_str} ({elapsed/60:.1f} min)", flush=True)

df = pd.DataFrame(records)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - overall_start
print(f"All {K_FOLDS} LOCO folds done in {total_elapsed/60:.1f} min", flush=True)

df_valid = df.dropna(subset=["bilinear_rr"]).copy()


def paired_bootstrap(sub, group_col, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    groups = sub[group_col].dropna().unique()
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["bilinear_rr"], weights=w) - np.average(resampled["blast_rr"], weights=w)
        deltas.append(d)
    return np.array(deltas)


summary_lines = ["=" * 80, f"LOCO ({K_FOLDS} folds, cist trening): SAM OuterProductBilinear vs SAM BLAST",
                  "=" * 80, "", f"Ukupno runtime: {total_elapsed/60:.1f} min", f"Ukupno upita: {len(df)}", "",
                  f"BLAST MRR (micro): {df_valid['blast_rr'].mean():.4f}",
                  f"Bilinear MRR (micro): {df_valid['bilinear_rr'].mean():.4f}",
                  f"Delta: {df_valid['bilinear_rr'].mean() - df_valid['blast_rr'].mean():+.4f}", ""]

for label, group_col in [("PO PARU (pair_id)", "pair_id"), ("PO IZVORU (reference, studijski nivo)", "reference")]:
    deltas = paired_bootstrap(df_valid, group_col, N_BOOTSTRAP, SEED)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
    summary_lines.append(f"{label}: mean delta={deltas.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
