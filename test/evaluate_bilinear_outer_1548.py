"""
Mentorov predlog: umesto Hadamard produkta (u*v, hvata SAMO interakcije
ISTE dimenzije, 1280 vrednosti), koristiti OUTER PRODUCT (u tensor v, hvata
SVE parove dimenzija u_i*v_j) pomnozen matricom tezina, pa MLP -- "dobices
delove koji su ti bitni za reakciju".

Problem: pun outer product 1280x1280=1.6M vrednosti bi eksplodirao broj
parametara (desetine miliona) sa samo ~785 cistih trening parova ->
garantovan overfit. Resenje: LOW-RANK bilinear -- prvo projektuj OBA
embeddinga na manju deljenu dimenziju r (naucenom projekcijom A, DELJENOM
za u i v -- ne odvojene A/B, da bi model bio SIMETRICAN: score(u,v)==
score(v,u), isto svojstvo kao SVI ostali modeli u projektu), PA outer
product (r x r, upravljivo), PA MLP.

Simetrija: O(u,v) = a⊗b + b⊗a gde je a=A(u), b=A(v) (DELJENA projekcija) --
ova suma je simetricna matrica (O[i,j]=O[j,i]), garantuje score(u,v)=score(v,u)
egzaktno, ne post-hoc usrednjavanjem.

Trening: training_eligible_pairs() (bez preostalih Inferred), ista cist-
trening disciplina kao MLP(hadamard) danas.

Testira se ISTOM uparenom metodologijom (test/paired_test_mlp_vs_blast_1548.py
stil) protiv BLAST SAM i MLP(hadamard) SAM, na istih 54 pacijenta.

Izlaz:
    test/evaluation_results_raw_bilinear.json
    output/paired_test_bilinear_vs_others_1548_summary.txt
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/test")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
RAW_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_bilinear.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/paired_test_bilinear_vs_others_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
RANK = 64
HIDDEN = 32
EMBED_DIM = 1280
N_PERM = 10000
N_BOOTSTRAP = 10000

torch.manual_seed(SEED)


class OuterProductBilinear(nn.Module):
    """Deljena projekcija A (EMBED_DIM->RANK), simetrican outer product,
    MLP na flatten-ovanoj (RANK x RANK) matrici."""

    def __init__(self, embed_dim=EMBED_DIM, rank=RANK, hidden=HIDDEN):
        super().__init__()
        self.proj = nn.Linear(embed_dim, rank, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(rank * rank, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
        self.rank = rank

    def pair_features(self, u, v):
        a = self.proj(u)  # (batch, rank)
        b = self.proj(v)  # (batch, rank)
        outer_ab = torch.einsum("bi,bj->bij", a, b)
        outer_ba = torch.einsum("bi,bj->bij", b, a)
        sym = (outer_ab + outer_ba).reshape(a.shape[0], -1)  # (batch, rank*rank), simetricna po (u,v)<->(v,u)
        return sym

    def forward(self, u, v):
        return self.mlp(self.pair_features(u, v)).squeeze(-1)

    @torch.no_grad()
    def score_against_all(self, query_vec, candidate_matrix):
        """query_vec: (embed_dim,), candidate_matrix: (N, embed_dim) -> (N,) skorovi."""
        a = self.proj(query_vec.unsqueeze(0))  # (1, rank)
        b_all = self.proj(candidate_matrix)  # (N, rank)
        n = b_all.shape[0]
        a_exp = a.expand(n, -1)
        outer_ab = torch.einsum("bi,bj->bij", a_exp, b_all)
        outer_ba = torch.einsum("bi,bj->bij", b_all, a_exp)
        sym = (outer_ab + outer_ba).reshape(n, -1)
        logits = self.mlp(sym).squeeze(-1)
        return torch.sigmoid(logits).numpy()


print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_pairs_clean = training_eligible_pairs(dataset.gold_pairs)
print(f"  Trening-podobnih (bez Inferred): {len(train_pairs_clean)}", flush=True)

train_negatives = sample_negative_pairs(dataset.all_ids, len(train_pairs_clean) * NEG_PER_POS, SEED,
                                          dataset.positive_pair_set)

embedding_matrix_t = torch.tensor(dataset.embedding_matrix, dtype=torch.float32)
# standardizacija -- isti razlog kao mlp.py DEFAULT_PARAMS (standardize=True za ne-hadamard modele);
# ovde OBA (u,v) prolaze kroz istu projekciju A pa standardizacija ulaznih embeddinga pomaze stabilnosti
mean = embedding_matrix_t.mean(dim=0, keepdim=True)
std = embedding_matrix_t.std(dim=0, keepdim=True) + 1e-8
embedding_matrix_std = (embedding_matrix_t - mean) / std


def get_vec(aid):
    return embedding_matrix_std[dataset.id_to_index[aid]]


train_u, train_v, train_y = [], [], []
for p in train_pairs_clean:
    train_u.append(get_vec(p["id_1"]))
    train_v.append(get_vec(p["id_2"]))
    train_y.append(1.0)
for a, b in train_negatives:
    train_u.append(get_vec(a))
    train_v.append(get_vec(b))
    train_y.append(0.0)

train_u = torch.stack(train_u)
train_v = torch.stack(train_v)
train_y = torch.tensor(train_y, dtype=torch.float32)
print(f"  Ukupno trening primera: {len(train_y)}", flush=True)

model = OuterProductBilinear()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

n = len(train_y)
n_val = max(int(0.15 * n), 10)
perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
val_idx, tr_idx = perm[:n_val], perm[n_val:]

best_val_loss = float("inf")
best_state = None
patience, no_improve = 20, 0
BATCH = 64

print("\nTreniram OuterProductBilinear model...", flush=True)
for epoch in range(300):
    model.train()
    epoch_perm = tr_idx[torch.randperm(len(tr_idx))]
    for start in range(0, len(epoch_perm), BATCH):
        idx = epoch_perm[start:start + BATCH]
        optimizer.zero_grad()
        logits = model(train_u[idx], train_v[idx])
        loss = loss_fn(logits, train_y[idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(train_u[val_idx], train_v[val_idx])
        val_loss = loss_fn(val_logits, train_y[val_idx]).item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break
    if epoch % 50 == 0:
        print(f"  epoch {epoch}: val_loss={val_loss:.4f}", flush=True)

model.load_state_dict(best_state)
model.eval()
print(f"Trening gotov (stopped epoch, best val_loss={best_val_loss:.4f}).", flush=True)

print("\nUcitavam CrossReactivityRanker...", flush=True)
ranker = CrossReactivityRanker()
assert set(dataset.all_ids) == set(ranker.pool)
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])
candidate_matrix_ranker_order = embedding_matrix_std[perm_dataset_to_ranker]


def bilinear_scores_in_ranker_order(aid):
    qi = dataset.id_to_index[aid]
    query_vec = embedding_matrix_std[qi]
    return model.score_against_all(query_vec, candidate_matrix_ranker_order)


def rank_for_patient_bilinear(known_positive_names, known_negative_names=None):
    def resolve(names):
        ids = []
        for name in names or []:
            a = ranker.name_to_id.get(name)
            if a is None or a not in ranker.id_to_index:
                continue
            ids.append(a)
        return ids

    positive_ids = resolve(known_positive_names)
    negative_ids = resolve(known_negative_names)
    if not positive_ids:
        raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

    exclude_idx = {ranker.id_to_index[aid] for aid in positive_ids + negative_ids}
    combined = np.zeros(ranker.n_pool, dtype=np.float64)

    for aid in positive_ids:
        scores = bilinear_scores_in_ranker_order(aid)
        order = np.argsort(scores)[::-1]
        ranks = np.empty(ranker.n_pool, dtype=np.int64)
        ranks[order] = np.arange(1, ranker.n_pool + 1)
        combined += 1.0 / (RRF_K + ranks)

    for idx in exclude_idx:
        combined[idx] = -np.inf

    order = np.argsort(combined)[::-1]
    result = pd.DataFrame({
        "candidate_id": [ranker.pool[i] for i in order],
        "candidate_name": [ranker.id_to_name.get(ranker.pool[i], ranker.pool[i]) for i in order],
        "priority_score": combined[order],
    })
    result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} pacijenata", flush=True)

records = []
for case in cases:
    pid = case["patient_id"]
    verif_status = case["verification"]["status"]
    resolvable = []
    for c in case["components"]:
        if c["result"] not in ("positive", "negative"):
            continue
        resolved = resolve_protein(c["protein"])
        if resolved is None:
            continue
        resolvable.append({"json_name": c["protein"], "pool_name": resolved, "result": c["result"]})

    if len(resolvable) < 2:
        continue

    for i, hidden in enumerate(resolvable):
        others = resolvable[:i] + resolvable[i + 1:]
        known_pos = [o["pool_name"] for o in others if o["result"] == "positive"]
        known_neg = [o["pool_name"] for o in others if o["result"] == "negative"]
        if not known_pos:
            continue

        result_df = rank_for_patient_bilinear(known_pos, known_negative_names=known_neg)
        row = result_df[result_df["candidate_name"] == hidden["pool_name"]]
        if len(row) == 0:
            continue
        rank = int(row.iloc[0]["rank"])
        n_cand = len(result_df)
        percentile = rank / n_cand * 100

        records.append({
            "patient_id": pid, "hidden_protein": hidden["pool_name"],
            "true_result": hidden["result"], "rank": rank, "n_candidates": n_cand,
            "percentile": percentile, "verification_status": verif_status,
        })

df_bilinear = pd.DataFrame(records)
df_bilinear["rr"] = 1.0 / df_bilinear["rank"]
print(f"Leave-one-out trials: {len(df_bilinear)}", flush=True)
df_bilinear.to_json(RAW_OUTPUT, orient="records", indent=2)
print(f"Saved: {RAW_OUTPUT}", flush=True)

# -------------------------------------------------------
# Upareni test protiv BLAST SAM i MLP(hadamard) SAM (isti stil kao
# paired_test_mlp_vs_blast_1548.py)
# -------------------------------------------------------
blast = pd.read_json("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
mlp = pd.read_json("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
blast["rr"] = 1.0 / blast["rank"]
mlp["rr"] = 1.0 / mlp["rank"]


def paired_summary(other_df, other_label):
    merged = df_bilinear[["patient_id", "hidden_protein", "true_result", "verification_status", "rr"]].merge(
        other_df[["patient_id", "hidden_protein", "rr"]], on=["patient_id", "hidden_protein"],
        suffixes=("_bilinear", "_other"))
    lines = [f"=== BILINEAR vs {other_label} ==="]
    for label, sub in [("SVI", merged), ("HARD", merged[merged["verification_status"] == "full_text_verified"])]:
        if len(sub) == 0:
            continue
        per_patient = sub.groupby("patient_id").agg(mrr_b=("rr_bilinear", "mean"), mrr_o=("rr_other", "mean"))
        diffs = per_patient["mrr_b"] - per_patient["mrr_o"]
        diffs_nz = diffs[diffs != 0]
        line = f"  {label} (n={len(sub)} upita, {sub['patient_id'].nunique()} pac.): mean(rr_bilinear-rr_other)={float((sub['rr_bilinear']-sub['rr_other']).mean()):+.4f}"
        if len(diffs_nz) >= 5:
            stat, pval = wilcoxon(diffs_nz)
            line += f", patient-Wilcoxon p={pval:.4f} ({'ZNACAJNO' if pval < 0.05 else 'nije znacajno'})"
        else:
            line += f", patient-Wilcoxon: n={len(diffs_nz)}<5 nepouzdano"
        lines.append(line)
    lines.append("")
    return lines


summary_lines = ["=" * 80, "OuterProductBilinear (low-rank, r=64) -- upareni testovi", "=" * 80,
                  f"MRR (micro, bilinear, svi upiti): {df_bilinear['rr'].mean():.4f}", ""]
summary_lines += paired_summary(blast, "BLAST SAM")
summary_lines += paired_summary(mlp, "MLP(hadamard) SAM")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"Saved: {SUMMARY_OUTPUT}")
