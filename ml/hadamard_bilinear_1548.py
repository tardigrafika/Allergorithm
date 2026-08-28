"""
Mentor-ov predlog: y = sigmoid(w . hadamard(u,v)) -- mnogo jednostavniji
model od MLP-a (1281->256->64->1, ~344k parametara) ili RF-a (stotine
stabala). Hadamard produkt (u elementwise* v) je VEC simetrican (u*v = v*u),
pa ne treba abs-diff trik za simetriju kao kod MLP-a. Model je bukvalno
"nauceni tezinski cosine" -- w=uniformno bi bio proporcionalan cosinusu,
w se uci da naglasi koje dimenzije embeddinga najvise nose signal.

Samo ~1280 parametara (+ bias) -- dramaticno manje od MLP/RF, mnogo bolje
uskladjeno sa ~44-47 nezavisnih komponenti koje imamo. Testira se na
TRENUTNOM (1548) datasetu, LOCO (isti standard kao sav kasniji rad u
sesiji), bootstrap CI protiv cosine baseline-a -- ne stara "sirova razlika
brojeva bez testa" metodologija.

Izlaz:
    output/hadamard_bilinear_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/hadamard_bilinear_l2_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/hadamard_bilinear_l2_1548_per_query.csv")

SEED = 42
NEG_PER_POS_TRAIN = 10
MAX_EPOCHS = 300
PATIENCE = 15
LR = 1e-2
WEIGHT_DECAY = 0.0  # namerno iskljuceno -- mentor trazi PRAVI L2 UNUTAR loss-a, ne AdamW-ov
                     # decoupled weight decay (to je drugaciji mehanizam, videti L2_LAMBDA nize)
L2_LAMBDA = 1e-3  # eksplicitan L2 penal DODAT u loss (Ridge-stil), ne u optimizator

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {aid: i for i, aid in enumerate(all_ids)}
n_candidates = len(all_ids)
embedding_matrix = np.array([embeddings_dict[aid] for aid in all_ids], dtype=np.float32)
embedding_dim = embedding_matrix.shape[1]
cosine_matrix = cosine_similarity(embedding_matrix)

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains("negative|Contested|Risky|NO cross", case=False, na=False)
gold = gold_raw.loc[~negative_mask].copy()

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})
print(f"Gold pairs: {len(gold_pairs)}")

positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}

# =====================================================
# LOCO FOLDS
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
print(f"LOCO folds (connected components): {len(component_list)}")


class HadamardBilinear(nn.Module):
    """y = sigmoid(w . (u*v) + b) -- u*v je Hadamard (elementwise) produkt."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, u, v):
        return self.w(u * v).squeeze(-1)


def sample_negatives(protein_pool, n_needed, seed):
    rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    negatives = set()
    attempts, max_attempts = 0, n_needed * 50 + 2000
    while len(negatives) < n_needed and attempts < max_attempts:
        a, b = rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1
        if pair in positive_pair_set or pair in negatives:
            continue
        negatives.add(pair)
    return sorted(negatives)


def train_fold(train_pairs, train_protein_pool, seed):
    n_neg = len(train_pairs) * NEG_PER_POS_TRAIN
    neg_pairs = sample_negatives(train_protein_pool, n_neg, seed)

    rows_u, rows_v, labels = [], [], []
    for p in train_pairs:
        rows_u.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_v.append(embedding_matrix[id_to_index[p["id_2"]]])
        labels.append(1.0)
    for a, b in neg_pairs:
        rows_u.append(embedding_matrix[id_to_index[a]])
        rows_v.append(embedding_matrix[id_to_index[b]])
        labels.append(0.0)

    U = torch.from_numpy(np.array(rows_u, dtype=np.float32))
    V = torch.from_numpy(np.array(rows_v, dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.float32)

    n = len(y)
    val_size = max(10, int(0.15 * n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx, fit_idx = perm[:val_size], perm[val_size:]

    model = HadamardBilinear(embedding_dim)
    pos_weight = torch.tensor((y[fit_idx] == 0).sum().item() / max((y[fit_idx] == 1).sum().item(), 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = np.inf
    best_state = None
    no_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(U[fit_idx], V[fit_idx])
        bce_loss = criterion(logits, y[fit_idx])
        l2_term = L2_LAMBDA * sum((p ** 2).sum() for p in model.parameters())
        loss = bce_loss + l2_term  # eksplicitan L2 UNUTAR loss-a, ne AdamW weight_decay
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(U[val_idx], V[val_idx])
            val_loss = criterion(val_logits, y[val_idx]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def ranks_from_scores(scores, self_index):
    s = scores.copy()
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


print("\nLOCO training + eval...")
records = []
t0 = time.time()
embedding_tensor = torch.from_numpy(embedding_matrix)

for fi, (root, test_ids) in enumerate(components.items()):
    train_pairs = [p for p in gold_pairs if p["id_1"] not in test_ids and p["id_2"] not in test_ids]
    test_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
    if not test_pairs or not train_pairs:
        continue
    train_pool = set(all_ids) - test_ids

    model = train_fold(train_pairs, train_pool, seed=SEED + fi)

    with torch.no_grad():
        for p in test_pairs:
            for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
                qidx, tidx = id_to_index[qid], id_to_index[tid]
                q_vec = embedding_tensor[qidx].unsqueeze(0).expand(n_candidates, -1)
                logits = model(q_vec, embedding_tensor).numpy()
                model_ranks = ranks_from_scores(logits, qidx)
                cos_ranks = ranks_from_scores(cosine_matrix[qidx].copy(), qidx)
                records.append({"pair_id": p["pair_id"], "fold": fi,
                                 "model_rank": int(model_ranks[tidx]), "cosine_rank": int(cos_ranks[tidx])})

    if (fi + 1) % 10 == 0 or (fi + 1) == len(component_list):
        print(f"  fold {fi+1}/{len(component_list)}  ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}")

model_mrr = (1.0 / df["model_rank"]).mean()
cosine_mrr = (1.0 / df["cosine_rank"]).mean()

rng = np.random.default_rng(SEED)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = resampled["w"].to_numpy()
    d = np.average(1.0 / resampled["model_rank"], weights=w) - np.average(1.0 / resampled["cosine_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
sig = (ci_lo > 0) or (ci_hi < 0)

summary_lines = [
    "=" * 70, "Hadamard bilinear model: y = sigmoid(w . (u*v)), LOCO, ~1280 parametara", "=" * 70, "",
    f"n queries = {len(df)}, n folds = {len(component_list)}", "",
    f"Cosine MRR:            {cosine_mrr:.4f}",
    f"Hadamard bilinear MRR: {model_mrr:.4f}",
    f"Delta: {model_mrr - cosine_mrr:+.4f}",
    f"Bootstrap 95% CI (2000 resample, po pair_id): [{ci_lo:+.4f}, {ci_hi:+.4f}] -- "
    f"{'ZNACAJNO' if sig else 'nije znacajno'}",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
