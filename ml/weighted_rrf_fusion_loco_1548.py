"""
Weighted RRF: umesto fiksnog RRF-3 (score = 1/(K+rang) podjednako za cosine,
BLAST, FoldseekTM), NAUCITI 3 tezine preko LOCO-a. Namerno MINIMALAN broj
parametara (samo 3 tezine, K=60 ostaje fiksno kao ustanovljena konstanta) --
za razliku od RF-a i XGBoost-a (koji su uvek gubili od cosine-a jer imaju
previse parametara za samo 44-47 nezavisnih komponenti), 3 parametra bi
trebalo da mogu pouzdano da se fituju cak i na ovoliko malo nezavisnog
signala.

Metod: pairwise logisticka (RankNet-stil) optimizacija. Za svaki trening
upit (oba pravca gold para), uzorkuje se N_NEG nasumicnih distraktora iz
CELOG pool-a; feature razlika (pravi_target_RRF_features - distraktor_RRF_
features) treba da ima w*diff > 0 (target treba da bude iznad distraktora).
Optimizuje se softplus pairwise loss, w >= 0 (vise slicnosti nikad ne sme
da SMANJI kombinovani skor).

STROGI LOCO: tezine za svaki fold se uce SAMO na parovima ciji OBA kraja
NISU u tom foldu (isti Union-Find fold-split kao svuda u sesiji) -- test
upiti tog folda nikad ne uticu na svoje sopstvene tezine.

Poredi se sa fiksnim RRF-3 (w=[1,1,1]) preko paired bootstrap-a na istim
upitima/foldovima.

Izlaz:
    output/weighted_rrf_fusion_loco_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/weighted_rrf_fusion_loco_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/weighted_rrf_fusion_loco_1548_per_query.csv")

RRF_K = 60
N_NEG_TRAIN = 30
SEED = 42

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

all_ids = metadata["allergen_id"].tolist()
id_to_index = {aid: i for i, aid in enumerate(all_ids)}
n_candidates = len(all_ids)
embedding_matrix = np.array([embeddings_dict[aid] for aid in all_ids], dtype=np.float64)
cosine_matrix = cosine_similarity(embedding_matrix)

perm = np.array([blast_id_to_index.get(aid, -1) for aid in all_ids])
valid = perm >= 0
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

foldseek_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in id_to_index and b in id_to_index:
        i, j = id_to_index[a], id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"], "evidence_level": row["evidence_level"]})
print(f"Gold pairs: {len(gold_pairs)}")

# =====================================================
# LOCO FOLDS (Union-Find over connected components -- identical method
# used throughout the session)
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

fold_of = {}
components = {}
for pid in parent:
    root = find(pid)
    components.setdefault(root, set()).add(pid)
for root, members in components.items():
    for m in members:
        fold_of[m] = root

print(f"LOCO folds (connected components): {len(components)}")


def rrf_features(qidx):
    """Return per-candidate reciprocal-rank feature matrix [n_candidates, 3] for query qidx."""
    def ranks_from_scores(scores, self_index):
        s = scores.astype(np.float64, copy=True)
        s[self_index] = -np.inf
        order = np.argsort(s)[::-1]
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[order] = np.arange(1, len(s) + 1)
        return ranks

    cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
    blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
    fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)
    feats = np.stack([
        1.0 / (RRF_K + cos_ranks),
        1.0 / (RRF_K + blast_ranks),
        1.0 / (RRF_K + fs_ranks),
    ], axis=1)  # [n_candidates, 3]
    return feats


# precompute RRF-3 features for every protein once (reused for both training diffs and eval)
print("Precomputing per-protein RRF feature matrices...")
t0 = time.time()
all_feats = {aid: rrf_features(id_to_index[aid]) for aid in all_ids}
print(f"  done in {(time.time()-t0)/60:.1f} min")

rng = np.random.default_rng(SEED)


def pairwise_loss_and_grad(w, diffs):
    z = diffs @ w
    # softplus(-z) = log(1+exp(-z)), stable version
    loss = np.mean(np.logaddexp(0, -z))
    sig = 1.0 / (1.0 + np.exp(z))  # d/dz softplus(-z) = -sigmoid(-z) = sig-1... derive carefully below
    grad = -(diffs * sig[:, None]).mean(axis=0)
    return loss, grad


def fit_weights(train_pairs):
    diffs = []
    for p in train_pairs:
        for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qidx = id_to_index[qid]
            tidx = id_to_index[tid]
            feats = all_feats[qid]
            pos_feat = feats[tidx]
            neg_idx = rng.choice(n_candidates, size=N_NEG_TRAIN, replace=False)
            neg_idx = neg_idx[(neg_idx != qidx) & (neg_idx != tidx)]
            for ni in neg_idx:
                diffs.append(pos_feat - feats[ni])
    diffs = np.array(diffs)

    def objective(w):
        return pairwise_loss_and_grad(w, diffs)

    res = minimize(objective, x0=np.array([1.0, 1.0, 1.0]), jac=True,
                    bounds=[(0, None)] * 3, method="L-BFGS-B")
    return res.x


print(f"\nRunning LOCO weight fitting + eval ({len(components)} folds)...")
records = []
t0 = time.time()
for fi, (root, test_ids) in enumerate(components.items()):
    train_pairs = [p for p in gold_pairs if fold_of[p["id_1"]] != root]
    test_pairs = [p for p in gold_pairs if fold_of[p["id_1"]] == root]
    if not test_pairs:
        continue

    w = fit_weights(train_pairs)

    for p in test_pairs:
        for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qidx, tidx = id_to_index[qid], id_to_index[tid]
            feats = all_feats[qid]
            combined = feats @ w
            combined[qidx] = -np.inf
            order = np.argsort(combined)[::-1]
            ranks = np.empty(len(combined), dtype=np.int64)
            ranks[order] = np.arange(1, len(combined) + 1)

            plain_rrf = feats @ np.array([1.0, 1.0, 1.0])
            plain_rrf[qidx] = -np.inf
            order2 = np.argsort(plain_rrf)[::-1]
            ranks2 = np.empty(len(plain_rrf), dtype=np.int64)
            ranks2[order2] = np.arange(1, len(plain_rrf) + 1)

            records.append({
                "pair_id": p["pair_id"], "evidence_level": p["evidence_level"],
                "fold": fi, "w_cosine": w[0], "w_blast": w[1], "w_foldseek": w[2],
                "weighted_rank": int(ranks[tidx]), "plain_rrf_rank": int(ranks2[tidx]),
            })

    if (fi + 1) % 10 == 0 or (fi + 1) == len(components):
        elapsed = time.time() - t0
        print(f"  fold {fi+1}/{len(components)}  ({elapsed/60:.1f} min elapsed)  "
              f"last w=[{w[0]:.3f},{w[1]:.3f},{w[2]:.3f}]", flush=True)

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}")

# =====================================================
# AGGREGATE + BOOTSTRAP
# =====================================================

weighted_mrr = (1.0 / df["weighted_rank"]).mean()
plain_mrr = (1.0 / df["plain_rrf_rank"]).mean()

summary_lines = ["=" * 70, "Weighted RRF (LOCO-fit, 3 parametra) vs plain RRF-3 (w=1,1,1)", "=" * 70, "",
                  f"n queries = {len(df)}, n folds = {len(components)}", "",
                  f"Plain RRF-3 MRR:    {plain_mrr:.4f}",
                  f"Weighted RRF MRR:   {weighted_mrr:.4f}",
                  f"Delta:              {weighted_mrr - plain_mrr:+.4f}", ""]

avg_w = df[["w_cosine", "w_blast", "w_foldseek"]].mean()
summary_lines.append(f"Prosecne naucene tezine (preko svih foldova): "
                      f"cosine={avg_w['w_cosine']:.3f}  blast={avg_w['w_blast']:.3f}  "
                      f"foldseek={avg_w['w_foldseek']:.3f}")
summary_lines.append("")

# paired bootstrap by pair_id
rng2 = np.random.default_rng(SEED)
pair_ids = df["pair_id"].unique()
N_BOOTSTRAP = 2000
deltas = []
for _ in range(N_BOOTSTRAP):
    sampled = rng2.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    ww = resampled["w"].to_numpy()
    d = (np.average(1.0 / resampled["weighted_rank"], weights=ww)
         - np.average(1.0 / resampled["plain_rrf_rank"], weights=ww))
    deltas.append(d)
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
frac_better = (deltas > 0).mean()
significant = (ci_lo > 0) or (ci_hi < 0)
verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
summary_lines.append(f"Paired bootstrap (2000 resample, po pair_id): mean delta = {deltas.mean():+.4f}, "
                      f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], weighted bolji u {frac_better:.1%} resample-ova -- {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
