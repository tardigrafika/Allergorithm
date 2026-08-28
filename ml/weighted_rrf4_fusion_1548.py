"""
Zajednicko optimizovanje SVA 4 signala (cosine, BLAST, FoldseekTM, graph-
propagation) -- do sad su weighted fusion (3 signala) i graph-propagation
testirani ODVOJENO. Ovde: nauci 4 tezine ZAJEDNO, isti pairwise-logisticki
metod kao ml/weighted_rrf_fusion_loco_1548.py, K=20 (ml/rrf_k_sensitivity
nalaz, transferabilno potvrdjeno split-half testom).

LOCO napomena: graph feature strukturno ne moze da radi pod leave-ONE-
COMPONENT-out (isti razlog kao ml/graph_propagation_signal_1548.py -- upit
u potpuno izbacenom foldu nema nijednog vidljivog komsije). Zato se ovde
koristi 10 NASUMICNIH edge-level foldova (ne po komponenti) -- svaki test
upit i dalje NIKAD ne vidi SVOJU sopstvenu ivicu pri fitovanju tezina
(njegov graph feature racuna se leave-ONE-EDGE-out), ali drugi upiti u
drugim foldovima slobodno koriste SVE svoje ostale poznate komsije.

Izlaz:
    output/weighted_rrf4_fusion_1548_summary.txt
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/weighted_rrf4_fusion_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/weighted_rrf4_fusion_1548_per_query.csv")

RRF_K = 20  # ml/rrf_k_sensitivity_1548.py nalaz (bolje od K=60, transferabilno)
N_FOLDS = 10
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
negative_mask = gold_raw["evidence_level"].str.contains("negative|Contested|Risky|NO cross", case=False, na=False)
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
valid_idx = np.where(perm >= 0)[0]
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
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
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})
print(f"Gold pairs: {len(gold_pairs)}")

adjacency = {}
for p in gold_pairs:
    adjacency.setdefault(p["id_1"], set()).add(p["id_2"])
    adjacency.setdefault(p["id_2"], set()).add(p["id_1"])


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


print("Precomputing per-protein RRF-3 raw score vectors (za graph feature)...")
t0 = time.time()
rrf3_score_vec = {}
for aid in all_ids:
    idx = id_to_index[aid]
    cr = ranks_from_scores(cosine_matrix[idx], idx)
    br = ranks_from_scores(blast_matrix[idx], idx)
    fr = ranks_from_scores(foldseek_matrix[idx], idx)
    rrf3_score_vec[aid] = 1.0 / (RRF_K + cr) + 1.0 / (RRF_K + br) + 1.0 / (RRF_K + fr)
print(f"  done in {(time.time()-t0)/60:.1f} min")

print("\nBuilding 4-feature vectors per query (leave-one-edge-out za graph)...")
t0 = time.time()
queries = []  # list of dicts: pair_id, qid, tid, feats (n_candidates x 4), has_graph
for p in gold_pairs:
    for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx, tidx = id_to_index[qid], id_to_index[tid]
        cr = ranks_from_scores(cosine_matrix[qidx], qidx)
        br = ranks_from_scores(blast_matrix[qidx], qidx)
        fr = ranks_from_scores(foldseek_matrix[qidx], qidx)
        cos_feat = 1.0 / (RRF_K + cr)
        blast_feat = 1.0 / (RRF_K + br)
        fs_feat = 1.0 / (RRF_K + fr)

        neighbors = adjacency.get(qid, set()) - {tid}
        has_graph = len(neighbors) > 0
        if has_graph:
            graph_score = np.max([rrf3_score_vec[n] for n in neighbors], axis=0)
            gr = ranks_from_scores(graph_score, qidx)
            graph_feat = 1.0 / (RRF_K + gr)
        else:
            graph_feat = np.zeros(n_candidates, dtype=np.float64)

        feats = np.stack([cos_feat, blast_feat, fs_feat, graph_feat], axis=1)
        queries.append({"pair_id": p["pair_id"], "qidx": qidx, "tidx": tidx, "feats": feats, "has_graph": has_graph})
print(f"  {len(queries)} upita, {(time.time()-t0)/60:.1f} min")

applicable = [q for q in queries if q["has_graph"]]
print(f"Upiti sa graph signalom (koriste se za fer poredjenje sa RRF-3/RRF-4): {len(applicable)}/{len(queries)}")

rng = np.random.default_rng(SEED)
fold_of_query = rng.integers(0, N_FOLDS, size=len(queries))


def pairwise_loss_and_grad(w, diffs):
    z = diffs @ w
    loss = np.mean(np.logaddexp(0, -z))
    sig = 1.0 / (1.0 + np.exp(z))
    grad = -(diffs * sig[:, None]).mean(axis=0)
    return loss, grad


def fit_weights(train_queries):
    diffs = []
    for q in train_queries:
        pos_feat = q["feats"][q["tidx"]]
        neg_idx = rng.choice(n_candidates, size=N_NEG_TRAIN, replace=False)
        neg_idx = neg_idx[(neg_idx != q["qidx"]) & (neg_idx != q["tidx"])]
        for ni in neg_idx:
            diffs.append(pos_feat - q["feats"][ni])
    diffs = np.array(diffs)

    def objective(w):
        return pairwise_loss_and_grad(w, diffs)

    res = minimize(objective, x0=np.array([1.0, 1.0, 1.0, 1.0]), jac=True,
                    bounds=[(0, None)] * 4, method="L-BFGS-B")
    return res.x


print(f"\nFitovanje tezina preko {N_FOLDS} nasumicnih edge-level foldova...")
records = []
t0 = time.time()
for fi in range(N_FOLDS):
    train_queries = [q for i, q in enumerate(queries) if fold_of_query[i] != fi]
    test_queries = [q for i, q in enumerate(queries) if fold_of_query[i] == fi and q["has_graph"]]
    if not test_queries:
        continue
    w = fit_weights(train_queries)

    for q in test_queries:
        weighted_score = q["feats"] @ w
        weighted_score[q["qidx"]] = -np.inf
        wranks = ranks_from_scores(weighted_score, q["qidx"])

        plain3 = q["feats"][:, :3].sum(axis=1)  # cosine+blast+foldseek, unweighted (RRF-3)
        plain3[q["qidx"]] = -np.inf
        r3 = ranks_from_scores(plain3, q["qidx"])

        plain4 = q["feats"].sum(axis=1)  # + graph, unweighted (RRF-4, validated)
        plain4[q["qidx"]] = -np.inf
        r4 = ranks_from_scores(plain4, q["qidx"])

        records.append({"pair_id": q["pair_id"], "fold": fi,
                         "w_cosine": w[0], "w_blast": w[1], "w_foldseek": w[2], "w_graph": w[3],
                         "weighted4_rank": int(wranks[q["tidx"]]),
                         "rrf3_rank": int(r3[q["tidx"]]), "rrf4_rank": int(r4[q["tidx"]])})
    print(f"  fold {fi+1}/{N_FOLDS}  w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]  "
          f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}")

# =====================================================
# AGGREGATE + BOOTSTRAP
# =====================================================

rrf3_mrr = (1.0 / df["rrf3_rank"]).mean()
rrf4_mrr = (1.0 / df["rrf4_rank"]).mean()
weighted4_mrr = (1.0 / df["weighted4_rank"]).mean()
avg_w = df[["w_cosine", "w_blast", "w_foldseek", "w_graph"]].mean()

summary_lines = ["=" * 70, "Zajednicko 4-signal fitovanje (cosine+BLAST+FoldseekTM+graph), K=20",
                  "=" * 70, "",
                  f"n upita (sa graph signalom, {N_FOLDS} nasumicnih edge foldova) = {len(df)}", "",
                  f"RRF-3 MRR (bez graph):          {rrf3_mrr:.4f}",
                  f"RRF-4 MRR (+graph, uniformno):   {rrf4_mrr:.4f}",
                  f"Weighted-4 MRR (naucene tezine): {weighted4_mrr:.4f}", "",
                  f"Prosecne naucene tezine: cosine={avg_w['w_cosine']:.3f}  blast={avg_w['w_blast']:.3f}  "
                  f"foldseek={avg_w['w_foldseek']:.3f}  graph={avg_w['w_graph']:.3f}", ""]


def bootstrap_delta(col_a, col_b, label):
    pair_ids = df["pair_id"].unique()
    deltas = []
    for _ in range(2000):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
        ww = resampled["w"].to_numpy()
        d = (np.average(1.0 / resampled[col_a], weights=ww) - np.average(1.0 / resampled[col_b], weights=ww))
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    sig = (ci_lo > 0) or (ci_hi < 0)
    verdict = "ZNACAJNO" if sig else "nije znacajno"
    return f"  {label}: mean delta = {deltas.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}"


summary_lines.append("Paired bootstrap (2000 resample, po pair_id):")
summary_lines.append(bootstrap_delta("weighted4_rank", "rrf4_rank", "Weighted-4 vs RRF-4 (uniformno)"))
summary_lines.append(bootstrap_delta("weighted4_rank", "rrf3_rank", "Weighted-4 vs RRF-3 (bez graph)"))
summary_lines.append(bootstrap_delta("rrf4_rank", "rrf3_rank", "RRF-4 (uniformno) vs RRF-3 -- provera da se poklapa sa ranijim nalazom"))

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
