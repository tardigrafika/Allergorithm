"""
Retrieve-then-rerank: umesto ucenja na CELOJ bazi od 1534 kandidata (gde su
RF/XGBoost/weighted-fusion svi propali - previse lak zadatak sa nasumicnim
negativima, ili premalo nezavisnih podataka za toliko parametara), suzi
problem: uzmi top-K=50 iz RRF-4 (K=20), pa nauci MALI model da SAMO njih
preuredi. Kljucna razlika od weighted_rrf4_fusion_1548.py: trening negativi
su sada TESKI (ostali top-50 kandidati ZA ISTI upit), ne nasumicni iz cele
baze -- potpuno drugaciji trening signal.

Plafon metoda: ako pravi cilj NIJE vec u top-K, reranking ga ne moze spasti
-- to se belezi odvojeno (ne meri se kao "neuspeh" rerankera).

LOCO: component-level (44 folda) -- ovde nema graph-propagation strukturne
prepreke (feature vektori se racunaju unapred po upitu, ne treba live pristup
susedima van folda za SAM rerank korak, samo za samu graph_feat vrednost
koja je vec izracunata van ovog skripta... zapravo NIJE, graph feature i
dalje zahteva leave-one-edge-out. Zato se, kao i ranije, koristi 10
nasumicnih edge-level foldova, ne po komponenti.

Izlaz:
    output/retrieve_rerank_1548_summary.txt
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/retrieve_rerank_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/retrieve_rerank_1548_per_query.csv")

RRF_K = 20
TOP_K_RETRIEVE = 50
N_FOLDS = 10
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
rrf3_score_vec = {}
for aid in all_ids:
    idx = id_to_index[aid]
    cr = ranks_from_scores(cosine_matrix[idx], idx)
    br = ranks_from_scores(blast_matrix[idx], idx)
    fr = ranks_from_scores(foldseek_matrix[idx], idx)
    rrf3_score_vec[aid] = 1.0 / (RRF_K + cr) + 1.0 / (RRF_K + br) + 1.0 / (RRF_K + fr)

print("\nRetrieving top-K per upit i gradim RAW feature vektore (cosine_sim, blast_score, fs_score, graph_score)...")
t0 = time.time()
queries = []
n_ceiling_miss = 0
for p in gold_pairs:
    for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx, tidx = id_to_index[qid], id_to_index[tid]

        cr = ranks_from_scores(cosine_matrix[qidx], qidx)
        br = ranks_from_scores(blast_matrix[qidx], qidx)
        fr = ranks_from_scores(foldseek_matrix[qidx], qidx)
        rrf3 = 1.0 / (RRF_K + cr) + 1.0 / (RRF_K + br) + 1.0 / (RRF_K + fr)

        neighbors = adjacency.get(qid, set()) - {tid}
        has_graph = len(neighbors) > 0
        if has_graph:
            graph_score = np.max([rrf3_score_vec[n] for n in neighbors], axis=0)
            gr = ranks_from_scores(graph_score, qidx)
            graph_feat_full = 1.0 / (RRF_K + gr)
        else:
            graph_feat_full = np.zeros(n_candidates)

        rrf4 = rrf3 + graph_feat_full
        rrf4[qidx] = -np.inf
        order = np.argsort(rrf4)[::-1]
        top_k_idx = order[:TOP_K_RETRIEVE]

        rrf4_rank = int(np.where(order == tidx)[0][0]) + 1 if tidx in order else None

        if tidx not in top_k_idx:
            n_ceiling_miss += 1
            queries.append({"pair_id": p["pair_id"], "qidx": qidx, "tidx": tidx,
                             "in_top_k": False, "rrf4_rank": rrf4_rank, "has_graph": has_graph})
            continue

        raw_feats = np.stack([
            cosine_matrix[qidx][top_k_idx],
            blast_matrix[qidx][top_k_idx],
            foldseek_matrix[qidx][top_k_idx],
            graph_feat_full[top_k_idx],
        ], axis=1)

        queries.append({"pair_id": p["pair_id"], "qidx": qidx, "tidx": tidx, "in_top_k": True,
                         "rrf4_rank": rrf4_rank, "has_graph": has_graph,
                         "top_k_idx": top_k_idx, "raw_feats": raw_feats,
                         "target_pos_in_topk": int(np.where(top_k_idx == tidx)[0][0])})

print(f"  {len(queries)} upita, {(time.time()-t0)/60:.1f} min")
print(f"  Plafon: {n_ceiling_miss}/{len(queries)} ({n_ceiling_miss/len(queries):.1%}) "
      f"pravih ciljeva NIJE u top-{TOP_K_RETRIEVE} -- reranking ih ne moze spasti")

rerankable = [q for q in queries if q["in_top_k"]]
print(f"  Rerankable (cilj u top-{TOP_K_RETRIEVE}): {len(rerankable)}")

# normalizacija raw feature-a (razlicite skale: cosine~[0,1], blast score
# neogranicen, foldseek~[0,1], graph~[0, mali])  -- z-score PO UPITU, preko
# top-K kandidata tog upita, da skale budu uporedive pre fitovanja tezina
for q in rerankable:
    mu = q["raw_feats"].mean(axis=0)
    sd = q["raw_feats"].std(axis=0) + 1e-9
    q["norm_feats"] = (q["raw_feats"] - mu) / sd

rng = np.random.default_rng(SEED)
fold_of = rng.integers(0, N_FOLDS, size=len(rerankable))


def pairwise_loss_and_grad(w, diffs):
    z = diffs @ w
    loss = np.mean(np.logaddexp(0, -z))
    sig = 1.0 / (1.0 + np.exp(z))
    grad = -(diffs * sig[:, None]).mean(axis=0)
    return loss, grad


def fit_reranker(train_qs):
    diffs = []
    for q in train_qs:
        pos = q["norm_feats"][q["target_pos_in_topk"]]
        for i in range(len(q["norm_feats"])):
            if i == q["target_pos_in_topk"]:
                continue
            diffs.append(pos - q["norm_feats"][i])  # TESKI negativi -- ostali top-K za ISTI upit
    diffs = np.array(diffs)

    def objective(w):
        return pairwise_loss_and_grad(w, diffs)

    res = minimize(objective, x0=np.array([1.0, 1.0, 1.0, 1.0]), jac=True,
                    bounds=[(0, None)] * 4, method="L-BFGS-B")
    return res.x


print(f"\nFitovanje rerankera preko {N_FOLDS} nasumicnih edge-level foldova (teski negativi)...")
records = []
for fi in range(N_FOLDS):
    train_qs = [q for i, q in enumerate(rerankable) if fold_of[i] != fi]
    test_qs = [q for i, q in enumerate(rerankable) if fold_of[i] == fi]
    if not test_qs:
        continue
    w = fit_reranker(train_qs)

    for q in test_qs:
        rerank_scores = q["norm_feats"] @ w
        order_local = np.argsort(rerank_scores)[::-1]
        new_rank_within_topk = int(np.where(order_local == q["target_pos_in_topk"])[0][0]) + 1
        records.append({"pair_id": q["pair_id"], "fold": fi,
                         "rrf4_rank": q["rrf4_rank"], "reranked_rank": new_rank_within_topk})
    print(f"  fold {fi+1}/{N_FOLDS}  w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]", flush=True)

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)

rrf4_mrr = (1.0 / df["rrf4_rank"]).mean()
reranked_mrr = (1.0 / df["reranked_rank"]).mean()

rng2 = np.random.default_rng(SEED)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng2.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w_ = resampled["w"].to_numpy()
    d = (np.average(1.0 / resampled["reranked_rank"], weights=w_) - np.average(1.0 / resampled["rrf4_rank"], weights=w_))
    deltas.append(d)
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
sig = (ci_lo > 0) or (ci_hi < 0)

summary_lines = ["=" * 70, f"Retrieve-then-rerank (top-{TOP_K_RETRIEVE}, teski negativi)", "=" * 70, "",
                  f"Ukupno upita: {len(queries)}",
                  f"Plafon (cilj NIJE u top-{TOP_K_RETRIEVE}): {n_ceiling_miss} ({n_ceiling_miss/len(queries):.1%}) "
                  f"-- reranking ih strukturno ne moze popraviti",
                  f"Rerankable upiti (cilj JESTE u top-{TOP_K_RETRIEVE}): {len(rerankable)}", "",
                  f"RRF-4 MRR (na rerankable podskupu, unutar top-{TOP_K_RETRIEVE}): {rrf4_mrr:.4f}",
                  f"Reranked MRR (unutar top-{TOP_K_RETRIEVE}):                     {reranked_mrr:.4f}",
                  f"Delta: {reranked_mrr - rrf4_mrr:+.4f}",
                  f"Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {'ZNACAJNO' if sig else 'nije znacajno'}"]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
