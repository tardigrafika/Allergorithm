"""
Graph-propagation signal: genuinski nova vrsta informacije, za razliku od
svega dosad probanog (cosine/BLAST/Foldseek/kmer/surface/Pfam/Ankh/glyko/
LoRA su SVI content-based -- mere slicnost sekvence/strukture DVA proteina).
Ovde: ako je upit Q vec POZNATO cross-reaktivan sa N(Q) (njegovim drugim
potvrdjenim partnerima), a kandidat C je jako sadrzajno slican nekom
n iz N(Q), to samo po sebi povecava verovatnocu Q-C, nezavisno od direktne
Q-C slicnosti (isti princip kao collaborative filtering).

METODOLOSKA NAPOMENA (vazno, drugacije od svih dosadasnjih LOCO skripti):
Ovaj signal STRUKTURNO ne moze da se testira pod punim leave-ONE-COMPONENT-
out LOCO -- ako se izbaci CEO fold, upit iz tog folda nema NIJEDNOG poznatog
komsiju (svi njegovi gold parovi su, po definiciji Union-Find komponente,
UNUTAR istog folda). Zato se ovde koristi leave-ONE-EDGE-out: iz grafa se
izbaci SAMO testirana ivica Q-T, ali OSTALI poznati partneri Q (ako postoje)
ostaju vidljivi. Ovo odgovara stvarnom slucaju upotrebe alata (pacijent vec
ima >=1 potvrdjenu alergiju, trazimo ostale) -- ali NIJE isti, striktniji
"cold start" test kao ranije u sesiji. Queries bez ijednog drugog poznatog
partnera (N(Q) prazno posle uklanjanja testirane ivice) nemaju graph signal
i prijavljuju se odvojeno.

Izlaz:
    output/graph_propagation_signal_1548_summary.txt
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/graph_propagation_signal_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/graph_propagation_signal_1548_per_query.csv")

RRF_K = 60
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

adjacency = {}
for p in gold_pairs:
    adjacency.setdefault(p["id_1"], set()).add(p["id_2"])
    adjacency.setdefault(p["id_2"], set()).add(p["id_1"])

deg = [len(v) for v in adjacency.values()]
print(f"Proteini sa >=1 poznatim partnerom: {len(adjacency)}, "
      f"prosecan stepen={np.mean(deg):.2f}, medijana={np.median(deg):.0f}, max={np.max(deg)}")


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


print("Precomputing per-protein RRF-3 raw score vectors (za graph propagation)...")
t0 = time.time()
rrf3_score_vec = {}
for aid in all_ids:
    idx = id_to_index[aid]
    cr = ranks_from_scores(cosine_matrix[idx], idx)
    br = ranks_from_scores(blast_matrix[idx], idx)
    fr = ranks_from_scores(foldseek_matrix[idx], idx)
    rrf3_score_vec[aid] = 1.0 / (RRF_K + cr) + 1.0 / (RRF_K + br) + 1.0 / (RRF_K + fr)
print(f"  done in {(time.time()-t0)/60:.1f} min")

print("\nScoring all queries (RRF-3 vs RRF-4 sa graph propagation)...")
records = []
for p in gold_pairs:
    for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx, tidx = id_to_index[qid], id_to_index[tid]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)
        rrf3_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)

        # leave-ONE-EDGE-out: remove just this tested edge from Q's neighbor set
        neighbors = adjacency.get(qid, set()) - {tid}
        has_graph_signal = len(neighbors) > 0

        if has_graph_signal:
            graph_score = np.max([rrf3_score_vec[n] for n in neighbors], axis=0)
            graph_ranks = ranks_from_scores(graph_score, qidx)
            rrf4_score = rrf3_score + 1.0 / (RRF_K + graph_ranks)
            rrf4_ranks = ranks_from_scores(rrf4_score, qidx)
        else:
            graph_ranks = None
            rrf4_ranks = rrf3_ranks  # no signal available, RRF-4 collapses to RRF-3

        records.append({
            "pair_id": p["pair_id"], "evidence_level": p["evidence_level"],
            "query_id": qid, "n_other_neighbors": len(neighbors),
            "has_graph_signal": has_graph_signal,
            "rrf3_rank": int(rrf3_ranks[tidx]),
            "rrf4_rank": int(rrf4_ranks[tidx]),
        })

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}")

applicable = df[df["has_graph_signal"]]
print(f"\nUpiti sa dostupnim graph signalom (>=1 drugi poznati partner): "
      f"{len(applicable)}/{len(df)} ({len(applicable)/len(df):.1%})")

# =====================================================
# AGGREGATE + BOOTSTRAP (samo na upitima gde je signal primenjiv)
# =====================================================

rrf3_mrr_all = (1.0 / df["rrf3_rank"]).mean()
rrf3_mrr_applicable = (1.0 / applicable["rrf3_rank"]).mean()
rrf4_mrr_applicable = (1.0 / applicable["rrf4_rank"]).mean()

summary_lines = ["=" * 70, "Graph-propagation signal (leave-ONE-EDGE-out) vs plain RRF-3", "=" * 70, "",
                  f"Ukupno upita: {len(df)}, sa dostupnim graph signalom: {len(applicable)} ({len(applicable)/len(df):.1%})",
                  f"Prosecan broj drugih poznatih partnera (kad ih ima): "
                  f"{applicable['n_other_neighbors'].mean():.2f}", "",
                  f"RRF-3 MRR (svi upiti, referenca): {rrf3_mrr_all:.4f}",
                  f"RRF-3 MRR (samo primenjivi upiti): {rrf3_mrr_applicable:.4f}",
                  f"RRF-4 MRR (samo primenjivi upiti, +graph): {rrf4_mrr_applicable:.4f}",
                  f"Delta (na primenjivim upitima): {rrf4_mrr_applicable - rrf3_mrr_applicable:+.4f}", ""]

rng = np.random.default_rng(SEED)
pair_ids = applicable["pair_id"].unique()
N_BOOTSTRAP = 2000
deltas = []
for _ in range(N_BOOTSTRAP):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = applicable.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = resampled["w"].to_numpy()
    d = (np.average(1.0 / resampled["rrf4_rank"], weights=w)
         - np.average(1.0 / resampled["rrf3_rank"], weights=w))
    deltas.append(d)
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
frac_better = (deltas > 0).mean()
significant = (ci_lo > 0) or (ci_hi < 0)
verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
summary_lines.append(f"Paired bootstrap (2000 resample, po pair_id, samo primenjivi upiti): "
                      f"mean delta = {deltas.mean():+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
                      f"RRF-4 bolji u {frac_better:.1%} resample-ova -- {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
