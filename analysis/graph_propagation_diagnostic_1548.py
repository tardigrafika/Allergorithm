"""
Dijagnostika graph-propagation dobitka (RRF-4 vs RRF-3, delta +0.0060,
CI [+0.0017,+0.0100] na leave-ONE-EDGE-out setting-u) -- da li signal
stvarno "premoscuje" ka drugacijim proteinima, ili samo pojacava
already-easy within-family/hub strukturu? Cetiri analize:

  1) Target edge same-family vs cross-family (RRF-3 vs RRF-4 po grupi)
  2) Sastav graph-komsija (koliko je same-family vs cross-family u odnosu
     na query)
  3) "Novi hit" analiza -- upiti gde RRF-4 drasticno popravlja RRF-3, sa
     identitetom "pobednickog" komsije koji je doprineo max graph_score-u
  4) Degree-controlled analiza (low/medium/high stepen cvora) -- da li je
     dobitak generalizovan ili samo hub-bias

Izlaz:
    output/graph_propagation_diagnostic_1548_summary.txt
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/graph_propagation_diagnostic_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/graph_propagation_diagnostic_1548_per_query.csv")

RRF_K = 60
SEED = 42
N_BOOTSTRAP = 2000

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
family_map = {}
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"], "evidence_level": row["evidence_level"]})
    f1, f2 = str(row["family_1"]).strip(), str(row["family_2"]).strip()
    if f1:
        family_map.setdefault(id1, f1)
    if f2:
        family_map.setdefault(id2, f2)
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


print("Precomputing per-protein RRF-3 raw score vectors...")
t0 = time.time()
rrf3_score_vec = {}
for aid in all_ids:
    idx = id_to_index[aid]
    cr = ranks_from_scores(cosine_matrix[idx], idx)
    br = ranks_from_scores(blast_matrix[idx], idx)
    fr = ranks_from_scores(foldseek_matrix[idx], idx)
    rrf3_score_vec[aid] = 1.0 / (RRF_K + cr) + 1.0 / (RRF_K + br) + 1.0 / (RRF_K + fr)
print(f"  done in {(time.time()-t0)/60:.1f} min")

id_to_name = {v: k for k, v in name_to_id.items()}

print("\nScoring all queries + tracking winning neighbor / family composition...")
records = []
for p in gold_pairs:
    for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx, tidx = id_to_index[qid], id_to_index[tid]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)
        rrf3_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)

        neighbors = sorted(adjacency.get(qid, set()) - {tid})
        n_neighbors = len(neighbors)
        if n_neighbors == 0:
            continue  # no graph signal possible for this query, exclude from diagnostic (as before)

        query_fam = family_map.get(qid)
        target_fam = family_map.get(tid)
        same_family_edge = (query_fam is not None and query_fam == target_fam)

        n_same_fam_neighbors = sum(1 for n in neighbors if family_map.get(n) == query_fam)
        frac_same_fam_neighbors = n_same_fam_neighbors / n_neighbors

        # graph score + which neighbor achieved the max AT THE TARGET INDEX specifically
        neighbor_scores_at_target = np.array([rrf3_score_vec[n][tidx] for n in neighbors])
        winner_idx = int(np.argmax(neighbor_scores_at_target))
        winner_id = neighbors[winner_idx]
        winner_fam = family_map.get(winner_id)
        winner_same_fam_as_query = (winner_fam is not None and winner_fam == query_fam)
        winner_same_fam_as_target = (winner_fam is not None and winner_fam == target_fam)

        graph_score = np.max([rrf3_score_vec[n] for n in neighbors], axis=0)
        graph_ranks = ranks_from_scores(graph_score, qidx)
        rrf4_score = rrf3_score + 1.0 / (RRF_K + graph_ranks)
        rrf4_ranks = ranks_from_scores(rrf4_score, qidx)

        records.append({
            "pair_id": p["pair_id"], "evidence_level": p["evidence_level"],
            "query_id": qid, "query_name": id_to_name.get(qid, qid),
            "target_id": tid, "target_name": id_to_name.get(tid, tid),
            "query_family": query_fam, "target_family": target_fam,
            "same_family_edge": same_family_edge,
            "n_other_neighbors": n_neighbors, "frac_same_fam_neighbors": frac_same_fam_neighbors,
            "winner_id": winner_id, "winner_name": id_to_name.get(winner_id, winner_id),
            "winner_family": winner_fam,
            "winner_same_fam_as_query": winner_same_fam_as_query,
            "winner_same_fam_as_target": winner_same_fam_as_target,
            "rrf3_rank": int(rrf3_ranks[tidx]), "rrf4_rank": int(rrf4_ranks[tidx]),
        })

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}  ({len(df)} upita sa dostupnim graph signalom)")


def bootstrap_delta(sub, rng):
    pair_ids = sub["pair_id"].unique()
    if len(pair_ids) < 5:
        return None
    deltas = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on="pair_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = (np.average(1.0 / resampled["rrf4_rank"], weights=w)
             - np.average(1.0 / resampled["rrf3_rank"], weights=w))
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    sig = (ci_lo > 0) or (ci_hi < 0)
    return deltas.mean(), ci_lo, ci_hi, sig


rng = np.random.default_rng(SEED)
summary_lines = ["=" * 70, "Graph-propagation dijagnostika (van cross-family/same-family povrsinske podele)",
                  "=" * 70, ""]

# --- Analiza 1: same-family vs cross-family TARGET edge ---
summary_lines.append("--- 1) Target edge: same-family vs cross-family ---")
for label, sub in [("same-family", df[df["same_family_edge"]]), ("cross-family", df[~df["same_family_edge"]])]:
    rrf3_mrr = (1.0 / sub["rrf3_rank"]).mean()
    rrf4_mrr = (1.0 / sub["rrf4_rank"]).mean()
    res = bootstrap_delta(sub, rng)
    if res is None:
        summary_lines.append(f"  {label}: n={len(sub)} -- PREMALO za bootstrap (underpowered, ne negativan rezultat)")
    else:
        mean_d, lo, hi, sig = res
        verdict = "ZNACAJNO" if sig else "nije znacajno"
        summary_lines.append(f"  {label}: n={len(sub)}  RRF-3={rrf3_mrr:.4f}  RRF-4={rrf4_mrr:.4f}  "
                              f"delta={mean_d:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}] -- {verdict}")
summary_lines.append("")

# --- Analiza 2: sastav graph-komsija ---
summary_lines.append("--- 2) Sastav graph-komsija (frac_same_fam_neighbors po upitu) ---")
summary_lines.append(f"  Prosecan udeo iste-familije komsija: {df['frac_same_fam_neighbors'].mean():.2%}")
summary_lines.append(f"  Medijana: {df['frac_same_fam_neighbors'].median():.2%}")
summary_lines.append(f"  Upiti sa ISKLJUCIVO iste-familije komsijama (frac=1.0): "
                      f"{(df['frac_same_fam_neighbors']==1.0).mean():.1%}")
summary_lines.append(f"  Upiti sa ISKLJUCIVO cross-familije komsijama (frac=0.0): "
                      f"{(df['frac_same_fam_neighbors']==0.0).mean():.1%}")
winner_stats = df["winner_same_fam_as_query"].value_counts(normalize=True)
summary_lines.append(f"  'Pobednicki' komsija (koji je doprineo max graph_score) je iste familije kao QUERY "
                      f"u {winner_stats.get(True,0):.1%} slucajeva")
winner_target_stats = df["winner_same_fam_as_target"].value_counts(normalize=True)
summary_lines.append(f"  'Pobednicki' komsija je iste familije kao TARGET u {winner_target_stats.get(True,0):.1%} slucajeva")
summary_lines.append("")

# --- Analiza 3: "novi hit" -- RRF-4 drasticno popravlja los RRF-3 rang ---
summary_lines.append("--- 3) 'Novi hit' analiza (RRF-3 rank >= 50, RRF-4 rank <= 20) ---")
big_wins = df[(df["rrf3_rank"] >= 50) & (df["rrf4_rank"] <= 20)].sort_values("rrf3_rank", ascending=False)
summary_lines.append(f"  Broj 'big win' slucajeva: {len(big_wins)} / {len(df)} ukupno")
if len(big_wins) > 0:
    cross_fam_bigwin = (~big_wins["same_family_edge"]).sum()
    winner_crossfam_query = (~big_wins["winner_same_fam_as_query"]).sum()
    summary_lines.append(f"  Od toga cross-family target edge: {cross_fam_bigwin}/{len(big_wins)}")
    summary_lines.append(f"  Od toga 'pobednicki' komsija je CROSS-family u odnosu na query: "
                          f"{winner_crossfam_query}/{len(big_wins)}")
    summary_lines.append("  Detalji (do 15 primera, sortirano po najvecem RRF-3 rangu):")
    for _, r in big_wins.head(15).iterrows():
        bridge = "CROSS-family bridge" if not r["winner_same_fam_as_query"] else "same-family reinforcement"
        summary_lines.append(
            f"    {r['query_name']} -> {r['target_name']} ({'same' if r['same_family_edge'] else 'CROSS'}-family edge): "
            f"RRF-3={r['rrf3_rank']} -> RRF-4={r['rrf4_rank']}  |  komsija={r['winner_name']} ({r['winner_family']})  "
            f"-- {bridge}"
        )
else:
    summary_lines.append("  Nema slucajeva koji zadovoljavaju prag.")
summary_lines.append("")

# --- Analiza 4: degree-controlled ---
summary_lines.append("--- 4) Degree-controlled analiza (n_other_neighbors) ---")
q1, q2 = df["n_other_neighbors"].quantile([0.33, 0.66])
print(f"Degree tercile granice: {q1:.1f}, {q2:.1f}")


def degree_bucket(n):
    if n <= q1:
        return "low"
    elif n <= q2:
        return "medium"
    else:
        return "high"


df["degree_bucket"] = df["n_other_neighbors"].apply(degree_bucket)
for bucket in ["low", "medium", "high"]:
    sub = df[df["degree_bucket"] == bucket]
    rrf3_mrr = (1.0 / sub["rrf3_rank"]).mean()
    rrf4_mrr = (1.0 / sub["rrf4_rank"]).mean()
    res = bootstrap_delta(sub, rng)
    rng_range = sub["n_other_neighbors"].min(), sub["n_other_neighbors"].max()
    if res is None:
        summary_lines.append(f"  {bucket} (n_neighbors {rng_range[0]}-{rng_range[1]}): n={len(sub)} -- PREMALO za bootstrap")
    else:
        mean_d, lo, hi, sig = res
        verdict = "ZNACAJNO" if sig else "nije znacajno"
        summary_lines.append(f"  {bucket} (n_neighbors {rng_range[0]}-{rng_range[1]}): n={len(sub)}  "
                              f"RRF-3={rrf3_mrr:.4f}  RRF-4={rrf4_mrr:.4f}  delta={mean_d:+.4f}  "
                              f"95% CI [{lo:+.4f},{hi:+.4f}] -- {verdict}")
summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
