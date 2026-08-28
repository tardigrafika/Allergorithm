"""
Conservation-weighted signal: SVI dosadasnji pristupi (cosine/BLAST/Foldseek/
surface/epitope) su PARNI -- porede tacno 2 proteina. Ovde koristimo da
POSTOJI vise poznatih clanova iste familije odjednom: napravi "center-star"
pseudo-MSA nsLTP familije (29 proteina, referenca = Pru p 3 -- literatura ga
vec zove "best marker of LTP sensitization", ne proizvoljan izbor), izracunaj
KONZERVIRANOST po poziciji preko cele familije, pa tezinski favorizuj
rezidue-slicnost BAS na konzervisanim pozicijama -- populacioni, ne parni
signal.

Nema pravog MSA alata lokalno (mafft/clustalo trazi sudo, nedostupno) --
koristi se Bio.Align.PairwiseAligner (BLOSUM62, global) svaki clan protiv
reference (aproksimacija punog MSA-a, "star alignment").

Test OGRANICEN na nsLTP-only pool (29 proteina, nsLTP-nsLTP gold parovi) --
metoda strukturno zahteva da OBA proteina budu alignable na referencu, pa
se ne moze fer testirati na celoj bazi od 1534 u jednom potezu.

Izlaz:
    output/conservation_weighted_nsltp_1548_summary.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
POOLED_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/conservation_weighted_nsltp_1548_summary.txt")

REFERENCE_NAME = "Pru p 3.0101"
SEED = 42

clean = pd.read_csv(CLEAN_ALLERGENS)
name_to_id = dict(zip(clean["official_name"], clean["allergen_id"]))
id_to_name = {v: k for k, v in name_to_id.items()}
id_to_seq = dict(zip(clean["allergen_id"], clean["fasta_sequence"]))

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains("negative|Contested|Risky|NO cross", case=False, na=False)
gold = gold_raw.loc[~negative_mask].copy()

family_map = {}
for _, row in gold.iterrows():
    for col_id, col_fam in [("allergen_id_1", "family_1"), ("allergen_id_2", "family_2")]:
        n = str(row[col_id]).strip()
        f = str(row[col_fam]).strip()
        aid = name_to_id.get(n)
        if aid and f:
            family_map.setdefault(aid, f)

nsltp_ids = sorted(aid for aid, f in family_map.items() if f == "nsLTP")
print(f"nsLTP proteini: {len(nsltp_ids)}")

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)
with open(POOLED_EMBEDDINGS, "rb") as f:
    pooled_embeddings = pickle.load(f)

nsltp_ids = [aid for aid in nsltp_ids if aid in residue_embeddings and aid in pooled_embeddings
             and id_to_seq.get(aid) and len(id_to_seq[aid]) == residue_embeddings[aid].shape[0]]
print(f"nsLTP proteini sa validnim rezidue-embeddinzima: {len(nsltp_ids)}")

ref_id = name_to_id[REFERENCE_NAME]
assert ref_id in nsltp_ids, "referenca mora biti u nsLTP skupu"
ref_seq = id_to_seq[ref_id]
ref_len = len(ref_seq)
print(f"Referenca: {REFERENCE_NAME} ({ref_id}), duzina {ref_len}")

aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.mode = "global"

# za svaki nsLTP protein: mapiranje ref_position -> original_seq_position (ili None ako gap)
ref_pos_to_seq_pos = {}
for aid in nsltp_ids:
    if aid == ref_id:
        ref_pos_to_seq_pos[aid] = {i: i for i in range(ref_len)}
        continue
    seq = id_to_seq[aid]
    aln = aligner.align(ref_seq, seq)[0]
    ref_aligned, query_aligned = aln[0], aln[1]
    mapping = {}
    ref_pos = -1
    seq_pos = -1
    for rc, qc in zip(ref_aligned, query_aligned):
        if rc != "-":
            ref_pos += 1
        if qc != "-":
            seq_pos += 1
        if rc != "-" and qc != "-":
            mapping[ref_pos] = seq_pos
    ref_pos_to_seq_pos[aid] = mapping

print("Poravnanje zavrseno. Racunam konzerviranost po poziciji...")

# konzerviranost[i] = udeo clanova cija rezidua na toj ref poziciji je IDENTICNA referentnoj
conservation = np.zeros(ref_len)
counts = np.zeros(ref_len)
for i in range(ref_len):
    ref_res = ref_seq[i]
    match = 0
    total = 0
    for aid in nsltp_ids:
        mapping = ref_pos_to_seq_pos[aid]
        if i in mapping:
            total += 1
            if id_to_seq[aid][mapping[i]] == ref_res:
                match += 1
    conservation[i] = match / total if total > 0 else 0.0
    counts[i] = total

print(f"Prosecna konzerviranost: {conservation.mean():.3f}, "
      f"pozicija sa >=80% konzerviranosti: {(conservation>=0.8).sum()}/{ref_len}")


def conservation_weighted_similarity(aid_a, aid_b):
    map_a = ref_pos_to_seq_pos[aid_a]
    map_b = ref_pos_to_seq_pos[aid_b]
    common_ref_pos = set(map_a.keys()) & set(map_b.keys())
    if not common_ref_pos:
        return 0.0
    emb_a = residue_embeddings[aid_a]
    emb_b = residue_embeddings[aid_b]
    sims, weights = [], []
    for i in common_ref_pos:
        va = emb_a[map_a[i]]
        vb = emb_b[map_b[i]]
        sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        sims.append(sim)
        weights.append(conservation[i])
    sims = np.array(sims)
    weights = np.array(weights)
    if weights.sum() == 0:
        return float(sims.mean())
    return float(np.average(sims, weights=weights))


print("Racunam conservation-weighted i plain-cosine matrice (29x29 nsLTP pool)...")
n = len(nsltp_ids)
cons_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        s = conservation_weighted_similarity(nsltp_ids[i], nsltp_ids[j])
        cons_matrix[i, j] = cons_matrix[j, i] = s

pooled_matrix = np.array([pooled_embeddings[aid] for aid in nsltp_ids])
plain_cosine_matrix = cosine_similarity(pooled_matrix)

id_to_pos = {aid: i for i, aid in enumerate(nsltp_ids)}

nsltp_set = set(nsltp_ids)
gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    id1, id2 = name_to_id.get(n1), name_to_id.get(n2)
    if id1 in nsltp_set and id2 in nsltp_set and id1 != id2:
        gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})
print(f"nsLTP-nsLTP gold parovi (oba u pool-u): {len(gold_pairs)}")


def rank_of(matrix, qpos, tpos):
    scores = matrix[qpos].copy()
    scores[qpos] = -np.inf
    order = np.argsort(scores)[::-1]
    return int(np.where(order == tpos)[0][0]) + 1


records = []
for p in gold_pairs:
    for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qpos, tpos = id_to_pos[qid], id_to_pos[tid]
        records.append({
            "pair_id": p["pair_id"],
            "plain_rank": rank_of(plain_cosine_matrix, qpos, tpos),
            "cons_rank": rank_of(cons_matrix, qpos, tpos),
        })

df = pd.DataFrame(records)
plain_mrr = (1.0 / df["plain_rank"]).mean()
cons_mrr = (1.0 / df["cons_rank"]).mean()

rng = np.random.default_rng(SEED)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    c = pd.Series(sampled).value_counts()
    r = df.merge(c.rename("w"), left_on="pair_id", right_index=True)
    w = r["w"].to_numpy()
    d = np.average(1.0 / r["cons_rank"], weights=w) - np.average(1.0 / r["plain_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
sig = (ci_lo > 0) or (ci_hi < 0)

summary_lines = ["=" * 70, "Conservation-weighted signal, nsLTP-only pool (center-star pseudo-MSA)",
                  "=" * 70, "",
                  f"nsLTP pool: {n} proteina, referenca: {REFERENCE_NAME}",
                  f"Prosecna konzerviranost: {conservation.mean():.3f}",
                  f"nsLTP-nsLTP upiti: {len(df)}", "",
                  f"Plain cosine MRR (29-pool):              {plain_mrr:.4f}",
                  f"Conservation-weighted MRR (29-pool):     {cons_mrr:.4f}",
                  f"Delta: {cons_mrr - plain_mrr:+.4f}",
                  f"Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {'ZNACAJNO' if sig else 'nije znacajno'}"]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
