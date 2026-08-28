"""
Pilot: BepiPred-3.0 predikovani B-cell epitopi kao tezine za pooling
rezidue-embeddinga, na istih 50 najgorih RRF upita (nsLTP/Profilin) gde su
i cosine, BLAST i FoldseekTM vec pali. Isti princip kao TM-align pilot --
brz, jeftin test PRE ulaganja u pun run preko celog dataseta, sa OBAVEZNIM
null-baseline poredjenjem (slucajni parovi istih proteina) da se izbegne
isti artefakt kao kod TM-align-a (kratki proteini daju lazno "umerene"
skorove bez obzira na pravu srodnost).

Metod: epitope-weighted pooled embedding = weighted average rezidue-
embeddinga, tezina = BepiPred epitope-verovatnoca po rezidui (umesto
uniformnog mean-pool-a ili surface-only maske koju smo vec probale).

Ulaz (nakon sto se BepiPred pokrene na VM i donese nazad):
    output/bepipred_pilot_1548_raw_output.csv  (BepiPred-ov raw_output.csv,
        preimenovan; kolone: ID, Residue, BepiPred-3.0 score, BepiPred-3.0
        score (rolling average) -- tacne nazive kolona proveri pri uvozu)
    embeddings/residue_embeddings.pkl  (vec postoji lokalno)

Izlaz:
    output/bepipred_pilot_worst50_1548_summary.txt
"""

import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BEPIPRED_RAW = Path("/home/lana/ALERGRAF/output/bepipred_pilot_1548_raw_output.csv")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
RANK_FUSION = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/bepipred_pilot_worst50_1548_summary.txt")

SEED = 42

if not BEPIPRED_RAW.exists():
    raise SystemExit(
        f"Nedostaje {BEPIPRED_RAW} -- pokreni BepiPred-3.0 na VM nad "
        f"output/bepipred_pilot_1548.fasta, donesi raw_output.csv nazad i "
        f"preimenuj/kopiraj na ovu putanju pre pokretanja skripte."
    )

print("Loading BepiPred output...")
bp = pd.read_csv(BEPIPRED_RAW)
print(f"  columns: {bp.columns.tolist()}")
# BepiPred raw_output.csv obicno ima kolone slicne: "Accession","Residue","BepiPred-3.0 score"
id_col = [c for c in bp.columns if "acc" in c.lower() or c.lower() == "id"][0]
score_col = [c for c in bp.columns if "score" in c.lower() and "rolling" not in c.lower()][0]
print(f"  koristim id_col={id_col!r}, score_col={score_col!r}")

epitope_scores = {}
for aid, group in bp.groupby(id_col):
    epitope_scores[aid] = group[score_col].to_numpy(dtype=np.float64)

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

pool = sorted(set(epitope_scores.keys()) & set(residue_embeddings.keys()))
print(f"Proteini sa i BepiPred skorom i rezidue-embeddingom: {len(pool)}")

mismatched = []
epitope_weighted_pooled = {}
mean_pooled = {}
for aid in pool:
    emb = residue_embeddings[aid]
    scores = epitope_scores[aid]
    if emb.shape[0] != len(scores):
        mismatched.append((aid, emb.shape[0], len(scores)))
        continue
    w = scores / (scores.sum() + 1e-12)
    epitope_weighted_pooled[aid] = (emb * w[:, None]).sum(axis=0)
    mean_pooled[aid] = emb.mean(axis=0)

if mismatched:
    print(f"  [upozorenje] {len(mismatched)} proteina sa razlicitom duzinom (residue_emb vs BepiPred), preskoceno")
usable = sorted(epitope_weighted_pooled.keys())
print(f"Usable za poredjenje: {len(usable)}")

# =====================================================
# WORST-50 PAROVI (isti skup kao TM-align pilot)
# =====================================================

df = pd.read_csv(RANK_FUSION)
gold_meta = pd.read_csv(GOLD)[["pair_id", "allergen_id_1", "allergen_id_2"]].drop_duplicates("pair_id")
merged = df.merge(gold_meta, on="pair_id", how="left")
worst = merged.sort_values("rrf_rank", ascending=False).head(50)

clean = pd.read_csv(CLEAN_ALLERGENS)
name_to_id = dict(zip(clean["official_name"], clean["allergen_id"]))

gold_raw = pd.read_csv(GOLD)
gold_pairs_set = set()
for _, r in gold_raw.iterrows():
    a, b = name_to_id.get(str(r["allergen_id_1"]).strip()), name_to_id.get(str(r["allergen_id_2"]).strip())
    if a and b:
        gold_pairs_set.add(frozenset({a, b}))


def cos(a, b):
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


records = []
for _, row in worst.iterrows():
    id1, id2 = name_to_id.get(row["allergen_id_1"]), name_to_id.get(row["allergen_id_2"])
    if id1 not in usable or id2 not in usable:
        continue
    ew_score = cos(epitope_weighted_pooled[id1], epitope_weighted_pooled[id2])
    mp_score = cos(mean_pooled[id1], mean_pooled[id2])
    records.append({"pair_id": row["pair_id"], "name_1": row["allergen_id_1"], "name_2": row["allergen_id_2"],
                     "epitope_weighted": ew_score, "mean_pooled": mp_score})

pair_df = pd.DataFrame(records)
print(f"\nWorst-50 parovi sa oba proteina usable: {len(pair_df)}")

# =====================================================
# NULL BASELINE (slucajni NE-gold parovi istih proteina) -- OBAVEZNO,
# isti razlog kao kod TM-align pilota
# =====================================================

random.seed(SEED)
null_records = []
attempts = 0
worst_ids = sorted(set(pd.concat([pair_df["name_1"], pair_df["name_2"]]).map(name_to_id)) & set(usable))
while len(null_records) < 30 and attempts < 500 and len(worst_ids) >= 2:
    attempts += 1
    a, b = random.sample(worst_ids, 2)
    if frozenset({a, b}) in gold_pairs_set:
        continue
    null_records.append({
        "epitope_weighted": cos(epitope_weighted_pooled[a], epitope_weighted_pooled[b]),
        "mean_pooled": cos(mean_pooled[a], mean_pooled[b]),
    })
null_df = pd.DataFrame(null_records)

summary_lines = ["=" * 70, "BepiPred-3.0 pilot: epitope-weighted pooling na 50 najgorih RRF upita",
                  "=" * 70, "",
                  f"Worst-50 parovi (oba proteina usable): {len(pair_df)}",
                  f"Null baseline (slucajni NE-gold parovi istih proteina): {len(null_df)}", ""]

for col, label in [("mean_pooled", "Mean-pooled cosine (baseline)"), ("epitope_weighted", "Epitope-weighted cosine")]:
    wm, ws = pair_df[col].mean(), pair_df[col].std()
    nm, ns = null_df[col].mean(), null_df[col].std()
    summary_lines.append(f"{label}:")
    summary_lines.append(f"  Worst-50 (poznato cross-reaktivni, ali tesko rangirani): mean={wm:.4f} std={ws:.4f}")
    summary_lines.append(f"  Null (slucajni, NE-gold):                              mean={nm:.4f} std={ns:.4f}")
    elevated = wm > nm
    summary_lines.append(f"  Worst-50 > null? {'DA -- mozda ima signala' if elevated else 'NE -- verovatno artefakt, isto kao TM-align nalaz'}")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
