"""
30-worst-error tabela (mentorkin predlog): za svaki od 30 najgore rangiranih
parova (cosine, ceo dataset), skupi sve postojece signale koje vec imamo
(BLAST identity, FoldseekTM, evidence_level/ccd_flag/who2001_pass/epitope_type
iz gold dataseta, IEDB structure_type ako postoji za bar jedan clan) --
trazi OBRAZAC, ne novi model.

SASA/povrsinska ekspozicija NIJE ukljucena -- nije sacuvana kao opsti
reusable lookup (racunata je ad-hoc za 29-protein nsLTP podskup u ranijem
eksperimentu), ne racuna se ponovo ovde radi vremena; ako obrazac iz ostalih
kolona bude ukazivao da SASA vredi proveriti, to je sledeci, odvojen korak.

Izlaz:
    output/worst30_error_table_1548.csv
    output/worst30_error_table_1548_summary.txt
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import retrieval_evaluate  # noqa: E402
from ml.pipeline.models.classifiers.cosine import CosineSimilarity  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
IEDB_STRUCTURE = Path("/home/lana/ALERGRAF/output/iedb_epitope_structure_types_v2_1548.csv")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/worst30_error_table_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/worst30_error_table_1548_summary.txt")

N_WORST = 30

print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
clf = CosineSimilarity()
clf.fit(dataset.gold_pairs, [], dataset.embedding_matrix, dataset.id_to_index)
retrieval_df = retrieval_evaluate(dataset.gold_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                    cosine_matrix=cosine_matrix)

# jedan red po PARU (ne po smeru) -- uzmi gori (visi) rang izmedju dva smera
retrieval_df["pair_key"] = retrieval_df.apply(
    lambda r: frozenset([r["query_allergen_id"], r["target_allergen_id"]]), axis=1)
worst_per_pair = retrieval_df.loc[retrieval_df.groupby("pair_key")["model_rank"].idxmax()]
worst30 = worst_per_pair.sort_values("model_rank", ascending=False).head(N_WORST)

with open(BLAST_MATRIX, "rb") as f:
    blast = pickle.load(f)
blast_id_to_index = {aid: i for i, aid in enumerate(blast["ids"])}
identity_matrix = blast["identity_matrix"]

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)


def blast_pct(a, b):
    if a in blast_id_to_index and b in blast_id_to_index:
        return round(float(identity_matrix[blast_id_to_index[a], blast_id_to_index[b]]), 1)
    return None


def foldseek_tm(a, b):
    return foldseek_lookup.get(frozenset((a, b)))


iedb = pd.read_csv(IEDB_STRUCTURE) if IEDB_STRUCTURE.exists() else pd.DataFrame()
iedb_types = {}
if len(iedb):
    for aid, group in iedb.groupby("allergen_id"):
        types = set(group["structure_type"].dropna())
        iedb_types[aid] = types

gold_df = pd.read_csv(GOLD)
gold_lookup = {}
for _, r in gold_df.iterrows():
    id1 = dataset.name_to_id.get(str(r["allergen_id_1"]).strip())
    id2 = dataset.name_to_id.get(str(r["allergen_id_2"]).strip())
    if id1 is None or id2 is None:
        continue
    key = frozenset([id1, id2])
    gold_lookup[key] = r

id_to_name = {v: k for k, v in dataset.name_to_id.items()}

rows = []
for _, r in worst30.iterrows():
    a, b = r["query_allergen_id"], r["target_allergen_id"]
    key = frozenset([a, b])
    gold_row = gold_lookup.get(key)
    epitope_info = []
    for x in (a, b):
        t = iedb_types.get(x)
        if t:
            epitope_info.append(f"{x}: {'/'.join(sorted(t))}")
    rows.append({
        "pair_id": gold_row["pair_id"] if gold_row is not None else None,
        "allergen_1": id_to_name.get(a, a), "allergen_2": id_to_name.get(b, b), "family": r["query_family"],
        "rank": r["model_rank"], "n_candidates": len(dataset.all_ids),
        "percentile": round(r["model_rank"] / len(dataset.all_ids) * 100, 1),
        "blast_identity_pct": blast_pct(a, b),
        "foldseek_tm": round(foldseek_tm(a, b), 3) if foldseek_tm(a, b) is not None else None,
        "evidence_level": gold_row["evidence_level"] if gold_row is not None else None,
        "ccd_flag": gold_row.get("ccd_flag") if gold_row is not None else None,
        "who2001_pass": gold_row.get("who2001_pass") if gold_row is not None else None,
        "iedb_epitope_info": "; ".join(epitope_info) if epitope_info else "no IEDB data",
    })

result_df = pd.DataFrame(rows)
result_df.to_csv(CSV_OUTPUT, index=False)
print(result_df.to_string(index=False))
print(f"\nSaved: {CSV_OUTPUT}")

# obrazac po familiji
summary_lines = ["=" * 80, "30-worst-error obrazac", "=" * 80, "",
                  "Raspodela po familiji (30 najgorih parova):",
                  result_df["family"].value_counts().to_string(), "",
                  f"Prosecan BLAST identity (30 najgorih): {result_df['blast_identity_pct'].mean():.1f}%",
                  f"Prosecan Foldseek TM-score (30 najgorih, gde postoji): {result_df['foldseek_tm'].mean():.3f}",
                  f"Broj sa IEDB epitope podatkom (bar 1 clan): {(result_df['iedb_epitope_info']!='no IEDB data').sum()}/30",
                  f"Broj sa 'Discontinuous' u IEDB podatku: {result_df['iedb_epitope_info'].str.contains('Discontinuous', na=False).sum()}/30",
                  ]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"Saved: {SUMMARY_OUTPUT}")
