import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

ISOFORM_IDENTITY_THRESHOLD = 80.0
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

allergens = pd.read_csv(CLEAN_ALLERGENS)
names = sorted(set(allergens["official_name"].astype(str)))
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id


def base_name(n):
    return n.rsplit(".", 1)[0] if "." in n else n


groups = defaultdict(list)
for n in names:
    groups[base_name(n)].append(n)
isoform_groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
print("Bet v 1 group:", isoform_groups.get("Bet v 1", [])[:5])

with open(BLAST_MATRIX, "rb") as f:
    blast = pickle.load(f)
blast_id_to_index = {aid: i for i, aid in enumerate(blast["ids"])}
identity_matrix = blast["identity_matrix"]
print("Bet v 1.0101 id:", name_to_id.get("Bet v 1.0101"))
print("in blast index?", name_to_id.get("Bet v 1.0101") in blast_id_to_index)

gold = pd.read_csv(GOLD)
print("gold columns:", gold.columns.tolist())
print("gold shape:", gold.shape)
strong = gold[gold["evidence_level"].isin(
    ["Confirmed", "Strong evidence", "Strong evidence (within-species paralogs)", "Strong evidence (congeneric species)"]
)]
print("strong shape:", strong.shape)
print(strong.head(2))
