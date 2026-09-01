"""
Zamenjuje sequence_identity_pct u cross_reactive_1548.csv sa UNIFORMNO
izracunatom vrednoscu (BLAST global % identity preko cele sekvence, isti
izvor kao ml/pipeline za sve modele -- output/blast_identity_matrix_1443.pkl),
umesto rucno prepisanih literaturnih vrednosti (koje su bile prazne kod
1158/1548 parova, ili opsezi poput "50-60" umesto jednog broja).

Mentorkin zahtev: "% identičnosti u sekvenci izračunaj sama, kako bi imala
uniformnost među parovima u bazi."

NE BRISE staru vrednost -- prebacuje je u novu kolonu
sequence_identity_pct_literature (audit trail), sequence_identity_pct
postaje uniformno izracunata BLAST vrednost.

Izlaz: prepisuje output/cross_reactive_1548.csv.
"""

import pickle
from pathlib import Path

import pandas as pd

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

with open(BLAST_MATRIX, "rb") as f:
    blast = pickle.load(f)
blast_ids = blast["ids"]
identity_matrix = blast["identity_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}

allergens = pd.read_csv(CLEAN_ALLERGENS)
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id

df = pd.read_csv(GOLD)
df["sequence_identity_pct_literature"] = df["sequence_identity_pct"]

computed = []
missing = 0
for _, row in df.iterrows():
    id1 = name_to_id.get(str(row["allergen_id_1"]).strip())
    id2 = name_to_id.get(str(row["allergen_id_2"]).strip())
    if id1 in blast_id_to_index and id2 in blast_id_to_index:
        i, j = blast_id_to_index[id1], blast_id_to_index[id2]
        computed.append(round(float(identity_matrix[i, j]), 1))
    else:
        computed.append(None)
        missing += 1

df["sequence_identity_pct"] = computed
df.to_csv(GOLD, index=False)

print(f"Uniformno izracunato za {len(df) - missing}/{len(df)} parova.")
print(f"Nedostaje BLAST pokrivenost za: {missing} parova.")
print(f"\nStatistika nove kolone:")
print(df["sequence_identity_pct"].describe())
