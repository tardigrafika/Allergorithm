"""
Dodaje epitope_type kolonu u cross_reactive_1548.csv na osnovu IEDB
structure_type podataka (output/iedb_epitope_structure_types_v2_1548.csv,
40/87 proteina sa poznatim epitope zapisima -- delimicno, fetch prekinut
zbog sporog API-ja i niskog dodatnog prinosa, videti napomenu u toj skripti).

VAZNO OGRANICENJE (transparentno, ne sakriveno): IEDB podatak kaze SAMO da
protein X ima BAR JEDAN dokumentovan epitop tog tipa (linearni ili
konformacioni) -- NE govori da li je BAS TAJ epitop odgovoran za
cross-reaktivnost sa drugim proteinom u paru. Ovo je najbolja dostupna
proxy informacija, ne direktan dokaz po paru.

Logika po paru (oba proteina moraju imati IEDB podatak, inace prazno/NaN):
  "linear_only"        -- oba proteina imaju SAMO linearne dokumentovane epitope
  "conformational_present" -- bar jedan protein ima BAR JEDAN dokumentovan
                               konformacioni (Discontinuous) epitop
  (prazno/NaN)          -- bar jedan protein NEMA IEDB podatak (91%+ dataseta)

Izlaz: dodaje epitope_type kolonu u output/cross_reactive_1548.csv.
"""

from pathlib import Path

import pandas as pd

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
IEDB_STRUCTURE = Path("/home/lana/ALERGRAF/output/iedb_epitope_structure_types_v2_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

allergens = pd.read_csv(CLEAN_ALLERGENS)
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id

iedb = pd.read_csv(IEDB_STRUCTURE)
iedb = iedb[iedb["structure_type"].notna()]

has_conformational = set(iedb.loc[iedb["structure_type"].str.contains("Discontinuous", na=False), "allergen_id"])
has_any_iedb = set(iedb["allergen_id"].unique())

df = pd.read_csv(GOLD)
epitope_types = []
for _, row in df.iterrows():
    id1 = name_to_id.get(str(row["allergen_id_1"]).strip())
    id2 = name_to_id.get(str(row["allergen_id_2"]).strip())
    if id1 not in has_any_iedb or id2 not in has_any_iedb:
        epitope_types.append(None)
    elif id1 in has_conformational or id2 in has_conformational:
        epitope_types.append("conformational_present")
    else:
        epitope_types.append("linear_only")

df["epitope_type"] = epitope_types
df.to_csv(GOLD, index=False)

print(f"Pokriveno (oba proteina imaju IEDB podatak): {df['epitope_type'].notna().sum()}/{len(df)} parova")
print(df["epitope_type"].value_counts(dropna=False))
