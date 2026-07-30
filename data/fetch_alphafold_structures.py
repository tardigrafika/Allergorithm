"""
Preuzima AlphaFold DB strukture (PDB fajlove) za sve proteine koji imaju
UniProt ID, kesira lokalno. Korak 1 za TM-score feature.

Ulaz:
    output/clean_allergens.csv (allergen_id, uniprot_id)

Izlaz:
    data/alphafold_structures/{allergen_id}.pdb  (jedan fajl po proteinu)
    output/alphafold_fetch_status_1443.csv  (allergen_id, uniprot_id, status)
"""

import time
from pathlib import Path

import pandas as pd
import requests

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
STRUCTURE_DIR = Path("/home/lana/ALERGRAF/data/alphafold_structures")
STATUS_OUTPUT = Path("/home/lana/ALERGRAF/output/alphafold_fetch_status_1443.csv")

STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_ALLERGENS)
df = df[df["uniprot_id"].notna()].copy()
# clean ids with trailing annotations like "(variant V123F)"
df["uniprot_clean"] = df["uniprot_id"].str.split().str[0]

print(f"Proteins with a UniProt ID: {len(df)}")

session = requests.Session()
results = []

for i, row in enumerate(df.itertuples(index=False), 1):
    allergen_id = row.allergen_id
    uniprot = row.uniprot_clean
    out_path = STRUCTURE_DIR / f"{allergen_id}.pdb"

    if out_path.exists():
        results.append({"allergen_id": allergen_id, "uniprot_id": uniprot, "status": "cached"})
    else:
        try:
            api_resp = session.get(
                f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}", timeout=15
            )
            if api_resp.status_code != 200 or not api_resp.json():
                results.append({"allergen_id": allergen_id, "uniprot_id": uniprot, "status": f"api_{api_resp.status_code}"})
            else:
                pdb_url = api_resp.json()[0]["pdbUrl"]
                pdb_resp = session.get(pdb_url, timeout=30)
                if pdb_resp.status_code == 200:
                    out_path.write_text(pdb_resp.text)
                    results.append({"allergen_id": allergen_id, "uniprot_id": uniprot, "status": "downloaded"})
                else:
                    results.append({"allergen_id": allergen_id, "uniprot_id": uniprot, "status": f"pdb_{pdb_resp.status_code}"})
        except Exception as e:
            results.append({"allergen_id": allergen_id, "uniprot_id": uniprot, "status": f"error_{type(e).__name__}"})

    if i % 50 == 0 or i == len(df):
        n_ok = sum(1 for r in results if r["status"] in ("downloaded", "cached"))
        print(f"  {i}/{len(df)} processed, {n_ok} structures available so far")

status_df = pd.DataFrame(results)
status_df.to_csv(STATUS_OUTPUT, index=False)

print("\nDone.")
print(status_df["status"].value_counts())
print(f"Saved status to: {STATUS_OUTPUT}")
print(f"Structures in: {STRUCTURE_DIR}")
