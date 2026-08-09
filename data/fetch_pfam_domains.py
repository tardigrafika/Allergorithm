"""
Preuzima Pfam domensku anotaciju po proteinu preko InterPro REST API-ja
(nezavisan izvor od naseg cross-reactivity grafa - HMM sekvencijalno
skeniranje protiv Pfam baze, NE cirkularno kao stari same_family feature).

Ulaz:
    output/clean_allergens.csv (allergen_id, uniprot_id)

Izlaz:
    output/pfam_domains_1548.csv  (allergen_id, uniprot_id, pfam_accessions
                                    [';'-separated, moze biti vise domena], status)
"""

import time
from pathlib import Path

import pandas as pd
import requests

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
OUTPUT = Path("/home/lana/ALERGRAF/output/pfam_domains_1548.csv")

df = pd.read_csv(CLEAN_ALLERGENS)
df = df[df["uniprot_id"].notna()].copy()
df["uniprot_clean"] = (
    df["uniprot_id"].astype(str).str.split(",").str[0].str.split().str[0].str.split("-").str[0]
)
print(f"Proteins with a UniProt ID: {len(df)}")

session = requests.Session()
results = []

for i, row in enumerate(df.itertuples(index=False), 1):
    allergen_id = row.allergen_id
    uniprot = row.uniprot_clean
    try:
        resp = session.get(
            f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{uniprot}/",
            timeout=15,
        )
        if resp.status_code != 200:
            results.append({"allergen_id": allergen_id, "uniprot_id": uniprot,
                             "pfam_accessions": "", "status": f"api_{resp.status_code}"})
        else:
            data = resp.json()
            accessions = [r["metadata"]["accession"] for r in data.get("results", [])]
            results.append({"allergen_id": allergen_id, "uniprot_id": uniprot,
                             "pfam_accessions": ";".join(accessions),
                             "status": "ok" if accessions else "no_pfam_hit"})
    except Exception as e:
        results.append({"allergen_id": allergen_id, "uniprot_id": uniprot,
                         "pfam_accessions": "", "status": f"error_{type(e).__name__}"})

    if i % 50 == 0 or i == len(df):
        n_ok = sum(1 for r in results if r["status"] == "ok")
        print(f"  {i}/{len(df)} processed, {n_ok} with a Pfam hit so far", flush=True)

status_df = pd.DataFrame(results)
status_df.to_csv(OUTPUT, index=False)
print("\nDone.")
print(status_df["status"].value_counts())
print(f"Saved: {OUTPUT}")
