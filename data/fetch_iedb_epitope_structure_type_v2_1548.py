"""
POPRAVLJENA verzija fetch_iedb_epitope_structure_type_1548.py -- prethodna
verzija je zahtevala starting_position/ending_position da bi zapis usao u
brojanje, sto STRUKTURNO iskljucuje SVE "Discontinuous peptide" (konformacione)
zapise (oni u IEDB-u nemaju linearnu poziciju, potvrdjeno direktnim upitom:
curated_source_antigens.starting_position/ending_position = None za sve
discontinuous zapise). Ova verzija broji structure_type po accession match-u
BEZ zahteva za pozicijama -- pozicije se cuvaju kad postoje, ali ne filtriraju.

Izlaz:
    output/iedb_epitope_structure_types_v2_1548.csv
        allergen_id, uniprot_id, structure_type, starting_position, ending_position
"""

import time
from pathlib import Path

import pandas as pd
import requests

EXISTING = Path("/home/lana/ALERGRAF/output/iedb_epitopes_1548.csv")
OUTPUT = Path("/home/lana/ALERGRAF/output/iedb_epitope_structure_types_v2_1548.csv")

PAGE_SIZE = 200
REQUEST_DELAY = 0.2
REQUEST_TIMEOUT = 40
MAX_RETRIES = 2

df = pd.read_csv(EXISTING)
df = df[df["n_positive_records"] > 0].copy()
print(f"Proteini sa poznatim pozitivnim epitope zapisima: {len(df)}")

session = requests.Session()
records = []
already_done = set()
if OUTPUT.exists():
    prior = pd.read_csv(OUTPUT)
    records = prior.to_dict("records")
    already_done = set(prior["allergen_id"].unique())
    print(f"Resume: {len(already_done)} proteina vec ima sacuvane rezultate, preskacem ih.")
    df = df[~df["allergen_id"].isin(already_done)].copy()
    print(f"Preostalo za fetch: {len(df)} proteina")


def fetch_page(uniprot, offset):
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(
                "https://query-api.iedb.org/epitope_search",
                params={
                    "parent_source_antigen_iri": f"ilike.*{uniprot}*",
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "order": "structure_id",
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json(), None
            return None, f"HTTP {resp.status_code}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None, f"{type(e).__name__}: {e}"


start = time.time()
for i, row in enumerate(df.itertuples(index=False), 1):
    allergen_id, uniprot = row.allergen_id, row.uniprot_id
    offset = 0
    protein_had_any_record = False
    while True:
        page, err = fetch_page(uniprot, offset)
        time.sleep(REQUEST_DELAY)
        if err is not None:
            print(f"  WARNING: {allergen_id} ({uniprot}) offset {offset}: {err}")
            break
        if not page:
            break
        for record in page:
            quals = record.get("qualitative_measures") or []
            if not any(str(q).startswith("Positive") for q in quals):
                continue
            structure_type = record.get("structure_type")
            for antigen in record.get("curated_source_antigens") or []:
                if str(uniprot).upper() not in str(antigen.get("accession", "")).upper():
                    continue
                protein_had_any_record = True
                records.append({
                    "allergen_id": allergen_id, "uniprot_id": uniprot,
                    "structure_type": structure_type,
                    "starting_position": antigen.get("starting_position"),
                    "ending_position": antigen.get("ending_position"),
                })
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not protein_had_any_record:
        # eksplicitno oznaci "obradjeno, nula rezultata" da se resume ne vrti u krug na njemu
        records.append({"allergen_id": allergen_id, "uniprot_id": uniprot,
                         "structure_type": None, "starting_position": None, "ending_position": None})

    if i % 10 == 0 or i == len(df):
        elapsed = time.time() - start
        pd.DataFrame(records).to_csv(OUTPUT, index=False)
        print(f"  {i}/{len(df)} ({elapsed/60:.1f} min elapsed) [checkpoint saved]", flush=True)

out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT, index=False)
print(f"\nDone. {len(out_df)} zapisa ukupno.")
print(out_df["structure_type"].value_counts(dropna=False))
print(f"Saved: {OUTPUT}")
