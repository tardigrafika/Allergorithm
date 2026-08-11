"""
Preuzima STVARNE (eksperimentalno mapirane) B-cell epitope pozicije iz IEDB
(Immune Epitope Database) za sve nase proteine sa UniProt ID-jem.

Koristi query-api.iedb.org/epitope_search, filtrirano po
parent_source_antigen_iri (sadrzi UniProt accession). Zadrzava samo
"Positive*" qualitative_measures zapise (iskljucuje "Negative" - testirano,
bez reaktivnosti - to bi bio pogresan signal da ukljucimo kao "epitope").

Za svaki protein sa bar jednim pozitivnim epitope zapisom, cuva UNIJU svih
(starting_position, ending_position) opsega kao "epitope masku" - bilo koja
rezidua pokrivena BAR JEDNIM pozitivnim epitope fragmentom se racuna kao
epitope pozicija.

Ulaz:
    output/clean_allergens.csv (allergen_id, uniprot_id, fasta_sequence)

Izlaz:
    output/iedb_epitopes_1548.csv
        allergen_id, uniprot_id, n_positive_records, epitope_ranges
        (';'-separated "start-end" parovi, 1-indeksirano kao u IEDB)
"""

import time
from pathlib import Path

import pandas as pd
import requests

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
OUTPUT = Path("/home/lana/ALERGRAF/output/iedb_epitopes_1548.csv")

PAGE_SIZE = 200
REQUEST_DELAY = 0.3  # not actually a rate-limit issue (see below) -- just polite spacing
# NAPOMENA: ono sto je izgledalo kao throttling bila je zapravo SPORA upita za
# proteine BEZ IEDB podudaranja (ilike '%X%' bez match-a izgleda radi punu
# pretragu tabele na njihovom serveru -- potvrdjeno: 22s da vrati 200 sa
# praznim rezultatom). 15s timeout je bio prekratak i lazno je izgledao kao
# blokada. Vecina od ~331 proteina verovatno NEMA IEDB podatke (~26% ukupna
# pokrivenost iz ranijeg uzorka), pa ce vecina upita ici ovim sporim putem -
# otud REQUEST_TIMEOUT=40 i realno ocekivano trajanje ~1-1.5h za ceo run.
REQUEST_TIMEOUT = 40
MAX_RETRIES = 1
CONNECTED_UNIVERSE_IDS_FILE = Path("/home/lana/ALERGRAF/output/connected_universe_ids_1548.txt")

df = pd.read_csv(CLEAN_ALLERGENS)
df = df[df["uniprot_id"].notna()].copy()
df["uniprot_clean"] = (
    df["uniprot_id"].astype(str).str.split(",").str[0].str.split().str[0].str.split("-").str[0]
)

# scope down to the connected universe only -- Experiment A only needs epitope
# data for proteins that actually appear in a gold cross-reactivity pair, not
# the full 1302-protein UniProt-mapped universe (cuts request volume ~4x,
# both faster and gentler on the API after the earlier throttling)
if CONNECTED_UNIVERSE_IDS_FILE.exists():
    connected_ids = set(CONNECTED_UNIVERSE_IDS_FILE.read_text().split())
    df = df[df["allergen_id"].isin(connected_ids)].copy()
    print(f"Scoped to connected universe: {len(df)} proteins")
else:
    print("WARNING: connected universe id file not found -- fetching for ALL UniProt-mapped proteins")

print(f"Proteins to fetch: {len(df)}")

session = requests.Session()
results = []

def fetch_page(uniprot, offset):
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(
                "https://query-api.iedb.org/epitope_search",
                params={
                    "parent_source_antigen_iri": f"ilike.*{uniprot}*",
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "order": "structure_id",  # required by the API whenever offset is used
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json(), None
            return None, f"HTTP {resp.status_code}: {resp.text[:150]}"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None, f"{type(e).__name__}: {e}"


start = time.time()
for i, row in enumerate(df.itertuples(index=False), 1):
    allergen_id = row.allergen_id
    uniprot = row.uniprot_clean

    ranges = []
    offset = 0
    while True:
        page, err = fetch_page(uniprot, offset)
        time.sleep(REQUEST_DELAY)
        if err is not None:
            print(f"  WARNING: {allergen_id} ({uniprot}) at offset {offset}: {err}")
            break
        if not page:
            break

        for record in page:
            quals = record.get("qualitative_measures") or []
            if not any(str(q).startswith("Positive") for q in quals):
                continue
            for antigen in record.get("curated_source_antigens") or []:
                if uniprot.upper() not in str(antigen.get("accession", "")).upper():
                    continue
                sp, ep = antigen.get("starting_position"), antigen.get("ending_position")
                if sp is not None and ep is not None:
                    ranges.append((int(sp), int(ep)))

        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    results.append({
        "allergen_id": allergen_id,
        "uniprot_id": uniprot,
        "n_positive_records": len(ranges),
        "epitope_ranges": ";".join(f"{a}-{b}" for a, b in ranges),
    })

    if i % 10 == 0 or i == len(df):
        n_with_epitopes = sum(1 for r in results if r["n_positive_records"] > 0)
        elapsed = time.time() - start
        rate = i / elapsed
        remaining = (len(df) - i) / rate if rate > 0 else float("inf")
        checkpoint = i % 50 == 0 or i == len(df)
        if checkpoint:
            pd.DataFrame(results).to_csv(OUTPUT, index=False)
        print(f"  {i}/{len(df)} processed, {n_with_epitopes} with >=1 positive epitope record "
              f"({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min remaining)"
              + (" [checkpoint saved]" if checkpoint else ""), flush=True)

status_df = pd.DataFrame(results)
status_df.to_csv(OUTPUT, index=False)

print("\nDone.")
print(f"Proteins with >=1 positive epitope: {(status_df['n_positive_records'] > 0).sum()}/{len(status_df)}")
print(f"Saved: {OUTPUT}")
