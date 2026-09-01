"""
Sistematska provera da li su neke sekvence u clean_allergens.csv fragmenti
(NON_TER flag u UniProt-u, ili nasa zabelezena duzina ne odgovara pravoj
UniProt duzini) -- pokrenuto posle otkrica da je Gly m 1 (Q9S8F3) potvrdjen
UniProt fragment (35/42 od pravih ~63-70 aa) koji je bio glavni uzrocnik
najgorih gresaka u worst-30 analizi.

UniProt REST bulk search (OR-spojeni accession upiti, batch od 50) --
mnogo brze od pojedinacnih .txt upita za 1250 razlicitih UniProt ID-jeva.

Izlaz:
    output/fragment_sequence_check_1548.csv
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
OUTPUT = Path("/home/lana/ALERGRAF/output/fragment_sequence_check_1548.csv")

BATCH_SIZE = 50
REQUEST_DELAY = 0.3

# standardni UniProt accession format (6 ili 10 karaktera, alfanumericki)
ACCESSION_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9](?:[A-Z][A-Z0-9]{2}[0-9])?|^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}")


def extract_accession(raw):
    """Neki zapisi imaju dodatni tekst ('X (27-367)', 'X, Y') koji kvari upit --
    izvuci samo PRVI validan accession token."""
    raw = str(raw).strip()
    token = raw.split(",")[0].split(" ")[0].strip()
    return token if ACCESSION_RE.match(token) else None


allergens = pd.read_csv(CLEAN_ALLERGENS)
allergens["uniprot_id_clean"] = allergens["uniprot_id"].dropna().apply(extract_accession)
n_dirty = allergens["uniprot_id"].notna().sum() - allergens["uniprot_id_clean"].notna().sum()
print(f"Neuspesno ociscenih (nestandardan format): {n_dirty}")
uniprot_ids = sorted(allergens["uniprot_id_clean"].dropna().unique())
print(f"Proveravam {len(uniprot_ids)} razlicitih (ociscenih) UniProt ID-jeva...")

session = requests.Session()
records = []

for i in range(0, len(uniprot_ids), BATCH_SIZE):
    batch = uniprot_ids[i:i + BATCH_SIZE]
    query = " OR ".join(f"accession:{uid}" for uid in batch)
    try:
        resp = session.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={"query": query, "fields": "accession,length,ft_non_ter", "format": "tsv", "size": 500},
            timeout=40,
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        for line in lines[1:]:
            parts = line.split("\t")
            acc = parts[0]
            length = int(parts[1]) if len(parts) > 1 and parts[1] else None
            non_ter = parts[2] if len(parts) > 2 else ""
            records.append({"uniprot_id": acc, "uniprot_length": length, "non_terminal_flag": non_ter})
    except Exception as e:
        print(f"  WARNING batch {i}: {e}")
    time.sleep(REQUEST_DELAY)
    if (i // BATCH_SIZE) % 5 == 0:
        print(f"  {i + len(batch)}/{len(uniprot_ids)}", flush=True)

uniprot_df = pd.DataFrame(records)
print(f"\nDobijeno {len(uniprot_df)} UniProt zapisa.")

our_lengths = allergens[["official_name", "allergen_id", "uniprot_id", "uniprot_id_clean", "sequence_length"]].copy()
merged = our_lengths.merge(uniprot_df, left_on="uniprot_id_clean", right_on="uniprot_id", how="left", suffixes=("", "_matched"))
merged["is_fragment"] = merged["non_terminal_flag"].astype(str).str.contains("NON_TER", na=False)
merged["length_mismatch"] = (merged["sequence_length"] != merged["uniprot_length"]) & merged["uniprot_length"].notna()

merged.to_csv(OUTPUT, index=False)

print(f"\nProteina oznacenih kao Fragment (NON_TER) u UniProt-u: {merged['is_fragment'].sum()}")
print(f"Proteina gde nasa duzina ne odgovara UniProt duzini: {merged['length_mismatch'].sum()}")
print()
flagged = merged[merged["is_fragment"] | merged["length_mismatch"]].sort_values("sequence_length")
print(flagged[["official_name", "sequence_length", "uniprot_length", "is_fragment"]].head(40).to_string(index=False))
print(f"\nSaved: {OUTPUT}")
