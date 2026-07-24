"""
Create a clean WHO/IUIS allergen dataset for downstream ML.

This script:
- cleans text fields
- cleans protein sequences
- removes invalid amino acids
- removes very short sequences
- removes duplicate protein sequences
- exports a standardized CSV for embedding generation
"""

import pandas as pd
import re

INPUT = "/home/lana/ALERGRAF/data/jointable.csv"
OUTPUT = "clean_allergens.csv"

# 20 standard amino acids
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

MIN_SEQUENCE_LENGTH = 30


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\xa0", " ").strip()


def clean_sequence(sequence):
    if pd.isna(sequence):
        return ""

    # keep only letters
    sequence = re.sub(r"[^A-Za-z]", "", str(sequence)).upper()

    # keep only standard amino acids
    sequence = "".join(
        aa for aa in sequence
        if aa in VALID_AA
    )

    return sequence


def make_id(row):
    allergen = clean_text(row["AllergenID"])
    iso = clean_text(row["IsoAllergenID"])

    if iso:
        return f"WHO_{allergen}_ISO_{iso}"

    return f"WHO_{allergen}"


print("Loading WHO/IUIS dataset...")

df = pd.read_csv(INPUT, low_memory=False)

# clean column names
df.columns = [clean_text(c) for c in df.columns]

rows = []

removed_short = 0

for _, row in df.iterrows():

    name = clean_text(row["IsoName"])
    if not name:
        name = clean_text(row["Name"])

    sequence = clean_sequence(row["Sequence"])

    if sequence and len(sequence) < MIN_SEQUENCE_LENGTH:
        removed_short += 1
        continue

    reference = "; ".join(
        filter(
            None,
            [
                clean_text(row["AllergenicityRef"]),
                clean_text(row["SequenceRef"])
            ]
        )
    )

    rows.append({
        "allergen_id": make_id(row),
        "official_name": name,
        "source_food": clean_text(row["Common"]),
        "organism": clean_text(row["Species"]),
        "protein_family": "",
        "uniprot_id": clean_text(row["AccUniProt"]),
        "fasta_sequence": sequence,
        "sequence_length": len(sequence),
        "reference": reference
    })

out = pd.DataFrame(rows)

initial_count = len(out)

# remove duplicate rows
out = out.drop_duplicates()

# remove duplicate protein sequences
before_seq = len(out)
out = out.drop_duplicates(subset=["fasta_sequence"])
duplicate_sequences = before_seq - len(out)

# sort alphabetically
out = out.sort_values("official_name").reset_index(drop=True)

# save
out.to_csv(
    OUTPUT,
    index=False
)

print("\n========== DATASET SUMMARY ==========")
print(f"Final allergens:          {len(out)}")
print(f"Removed short proteins:   {removed_short}")
print(f"Duplicate sequences:      {duplicate_sequences}")
print(f"Missing UniProt IDs:      {(out['uniprot_id'] == '').sum()}")
print(f"Missing sequences:        {(out['fasta_sequence'] == '').sum()}")
print(f"Average sequence length:  {out['sequence_length'].mean():.1f} aa")
print(f"Longest sequence:         {out['sequence_length'].max()} aa")
print(f"Shortest sequence:        {out['sequence_length'].min()} aa")

print("\nSaved to:", OUTPUT)