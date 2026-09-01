"""
Konvertuje output/clean_allergens.csv u standardan FASTA fajl -- isti
protein dataset kao za ESM-2/ESM-1b embeddinge, sad u formatu koji
generate_alphafold_trunk_embeddings.py ocekuje.

ID u FASTA zaglavlju je allergen_id (isti identifikator koriscen svuda u
ml pipeline-u -- id_to_index, embeddings.pkl kljucevi, itd.) da se trunk
embeddinzi mogu direktno spojiti sa ostatkom pipeline-a bez re-mapiranja.

Pokretanje (lokalno, PRE transfera na klaster):
    python3 hpclab/csv_to_fasta.py
"""

from pathlib import Path

import pandas as pd

INPUT_CSV = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
OUTPUT_FASTA = Path("/home/lana/ALERGRAF/hpclab/proteins.fasta")

df = pd.read_csv(INPUT_CSV)
df = df[df["fasta_sequence"].notna()]
df = df[df["fasta_sequence"] != ""]
df = df.reset_index(drop=True)

with open(OUTPUT_FASTA, "w") as f:
    for row in df.itertuples(index=False):
        f.write(f">{row.allergen_id}\n")
        seq = row.fasta_sequence
        for i in range(0, len(seq), 60):  # standardna FASTA linija 60 karaktera
            f.write(seq[i:i + 60] + "\n")

print(f"Napisano {len(df)} sekvenci u {OUTPUT_FASTA}")
