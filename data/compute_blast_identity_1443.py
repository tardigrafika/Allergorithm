"""
Racuna punu simetricnu matricu BLAST-style % identity 
(lokalno poravnanje,BLOSUM62, Biopython PairwiseAligner
++++ blastp binarni alat nije dostupan pa ima python zamenu)  1534

Kesira se jednom (traje ~25 minuta)  RF skripta samo ucitava gotovu matricu.

Ulaz:
    output/clean_allergens.csv
    embeddings/embeddings.parquet 
Izlaz:
    output/blast_identity_matrix_1443.pkl
        dict: {"ids": [...], "identity_matrix": NxN float32, "score_matrix": NxN float32}
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
OUTPUT = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

print("Loading data...")
metadata = pd.read_parquet(METADATA)
all_ids = metadata["allergen_id"].tolist()

clean = pd.read_csv(CLEAN_ALLERGENS)
clean = clean[clean["fasta_sequence"].notna() & (clean["fasta_sequence"] != "")]
id_to_seq = dict(zip(clean["allergen_id"], clean["fasta_sequence"]))

sequences = [id_to_seq[allergen_id] for allergen_id in all_ids]
n = len(all_ids)
print(f"Proteins: {n}")

aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.mode = "local"
aligner.open_gap_score = -11
aligner.extend_gap_score = -1

identity_matrix = np.zeros((n, n), dtype=np.float32)
score_matrix = np.zeros((n, n), dtype=np.float32)

total_pairs = n * (n - 1) // 2
done = 0
start = time.time()

print(f"Computing {total_pairs} pairwise local alignments...")

for i in range(n):
    for j in range(i + 1, n):
        aln = aligner.align(sequences[i], sequences[j])[0]
        counts = aln.counts()
        aligned_length = counts.identities + counts.mismatches + counts.gaps
        identity_pct = (counts.identities / aligned_length * 100) if aligned_length > 0 else 0.0

        identity_matrix[i, j] = identity_matrix[j, i] = identity_pct
        score_matrix[i, j] = score_matrix[j, i] = aln.score

        done += 1

    if (i + 1) % 50 == 0 or i == n - 1:
        elapsed = time.time() - start
        rate = done / elapsed
        remaining = (total_pairs - done) / rate if rate > 0 else float("inf")
        print(f"  protein {i+1}/{n}, {done}/{total_pairs} pairs done "
              f"({rate:.0f} pairs/s, ~{remaining/60:.1f} min remaining)")

# diagonal = self, define as 100% identity for completeness (never used at retrieval time, self excluded)
np.fill_diagonal(identity_matrix, 100.0)

print(f"\nTotal time: {(time.time()-start)/60:.1f} minutes")

with open(OUTPUT, "wb") as f:
    pickle.dump({"ids": all_ids, "identity_matrix": identity_matrix, "score_matrix": score_matrix}, f)

print(f"Saved: {OUTPUT}")
print(f"Identity matrix stats: mean={identity_matrix[identity_matrix<100].mean():.2f}, "
      f"max(non-self)={np.max(identity_matrix - np.eye(n)*100):.2f}")
