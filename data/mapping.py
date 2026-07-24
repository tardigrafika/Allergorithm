"""
Map gold standard cross-reactive pairs to ESM embeddings
and calculate cosine similarity baseline.

Input:
    embeddings/embeddings.pkl
    output/gold_standard_cross_reactivity.csv

Output:
    output/cosine_baseline_results.csv
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Paths
# --------------------------------------------------

EMBEDDINGS = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
)

GOLD = Path(
    "/home/lana/ALERGRAF/data/create_gold_standard.py"
)

OUTPUT = Path(
    "/home/lana/ALERGRAF/output/cosine_baseline_results.csv"
)


# --------------------------------------------------
# Load
# --------------------------------------------------

print("Loading embeddings...")

with open(EMBEDDINGS, "rb") as f:
    embeddings = pickle.load(f)


print(f"Embeddings loaded: {len(embeddings)}")


gold = pd.read_csv(GOLD)

print(f"Gold pairs: {len(gold)}")


# --------------------------------------------------
# Create lookup table
# --------------------------------------------------

def normalize_name(name):
    """
    Convert WHO/IUIS family name to embedding names.
    """

    name = str(name).strip()

    return name


def find_embedding(name):
    """
    Find matching WHO/IUIS allergen ID.
    """

    name = normalize_name(name)

    # exact match
    if name in embeddings:
        return name


    # remove isoform numbers
    matches = []

    for key in embeddings.keys():

        base = key.split(".")[0]

        if base == name:
            matches.append(key)


    if len(matches) > 0:
        return matches[0]


    return None



# --------------------------------------------------
# Map pairs
# --------------------------------------------------

results = []

missing = []


for _, row in gold.iterrows():

    a_original = row["allergen_1"]
    b_original = row["allergen_2"]


    a_id = find_embedding(a_original)
    b_id = find_embedding(b_original)


    if a_id is None or b_id is None:

        missing.append(
            (
                a_original,
                b_original
            )
        )

        continue


    similarity = cosine_similarity(
        [embeddings[a_id]],
        [embeddings[b_id]]
    )[0][0]


    results.append({

        "pair_id": row["pair_id"],

        "allergen_1_original": a_original,
        "allergen_2_original": b_original,

        "allergen_1_embedding": a_id,
        "allergen_2_embedding": b_id,

        "cosine_similarity": similarity,

        "family_1": row["family_1"],
        "family_2": row["family_2"],

        "evidence_level": row["evidence_level"]

    })



# --------------------------------------------------
# Save
# --------------------------------------------------

out = pd.DataFrame(results)

out.to_csv(
    OUTPUT,
    index=False
)


print("\n==============================")
print("RESULT")
print("==============================")

print(f"Mapped pairs : {len(out)}")
print(f"Missing pairs: {len(missing)}")


if missing:

    print("\nMissing:")
    for x in missing[:20]:
        print(x)


print("\nCosine statistics:")
print(out["cosine_similarity"].describe())


print("\nSaved:")
print(OUTPUT)