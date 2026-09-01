import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

/home/lana/ALERGRAF/CLAUDE.md
# -----------------------------
# Paths
# -----------------------------

EMBEDDINGS = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
)

EMBEDDING_META = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.parquet"
)

GOLD = Path(
    "/home/lana/ALERGRAF/output/cross_reactive_combined.csv"
)

OUTPUT = Path(
    "/home/lana/ALERGRAF/output/cosine_baseline_results.csv"
)


# -----------------------------
# Load embeddings
# -----------------------------

print("Loading embeddings...")

with open(EMBEDDINGS, "rb") as f:
    embeddings = pickle.load(f)

print("Embeddings:", len(embeddings))


meta = pd.read_parquet(
    EMBEDDING_META
)


print("Metadata:")
print(meta.head())


# -----------------------------
# Create name lookup
# -----------------------------

name_to_id = {}

for _, row in meta.iterrows():

    name = str(row["official_name"]).strip()

    name_to_id[name] = row["allergen_id"]


print(
    "Names available:",
    len(name_to_id)
)


# -----------------------------
# Load gold
# -----------------------------

gold = pd.read_csv(GOLD)

print(
    "Gold pairs:",
    len(gold)
)


# -----------------------------
# Matching
# -----------------------------

results = []

missing = []


for _, row in gold.iterrows():

    a_name = row["allergen_id_1"]
    b_name = row["allergen_id_2"]


    if (
        a_name not in name_to_id
        or b_name not in name_to_id
    ):

        missing.append(
            (
                a_name,
                b_name
            )
        )

        continue


    a_id = name_to_id[a_name]
    b_id = name_to_id[b_name]


    score = cosine_similarity(
        [embeddings[a_id]],
        [embeddings[b_id]]
    )[0][0]


    results.append({

        "pair_id": row["pair_id"],

        "allergen_1": a_name,
        "allergen_2": b_name,

        "cosine_similarity": score,

        "family_1": row["family_1"],
        "family_2": row["family_2"],

        "evidence_level":
            row["evidence_level"]

    })


# -----------------------------
# Save
# -----------------------------

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT,
    index=False
)


print("\n==============================")
print("RESULT")
print("==============================")

print(
    "Valid pairs:",
    len(df)
)

print(
    "Missing:",
    len(missing)
)


if len(df) > 0:

    print("\nCosine statistics:")
    print(
        df["cosine_similarity"].describe()
    )


print("\nSaved:")
print(OUTPUT)