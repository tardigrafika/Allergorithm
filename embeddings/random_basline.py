"""
Evaluate ESM embedding cosine similarity baseline.

Creates:
    Positive pairs = confirmed cross-reactive allergens
    Negative pairs = random non-cross-reactive pairs

Outputs:
    output/cosine_baseline_evaluation.csv
"""

import pickle
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score


# --------------------------------------------------
# Paths
# --------------------------------------------------

EMBEDDINGS = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
)

META = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.parquet"
)

GOLD = Path(
    "/home/lana/ALERGRAF/output/gold_standard_cross_reactivity.csv"
)

OUTPUT = Path(
    "/home/lana/ALERGRAF/output/cosine_baseline_evaluation.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading embeddings...")

with open(EMBEDDINGS, "rb") as f:
    embeddings = pickle.load(f)

print(
    "Embeddings:",
    len(embeddings)
)


print("Loading metadata...")

meta = pd.read_parquet(META)


# official_name -> WHO embedding id

name_to_id = dict(
    zip(
        meta["official_name"],
        meta["allergen_id"]
    )
)


print(
    "Names available:",
    len(name_to_id)
)


print("Loading gold standard...")

gold = pd.read_csv(GOLD)

print(
    "Gold pairs:",
    len(gold)
)



# --------------------------------------------------
# Cosine similarity
# --------------------------------------------------

def cosine(a, b):

    return cosine_similarity(
        [embeddings[a]],
        [embeddings[b]]
    )[0][0]



# --------------------------------------------------
# Positive pairs
# --------------------------------------------------

results = []

positive_pairs = set()


for _, row in gold.iterrows():

    name_a = row["allergen_id_1"]
    name_b = row["allergen_id_2"]


    if (
        name_a in name_to_id
        and name_b in name_to_id
    ):

        a = name_to_id[name_a]
        b = name_to_id[name_b]


        results.append({

            "allergen_1": name_a,
            "allergen_2": name_b,

            "embedding_id_1": a,
            "embedding_id_2": b,

            "cosine_similarity":
                cosine(a, b),

            "label": 1

        })


        positive_pairs.add(
            tuple(sorted([a,b]))
        )



print(
    "Positive added:",
    len(results)
)



# --------------------------------------------------
# Negative random pairs
# --------------------------------------------------

names = list(
    embeddings.keys()
)


negative_needed = len(results)

negative_count = 0


while negative_count < negative_needed:


    a, b = random.sample(
        names,
        2
    )


    pair = tuple(
        sorted([a,b])
    )


    # skip known cross-reactive pairs

    if pair in positive_pairs:
        continue


    results.append({

        "allergen_1": a,
        "allergen_2": b,

        "embedding_id_1": a,
        "embedding_id_2": b,

        "cosine_similarity":
            cosine(a,b),

        "label": 0

    })


    negative_count += 1



print(
    "Negative added:",
    negative_count
)



# --------------------------------------------------
# Save results
# --------------------------------------------------

df = pd.DataFrame(results)


df.to_csv(
    OUTPUT,
    index=False
)



print("\n==============================")
print("BASELINE RESULT")
print("==============================")


print(
    df.groupby("label")["cosine_similarity"].describe()
)



# --------------------------------------------------
# Metrics
# --------------------------------------------------

roc = roc_auc_score(
    df["label"],
    df["cosine_similarity"]
)


pr = average_precision_score(
    df["label"],
    df["cosine_similarity"]
)


print("\n==============================")
print("METRICS")
print("==============================")


print(
    f"ROC-AUC: {roc:.4f}"
)

print(
    f"PR-AUC : {pr:.4f}"
)



print("\nSaved:")
print(OUTPUT)