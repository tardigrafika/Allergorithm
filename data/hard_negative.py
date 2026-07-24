"""
Hard negative evaluation for ESM embeddings.

Finds proteins that are highly similar in embedding space
but are NOT known cross-reactive.

"""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# =========================
# PATHS
# =========================

EMBEDDINGS = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
)

METADATA = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.parquet"
)

GOLD = Path(
    "/home/lana/ALERGRAF/output/gold_standard_cross_reactivity.csv"
)

OUTPUT = Path(
    "/home/lana/ALERGRAF/output/hard_negative_results.csv"
)



# =========================
# LOAD
# =========================


print("Loading embeddings...")

with open(EMBEDDINGS,"rb") as f:
    embeddings = pickle.load(f)



meta = pd.read_parquet(METADATA)

gold = pd.read_csv(GOLD)



print("Proteins:", len(embeddings))
print("Gold pairs:", len(gold))



# =========================
# MATRIX
# =========================


ids = list(embeddings.keys())

X = np.vstack(
    [
        embeddings[x]
        for x in ids
    ]
)


similarity_matrix = cosine_similarity(X)



id_to_index = {
    x:i for i,x in enumerate(ids)
}



# =========================
# NAME MAPPING
# =========================


name_to_id = dict(
    zip(
        meta["official_name"],
        meta["allergen_id"]
    )
)



# =========================
# KNOWN POSITIVE SET
# =========================


positive_pairs=set()


for _,row in gold.iterrows():

    a=name_to_id.get(
        row["allergen_id_1"]
    )

    b=name_to_id.get(
        row["allergen_id_2"]
    )


    if a and b:

        positive_pairs.add(
            tuple(
                sorted([a,b])
            )
        )



print(
    "Known pairs:",
    len(positive_pairs)
)



# =========================
# POSITIVES
# =========================


results=[]


for _,row in gold.iterrows():

    a=name_to_id.get(
        row["allergen_id_1"]
    )

    b=name_to_id.get(
        row["allergen_id_2"]
    )


    if a and b:

        results.append({

            "protein_1":a,
            "protein_2":b,

            "cosine":
            similarity_matrix[
                id_to_index[a],
                id_to_index[b]
            ],

            "type":"positive",

            "label":1

        })



# =========================
# HARD NEGATIVES
# =========================


print(
    "Searching hard negatives..."
)



for _,row in gold.iterrows():

    anchor=name_to_id.get(
        row["allergen_id_1"]
    )


    if anchor is None:
        continue


    idx=id_to_index[anchor]


    similarities = similarity_matrix[idx]


    ranked=np.argsort(
        similarities
    )[::-1]


    found=0


    for j in ranked:


        candidate=ids[j]


        if candidate==anchor:
            continue


        pair=tuple(
            sorted(
                [
                    anchor,
                    candidate
                ]
            )
        )


        # skip known reactions
        if pair in positive_pairs:
            continue


        results.append({

            "protein_1":anchor,
            "protein_2":candidate,

            "cosine":
            similarities[j],

            "type":"hard_negative",

            "label":0

        })


        found+=1


        if found==1:
            break



# =========================
# SAVE
# =========================


df=pd.DataFrame(results)


df.to_csv(
    OUTPUT,
    index=False
)



print("\n===================")
print(df.groupby("type")["cosine"].describe())
print("===================")

print(
    "Saved:",
    OUTPUT
)