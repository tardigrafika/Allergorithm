"""
PCA visualization of ESM protein embeddings

Input:
    embeddings.parquet

Output:
    pca_embedding_space.png
"""


from pathlib import Path

import ast
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# ======================================================
# Configuration
# ======================================================

INPUT_FILE = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.parquet"
)

OUTPUT_FILE = Path(
    "/home/lana/ALERGRAF/output/pca_embedding_space.png"
)



# ======================================================
# Load embeddings
# ======================================================

print("==============================")
print("LOADING EMBEDDINGS")
print("==============================")


df = pd.read_parquet(INPUT_FILE)


print(
    f"Loaded {len(df)} proteins"
)


print(
    df.columns.tolist()
)



# ======================================================
# Convert embeddings
# ======================================================

print()
print("==============================")
print("PREPARING MATRIX")
print("==============================")


X = np.vstack(
    df["embedding"].apply(
        lambda x: np.array(x)
    )
)


print(
    "Embedding matrix:",
    X.shape
)



# ======================================================
# PCA
# ======================================================

print()
print("==============================")
print("RUNNING PCA")
print("==============================")


# Standardization helps PCA
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


pca = PCA(
    n_components=2,
    random_state=42
)


X_pca = pca.fit_transform(
    X_scaled
)


print(
    "Explained variance:"
)

print(
    pca.explained_variance_ratio_
)

print(
    "Total:",
    pca.explained_variance_ratio_.sum()
)



# ======================================================
# Plot
# ======================================================

print()
print("==============================")
print("PLOTTING")
print("==============================")


plt.figure(
    figsize=(12,9)
)


families = (
    df["protein_family"]
    .fillna("Unknown")
    .unique()
)


for family in families:

    mask = (
        df["protein_family"]
        .fillna("Unknown")
        ==
        family
    )

    plt.scatter(
        X_pca[mask,0],
        X_pca[mask,1],
        label=family,
        s=80,
        alpha=0.8
    )



# Add labels

for i, name in enumerate(
    df["official_name"]
):

    plt.annotate(
        name,
        (
            X_pca[i,0],
            X_pca[i,1]
        ),
        fontsize=7,
        alpha=0.7
    )



plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.title(
    "PCA Visualization of ESM-2 Allergen Embeddings"
)


plt.legend(
    bbox_to_anchor=(1.05,1),
    loc="upper left"
)


plt.tight_layout()



plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)


plt.show()



print()
print("==============================")
print("DONE")
print("==============================")

print(
    "Saved:",
    OUTPUT_FILE
)