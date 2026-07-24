"""
Plot cosine similarity distributions.

Input:
    output/cosine_baseline_evaluation.csv

Output:
    output/cosine_similarity_histogram.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =====================================================
# PATHS
# =====================================================

INPUT = Path(
    "/home/lana/ALERGRAF/output/cosine_baseline_evaluation.csv"
)

OUTPUT = Path(
    "/home/lana/ALERGRAF/output/cosine_similarity_histogram.png"
)


# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(INPUT)

positive = df[df["label"] == 1]["cosine_similarity"]

negative = df[df["label"] == 0]["cosine_similarity"]


# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(9,6))

plt.hist(
    negative,
    bins=20,
    alpha=0.6,
    label="Random pairs",
)

plt.hist(
    positive,
    bins=20,
    alpha=0.6,
    label="Known cross-reactive pairs",
)

plt.xlabel("Cosine similarity")
plt.ylabel("Number of allergen pairs")
plt.title("Distribution of cosine similarities")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300
)

plt.show()

print("\nSaved:", OUTPUT)