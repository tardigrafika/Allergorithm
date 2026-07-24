"""
Statistical significance of cosine similarity baseline.

Calculates:
- mean
- standard deviation
- Welch's t-test
- p-value
"""

from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind


# =====================================================
# PATH
# =====================================================

INPUT = Path(
    "/home/lana/ALERGRAF/output/cosine_baseline_evaluation.csv"
)


# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(INPUT)

positive = df[df["label"] == 1]["cosine_similarity"]

negative = df[df["label"] == 0]["cosine_similarity"]


# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

print("\n==============================")
print("DESCRIPTIVE STATISTICS")
print("==============================")

print(f"Positive pairs: {len(positive)}")
print(f"Mean           : {positive.mean():.6f}")
print(f"Std            : {positive.std():.6f}")

print()

print(f"Negative pairs: {len(negative)}")
print(f"Mean           : {negative.mean():.6f}")
print(f"Std            : {negative.std():.6f}")


# =====================================================
# WELCH T-TEST
# =====================================================

t_stat, p_value = ttest_ind(
    positive,
    negative,
    equal_var=False
)


print("\n==============================")
print("WELCH T-TEST")
print("==============================")

print(f"T statistic : {t_stat:.6f}")
print(f"P value     : {p_value:.10e}")


if p_value < 0.05:
    print("\nResult: Difference is statistically significant.")
else:
    print("\nResult: Difference is NOT statistically significant.")