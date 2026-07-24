"""
ALERGRAF Cross-Reactivity Ranking Benchmark

Evaluates whether known cross-reactive allergens
are ranked higher than random proteins using
ESM embedding cosine similarity.

Metrics:
- Mean Rank
- MRR
- Hits@1
- Hits@3
- Hits@10
"""


from pathlib import Path
import pandas as pd
import numpy as np
import random

from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS_PATH = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
)

POSITIVE_PAIRS_PATH = Path(
    "/home/lana/ALERGRAF/data/cross_reactive_pairs.csv"
)


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

print("Loading embeddings...")

emb_df = pd.read_csv(EMBEDDINGS_PATH)


protein_ids = emb_df["protein_id"].tolist()


embedding_matrix = emb_df.drop(
    columns=["protein_id"]
).values


embedding_dict = {
    protein: embedding_matrix[i]
    for i, protein in enumerate(protein_ids)
}


print(f"Loaded embeddings: {len(embedding_dict)} proteins")


# =====================================================
# LOAD POSITIVE PAIRS
# =====================================================

pairs_df = pd.read_csv(
    POSITIVE_PAIRS_PATH
)


positive_pairs = list(
    zip(
        pairs_df["protein_1"],
        pairs_df["protein_2"]
    )
)


print(f"Positive pairs: {len(positive_pairs)}")


# =====================================================
# RANK STORAGE
# =====================================================


class Ranks:

    def __init__(self):
        self.ranks = []


    def add_rank(self, rank):
        self.ranks.append(rank)


    def mean_rank(self):
        return np.mean(self.ranks)


    def mrr(self):
        return np.mean(
            [
                1.0 / r
                for r in self.ranks
            ]
        )


    def hits_at_k(self, k):
        return np.mean(
            [
                1.0 if r <= k else 0.0
                for r in self.ranks
            ]
        )


    def add_scores(self, scores):

        target_score = scores[0]

        rank = 1 + sum(
            score > target_score
            for score in scores[1:]
        )

        self.add_rank(rank)



# =====================================================
# COSINE SCORE
# =====================================================


def cosine_score(protein_a, protein_b):

    emb_a = embedding_dict[protein_a]
    emb_b = embedding_dict[protein_b]

    return cosine_similarity(
        [emb_a],
        [emb_b]
    )[0][0]



# =====================================================
# GENERATE NEGATIVE CANDIDATES
# =====================================================


def generate_negative_samples(
        protein,
        true_partner,
        n_samples=500
):

    negatives = []

    while len(negatives) < n_samples:

        candidate = random.choice(
            protein_ids
        )

        if (
            candidate != protein
            and candidate != true_partner
        ):
            negatives.append(candidate)

    return negatives



# =====================================================
# BENCHMARK
# =====================================================


def ranking_benchmark(
        positive_pairs,
        negative_samples=500,
        max_pairs=1000
):

    ranks = Ranks()


    test_pairs = random.sample(
        positive_pairs,
        min(
            len(positive_pairs),
            max_pairs
        )
    )


    for protein_a, protein_b in test_pairs:


        if (
            protein_a not in embedding_dict
            or protein_b not in embedding_dict
        ):
            continue


        scores = []


        # TRUE EDGE FIRST
        true_score = cosine_score(
            protein_a,
            protein_b
        )

        scores.append(true_score)



        # NEGATIVE EDGES

        negatives = generate_negative_samples(
            protein_a,
            protein_b,
            negative_samples
        )


        for neg in negatives:

            score = cosine_score(
                protein_a,
                neg
            )

            scores.append(score)



        ranks.add_scores(scores)



    return ranks



# =====================================================
# RUN
# =====================================================


results = ranking_benchmark(
    positive_pairs,
    negative_samples=500
)


print("\n==============================")
print("ALERGRAF RANKING BENCHMARK")
print("==============================")


print(
    f"Evaluated pairs: {len(results.ranks)}"
)


print(
    f"Mean Rank : {results.mean_rank():.3f}"
)


print(
    f"MRR       : {results.mrr():.4f}"
)


print(
    f"Hits@1    : {results.hits_at_k(1):.4f}"
)


print(
    f"Hits@3    : {results.hits_at_k(3):.4f}"
)


print(
    f"Hits@10   : {results.hits_at_k(10):.4f}"
)