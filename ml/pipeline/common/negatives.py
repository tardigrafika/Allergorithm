"""
Negative sampling -- identicna funkcija u SVIM ml/*.py skriptovima koji
treniraju klasifikator (RF, MLP, XGBoost, Hadamard). Nasumicno bira parove
proteina iz DATOG pool-a (train ili test proteini, nikad mesano) koji NISU
dokumentovan pozitivan par.
"""

import numpy as np


def sample_negative_pairs(protein_pool, n_needed: int, seed: int, positive_pair_set: set):
    """
    sorted(), ne list(): Python randomizuje redosled hash-a stringova po
    procesu, pa list(neki_skup) NIJE reproducibilno ni sa fiksnim numpy
    seed-om -- sortiranje fiksira determinsticki redosled elemenata.
    """
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    negatives = set()

    max_attempts = n_needed * 50 + 2000
    attempts = 0

    while len(negatives) < n_needed and attempts < max_attempts:
        a, b = local_rng.choice(pool, size=2, replace=False)
        pair = tuple(sorted((a, b)))
        attempts += 1

        if pair in positive_pair_set or pair in negatives:
            continue

        negatives.add(pair)

    if len(negatives) < n_needed:
        print(f"WARNING: only sampled {len(negatives)}/{n_needed} negatives "
              f"(pool too small or too many collisions)")

    # sorted(), ne list(): isti razlog kao gore -- RF-ovo bootstrap sampling je
    # index-based, pa redosled reda u feature matrici inace tiho menja koji
    # se stabla treniraju cak i sa fiksnim seed-om.
    return sorted(negatives)
