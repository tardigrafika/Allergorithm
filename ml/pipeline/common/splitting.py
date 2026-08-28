"""
Group-aware protein-level split (Union-Find) -- identicna funkcija u SVIM
ml/*.py skriptovima (random_forest_baseline.py, mlp_baseline.py, itd.).
Povezane komponente gold-standard grafa se drze CELE u jednom splitu, da
nijedan protein sa dokumentovanim cross-reactive partnerom ne zavrsi i u
train i u test skupu (data leakage).

Takodje sadrzi LOCO (leave-one-connected-component-out) varijantu -- kasnije
u sesiji ustanovljen rigor standard (ml/loco_*.py, ml/graph_propagation_*.py)
kad se ispostavilo da K=5 slucajni fold unosi previse suma na ovoliko malom
grafu (~44-47 nezavisnih komponenti).
"""

import numpy as np


def find_connected_components(pairs: list) -> list:
    """pairs: lista dict-ova sa 'id_1'/'id_2'. Vraca listu skupova (komponenti)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pairs:
        union(p["id_1"], p["id_2"])

    components = {}
    for pid in parent:
        components.setdefault(find(pid), set()).add(pid)
    return list(components.values())


def group_aware_split(gold_pairs: list, all_ids: list, test_fraction: float, seed: int):
    """Identicna logika svuda: cele povezane komponente idu u train ILI test
    (nikad podeljene), pa se 'slobodni' proteini (bez gold para) dele nezavisno
    da se pogodi ciljani test_fraction. Vraca (train_ids, test_ids) kao skupove."""
    component_list = find_connected_components(gold_pairs)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(component_list))
    gold_protein_count = sum(len(c) for c in component_list)
    target_component_test = round(test_fraction * gold_protein_count)

    train_ids, test_ids = set(), set()
    running_test = 0
    for idx in order:
        component = component_list[idx]
        if running_test < target_component_test:
            test_ids |= component
            running_test += len(component)
        else:
            train_ids |= component

    free_proteins = [pid for pid in all_ids if pid not in train_ids and pid not in test_ids]
    free_proteins = list(free_proteins)
    rng.shuffle(free_proteins)
    n_free_test = round(test_fraction * len(free_proteins))
    test_ids |= set(free_proteins[:n_free_test])
    train_ids |= set(free_proteins[n_free_test:])

    assert train_ids.isdisjoint(test_ids), "protein split is not disjoint (bug)"
    assert len(train_ids) + len(test_ids) == len(all_ids), "split does not cover all proteins"

    return train_ids, test_ids


def split_pairs(gold_pairs: list, train_ids: set, test_ids: set):
    """Podeli gold_pairs na train/test na osnovu kojem skupu (train_ids/test_ids)
    OBA kraja para pripadaju. Baca AssertionError ako neki par ostane 'preko
    granice' (curenje -- ne bi trebalo da se ikad desi ako su train/test
    dobijeni preko group_aware_split iznad)."""
    train_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
    test_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
    cross_split = len(gold_pairs) - len(train_pairs) - len(test_pairs)
    assert cross_split == 0, "a gold pair spans both splits (leakage, should be impossible)"
    return train_pairs, test_pairs


def loco_folds(gold_pairs: list):
    """Leave-one-connected-component-out: svaka povezana komponenta je
    sopstveni test fold, train = SVE OSTALE komponente. Vraca listu
    (train_pairs, test_pairs, test_ids) -- jedna stavka po foldu.
    (ml/graph_propagation_signal_1548.py, ml/weighted_rrf4_fusion_1548.py, itd.)"""
    component_list = find_connected_components(gold_pairs)
    folds = []
    for test_ids in component_list:
        train_pairs = [p for p in gold_pairs if p["id_1"] not in test_ids and p["id_2"] not in test_ids]
        test_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
        if not test_pairs or not train_pairs:
            continue
        folds.append((train_pairs, test_pairs, test_ids))
    return folds
