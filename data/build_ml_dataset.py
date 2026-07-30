#!/usr/bin/env python3
"""
Pretvara gold standard cross-reactivnosti u skup podataka za treniranje MLP modela.

Ucitava sve alergene iz clean_allergens.csv
Ucitava pozitivne parove i dodeljuje tezine prema jacini dokaza
Grupise alergene u povezane klastere
Deli klastere na train/val/test bez preklapanja
Generise negativne parove (lake i teske)
Output:
 train_pairs.csv, 
 val_pairs.csv 
 test_pairs.csv sa kolonama: allergen_id_1, allergen_id_2, label, weight, pair_type, split.
"""

import argparse
import csv
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# nivo dokazanosti > confidence weight for positives
# ono sto nije pomenuto DEFAULT_POS_WEIGHT.
# "they're a starting point, not gospel"
# ---------------------------------------------------------------------------
EVIDENCE_WEIGHTS = {
    "Confirmed": 1.0,
    "Confirmed (with epitope-level caveat)": 0.9,
    "Confirmed (partial epitope overlap)": 0.85,
    "Confirmed but low real-world prevalence": 0.8,
    "Confirmed in this population (see NEG016 for a contradicting temperate-climate finding)": 0.7,
    "Strong evidence": 0.85,
    "Strong evidence (within-species paralogs)": 0.8,
    "Strong evidence (congeneric species)": 0.8,
    "Suspected (homology-based)": 0.6,
    "Suspected": 0.55,
    "Suspected (homology-based; reduced cross-subfamily reactivity)": 0.45,
    "Suspected (homology-based; reduced cross-reactivity)": 0.45,
    "Suspected (homology-based; reduced cross-reactivity reported)": 0.45,
    "Suspected (homology-based; low sequence identity)": 0.4,
    "Suspected (homology-based; weak/limited cross-reactivity)": 0.35,
    "Suspected (homology-based; moderate)": 0.5,
    "Suspected (homology-based; amphibian-fish)": 0.45,
    "Suspected (homology-based; within-species isoforms)": 0.55,
    "Suspected (homology-based only)": 0.5,
    "Suspected (homology-based; clinical significance under-researched)": 0.4,
    "Suspected (low confidence by design of source review)": 0.35,
    "Inferred (family-level homology)": 0.4,
}
DEFAULT_POS_WEIGHT = 0.5


NEGATIVE_EVIDENCE_MARKERS = (
    "Reported negative",
    "Reported NO cross-reactivity",
    "Reported low/weak",
    "Risky/Contested",
    "Suspected/Contested",
)


def is_negative_evidence(evidence_level: str) -> bool:
    return any(evidence_level.strip().startswith(m) for m in NEGATIVE_EVIDENCE_MARKERS)


# ---------------------------------------------------------------------------
# Union-Find 
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_allergens(path):
    """vraca dict official_name > {source_food, organism, sequence_length}.
    NOTE: keyed by official_name (e.g. "Bet v 1.0101"), not the WHO/IUIS
    allergen_id used inside embeddings.pkl -- this matches how
    cross_reactive_combined.csv's "allergen_id_1"/"allergen_id_2" columns
    actually contain official names, not WHO IDs (see other *_1443 scripts'
    name_to_id mapping for the same convention)."""
    allergens = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["official_name"].strip()
            allergens[name] = {
                "source_food": row.get("source_food", "").strip(),
                "organism": row.get("organism", "").strip(),
                "sequence_length": row.get("sequence_length", "").strip(),
            }
    return allergens


def load_embeddings(pickle_path):
    """vraca dict WHO/IUIS allergen_id > 1280 np.array embedding."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def build_name_to_embedding(embeddings_pkl_path, embeddings_parquet_path):
    """
    Returns dict: official_name -> raw embedding vector, by joining
    embeddings.pkl (keyed by WHO/IUIS allergen_id) through
    embeddings.parquet's allergen_id<->official_name mapping
    """
    import pandas as pd

    embeddings = load_embeddings(embeddings_pkl_path)
    meta = pd.read_parquet(embeddings_parquet_path)
    meta = meta[meta["allergen_id"].isin(embeddings.keys())]

    name_to_embedding = {}
    for official_name, allergen_id in zip(meta["official_name"], meta["allergen_id"]):
        name = str(official_name).strip()
        if name and name not in name_to_embedding:
            name_to_embedding[name] = embeddings[allergen_id]

    return embeddings, name_to_embedding


def load_positive_pairs(path, known_allergens):
    """
    vraca listu dicts {a, b, evidence_level, weight, family}
    """
    pairs = []
    seen = set()
    n_dropped_negative = 0
    n_dropped_dup = 0
    n_missing_allergen = 0

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            a = row["allergen_id_1"].strip()
            b = row["allergen_id_2"].strip()
            level = row.get("evidence_level", "").strip()

            if is_negative_evidence(level):
                n_dropped_negative += 1
                continue

            if a not in known_allergens or b not in known_allergens:
                n_missing_allergen += 1
                continue

            key = frozenset([a, b])
            if key in seen:
                n_dropped_dup += 1
                continue
            seen.add(key)

            weight = EVIDENCE_WEIGHTS.get(level, DEFAULT_POS_WEIGHT)
            pairs.append({
                "a": a, "b": b,
                "evidence_level": level,
                "weight": weight,
                "family": row.get("family_1", "") or row.get("family_2", ""),
            })

    print(f"[positives] loaded {len(pairs)} usable positive pairs "
          f"(dropped {n_dropped_negative} documented-negative rows, "
          f"{n_dropped_dup} exact duplicates, "
          f"{n_missing_allergen} referencing an allergen not in clean_allergens.csv)")
    return pairs


def build_clusters(all_allergens, positive_pairs):
    """Union-Find over all allergens using positive edges. Every allergen
    with no positive edge ends up in its own singleton cluster."""
    uf = UnionFind(all_allergens)
    for p in positive_pairs:
        uf.union(p["a"], p["b"])

    clusters = defaultdict(list)
    for a in all_allergens:
        clusters[uf.find(a)].append(a)
    return list(clusters.values())


def _greedy_bin_pack(clusters, train_frac, val_frac, rng):
    """Greedy bin-pack clusters (by size, largest first, ties shuffled) into
    train/val/test so the *allergen count* ratio approximates
    train_frac/val_frac/test_frac, without ever splitting a cluster."""
    clusters = list(clusters)
    rng.shuffle(clusters)
    clusters.sort(key=len, reverse=True)

    total = sum(len(c) for c in clusters)
    target = {
        "train": train_frac * total,
        "val": val_frac * total,
        "test": (1 - train_frac - val_frac) * total,
    }
    bins = {"train": [], "val": [], "test": []}
    filled = {"train": 0, "val": 0, "test": 0}

    for c in clusters:
        deficit = {k: target[k] - filled[k] for k in bins}
        choice = max(deficit, key=deficit.get)
        bins[choice].append(c)
        filled[choice] += len(c)

    return bins, filled


def split_clusters(clusters, train_frac, val_frac, seed):
    """
    Splits clusters into train/val/test, handling two groups SEPARATELY:

      - "informative" clusters (size > 1, i.e. contain at least one positive
        pair) are bin-packed first, on their own, so that val and test are
        each guaranteed a fair share of the signal that actually exists.
        Without this, a dataset where positives cluster into a small number
        of dense families (as this one does: ~29 informative clusters out
        of 1300+) will greedily starve val/test of ANY positive examples,
        because a proportional-to-total-size packing always prefers
        whichever split has the biggest raw deficit -- which for a while is
        always train, since train's target is ~4-5x val/test's.
      - "singleton" clusters (allergens with no known positive pair) are
        distributed afterwards, independently, just to size up each split's
        negative-sampling pool. They carry no label-leakage risk since they
        have no edges at all.
    """
    rng = random.Random(seed)

    informative = [c for c in clusters if len(c) > 1]
    singletons = [c for c in clusters if len(c) == 1]

    info_bins, info_filled = _greedy_bin_pack(informative, train_frac, val_frac, rng)
    n_info_val = sum(1 for c in info_bins["val"])
    n_info_test = sum(1 for c in info_bins["test"])
    if n_info_val == 0 or n_info_test == 0:
        print(f"  [warn] only {len(informative)} informative clusters total -- "
              f"val got {n_info_val}, test got {n_info_test}. Consider lowering "
              f"--val-frac/--test-frac allocation expectations, or accept that "
              f"one split may have very few/no positives.")

    single_bins, single_filled = _greedy_bin_pack(singletons, train_frac, val_frac, rng)

    bins = {k: info_bins[k] + single_bins[k] for k in ("train", "val", "test")}
    filled = {k: info_filled[k] + single_filled[k] for k in ("train", "val", "test")}

    print(f"  [clusters] informative clusters by split -> "
          f"train={len(info_bins['train'])}, val={len(info_bins['val'])}, test={len(info_bins['test'])} "
          f"(allergens in those: train={info_filled['train']}, val={info_filled['val']}, test={info_filled['test']})")

    return bins, filled


def load_family_map(positive_pairs):
    """Best-effort allergen -> family map, built from the family column of
    whichever positive pairs mention each allergen. Allergens that never
    appear in a positive pair (the vast majority, ~1270/1535 here) have no
    known family and are treated as 'unknown' -- they can still be used for
    EASY negatives, just not for family-based HARD negatives, since we have
    no family label to compare against."""
    fam = {}
    for p in positive_pairs:
        f = p.get("family", "").strip()
        if not f:
            continue
        fam.setdefault(p["a"], f)
        fam.setdefault(p["b"], f)
    return fam


def load_documented_negatives(path, known_allergens):
    """Loads a hand-curated table of pairs with PUBLISHED evidence of
    absent/reduced cross-reactivity (documented_negatives.csv). These are
    the strongest possible negatives -- unlike randomly/hard-mined negatives,
    someone actually ran the assay and it came back negative (or reduced).
    Returns a list of (a, b, notes) and is silently skipped if the file
    doesn't exist, so this script still works without it."""
    import os
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            a = row["allergen_id_1"].strip()
            b = row["allergen_id_2"].strip()
            if a not in known_allergens or b not in known_allergens:
                continue
            out.append((a, b, row.get("finding_type", "")))
    print(f"[documented negatives] loaded {len(out)} literature-verified negative/reduced pairs")
    return out


def food_keyword(source_food):
    """Crude bucketing heuristic for 'hard negative' sampling: first
    significant word of the source_food string, lowercased."""
    if not source_food:
        return ""
    first = source_food.split(",")[0].split("(")[0].strip().lower()
    return first.split()[0] if first else ""


def sample_negatives(allergen_pool, positive_pairs_set, allergen_meta, family_map,
                      n_needed, hard_frac, rng, name_to_embedding=None,
                      hard_candidates_per_draw=10, max_attempts_factor=50):
    """
   Generise zadati broj negativnih parova, izuzimajuci poznate pozitivne.

* Pravi lake i teske negativne parove.
* Teski parovi moraju biti iz razlicitih poznatih proteinskih familija.
* Parovi iz iste ili nepoznate familije se ne koriste kao teski negativni.
* Ako postoje embedding vektori, bira najteze parove sa najvecom kosinusnom slicnoscu.
* Ako embedding nije dostupan, koristi samo pravilo razlicitih familija.

    """
    pool = list(allergen_pool)
    n_pool = len(pool)
    if n_pool < 2:
        return []

    # Hard-negative candidates: allergens in this pool that HAVE a known
    # family, bucketed by that family.
    labeled = [a for a in pool if a in family_map]
    fam_buckets = defaultdict(list)
    for a in labeled:
        fam_buckets[family_map[a]].append(a)
    fam_names = list(fam_buckets.keys())

    negatives = []
    seen = set()
    n_hard_target = int(n_needed * hard_frac)
    max_attempts = n_needed * max_attempts_factor

    def cosine_sim(name_a, name_b):
        ea, eb = name_to_embedding[name_a], name_to_embedding[name_b]
        denom = (np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-12
        return float(np.dot(ea, eb) / denom)

    use_geometric_hardness = name_to_embedding is not None

    # --- hard negatives: sample two DIFFERENT families, then one allergen
    # from each; if embeddings are available, keep the most cosine-similar
    # of several random draws ("hardest of K") instead of the first valid one ---
    attempts = 0
    while len(negatives) < n_hard_target and len(fam_names) >= 2 and attempts < max_attempts:
        best_pair = None
        best_similarity = -2.0  # cosine similarity is always > -2

        n_draws = hard_candidates_per_draw if use_geometric_hardness else 1
        for _ in range(n_draws):
            attempts += 1
            fam_a, fam_b = rng.sample(fam_names, 2)
            a = rng.choice(fam_buckets[fam_a])
            b = rng.choice(fam_buckets[fam_b])
            key = frozenset([a, b])
            if key in positive_pairs_set or key in seen:
                continue

            if not use_geometric_hardness:
                best_pair = (a, b)
                break

            if a in name_to_embedding and b in name_to_embedding:
                similarity = cosine_sim(a, b)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (a, b)

        if best_pair is None:
            continue
        a, b = best_pair
        seen.add(frozenset([a, b]))
        tag = "hard_cross_family_geometric" if use_geometric_hardness else "hard_cross_family"
        negatives.append((a, b, tag))

    # --- easy (uniform random) negatives fill the rest ---
    attempts = 0
    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(pool, 2)
        key = frozenset([a, b])
        if key in positive_pairs_set or key in seen:
            continue
        seen.add(key)
        negatives.append((a, b, "easy_random"))

    if len(negatives) < n_needed:
        print(f"  [warn] could only sample {len(negatives)}/{n_needed} negatives "
              f"for a pool of {n_pool} allergens ({len(labeled)} with a known family)")

    return negatives


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allergens", required=True, help="path to clean_allergens.csv")
    ap.add_argument("--pairs", required=True, help="path to cross_reactive_combined.csv")
    ap.add_argument("--embeddings-pkl", default="embeddings/embeddings.pkl",
                     help="path to embeddings.pkl (WHO/IUIS allergen_id -> ESM-2 vector). "
                          "Used to (a) drop any allergen with no generated embedding, since "
                          "it would be unusable downstream, and (b) select GEOMETRICALLY hard "
                          "negatives (see --hard-frac). Pass an empty string to disable both.")
    ap.add_argument("--embeddings-parquet", default="embeddings/embeddings.parquet",
                     help="path to embeddings.parquet, used only to map official_name -> "
                          "allergen_id for the --embeddings-pkl lookup above.")
    ap.add_argument("--documented-negatives", default=None,
                     help="optional path to documented_negatives.csv (literature-verified "
                          "negative/reduced cross-reactivity pairs). If given, these are "
                          "included as negatives with top priority before hard/easy sampling.")
    ap.add_argument("--outdir", default="ml_dataset", help="output directory")
    ap.add_argument("--neg-ratio", type=float, default=5.0,
                     help="negatives sampled per positive, per split (default 5x)")
    ap.add_argument("--hard-frac", type=float, default=0.3,
                     help="fraction of negatives that are 'hard' (default 0.3)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    allergen_meta = load_allergens(args.allergens)
    all_allergens = list(allergen_meta.keys())
    print(f"[allergens] {len(all_allergens)} total allergens in {args.allergens}")

    name_to_embedding = {}
    if args.embeddings_pkl:
        _, name_to_embedding = build_name_to_embedding(args.embeddings_pkl, args.embeddings_parquet)
        missing_embedding = [a for a in all_allergens if a not in name_to_embedding]
        if missing_embedding:
            print(f"[embeddings] dropping {len(missing_embedding)} allergen(s) with no generated "
                  f"embedding (unusable by any downstream model): {missing_embedding}")
            all_allergens = [a for a in all_allergens if a in name_to_embedding]
        print(f"[embeddings] {len(name_to_embedding)} allergens have a usable embedding "
              f"-- these are eligible for GEOMETRIC hard-negative mining (see --hard-frac)")
    else:
        print("[embeddings] --embeddings-pkl disabled -- hard negatives will fall back to "
              "family-only selection (no cosine-similarity-based hardness)")

    positive_pairs = load_positive_pairs(args.pairs, set(all_allergens))
    positive_set_global = {frozenset([p["a"], p["b"]]) for p in positive_pairs}
    family_map = load_family_map(positive_pairs)
    print(f"[family map] {len(family_map)} allergens have a known family label "
          f"(from appearing in a positive pair) -- these are the only ones eligible "
          f"as HARD-negative candidates. The other {len(all_allergens) - len(family_map)} "
          f"allergens have no family label and are only used for easy/random negatives.")

    documented_negatives = load_documented_negatives(args.documented_negatives, set(all_allergens))

    clusters = build_clusters(all_allergens, positive_pairs)
    n_multi = sum(1 for c in clusters if len(c) > 1)
    n_singleton = sum(1 for c in clusters if len(c) == 1)
    print(f"[clusters] {len(clusters)} connected components "
          f"({n_multi} multi-allergen clusters, {n_singleton} singletons "
          f"with no known positive pair)")

    bins, filled = split_clusters(clusters, args.train_frac, args.val_frac, args.seed)
    print(f"[split] allergen counts -> train={filled['train']}, "
          f"val={filled['val']}, test={filled['test']}")

    # allergen -> split lookup, and per-split allergen pools
    allergen_to_split = {}
    split_pools = {"train": [], "val": [], "test": []}
    for split_name, cluster_list in bins.items():
        for c in cluster_list:
            for a in c:
                allergen_to_split[a] = split_name
                split_pools[split_name].append(a)

    # positives per split (an edge's split = the split of its cluster; both
    # endpoints are guaranteed to be in the same split by construction)
    split_positives = {"train": [], "val": [], "test": []}
    for p in positive_pairs:
        s = allergen_to_split[p["a"]]
        assert s == allergen_to_split[p["b"]], "positive edge crossed a split boundary (bug)"
        split_positives[s].append(p)

    # documented negatives per split -- both endpoints must land in the same
    # split (from the earlier cluster-based partition) or we skip the pair,
    # since we can't guarantee it without introducing cross-split leakage.
    split_documented_neg = {"train": [], "val": [], "test": []}
    n_doc_skipped_cross_split = 0
    for a, b, finding_type in documented_negatives:
        sa, sb = allergen_to_split.get(a), allergen_to_split.get(b)
        if sa is None or sb is None or sa != sb:
            n_doc_skipped_cross_split += 1
            continue
        split_documented_neg[sa].append((a, b, finding_type))
    if n_doc_skipped_cross_split:
        print(f"[documented negatives] skipped {n_doc_skipped_cross_split} pair(s) whose "
              f"two allergens landed in different splits (can't place without leakage)")

    import os
    os.makedirs(args.outdir, exist_ok=True)

    grand_total_rows = 0
    for split_name in ("train", "val", "test"):
        pool = split_pools[split_name]
        pos = split_positives[split_name]
        doc_neg = split_documented_neg[split_name]
        n_neg_needed = max(0, int(round(len(pos) * args.neg_ratio)) - len(doc_neg))

        mined_negatives = sample_negatives(
            pool, positive_set_global, allergen_meta, family_map,
            n_neg_needed, args.hard_frac, rng,
            name_to_embedding=name_to_embedding or None,
        )

        out_path = os.path.join(args.outdir, f"{split_name}_pairs.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["allergen_id_1", "allergen_id_2", "label", "weight",
                        "pair_type", "evidence_level", "split"])
            for p in pos:
                w.writerow([p["a"], p["b"], 1, round(p["weight"], 3),
                            "positive", p["evidence_level"], split_name])
            for a, b, finding_type in doc_neg:
                w.writerow([a, b, 0, 1.0, "negative_documented", finding_type, split_name])
            for a, b, kind in mined_negatives:
                w.writerow([a, b, 0, 1.0, kind, "", split_name])

        n_rows = len(pos) + len(doc_neg) + len(mined_negatives)
        grand_total_rows += n_rows
        n_hard = sum(1 for *_, k in mined_negatives if k.startswith("hard_cross_family"))
        n_easy = sum(1 for *_, k in mined_negatives if k == "easy_random")
        print(f"[{split_name}] {len(pool)} allergens available | "
              f"{len(pos)} positives | {len(doc_neg)} documented negatives | "
              f"{len(mined_negatives)} mined negatives ({n_hard} hard_cross_family / {n_easy} easy_random) "
              f"-> {out_path}")

    print(f"\nDone. {grand_total_rows} total rows written across train/val/test "
          f"into {args.outdir}/")
    print("\nReminder: 'weight' is a soft-label confidence multiplier for positives "
          "(1.0 = Confirmed ... down to ~0.25-0.4 for Inferred/family-level pairs). "
          "Negatives are all weight=1.0 by construction. Use it as a per-sample loss "
          "weight, or threshold it to build a stricter positive set if you'd rather "
          "not trust the family-level-inferred pairs at all.")


if __name__ == "__main__":
    main()
