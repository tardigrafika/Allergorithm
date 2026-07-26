"""
Konacni izvestaj koji poredi rezultate za originalni dataset (296 parova) i prosireni dataset (1.443 reda).

Koristi vec sacuvane rezultate svih 10 eksperimenata (5 metoda × 2 velicine dataseta) i pravi jedan objedinjeni izvestaj. 
Ne pokrece ponovo eksperimente niti trenira modele

Izlaz:
`output/comparison_296_vs_1443_summary.txt`

"""

from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
GOLD_296 = OUTPUT_DIR / "cross_reactive_combined.csv"
GOLD_1443 = OUTPUT_DIR / "cross_reactive_1443.csv"

TOP_K = [1, 5, 10, 20]

lines = []


def add(line=""):
    lines.append(line)
    print(line)


def load_hits_mrr(path, hits_col_fn, mrr_col):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {k: df[hits_col_fn(k)].mean() for k in TOP_K}, df[mrr_col].mean(), len(df)


def fmt_row(label, result, width_label=30):
    if result is None:
        return f"{label:<{width_label}}(not found)"
    hits, mrr, n = result
    return (f"{label:<{width_label}}"
            f"{hits[1]:<10.4f}{hits[5]:<10.4f}{hits[10]:<10.4f}{hits[20]:<10.4f}{mrr:<10.4f}"
            f"  (n={n})")


# =====================================================
# DATASET OVERVIEW
# =====================================================

add("=" * 78)
add("FINAL COMPARISON: 296-pair (original) vs 1,443-row (extended) gold standard")
add("=" * 78)
add("")
add("--- DATASET SIZES ---")

gold_296 = pd.read_csv(GOLD_296)
add(f"296-pair gold file (output/cross_reactive_combined.csv): {len(gold_296)} rows, "
    f"{len(gold_296)} used directly (no evidence_level exclusions applied in that experiment family)")

gold_1443_raw = pd.read_csv(GOLD_1443)
neg_mask = gold_1443_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
excluded = gold_1443_raw.loc[neg_mask]
gold_1443 = gold_1443_raw.loc[~neg_mask]

add(f"1443-row gold file (output/cross_reactive_1443.csv): {len(gold_1443_raw)} rows")
add(f"  Excluded as negative/contested/risky: {len(excluded)}")
add(f"  Positive gold-standard pairs retained: {len(gold_1443)}")
add("")
add("--- PAIRS PER EVIDENCE LEVEL (1443 dataset, retained positives only) ---")


def evidence_bucket(value):
    if value.startswith("Confirmed") or value.startswith("Strong evidence"):
        return "Confirmed/Strong"
    if value.startswith("Suspected"):
        return "Suspected"
    if value.startswith("Inferred"):
        return "Inferred/family-level"
    return "UNMAPPED"


buckets = gold_1443["evidence_level"].map(evidence_bucket)
for bucket in ["Confirmed/Strong", "Suspected", "Inferred/family-level"]:
    count = int((buckets == bucket).sum())
    add(f"  {bucket:<24}: {count:5d}  ({count/len(gold_1443):.1%})")
add(f"  {'TOTAL':<24}: {len(gold_1443):5d}")
add("")
add("Excluded rows (not counted as cross-reactive evidence in ANY 1443 experiment):")
for _, row in excluded.iterrows():
    add(f"  {row['pair_id']}: {row['allergen_id_1']} <-> {row['allergen_id_2']} "
        f"-- \"{row['evidence_level']}\"")


# =====================================================
# TRAIN/TEST SPLIT COMPARISON
# =====================================================

add("")
add("--- LEAKAGE-SAFE PROTEIN-LEVEL SPLIT (same algorithm, seed=42, for RF/MLP/embed-transform) ---")
add(f"{'':<28}{'296-pair (old)':<20}{'1432-pair (new)':<20}")
add(f"{'Train proteins':<28}{'1224 (79.8%)':<20}{'1227 (80.0%)':<20}")
add(f"{'Test proteins':<28}{'310 (20.2%)':<20}{'307 (20.0%)':<20}")
add(f"{'Train positive pairs':<28}{'241':<20}{'1267':<20}")
add(f"{'Test positive pairs':<28}{'55':<20}{'165':<20}")
add(f"{'Test retrieval queries':<28}{'110':<20}{'330':<20}")
add("(Values for the '1432-pair' column are read from the runtime output of "
    "ml/random_forest_baseline_1443.py / ml/mlp_baseline_1443.py / "
    "ml/mlp_embedding_transform_1443.py, all three of which were verified via an "
    "in-script consistency check to have produced the IDENTICAL held-out test split.)")


# =====================================================
# RESULTS TABLE -- NEW (1443) DATASET, ALL 5 METHODS
# =====================================================

add("")
add("=" * 78)
add("RESULTS ON THE NEW (1432-pair) DATASET -- all 5 methods")
add("=" * 78)
add("NOTE on query-set scope: Cosine and PCA are unsupervised (no train/test split "
    "needed) and are evaluated on ALL 1432 pairs / 2864 queries below ('full'). RF, "
    "MLP classifier, and MLP embedding-transform are supervised and can only be "
    "evaluated on the 165 held-out TEST pairs / 330 queries ('test-only') without "
    "leakage. A 'Cosine (test-only)' row is also included, recomputed on the exact "
    "same 330 queries as the supervised methods, for a fair apples-to-apples read.")
add("")

cosine_full = load_hits_mrr(OUTPUT_DIR / "hits_mrr_results_1443.csv",
                             lambda k: f"hits_at_{k}", "reciprocal_rank")
pca_full = load_hits_mrr(OUTPUT_DIR / "pca_euclidean_retrieval_results_1443.csv",
                          lambda k: f"hits_at_{k}", "reciprocal_rank")
rf_df_path = OUTPUT_DIR / "random_forest_retrieval_results_1443.csv"
mlp_df_path = OUTPUT_DIR / "mlp_retrieval_results_1443.csv"
embed_df_path = OUTPUT_DIR / "mlp_embedding_transform_retrieval_results_1443.csv"

cosine_test = load_hits_mrr(rf_df_path, lambda k: f"cosine_hits_at_{k}", "cosine_reciprocal_rank")
rf_test = load_hits_mrr(rf_df_path, lambda k: f"rf_hits_at_{k}", "rf_reciprocal_rank")
mlp_test = load_hits_mrr(mlp_df_path, lambda k: f"mlp_hits_at_{k}", "mlp_reciprocal_rank")
embed_test = load_hits_mrr(embed_df_path, lambda k: f"mlp_hits_at_{k}", "mlp_reciprocal_rank")

header = f"{'Method':<30}{'Hits@1':<10}{'Hits@5':<10}{'Hits@10':<10}{'Hits@20':<10}{'MRR':<10}"
add(header)
add("-" * len(header))
add(fmt_row("Cosine (full, 2864q)", cosine_full))
add(fmt_row("PCA(128)+Eucl (full, 2864q)", pca_full))
add(fmt_row("Cosine (test-only, 330q)", cosine_test))
add(fmt_row("Random Forest (test, 330q)", rf_test))
add(fmt_row("MLP classifier (test, 330q)", mlp_test))
add(fmt_row("MLP embed.transform (test)", embed_test))


# =====================================================
# RESULTS TABLE -- OLD (296) DATASET, ALL 5 METHODS (for reference)
# =====================================================

add("")
add("=" * 78)
add("RESULTS ON THE OLD (296-pair) DATASET -- all 5 methods (for reference)")
add("=" * 78)
add("")

cosine_full_old = load_hits_mrr(OUTPUT_DIR / "hits_mrr_results.csv",
                                 lambda k: f"hits_at_{k}", "reciprocal_rank")
pca_full_old = load_hits_mrr(OUTPUT_DIR / "pca_euclidean_retrieval_results.csv",
                              lambda k: f"hits_at_{k}", "reciprocal_rank")
rf_df_path_old = OUTPUT_DIR / "random_forest_retrieval_results.csv"
mlp_df_path_old = OUTPUT_DIR / "mlp_retrieval_results.csv"
embed_df_path_old = OUTPUT_DIR / "mlp_embedding_transform_retrieval_results.csv"

cosine_test_old = load_hits_mrr(rf_df_path_old, lambda k: f"cosine_hits_at_{k}", "cosine_reciprocal_rank")
rf_test_old = load_hits_mrr(rf_df_path_old, lambda k: f"rf_hits_at_{k}", "rf_reciprocal_rank")
mlp_test_old = load_hits_mrr(mlp_df_path_old, lambda k: f"mlp_hits_at_{k}", "mlp_reciprocal_rank")
embed_test_old = load_hits_mrr(embed_df_path_old, lambda k: f"mlp_hits_at_{k}", "mlp_reciprocal_rank")

add(header)
add("-" * len(header))
add(fmt_row("Cosine (full, 592q)", cosine_full_old))
add(fmt_row("PCA(128)+Eucl (full, 592q)", pca_full_old))
add(fmt_row("Cosine (test-only, 110q)", cosine_test_old))
add(fmt_row("Random Forest (test, 110q)", rf_test_old))
add(fmt_row("MLP classifier (test, 110q)", mlp_test_old))
add(fmt_row("MLP embed.transform (test)", embed_test_old))


# =====================================================
# OLD vs NEW: DOES MORE DATA HELP?
# =====================================================

add("")
add("=" * 78)
add("DOES THE LARGER (1432-pair) DATASET IMPROVE THE SUPERVISED MODELS?")
add("=" * 78)
add("Comparing each method's TEST-set MRR: 296-pair test (110q) vs 1432-pair test (330q).")
add("(Test sets differ in size/composition between the two dataset versions, so this is")
add(" a directional comparison, not a controlled ablation on the identical test set.)")
add("")

comparisons = [
    ("Cosine (test-only)", cosine_test_old, cosine_test),
    ("Random Forest", rf_test_old, rf_test),
    ("MLP classifier", mlp_test_old, mlp_test),
    ("MLP embed.transform", embed_test_old, embed_test),
]

header2 = f"{'Method':<22}{'MRR (296)':<14}{'MRR (1432)':<14}{'Delta':<12}{'Direction'}"
add(header2)
add("-" * len(header2))
for label, old_res, new_res in comparisons:
    if old_res is None or new_res is None:
        add(f"{label:<22}(missing data)")
        continue
    old_mrr = old_res[1]
    new_mrr = new_res[1]
    delta = new_mrr - old_mrr
    direction = "IMPROVED" if delta > 0 else ("WORSE" if delta < 0 else "UNCHANGED")
    add(f"{label:<22}{old_mrr:<14.4f}{new_mrr:<14.4f}{delta:<+12.4f}{direction}")


# =====================================================
# DOES COSINE STILL WIN?
# =====================================================

add("")
add("=" * 78)
add("DOES COSINE SIMILARITY STILL GIVE THE BEST RETRIEVAL RESULTS?")
add("=" * 78)

if all(r is not None for r in [cosine_test, rf_test, mlp_test, embed_test]):
    methods_new = {
        "Cosine": cosine_test,
        "Random Forest": rf_test,
        "MLP classifier": mlp_test,
        "MLP embed.transform": embed_test,
    }
    best_new = max(methods_new.items(), key=lambda kv: kv[1][1])
    add(f"On the 1432-pair dataset (test-only, 330 queries), the method with the "
        f"highest MRR is: {best_new[0]} (MRR={best_new[1][1]:.4f})")
    ranked_new = sorted(methods_new.items(), key=lambda kv: kv[1][1], reverse=True)
    add("Full ranking by MRR (1432-pair, test-only):")
    for i, (name, res) in enumerate(ranked_new, 1):
        add(f"  {i}. {name:<22} MRR={res[1]:.4f}")

if all(r is not None for r in [cosine_test_old, rf_test_old, mlp_test_old, embed_test_old]):
    methods_old = {
        "Cosine": cosine_test_old,
        "Random Forest": rf_test_old,
        "MLP classifier": mlp_test_old,
        "MLP embed.transform": embed_test_old,
    }
    best_old = max(methods_old.items(), key=lambda kv: kv[1][1])
    add(f"\nOn the 296-pair dataset (test-only, 110 queries), the method with the "
        f"highest MRR is: {best_old[0]} (MRR={best_old[1][1]:.4f})")
    ranked_old = sorted(methods_old.items(), key=lambda kv: kv[1][1], reverse=True)
    add("Full ranking by MRR (296-pair, test-only):")
    for i, (name, res) in enumerate(ranked_old, 1):
        add(f"  {i}. {name:<22} MRR={res[1]:.4f}")


# =====================================================
# CONCLUSION
# =====================================================

add("")
add("=" * 78)
add("CONCLUSION")
add("=" * 78)
add(
    "On the small 296-pair dataset, cosine similarity was the strongest retrieval "
    "method on the held-out test split (highest MRR/Hits@1/5/10 among all 4 "
    "comparable methods); Random Forest and the MLP classifier trailed it, and the "
    "MLP embedding-transform (Approach B) collapsed almost to random."
)
add("")
add(
    "On the larger, noisier 1432-pair dataset -- which is ~5x bigger but ~78% "
    "'Inferred (family-level homology)' evidence rather than confirmed/strong "
    "evidence -- Random Forest overtakes cosine similarity on every retrieval "
    "metric on its held-out test split, while the MLP classifier remains a step "
    "behind cosine, and the MLP embedding-transform gets markedly WORSE, not "
    "better, despite having ~5x more training pairs (this mirrors the classic "
    "\"more data doesn't help a fundamentally mismatched objective\" pattern flagged "
    "for this approach in plan/firststeps.md: MSE regression in embedding space "
    "still does not translate into correct cosine-similarity ranking against 1534 "
    "candidates)."
)
add("")
add(
    "So: more data helps Random Forest (and to a much lesser extent, does not "
    "rescue the MLP embedding-transform), but it comes at the cost of overall "
    "retrieval quality across the board -- every method's absolute Hits@K/MRR is "
    "lower on the 1432-pair benchmark than on the 296-pair one, including cosine "
    "itself (compare the two 'RESULTS ON...' tables above). This is expected: the "
    "extended dataset is dominated by weaker, family-inferred evidence that is "
    "intrinsically harder to retrieve than the original curated confirmed/strong "
    "296 pairs, and the retrieval task itself got harder (330 or 2864 queries vs. "
    "110/592, more distinct correct answers to rank among the same 1534-protein "
    "pool). Cosine similarity is no longer the single best method once real "
    "supervision (Random Forest) is given a large enough labeled set to learn "
    "from, but it remains a very strong, free (no training) baseline that a "
    "trained model does not dramatically outperform."
)

summary_text = "\n".join(lines)
with open(OUTPUT_DIR / "comparison_296_vs_1443_summary.txt", "w") as f:
    f.write(summary_text + "\n")

print(f"\n\nFull report saved to: {OUTPUT_DIR / 'comparison_296_vs_1443_summary.txt'}")
