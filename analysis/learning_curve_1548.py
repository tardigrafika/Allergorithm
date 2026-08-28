"""
Learning curve: RF+BLAST MRR u zavisnosti od kolicine trening parova
(25% / 50% / 75% / 100% od train split-a) - 1548 dataset.

Direktan empirijski test pitanja "da li je plafon problem malo podataka":
ako MRR i dalje raste sa N pri 100%, verovatno JOS ima prostora da vise
podataka pomogne. Ako je vec ravno (plato) mnogo pre 100%, dodavanje jos
istovrsnih trening parova verovatno nece pomoci - to bi bio argument u
prilog "reprezentacija je plafon" hipotezi, ne "malo podataka".

Cosine je training-free (ne koristi gold parove za treniranje), pa je
prikazan kao fiksna referentna linija, ne kao deo krive.

Podskupovi su UGNJEZDENI (25% subset je podskup 50% subset-a itd, fiksnim
seed-om) da bi kriva bila monotono uporediva, ne 4 nezavisna nasumicna uzorka.

NAPOMENA o obimu: jedan run po frakciji (ne ponavljanja/multiple seeds) -
isti "umeren obim, ne iscrpna pretraga" princip kao svuda u ovoj rundi due
diligence-a. Kod malih frakcija ocekivati vecu varijansu nego sto je ovaj
jedan run pokazuje.

Izlaz:
    output/learning_curve_1548_summary.txt
    output/learning_curve_1548_results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import bootstrap_ci, retrieval_evaluate, summarize_retrieval  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import group_aware_split, split_pairs  # noqa: E402
from ml.pipeline.models.classifiers.random_forest import RandomForestPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/learning_curve_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/learning_curve_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10
FRACTIONS = [0.25, 0.5, 0.75, 1.0]
RF_PARAMS = dict(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", n_jobs=-1)

print("Loading dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs_full, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
print(f"Train parova (100%): {len(train_pairs_full)}  Test parova: {len(test_pairs)}")

cosine_matrix = cosine_similarity(dataset.embedding_matrix)

# Fiksan ugnjezden redosled train parova (isti shuffle za sve frakcije -> nested subsets)
rng = np.random.default_rng(SEED)
shuffled_order = rng.permutation(len(train_pairs_full))

print(f"\nProlazim kroz {len(FRACTIONS)} frakcije treninga...\n")
results = []
cosine_mrr_ref = None
for frac in FRACTIONS:
    n_take = max(2, int(round(frac * len(train_pairs_full))))
    subset_idx = shuffled_order[:n_take]
    train_pairs = [train_pairs_full[i] for i in subset_idx]

    n_train_neg = len(train_pairs) * NEG_PER_POS
    train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)

    clf = RandomForestPairClassifier(params=RF_PARAMS, extra_features=["blast_identity", "blast_score"],
                                       blast_matrix_path=BLAST_MATRIX, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")
    cosine_mrr_ref = summary["cosine_mrr"]

    results.append({
        "fraction": frac, "n_train_pairs": len(train_pairs), "n_train_negatives": len(train_negatives),
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  frac={frac:.2f}  n_train={len(train_pairs):4d}  MRR={summary['mrr']:.4f}  "
          f"delta={delta_stats['mean']:+.4f}  CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}",
          flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "Learning curve: RF+BLAST MRR vs kolicina trening parova (1548)", "=" * 80, "",
                  f"Cosine baseline MRR (fiksno, ne zavisi od N): {cosine_mrr_ref:.4f}", ""]
summary_lines.append(f"{'Frakcija':<10}{'N train':<10}{'MRR':<10}{'Delta':<12}{'95% CI':<24}{'Znacajno?'}")
for _, r in results_df.iterrows():
    summary_lines.append(
        f"{r['fraction']:<10.2f}{r['n_train_pairs']:<10.0f}{r['mrr']:<10.4f}{r['delta']:<+12.4f}"
        f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{'  DA' if r['significant'] else '  ne'}"
    )

mrr_25 = results_df.loc[results_df["fraction"] == 0.25, "mrr"].iloc[0]
mrr_100 = results_df.loc[results_df["fraction"] == 1.0, "mrr"].iloc[0]
slope_note = (
    f"\nMRR na 25% train podataka: {mrr_25:.4f}  ->  MRR na 100%: {mrr_100:.4f}  "
    f"(razlika {mrr_100 - mrr_25:+.4f} preko 4x vise podataka)"
)
summary_lines.append(slope_note)
if mrr_100 - mrr_25 < 0.01:
    summary_lines.append("Kriva je VEC RAVNA (plato) mnogo pre 100% - dodatni istovrsni trening parovi "
                          "verovatno NE bi pomogli. Argument u prilog 'reprezentacija je plafon', ne 'malo podataka'.")
else:
    summary_lines.append("Kriva JOS RASTE ka 100% - moguce da bi vise istovrsnih trening parova jos pomoglo.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
