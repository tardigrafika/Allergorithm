"""
Per-familija MRR breakdown -- mentorkin predlog: da li je plafon GLOBALAN
(whole-protein similarity nosi skoro sav dostupan signal) ili
FAMILY-SPECIFIC (neke familije katastrofalne, druge dobre)?

Cosine (training-free) preko CELOG prosirenog dataseta (1910/1884 posle
filtera), grupisano po family_1 (upit-strana familija).

Izlaz:
    output/mrr_by_family_1548.csv
    output/mrr_by_family_1548_summary.txt
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import retrieval_evaluate  # noqa: E402
from ml.pipeline.models.classifiers.cosine import CosineSimilarity  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mrr_by_family_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mrr_by_family_1548_summary.txt")

MIN_QUERIES_FOR_REPORTING = 8  # ispod ovoga MRR je previse sumovit da bi bio informativan

print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
print(f"Gold parova (posle filtera): {len(dataset.gold_pairs)}")

cosine_matrix = cosine_similarity(dataset.embedding_matrix)
clf = CosineSimilarity()
clf.fit(dataset.gold_pairs, [], dataset.embedding_matrix, dataset.id_to_index)

retrieval_df = retrieval_evaluate(dataset.gold_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                    cosine_matrix=cosine_matrix)

print(f"Ukupno upita: {len(retrieval_df)}")
print("Kolone:", retrieval_df.columns.tolist())

by_family = retrieval_df.groupby("query_family").agg(
    n_queries=("model_reciprocal_rank", "size"),
    mrr=("model_reciprocal_rank", "mean"),
    hits_at_10=("model_hits_at_10", "mean"),
).sort_values("mrr")

by_family.to_csv(CSV_OUTPUT)

reportable = by_family[by_family["n_queries"] >= MIN_QUERIES_FOR_REPORTING]
overall_mrr = retrieval_df["model_reciprocal_rank"].mean()

summary_lines = ["=" * 80, "MRR po proteinskoj familiji (cosine, ceo dataset)", "=" * 80, "",
                  f"Ukupan (globalni) MRR preko svih upita: {overall_mrr:.4f}",
                  f"Familija sa >={MIN_QUERIES_FOR_REPORTING} upita (pouzdanije): {len(reportable)}/{len(by_family)}", ""]
summary_lines.append(f"{'Family':<55}{'n_queries':<12}{'MRR':<10}{'Hits@10'}")
for fam, row in reportable.iterrows():
    summary_lines.append(f"{str(fam):<55}{int(row['n_queries']):<12}{row['mrr']:<10.4f}{row['hits_at_10']:.4f}")

summary_lines.append("")
worst5 = reportable.head(5)
best5 = reportable.tail(5)
summary_lines.append(f"NAJGORE 5 familija (n>={MIN_QUERIES_FOR_REPORTING}):")
for fam, row in worst5.iterrows():
    summary_lines.append(f"  {fam}: MRR={row['mrr']:.4f} (n={int(row['n_queries'])})")
summary_lines.append(f"\nNAJBOLJE 5 familija (n>={MIN_QUERIES_FOR_REPORTING}):")
for fam, row in best5.iterrows():
    summary_lines.append(f"  {fam}: MRR={row['mrr']:.4f} (n={int(row['n_queries'])})")

spread = reportable["mrr"].max() - reportable["mrr"].min()
summary_lines.append(f"\nRaspon MRR preko familija (n>={MIN_QUERIES_FOR_REPORTING}): {spread:.4f} "
                      f"(od {reportable['mrr'].min():.4f} do {reportable['mrr'].max():.4f})")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
