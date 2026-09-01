"""
Error taxonomy: sistematska kategorizacija SVIH lose rangiranih RRF-4 upita
po VEROVATNOM uzroku, umesto nagadjanja familija-po-familija (kako je radjeno
za PR-10 pojedinacno). Mentorov predlog, sad primenjen na ceo dataset.

Koristi POSTOJECI graph_propagation_signal_1548_per_query.csv (RRF-4 rang
po upitu preko celog gold dataseta) -- NAPOMENA: ovaj fajl je malo zastareo
(1537 parova, dataset je sad 1922 posle D1/D2/E3 dodataka) ali strukturalno
reprezentativan za dijagnostiku (nije headline broj, vec pitanje GDE
greske klasteruju).

Kategorije (nisu medjusobno iskljucive, upit moze imati vise oznaka):
  - cold_start: n_other_neighbors == 0 (nema drugih poznatih partnera u
    grafu, "graph-propagation" signal strukturno nedostupan)
  - sparse: 1 <= n_other_neighbors <= 2 (malo, ali ne nula)
  - ccd_driven: ccd_flag == ccd_glycan_confirmed ili ccd_possible_unverified
  - inferred_tier: evidence_level pocinje sa "Inferred"
  - who2001_borderline_expected: who2001_pass==False ALI familija je
    poznata kao "nizak identitet, pravi fold-conserved cross-reactivity"
    (nsLTP/2S albumin/Lipocalin -- vec potvrdjeno literaturom u mi_lse
    fazi, NIJE sum)
  - who2001_fail_suspect: who2001_pass==False i familija NIJE u gornjoj
    listi (stvarni kandidat za sumnjiv/slab par)
  - low_identity: sequence_identity_pct < 30 (bez obzira na WHO2001 flag)
  - directionality_known: par je medju 8 poznatih direkciono-oznacenih
  - unexplained: nijedna gornja kategorija se ne primenjuje -- OVO je
    najvazniji bucket, pravi kandidat za nepoznat reprezentacioni problem

Izlaz:
    output/error_taxonomy_1548_summary.txt
    output/error_taxonomy_1548_per_query.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

PER_QUERY = Path("/home/lana/ALERGRAF/output/graph_propagation_signal_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/error_taxonomy_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/error_taxonomy_1548_per_query.csv")

FAIL_RANK_THRESHOLD = 100  # rang > 100 (od ~1534 kandidata) = "los" rezultat, isti red velicine kao worst30

LOW_IDENTITY_KNOWN_OK_FAMILIES = {"nsLTP", "2S albumin", "Lipocalin"}  # potvrdjeno u mi_lse fazi: nizak
                                                                          # identitet OCEKIVAN, nije sum

df = pd.read_csv(PER_QUERY)
gold = pd.read_csv(GOLD)[["pair_id", "family_1", "ccd_flag", "who2001_pass", "sequence_identity_pct",
                            "directionality_note"]].drop_duplicates(subset="pair_id")
merged = df.merge(gold, on="pair_id", how="left")

failing = merged[merged["rrf4_rank"] > FAIL_RANK_THRESHOLD].copy()
print(f"Ukupno upita: {len(merged)}, losih (rang > {FAIL_RANK_THRESHOLD}): {len(failing)} "
      f"({len(failing)/len(merged)*100:.1f}%)")

failing["cold_start"] = failing["n_other_neighbors"] == 0
failing["sparse"] = failing["n_other_neighbors"].between(1, 2)
failing["ccd_driven"] = failing["ccd_flag"].isin(["ccd_glycan_confirmed", "ccd_possible_unverified"])
failing["inferred_tier"] = failing["evidence_level"].astype(str).str.startswith("Inferred")
failing["who2001_borderline_expected"] = (~failing["who2001_pass"].fillna(True)) & \
                                          (failing["family_1"].isin(LOW_IDENTITY_KNOWN_OK_FAMILIES))
failing["who2001_fail_suspect"] = (~failing["who2001_pass"].fillna(True)) & \
                                   (~failing["family_1"].isin(LOW_IDENTITY_KNOWN_OK_FAMILIES))
failing["low_identity"] = failing["sequence_identity_pct"] < 30
failing["directionality_known"] = failing["directionality_note"].notna()

tag_cols = ["cold_start", "sparse", "ccd_driven", "inferred_tier", "who2001_borderline_expected",
            "who2001_fail_suspect", "low_identity", "directionality_known"]
failing["any_tag"] = failing[tag_cols].any(axis=1)
failing["unexplained"] = ~failing["any_tag"]

failing.to_csv(PER_QUERY_OUTPUT, index=False)

summary_lines = ["=" * 80, f"Error taxonomy -- RRF-4 upiti sa rangom > {FAIL_RANK_THRESHOLD} "
                  f"(od ~1534 kandidata)", "=" * 80, "",
                  f"Ukupno upita u datasetu: {len(merged)}",
                  f"Losih upita (rang > {FAIL_RANK_THRESHOLD}): {len(failing)} ({len(failing)/len(merged)*100:.1f}%)",
                  ""]

summary_lines.append("--- Udeo losih upita po kategoriji (nisu iskljucive, mogu se preklapati) ---")
for tag in tag_cols + ["unexplained"]:
    n = failing[tag].sum()
    summary_lines.append(f"  {tag:30s}: {n:4d} ({n/len(failing)*100:5.1f}%)")

summary_lines.append("")
summary_lines.append(f"  BAR JEDNA objasnjavajuca kategorija: {failing['any_tag'].sum()} "
                      f"({failing['any_tag'].sum()/len(failing)*100:.1f}%)")
summary_lines.append(f"  NEOBJASNJENO (nijedna kategorija): {failing['unexplained'].sum()} "
                      f"({failing['unexplained'].sum()/len(failing)*100:.1f}%)")

summary_lines.append("")
summary_lines.append("--- 'unexplained' upiti po familiji (top 15) ---")
unexplained = failing[failing["unexplained"]]
summary_lines.append(unexplained["family_1"].value_counts().head(15).to_string())

summary_lines.append("")
summary_lines.append("--- 'unexplained' upiti: raspodela n_other_neighbors (da li su ipak blizu cold-start) ---")
summary_lines.append(unexplained["n_other_neighbors"].describe().to_string())

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
print(f"Saved: {PER_QUERY_OUTPUT}")
