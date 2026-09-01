"""
Failure analysis: MLP(hadamard) SAM vs BLAST SAM na svih 57 CRD pacijenata
(test/test_cases.json). CISTO ANALITICKI fajl -- ne trenira se nista, ne
implementira se nikakav novi model ili heuristika, samo se obradjuju vec
postojeci rezultati iz test/evaluation_results_raw_mlponly.json i
test/evaluation_results_raw_blastonly.json (generisani od
test/evaluate_mlp_only_vs_blast_only_patients_1548.py).

Za svaki od 176 trial-ova izvlaci se: patient_id, hidden protein (kandidat),
positive/negative, organism (output/clean_allergens.csv), protein_family
(direktno iz test_cases.json component-level polja -- 280/280 popunjeno,
pouzdanije od gold-pair-izvedene family mape koja pokriva samo 445/1536
proteina), MLP rank/percentile, BLAST rank/percentile, i ko je bolji
(uzimajuci u obzir smer: kod pozitiva manji percentil je bolji, kod
negativa veci percentil je bolji).

Kategorije gresaka (nisu medjusobno iskljucive, trial moze upasti u vise):
  - same_organism_diff_family: hidden protein deli 'organism'
    (output/clean_allergens.csv) sa BAREM JEDNIM poznatim pozitivnim
    komponentom ISTOG pacijenta, a ima RAZLICITU protein_family -- ovo je
    tacno ista definicija kao ml/loco_targeted_hardneg_mlp_hadamard_1548.py
    kandidatski kriterijum, ovde primenjena na STVARNE pacijentske trial-ove
    (ne na training kandidate) da se proveri da li je Pru p1/Pru p3 izolovan
    slucaj ili deo sireg obrasca.
  - family_crowding: protein_family hidden proteina je jedna od vec
    dijagnostikovanih "crowded" porodica (nsLTP, Profilin, PR-10 --
    analysis/mrr_by_family_1548.py, 13x MRR spread nalaz).
  - profilin: family == Profilin (izdvojeno posebno na trazenje korisnika,
    i pored preklapanja sa family_crowding).
  - storage_protein: family u {2S albumin, Cupin (legumin/11S globulin),
    Cupin (vicilin/7S globulin), Oleosin} -- standardna WHO/IUIS storage
    protein grupa.
  - other: ne upada ni u jednu od gore navedenih.

Izlaz:
    output/mlp_blast_crd_failure_analysis_1548_per_trial.csv
    output/mlp_blast_crd_failure_analysis_1548_summary.txt
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MLP_RAW = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
BLAST_RAW = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
PER_TRIAL_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_blast_crd_failure_analysis_1548_per_trial.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_blast_crd_failure_analysis_1548_summary.txt")

# NAPOMENA: test_cases.json koristi SVOJU family-nomenklaturu (proveri
# df['protein_family'].value_counts() pre izmene) -- NIJE identicna
# gold-dataset (cross_reactive_1548.csv) family_1/family_2 stringovima.
STORAGE_FAMILIES = {"storage_protein_7S", "storage_protein_11S", "storage_protein_2S", "storage_protein_gliadin"}
CROWDED_FAMILIES = {"nsLTP", "profilin", "PR-10"}

# ---------------------------------------------------------------------------
# 1) Ucitaj i spoji MLP/BLAST rezultate (identicna logika kao
#    test/paired_test_mlp_vs_blast_1548.py).
# ---------------------------------------------------------------------------
mlp = pd.read_json(MLP_RAW)
blast = pd.read_json(BLAST_RAW)
mlp["rr"] = 1.0 / mlp["rank"]
blast["rr"] = 1.0 / blast["rank"]

merged = mlp[["patient_id", "hidden_protein", "true_result", "verification_status", "rank", "percentile"]].merge(
    blast[["patient_id", "hidden_protein", "rank", "percentile"]],
    on=["patient_id", "hidden_protein"], suffixes=("_mlp", "_blast"))
assert len(merged) == len(mlp) == len(blast), "Spajanje MLP/BLAST nije 1:1"
print(f"Ukupno trial-ova: {len(merged)}, pacijenata: {merged['patient_id'].nunique()}")

# ---------------------------------------------------------------------------
# 2) organism (clean_allergens.csv) + protein_family (test_cases.json,
#    component-level, 280/280 popunjeno -- pouzdanije od gold-pair mape).
# ---------------------------------------------------------------------------
allergens = pd.read_csv(ALLERGENS)
name_to_organism = dict(zip(allergens["official_name"], allergens["organism"]))

with open(TEST_CASES) as f:
    cases = json.load(f)

import sys
sys.path.insert(0, "/home/lana/ALERGRAF/test")
from protein_resolution import resolve_protein  # noqa: E402

pool_names = sorted(allergens["official_name"].dropna().unique().tolist())

# patient_id -> list of {pool_name, result, protein_family}
patient_components = {}
name_to_family_by_patient_component = {}
for case in cases:
    pid = case["patient_id"]
    comps = []
    for c in case["components"]:
        if c["result"] not in ("positive", "negative"):
            continue
        resolved = resolve_protein(c["protein"], pool_names)
        if resolved is None:
            continue
        comps.append({"pool_name": resolved, "result": c["result"], "protein_family": c["protein_family"]})
    patient_components[pid] = comps

# pool_name -> protein_family: take the family recorded wherever this exact
# protein appears as a component (should be consistent; if a protein appears
# under >1 family label across patients, flag it rather than silently pick one)
family_votes = {}
for pid, comps in patient_components.items():
    for c in comps:
        family_votes.setdefault(c["pool_name"], set()).add(c["protein_family"])
family_inconsistent = {k: v for k, v in family_votes.items() if len(v) > 1}
if family_inconsistent:
    print(f"WARNING: {len(family_inconsistent)} proteins have inconsistent protein_family "
          f"across patients in test_cases.json: {family_inconsistent}")
name_to_family = {k: sorted(v)[0] for k, v in family_votes.items()}


def organism_of(pool_name):
    return name_to_organism.get(pool_name)


def family_of(pool_name):
    return name_to_family.get(pool_name)


# ---------------------------------------------------------------------------
# 3) Per-trial enrichment: organism, protein_family, same_organism_diff_family
#    (checked against the SAME patient's OTHER known positive components --
#    this is what the ranker actually conditions on, not a single "query").
# ---------------------------------------------------------------------------
def compute_row_extras(row):
    pid = row["patient_id"]
    hidden = row["hidden_protein"]
    hidden_org = organism_of(hidden)
    hidden_fam = family_of(hidden)

    other_known_pos = [c for c in patient_components.get(pid, [])
                        if c["pool_name"] != hidden and c["result"] == "positive"]
    same_org_diff_fam = False
    matched_positive = None
    for c in other_known_pos:
        pos_org = organism_of(c["pool_name"])
        pos_fam = c["protein_family"]
        if pos_org is not None and pos_org == hidden_org and pos_fam != hidden_fam:
            same_org_diff_fam = True
            matched_positive = c["pool_name"]
            break

    return pd.Series({
        "organism": hidden_org,
        "protein_family": hidden_fam,
        "same_organism_diff_family": same_org_diff_fam,
        "matched_known_positive": matched_positive,
        "family_crowding": hidden_fam in CROWDED_FAMILIES,
        "profilin": hidden_fam == "profilin",
        "storage_protein": hidden_fam in STORAGE_FAMILIES,
    })


extras = merged.apply(compute_row_extras, axis=1)
merged = pd.concat([merged, extras], axis=1)
merged["error_other"] = ~(merged["same_organism_diff_family"] | merged["family_crowding"] | merged["storage_protein"])


def better_model(row):
    if row["true_result"] == "positive":
        if row["percentile_mlp"] < row["percentile_blast"]:
            return "MLP"
        if row["percentile_mlp"] > row["percentile_blast"]:
            return "BLAST"
        return "tie"
    else:
        if row["percentile_mlp"] > row["percentile_blast"]:
            return "MLP"
        if row["percentile_mlp"] < row["percentile_blast"]:
            return "BLAST"
        return "tie"


merged["winner"] = merged.apply(better_model, axis=1)
merged.to_csv(PER_TRIAL_OUTPUT, index=False)
print(f"Saved per-trial table: {PER_TRIAL_OUTPUT} ({len(merged)} redova)")

# ---------------------------------------------------------------------------
# 4) Agregacije
# ---------------------------------------------------------------------------
lines = ["=" * 90, "MLP(hadamard) vs BLAST -- failure analysis na svih 57 CRD pacijenata "
         f"({len(merged)} trial-ova, {merged['patient_id'].nunique()} pacijenata)", "=" * 90, ""]


def summarize_group(df, label, min_n=1):
    lines_local = [f"--- {label} (n={len(df)}) ---"]
    if len(df) == 0:
        lines_local.append("  (nema trial-ova)")
        return lines_local
    win_counts = df["winner"].value_counts()
    lines_local.append(f"  Pobednik: MLP={win_counts.get('MLP', 0)}  BLAST={win_counts.get('BLAST', 0)}  "
                        f"tie={win_counts.get('tie', 0)}")
    for res in ("positive", "negative"):
        sub = df[df["true_result"] == res]
        if len(sub) == 0:
            continue
        lines_local.append(f"  {res}: n={len(sub)}  MLP medijan percentil={sub['percentile_mlp'].median():.2f}%  "
                            f"BLAST medijan percentil={sub['percentile_blast'].median():.2f}%  "
                            f"(mean diff MLP-BLAST={ (sub['percentile_mlp']-sub['percentile_blast']).mean():+.2f} "
                            f"pp, {'MLP gori (visi=lose za pozitiv, nizi=lose za negativ)' if res=='positive' else ''})")
    return lines_local


# 1) Kategorije gresaka
lines.append("1) GRUPISANJE PO TIPU GRESKE (kategorije se preklapaju, trial moze biti u vise)")
lines.append("")
for cat_col, cat_label in [("same_organism_diff_family", "Isti organizam / razlicita familija"),
                             ("family_crowding", "Family crowding (nsLTP/Profilin/PR-10)"),
                             ("profilin", "Profilin (podskup crowding-a, izdvojeno posebno)"),
                             ("storage_protein", "Storage proteini (2S albumin/Cupin/Oleosin)"),
                             ("error_other", "Ostalo (ne upada ni u jednu gornju kategoriju)")]:
    sub = merged[merged[cat_col]]
    lines.extend(summarize_group(sub, cat_label))
    lines.append("")

# 2) Po protein_family
lines.append("2) MLP vs BLAST PO PROTEIN FAMILY (n>=4 trial-ova)")
lines.append("")
fam_counts = merged["protein_family"].value_counts()
for fam in fam_counts[fam_counts >= 4].index:
    sub = merged[merged["protein_family"] == fam]
    lines.extend(summarize_group(sub, f"Family = {fam}"))
    lines.append("")

small_fams = fam_counts[fam_counts < 4]
if len(small_fams) > 0:
    lines.append(f"  (Porodice sa <4 trial-a, izostavljene iz detaljne tabele radi pouzdanosti: "
                 f"{dict(small_fams)})")
    lines.append("")

# 3) Sistematski dobici/gubici (na nivou familije, sortiran po mean percentile diff za pozitive)
lines.append("3) SISTEMATSKI DOBICI/GUBICI PO FAMILY (mean percentile diff MLP-BLAST, pozitivi -- "
              "negativan broj = MLP BOLJI na pozitivima)")
lines.append("")
fam_stats = []
for fam in fam_counts[fam_counts >= 4].index:
    sub_pos = merged[(merged["protein_family"] == fam) & (merged["true_result"] == "positive")]
    sub_neg = merged[(merged["protein_family"] == fam) & (merged["true_result"] == "negative")]
    pos_diff = (sub_pos["percentile_mlp"] - sub_pos["percentile_blast"]).mean() if len(sub_pos) else np.nan
    neg_diff = (sub_neg["percentile_mlp"] - sub_neg["percentile_blast"]).mean() if len(sub_neg) else np.nan
    fam_stats.append((fam, len(sub_pos), pos_diff, len(sub_neg), neg_diff))
fam_stats.sort(key=lambda x: (x[2] if not np.isnan(x[2]) else 0))
for fam, n_pos, pos_diff, n_neg, neg_diff in fam_stats:
    pos_str = f"{pos_diff:+.2f}pp (n={n_pos})" if not np.isnan(pos_diff) else "n/a"
    neg_str = f"{neg_diff:+.2f}pp (n={n_neg}, pozitivan broj=MLP BOLJI na negativima)" if not np.isnan(neg_diff) else "n/a"
    lines.append(f"  {fam:55s} pozitivi: {pos_str:25s} negativi: {neg_str}")
lines.append("")

# 4) Pru p1/Pru p3 -- da li je deo sireg obrasca
lines.append("4) Da li je Pru p1/Pru p3 IZOLOVAN slucaj ili deo sireg 'same_organism_diff_family' obrasca?")
lines.append("")
sod_neg = merged[(merged["same_organism_diff_family"]) & (merged["true_result"] == "negative")]
sod_pos = merged[(merged["same_organism_diff_family"]) & (merged["true_result"] == "positive")]
lines.append(f"  Svi same_organism_diff_family trial-ovi: {len(merged[merged['same_organism_diff_family']])} "
             f"({sod_neg['patient_id'].nunique() + sod_pos['patient_id'].nunique() if len(sod_pos)+len(sod_neg) else 0} "
             f"pacijenata dodiruje ovu kategoriju)")
lines.append(f"  -- Negativni trial-ovi (ovo je Pru p1-tip slucaj): n={len(sod_neg)}")
if len(sod_neg) > 0:
    lines.append(f"     MLP medijan percentil={sod_neg['percentile_mlp'].median():.2f}%  "
                  f"BLAST medijan percentil={sod_neg['percentile_blast'].median():.2f}%")
    lines.append(f"     MLP pobedjuje (bolje potiskuje): {(sod_neg['winner']=='MLP').sum()}/{len(sod_neg)}")
    lines.append(f"     Uporedjeno sa SVIM negativnim trial-ovima (n={len(merged[merged['true_result']=='negative'])}): "
                  f"MLP medijan={merged[merged['true_result']=='negative']['percentile_mlp'].median():.2f}%  "
                  f"BLAST medijan={merged[merged['true_result']=='negative']['percentile_blast'].median():.2f}%")
    lines.append("     Distinktni proteini/parovi u ovoj kategoriji:")
    distinct_pairs = sod_neg[["hidden_protein", "matched_known_positive"]].drop_duplicates()
    for _, r in distinct_pairs.iterrows():
        pair_sub = sod_neg[(sod_neg["hidden_protein"] == r["hidden_protein"]) &
                             (sod_neg["matched_known_positive"] == r["matched_known_positive"])]
        lines.append(f"       {r['matched_known_positive']} (poznat pozitiv) -> {r['hidden_protein']} (skriven negativ): "
                      f"n={len(pair_sub)}, MLP medijan%={pair_sub['percentile_mlp'].median():.1f}, "
                      f"BLAST medijan%={pair_sub['percentile_blast'].median():.1f}")
lines.append("")

summary_text = "\n".join(lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
