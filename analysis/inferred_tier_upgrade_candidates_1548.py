"""
Identifikuje Inferred-tier parove koji se mogu nadograditi na pravu,
kvantifikovanu literaturu umesto generickog blanket citata ("Radauer &
Breiteneder 2018 (panallergen family review); Aalberse RC 2000...", koji
pokriva 100% od 1112 Inferred parova / 72.3% celog gold dataseta -- vidi
konverzaciju/memoriju za taj nalaz).

Prvi pronadjen "kisobran" izvor: Scala E, Alessandri C, Palazzo P, et al.
"IgE Recognition Patterns of Profilin, PR-10, and Tropomyosin Panallergens
Tested in 3,113 Allergic Patients by Allergen Microarray-Based Technology."
PLoS ONE. 2011;6(9):e24912. https://doi.org/10.1371/journal.pone.0024912
(PMC3174236, otvoren pristup, verifikovano preko WebFetch).

Pokriva 22 imenovane komponente sa STVARNIM Pearson korelacijama (Bivariate
Correlation analiza, n=3113 pacijenata):
  Profilini (9): Bet v 2, Cyn d 12, Hel a 2, Hev b 8, Mer a 1, Ole e 2,
                 Par j 3, Phl p 12, Pho d 2 -- sve parne korelacije >0.70
  PR-10 (7):     Aln g 1, Api g 1, Bet v 1.0101, Bet v 1.0401, Cor a 1,
                 Dau c 1, Mal d 1.0108 -- NIJANSIRANIJE (Bet v1/Aln g1/Cor a1
                 jak klaster; Api g1/Dau c1 odvojen klaster -- ne svi parovi
                 podjednako korelisani, vrednija informacija od blanket
                 pretpostavke)
  Tropomiozini (6): Ani s 3, Der p 10, Hel as 1, Pen i 1, Pen m 1, Per a 7
                 -- sve parne korelacije >0.70

VAZNO: ovo samo IDENTIFIKUJE kandidate za rucnu nadogradnju (koja tacna
evidence_level/reference vrednost da se upise treba ODLUKA, ne automatska
izmena cross_reactive_1548.csv u ovoj skripti -- ne menjamo gold dataset
bez eksplicitne odluke).

Izlaz:
    output/inferred_tier_upgrade_candidates_1548.csv
    output/inferred_tier_upgrade_candidates_1548_summary.txt
"""

from pathlib import Path

import pandas as pd

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/inferred_tier_upgrade_candidates_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/inferred_tier_upgrade_candidates_1548_summary.txt")

SCALA_2011_CITATION = (
    "Scala E, Alessandri C, Palazzo P, et al. IgE Recognition Patterns of Profilin, "
    "PR-10, and Tropomyosin Panallergens Tested in 3,113 Allergic Patients by Allergen "
    "Microarray-Based Technology. PLoS ONE. 2011;6(9):e24912. "
    "https://doi.org/10.1371/journal.pone.0024912 -- Bivariate Pearson correlation "
    "analysis; profilins and tropomyosins all pairwise r>0.70; PR-10 shows distinct "
    "clusters (Bet v 1/Aln g 1/Cor a 1 strong; Api g 1/Dau c 1 separate)."
)

SCALA_2011_COMPONENTS = [
    "Bet v 2", "Cyn d 12", "Hel a 2", "Hev b 8", "Mer a 1", "Ole e 2", "Par j 3", "Phl p 12", "Pho d 2",
    "Aln g 1", "Api g 1", "Bet v 1", "Cor a 1", "Dau c 1", "Mal d 1",
    "Ani s 3", "Der p 10", "Hel as 1", "Pen i 1", "Pen m 1", "Per a 7",
]


def matches(name, components):
    return any(str(name).startswith(c) for c in components)


gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains("negative|Contested|Risky|NO cross", case=False, na=False)
gold = gold_raw.loc[~negative_mask].copy()
inferred = gold[gold["evidence_level"].str.startswith("Inferred", na=False)].copy()

mask = inferred["allergen_id_1"].apply(lambda n: matches(n, SCALA_2011_COMPONENTS)) & \
       inferred["allergen_id_2"].apply(lambda n: matches(n, SCALA_2011_COMPONENTS))
upgradeable = inferred.loc[mask, ["pair_id", "allergen_id_1", "allergen_id_2", "family_1",
                                   "evidence_level", "reference"]].copy()
upgradeable["current_reference"] = upgradeable.pop("reference")
upgradeable["suggested_new_reference"] = SCALA_2011_CITATION
upgradeable["suggested_new_tier"] = "Strong evidence (population correlation, n=3113, not individually-inspected pair)"
upgradeable["status"] = "CANDIDATE -- not yet applied to gold dataset, needs review"

upgradeable.to_csv(CSV_OUTPUT, index=False)

summary_lines = [
    "=" * 70,
    "Inferred-tier upgrade kandidati (1. runda -- Scala et al. 2011)",
    "=" * 70,
    "",
    f"Ukupno Inferred parova u datasetu: {len(inferred)} (72.3% celog gold dataseta)",
    f"Svih 1112 deli identican blanket citat (Radauer & Breiteneder 2018 + Aalberse 2000)",
    "",
    f"Nadogradivo ovim JEDNIM izvorom (Scala et al. 2011, PLoS ONE, n=3113 pacijenata): "
    f"{len(upgradeable)} parova ({len(upgradeable)/len(inferred):.1%} od Inferred tier-a)",
    "",
    str(upgradeable["family_1"].value_counts()),
    "",
    "Napomena: PR-10 par (Api g 1 / Mal d 1) i Oleosin par (Cor a 12/13) zahtevaju "
    "dodatnu proveru pre primene -- Oleosin nije direktno pokriven Scala 2011 (verovatno "
    "false match na prefix, proveriti rucno pre upisa).",
    "",
    "SLEDECI KORAK (nije jos uradjeno): rucna odluka o promeni evidence_level/reference "
    "u cross_reactive_1548.csv za ove parove, plus trazenje dodatnih 'kisobran' izvora "
    "za nsLTP, tropomiozin (dalji clanovi), parvalbumin, 2S albumin familije.",
]
summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
