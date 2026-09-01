"""
Upisuje NOVE, resolvable kandidate iz maug_candidate_pairs_1548.py u
cross_reactive_1548.csv -- SAMO one koji ne postoje vec (provereno preko
check_maug_candidates_1548.py) i cija su oba imena resolvable u pool-u.

family_1/family_2/source_food_1/source_food_2 preuzimaju se iz POSTOJECIH
redova gde se isti protein vec pojavljuje (doslednost sa vec uspostavljenom
konvencijom imenovanja u fajlu), ne izmisljaju se.

Nove kolone (ccd_flag, epitope_type, who2001_*, sequence_identity_pct_literature)
ostaju prazne za ove redove -- nisu jos pregledane istim procesom kao ostatak
dataseta (buduci korak, ne ovaj).

Izlaz: prepisuje output/cross_reactive_1548.csv, dodaje nove redove.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/data")
from maug_candidate_pairs_1548 import MAUG_CITATION, NEGATIVE_CANDIDATES, POSITIVE_CANDIDATES  # noqa: E402

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")

allergens = pd.read_csv(CLEAN_ALLERGENS)
pool_names = sorted(set(allergens["official_name"].astype(str)))


def resolve(name):
    matches = [n for n in pool_names if n == name or n.startswith(name + ".")]
    return sorted(matches)[0] if matches else None


gold = pd.read_csv(GOLD)
existing_pairs = set()
for _, row in gold.iterrows():
    a, b = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    existing_pairs.add(frozenset([a, b]))


allergen_family_lookup = dict(zip(allergens["official_name"], allergens["protein_family"]))
allergen_source_lookup = dict(zip(allergens["official_name"], allergens["source_food"]))

# Fallback za proteine bez family podatka NI u gold datasetu NI u clean_allergens.csv
# (518/1637 allergena u masterskoj bazi nema popunjeno protein_family, vidi
# output/validation_report.txt) -- porodica ovde dolazi DIREKTNO iz MAUG 2.0
# poglavlja u kom je protein opisan (C02=PR-10, C07=Lipocalin, C08=Cupin/vicilin,
# C09=GRP, C11=Parvalbumin, C06=EF-hand/polcalcin-vezano) -- stvaran izvor, ne
# nagadjanje po analogiji.
MAUG_FAMILY_OVERRIDE = {
    "Aln g 1.0101": "PR-10",  # C02, Bet v 1-homologue navedeno u istom pasusu
    "Fel d 7.0101": "Lipocalin",  # C07
    "Can f 2.0101": "Lipocalin",  # C07
    "Jug r 6.0101": "Cupin (vicilin/7S globulin)",  # C08, cross-reaguje sa drugim 7S globulinima
    "Pis v 3.0101": "Cupin (vicilin/7S globulin)",  # C08, eksplicitno "7S globulin... pistachio"
    "Cap a 7.0101": "Gibberellin-regulated protein (GRP)",  # C09
    "Gad m 1.0101": "Parvalbumin",  # C11
    "Cro p 1.0101": "Parvalbumin",  # C11, "crocodile parvalbumin Cro p 1"
    "Gal d 2.0101": "Ovalbumin (serpin family)",  # C04 kontekst, egg white storage protein
    "Bet v 3.0101": "Polcalcin-related (3-EF-hand, distinct subclass)",  # C06, eksplicitno RAZLICITA
    # podklasa od 2-EF-hand pravih polkalcina (Phl p 7/Bet v 4) -- izvor sam to naglasava
}


def lookup_family_source(resolved_name):
    """Nadji family_1/source_food_1 iz postojeceg reda gde se protein vec pominje
    (prioritet -- dosledno sa vec uspostavljenom konvencijom imenovanja u fajlu),
    inace fallback na clean_allergens.csv protein_family/source_food kolone,
    inace MAUG_FAMILY_OVERRIDE (vidi napomenu iznad)."""
    hit = gold[gold["allergen_id_1"] == resolved_name]
    if len(hit):
        return hit.iloc[0]["family_1"], hit.iloc[0]["source_food_1"]
    hit = gold[gold["allergen_id_2"] == resolved_name]
    if len(hit):
        return hit.iloc[0]["family_2"], hit.iloc[0]["source_food_2"]
    fam = allergen_family_lookup.get(resolved_name)
    src = allergen_source_lookup.get(resolved_name)
    if fam is not None and str(fam).strip() and str(fam).lower() != "nan":
        return fam, src
    if resolved_name in MAUG_FAMILY_OVERRIDE:
        return MAUG_FAMILY_OVERRIDE[resolved_name], src
    return None, None


# id_1 x id_2 (sorted za konzistentnost) hardkodirani skip (poznati sukob, resen u komentaru)
SKIP_NEGATIVE = set()

new_rows = []
next_cr_num = max(int(pid[2:]) for pid in gold["pair_id"] if str(pid).startswith("CR") and pid[2:].isdigit()) + 1

skipped_no_family = []

for c in POSITIVE_CANDIDATES:
    r1, r2 = resolve(c["id_1"]), resolve(c["id_2"])
    if r1 is None or r2 is None or frozenset([r1, r2]) in existing_pairs:
        continue
    fam1, src1 = lookup_family_source(r1)
    fam2, src2 = lookup_family_source(r2)
    if fam1 is None or fam2 is None:
        skipped_no_family.append((r1, r2))
        continue
    new_rows.append({
        "pair_id": f"CR{next_cr_num:03d}", "allergen_id_1": r1, "source_food_1": src1, "family_1": fam1,
        "allergen_id_2": r2, "source_food_2": src2, "family_2": fam2,
        "evidence_type": "IgE cross-reactivity / homology (MAUG 2.0)", "evidence_level": c["evidence"],
        "sequence_identity_pct": None, "reference": MAUG_CITATION,
        "isoform_note": None, "notes": c["note"],
    })
    existing_pairs.add(frozenset([r1, r2]))
    next_cr_num += 1

for c in NEGATIVE_CANDIDATES:
    r1, r2 = resolve(c["id_1"]), resolve(c["id_2"])
    if r1 is None or r2 is None or frozenset([r1, r2]) in existing_pairs:
        continue
    fam1, src1 = lookup_family_source(r1)
    fam2, src2 = lookup_family_source(r2)
    if fam1 is None or fam2 is None:
        skipped_no_family.append((r1, r2))
        continue
    new_rows.append({
        "pair_id": f"NEG{next_cr_num:03d}", "allergen_id_1": r1, "source_food_1": src1, "family_1": fam1,
        "allergen_id_2": r2, "source_food_2": src2, "family_2": fam2,
        "evidence_type": "Reported negative / explicit non-cross-reactivity (MAUG 2.0)",
        "evidence_level": "Reported negative", "sequence_identity_pct": None, "reference": MAUG_CITATION,
        "isoform_note": None, "notes": c["note"],
    })
    existing_pairs.add(frozenset([r1, r2]))
    next_cr_num += 1

print(f"Preskoceno (nema postojeceg reda za lookup family/source): {skipped_no_family}")
print(f"Dodajem {len(new_rows)} novih redova.")

new_df = pd.DataFrame(new_rows)
combined = pd.concat([gold, new_df], ignore_index=True)
combined.to_csv(GOLD, index=False)

print(f"Novi ukupan broj redova u {GOLD}: {len(combined)}")
print(new_df[["pair_id", "allergen_id_1", "allergen_id_2", "evidence_level"]].to_string(index=False))
