"""
D2: Izoform ekspanzija -- za svaki Confirmed/Strong par A-B, nadji izoforme
A1 (ili B1) sa >=80% BLAST identicnoscu naspram vec koriscenog clana, JOS
neuparene sa drugim clanom para. OGRANICENO na max 3 nova para PO
POSTOJECEM baznom paru (najvisa identicnost prvo) -- izbegava koncentraciju
na proteine sa mnogo registrovanih izoformi (npr. Bet v 1 ima 16+, bez
ogranicenja bi sam cinio ~40% svih novih parova).

Evidence tier NASLEDJUJE se od baznog para, ali JEDAN nivo nize (izvedeno,
ne direktno potvrdjeno):
  Confirmed -> "Strong evidence (isoform-inferred)"
  Strong evidence* -> "Suspected (isoform-inferred)"

Epitope-preklapanje (mentorkin uslov "ako se razlike ne nalaze u epitopu")
NIJE moguce sistematski proveriti (IEDB pokriva samo 87/1534 proteina) --
ovo je EKSPLICITNO navedeno u notes koloni svakog novog reda, ne prikriveno.

Izlaz: prepisuje output/cross_reactive_1548.csv.
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")

ISOFORM_IDENTITY_THRESHOLD = 80.0
MAX_NEW_PER_BASE_PAIR = 3
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")

allergens = pd.read_csv(CLEAN_ALLERGENS)
names = sorted(set(allergens["official_name"].astype(str)))
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id


def base_name(n):
    return n.rsplit(".", 1)[0] if "." in n else n


groups = defaultdict(list)
for n in names:
    groups[base_name(n)].append(n)
isoform_groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

with open(BLAST_MATRIX, "rb") as f:
    blast = pickle.load(f)
blast_id_to_index = {aid: i for i, aid in enumerate(blast["ids"])}
identity_matrix = blast["identity_matrix"]


def high_identity_isoforms(name):
    base = base_name(name)
    if base not in isoform_groups:
        return []
    pid = name_to_id.get(name)
    if pid not in blast_id_to_index:
        return []
    pidx = blast_id_to_index[pid]
    results = []
    for iso in isoform_groups[base]:
        if iso == name:
            continue
        iid = name_to_id.get(iso)
        if iid not in blast_id_to_index:
            continue
        iidx = blast_id_to_index[iid]
        pct = float(identity_matrix[pidx, iidx])
        if pct >= ISOFORM_IDENTITY_THRESHOLD:
            results.append((iso, pct))
    return results


gold = pd.read_csv(GOLD)
strong = gold[gold["evidence_level"].isin(
    ["Confirmed", "Strong evidence", "Strong evidence (within-species paralogs)", "Strong evidence (congeneric species)"]
)]

existing_pairs = set()
for _, row in gold.iterrows():
    a, b = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    existing_pairs.add(frozenset([a, b]))


def downgrade_tier(base_level):
    if base_level == "Confirmed":
        return "Strong evidence (isoform-inferred)"
    return "Suspected (isoform-inferred)"


allergen_family_lookup = dict(zip(allergens["official_name"], allergens["protein_family"]))
allergen_source_lookup = dict(zip(allergens["official_name"], allergens["source_food"]))


def lookup_family_source_exact(resolved_name):
    hit = gold[gold["allergen_id_1"] == resolved_name]
    if len(hit):
        return hit.iloc[0]["family_1"], hit.iloc[0]["source_food_1"]
    hit = gold[gold["allergen_id_2"] == resolved_name]
    if len(hit):
        return hit.iloc[0]["family_2"], hit.iloc[0]["source_food_2"]
    fam, src = allergen_family_lookup.get(resolved_name), allergen_source_lookup.get(resolved_name)
    return (fam, src) if fam is not None and str(fam).lower() != "nan" else (None, src)


def lookup_family_source(resolved_name):
    """Prvo tacno ime, inace BILO KOJA druga izoforma istog baznog proteina --
    izoforme (npr. Bet v 1.0102) po definiciji nikad ranije nisu koriscene u
    gold datasetu (uvek se koristila primarna .0101), pa tacan lookup uvek
    promasi za novododate izoforme -- ali dele istu porodicu kao ostale
    izoforme istog proteina."""
    fam, src = lookup_family_source_exact(resolved_name)
    if fam is not None:
        return fam, src
    base = base_name(resolved_name)
    for sibling in isoform_groups.get(base, []):
        if sibling == resolved_name:
            continue
        fam2, src2 = lookup_family_source_exact(sibling)
        if fam2 is not None:
            return fam2, (src if src is not None and str(src).lower() != "nan" else src2)
    return None, src


added_this_run = set()
new_rows = []
next_num = max(int(pid[2:]) for pid in gold["pair_id"] if str(pid).startswith("CR") and pid[2:].isdigit()) + 1

for _, row in strong.iterrows():
    a, b = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    base_level = row["evidence_level"]
    candidates_for_this_base_pair = []
    for iso, pct in high_identity_isoforms(a):
        key = frozenset([iso, b])
        if key not in existing_pairs and key not in added_this_run:
            candidates_for_this_base_pair.append((iso, b, a, pct))
    for iso, pct in high_identity_isoforms(b):
        key = frozenset([a, iso])
        if key not in existing_pairs and key not in added_this_run:
            candidates_for_this_base_pair.append((a, iso, b, pct))

    candidates_for_this_base_pair.sort(key=lambda x: -x[3])
    for c1, c2, orig, pct in candidates_for_this_base_pair[:MAX_NEW_PER_BASE_PAIR]:
        fam1, src1 = lookup_family_source(c1)
        fam2, src2 = lookup_family_source(c2)
        if fam1 is None or fam2 is None:
            continue
        new_rows.append({
            "pair_id": f"CR{next_num:03d}", "allergen_id_1": c1, "source_food_1": src1, "family_1": fam1,
            "allergen_id_2": c2, "source_food_2": src2, "family_2": fam2,
            "evidence_type": "Isoform inference from confirmed base pair",
            "evidence_level": downgrade_tier(base_level),
            "sequence_identity_pct": None, "reference": f"Izvedeno iz para {a} x {b} ({base_level})",
            "isoform_note": (f"{pct:.1f}% BLAST identicnost naspram {orig} (koji je deo potvrdjenog para). "
                              "Epitope-preklapanje NIJE sistematski provereno (IEDB pokriva samo mali deo pool-a) -- "
                              "pretpostavka iz WHO(2001)-stil homologije, ne direktan dokaz za ovaj konkretan par."),
            "notes": None,
        })
        added_this_run.add(frozenset([c1, c2]))
        next_num += 1

print(f"Dodajem {len(new_rows)} novih redova (max {MAX_NEW_PER_BASE_PAIR} po baznom paru, {len(strong)} baznih parova pregledano).")
new_df = pd.DataFrame(new_rows)
combined = pd.concat([gold, new_df], ignore_index=True)
combined.to_csv(GOLD, index=False)
print(f"Novi ukupan broj redova: {len(combined)}")
print()
print("Raspodela po nasledjenom evidence_level:")
print(new_df["evidence_level"].value_counts())
