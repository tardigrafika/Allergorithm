"""
Proverava maug_candidate_pairs_1548.py kandidate protiv postojeceg gold
dataseta i protiv allergen pool-a -- NE upisuje nista, samo izvestava
sta je novo/resolvable/vec-postoji/nerazresivo, radi ljudske provere
pre bilo kakvog upisa u cross_reactive_1548.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/data")
from maug_candidate_pairs_1548 import NEGATIVE_CANDIDATES, POSITIVE_CANDIDATES  # noqa: E402

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


def check_list(candidates, label):
    print(f"\n{'='*70}\n{label} ({len(candidates)} kandidata)\n{'='*70}")
    new_resolvable, already_exists, unresolvable = [], [], []
    for c in candidates:
        r1, r2 = resolve(c["id_1"]), resolve(c["id_2"])
        if r1 is None or r2 is None:
            unresolvable.append((c, r1, r2))
            continue
        if frozenset([r1, r2]) in existing_pairs:
            already_exists.append((c, r1, r2))
        else:
            new_resolvable.append((c, r1, r2))

    print(f"\n--- NOVO, RESOLVABLE, kandidat za dodavanje ({len(new_resolvable)}) ---")
    for c, r1, r2 in new_resolvable:
        print(f"  {r1}  <->  {r2}   [{c.get('evidence', 'NEGATIVE')}]")
    print(f"\n--- VEC POSTOJI u gold datasetu ({len(already_exists)}) ---")
    for c, r1, r2 in already_exists:
        print(f"  {r1}  <->  {r2}")
    print(f"\n--- NERAZRESIVO (ime ne postoji u pool-u) ({len(unresolvable)}) ---")
    for c, r1, r2 in unresolvable:
        print(f"  {c['id_1']!r} -> {r1}   |   {c['id_2']!r} -> {r2}")

    return new_resolvable


new_pos = check_list(POSITIVE_CANDIDATES, "POZITIVNI KANDIDATI")
new_neg = check_list(NEGATIVE_CANDIDATES, "NEGATIVNI KANDIDATI")

print(f"\n\nUKUPNO: {len(new_pos)} novih pozitivnih, {len(new_neg)} novih negativnih kandidata za upis.")
