"""
E2: Familije poznate da NE reaguju ukrsteno -- Phl p 7 (pravi 2-EF-hand
polkalcin) eksplicitno NE deli epitope sa 3-EF-hand (Bet v 3, Gad c 1,
Cyp c 1) ni 4-EF-hand (Ole e 8, Jun o 4, Amb a 10) kalcijum-vezujucim
proteinima -- MAUG 2.0 C06 poglavlje, citat vec izvucen u D1 fazi ali
iskoriscen samo za Phl p 7 x Bet v 3 par; ostalih 4 para nije bilo dodato.

Citat (MAUG 2.0, C06): "Phl p 7 and related two EF-hand allergens do not
share epitopes with other 3-EF-hand calcium binding proteins allergens
(e.g., Bet v 3, parvalbumins Gad c 1 or carp Cyp c 1) or 4-EF hand calcium
binding proteins allergens (Ole e 8, Jun o 4, Amb a 10)."

Izlaz: dodaje nove negativne redove u cross_reactive_1548.csv.
"""

from pathlib import Path

import pandas as pd

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CITATION = ("EAACI Molecular Allergology User's Guide 2.0 (C06, Polcalcins): \"Phl p 7 and related two EF-hand "
            "allergens do not share epitopes with other 3-EF-hand calcium binding proteins allergens (e.g., "
            "Bet v 3, parvalbumins Gad c 1 or carp Cyp c 1) or 4-EF hand calcium binding proteins allergens "
            "(Ole e 8, Jun o 4, Amb a 10).\"")

PAIRS = ["Gad c 1", "Cyp c 1", "Ole e 8", "Jun o 4", "Amb a 10"]

allergens = pd.read_csv("/home/lana/ALERGRAF/output/clean_allergens.csv")
names = sorted(set(allergens["official_name"].astype(str)))


def resolve(n):
    m = [x for x in names if x == n or x.startswith(n + ".")]
    return sorted(m)[0] if m else None


df = pd.read_csv(GOLD)
existing_pairs = {frozenset([str(r["allergen_id_1"]), str(r["allergen_id_2"])]) for _, r in df.iterrows()}
phl_p7 = "Phl p 7.0101"
fam_hit = df[df["allergen_id_1"] == phl_p7]
phl_fam, phl_src = (fam_hit.iloc[0]["family_1"], fam_hit.iloc[0]["source_food_1"]) if len(fam_hit) else (None, None)

allergen_family_lookup = dict(zip(allergens["official_name"], allergens["protein_family"]))
allergen_source_lookup = dict(zip(allergens["official_name"], allergens["source_food"]))

next_num = max(int(pid[3:]) for pid in df["pair_id"] if str(pid).startswith("NEG") and pid[3:].isdigit()) + 1
new_rows = []
for name in PAIRS:
    resolved = resolve(name)
    if resolved is None or frozenset([phl_p7, resolved]) in existing_pairs:
        print(f"Preskoceno: {name} -> {resolved}")
        continue
    fam2 = allergen_family_lookup.get(resolved)
    src2 = allergen_source_lookup.get(resolved)
    new_rows.append({
        "pair_id": f"NEG{next_num:03d}", "allergen_id_1": phl_p7, "source_food_1": phl_src, "family_1": phl_fam,
        "allergen_id_2": resolved, "source_food_2": src2,
        "family_2": fam2 if fam2 is not None and str(fam2).lower() != "nan" else "EF-hand calcium-binding protein (non-polcalcin subclass)",
        "evidence_type": "Explicit non-cross-reactivity (structurally distinct EF-hand subclass)",
        "evidence_level": "Reported negative", "sequence_identity_pct": None, "reference": CITATION,
        "isoform_note": None,
        "notes": f"Phl p 7 (2-EF-hand true polcalcin) does NOT share epitopes with {name} (3/4-EF-hand subclass).",
    })
    existing_pairs.add(frozenset([phl_p7, resolved]))
    next_num += 1

new_df = pd.DataFrame(new_rows)
combined = pd.concat([df, new_df], ignore_index=True)
combined.to_csv(GOLD, index=False)
print(f"Dodato {len(new_rows)} novih EF-hand negativnih parova.")
print(f"Novi ukupan broj redova: {len(combined)}")
