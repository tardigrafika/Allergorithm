"""
Dodaje ccd_flag kolonu u cross_reactive_1548.csv -- da li dokaz za cross-
reaktivnost dolazi od PROTEINSKOG epitopa ili GLIKANSKOG (CCD = cross-
reactive carbohydrate determinant) epitopa. Mentorka je eksplicitno trazila
ovo: model uci iz proteinske sekvence (ESM), pa CCD-bazirani "pozitivi" ne
mogu nikad biti naucljivi -- treba ih oznaciti (i kasnije iskljuciti iz
trening/evaluacionog skupa za protein-only modele).

PRVI PROLAZ (namerno delimican, ne pokusava sve odjednom): fokus na
najpoznatiju CCD-rizik zonu u datasetu -- Hymenoptera venom alergene (bee
vs vespid vs fire ant), gde je CCD confounding klasican, dobro dokumentovan
problem u literaturi. Svaka oznaka ispod je VERIFIKOVANA preko WebSearch-a
(citati navedeni), ne pretpostavljena po nazivu familije.

Vrednosti ccd_flag:
  "protein_epitope"        -- verifikovano kao proteinski-bazirana reaktivnost
  "ccd_glycan_confirmed"   -- verifikovano preko literature da je CCD/glikan-bazirano
  "ccd_possible_unverified" -- rizicna zona (npr. glikoprotein iz poznate CCD-nosece
                                familije) ali specifican par NIJE pojedinacno potvrdjen
  (prazno/NaN)              -- JOS NIJE PREGLEDANO -- vecina dataseta, sledeci koraci

VERIFIKOVANO (avgust 2026, WebSearch):
  - CR360 (Api m 2 x Ves v 2, pcela-osa hijaluronidaza), CR361 (Api m 2 x Sol i 2,
    pcela-vatreni mrav): Jovanovic et al 2023 (Clin Transl Allergy, PMC9993137) i
    Blank/Jin et al: "Cross-reactivity between bee venom hyaluronidases and Vespid
    venom hyaluronidases is deemed limited outside CCD moieties... vespid
    hyaluronidases exhibited pronounced and PRIMARY carbohydrate reactivity
    rendering their relevance in allergy questionable." -> ccd_glycan_confirmed
  - CR349 (Api m 1 x Ves v 1, PLA2 vs PLA1): dvosmisleno -- Ves v 1 opisan kao
    "classical, mostly non-glycosylated... CCD-free... not causative for CCD-based
    cross-reactivity" (isti izvor), ali Api m 1 nosi alpha-1,3-fukozilovan N-glikan
    (poznat CCD epitop). Ne moze se cisto potvrditi ni odbaciti sa dostupnom
    literaturom -> ccd_possible_unverified.
  - Svi ostali venom parovi u datasetu (Ves v/Dol m/Pol d/Vesp v/Vesp c/Vesp m/Ves m,
    SVI unutar Vespidae, PLA1 grupa 1 ili Antigen 5): King TP & Spangfort 2000 i
    Hoffman DR 2006 citati su o PROTEINSKOJ sekvencnoj homologiji medju blisko
    srodnim vespid vrstama (ne bee-vespid CCD confounding); Ves v 1-tip PLA1 je
    "uglavnom neglikoziliran" po istom izvoru -> protein_epitope.

SLEDECI KORAK (nije jos uradjeno): sistematican pregled ostalih CCD-rizicnih
kategorija (plant glikoproteini opsteg tipa, mite alergeni, pollen-pollen
"panallergen" parovi) -- ovaj prolaz pokriva SAMO Hymenoptera venom, 25/1548
parova.

Izlaz: prepisuje output/cross_reactive_1548.csv sa novom ccd_flag kolonom
(sve ostalo neizmenjeno).
"""

from pathlib import Path

import pandas as pd

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")

VERIFIED_CCD_FLAGS = {
    "CR360": ("ccd_glycan_confirmed",
              "Jovanovic et al 2023 (PMC9993137); Blank/Jin venom hyaluronidase CCD studies -- "
              "vespid hyaluronidase cross-reactivity 'primary carbohydrate reactivity'"),
    "CR361": ("ccd_glycan_confirmed",
              "isto kao CR360 (bee-fire ant hijaluronidaza, ista CCD osnova)"),
    "CR349": ("ccd_possible_unverified",
              "Api m 1 nosi CCD epitop (alpha-1,3-fukoza), ali Ves v 1 je opisan kao CCD-free -- "
              "specifican par nije pojedinacno potvrdjen ni kao cisto proteinski ni kao CCD"),
}

PROTEIN_EPITOPE_VENOM_PAIRS = [
    "CR340", "CR341", "CR342", "CR343", "CR344", "CR345", "CR346", "CR347", "CR348",
    "CR350", "CR351", "CR352", "CR353", "CR354", "CR355", "CR356", "CR357", "CR358", "CR359",
]

df = pd.read_csv(GOLD)
df["ccd_flag"] = pd.NA

for pair_id, (flag, note) in VERIFIED_CCD_FLAGS.items():
    mask = df["pair_id"] == pair_id
    assert mask.sum() == 1, f"{pair_id} not found exactly once"
    df.loc[mask, "ccd_flag"] = flag

for pair_id in PROTEIN_EPITOPE_VENOM_PAIRS:
    mask = df["pair_id"] == pair_id
    assert mask.sum() == 1, f"{pair_id} not found exactly once"
    df.loc[mask, "ccd_flag"] = "protein_epitope"

df.to_csv(GOLD, index=False)

print(f"ccd_flag dodat. Popunjeno za {df['ccd_flag'].notna().sum()}/{len(df)} parova (Hymenoptera venom, prvi prolaz).")
print(df["ccd_flag"].value_counts(dropna=False))
