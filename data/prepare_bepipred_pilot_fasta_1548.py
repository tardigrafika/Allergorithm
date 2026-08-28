"""
Priprema FASTA fajla za BepiPred-3.0 pilot -- isti skup proteina kao
TM-align pilot (analysis/tmalign_pilot_worst50_1548.py): svi proteini
ukljuceni u 50 najgorih RRF upita (nsLTP/Profilin, Gly m 1 hub), da direktno
uporedimo da li predikcija epitopa (umesto stvarnih retkih IEDB podataka --
samo 87/331 proteina, N=80 parova, neuverljivo) daje diskriminativan signal
tamo gde su svi globalni signali (cosine/BLAST/FoldseekTM) vec pali.

Pokrenuti LOKALNO (ne treba VM za ovaj korak, samo za sam BepiPred).

Izlaz:
    output/bepipred_pilot_1548.fasta   -- prebaciti na VM
"""

from pathlib import Path

import pandas as pd

RANK_FUSION = Path("/home/lana/ALERGRAF/output/rank_fusion_1548_per_query.csv")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
FASTA_OUTPUT = Path("/home/lana/ALERGRAF/output/bepipred_pilot_1548.fasta")

df = pd.read_csv(RANK_FUSION)
gold = pd.read_csv(GOLD)[["pair_id", "allergen_id_1", "allergen_id_2"]].drop_duplicates("pair_id")
merged = df.merge(gold, on="pair_id", how="left")
worst = merged.sort_values("rrf_rank", ascending=False).head(50)

names_needed = sorted(set(worst["allergen_id_1"]) | set(worst["allergen_id_2"]))
print(f"Jedinstveni proteini u worst-50: {len(names_needed)}")

clean = pd.read_csv(CLEAN_ALLERGENS)
name_to_seq = dict(zip(clean["official_name"], clean["fasta_sequence"]))
name_to_id = dict(zip(clean["official_name"], clean["allergen_id"]))

with open(FASTA_OUTPUT, "w") as f:
    written = 0
    for name in names_needed:
        seq = name_to_seq.get(name)
        aid = name_to_id.get(name)
        if not seq or not aid:
            print(f"  [upozorenje] '{name}' nema sekvencu, preskacem")
            continue
        f.write(f">{aid}\n{seq}\n")
        written += 1

print(f"Napisano {written} sekvenci u {FASTA_OUTPUT}")
