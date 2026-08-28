"""
Priprema FASTA za PUN BepiPred-3.0 run -- svi proteini sa rezidue-
embeddinzima (isti pool koji analysis/residue_topk_nsltp_profilin_1548.py
koristi kao candidate universe), da uradimo PRAVI test (rang u punom pool-u,
bootstrap CI) umesto brzog pilota sa sirovim cosine vrednostima na malom
podskupu (koji je imao poznatu manu: anisotropy efekat unutar uske,
familijski-homogene grupe cini null baseline vestacki visokim).

BepiPred ne zavisi od AlphaFold strukture (samo sekvenca), pa mozemo pokriti
CEO pool (1534), ne samo 1016 koliko je surface-residue pristup imao zbog
nedostajucih struktura -- ovo je sira, kompletnija verzija tog ranijeg testa.

Pokrenuti LOKALNO. Izlaz:
    output/bepipred_full_1548.fasta   -- prebaciti na VM
"""

import pickle
from pathlib import Path

import pandas as pd

RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
FASTA_OUTPUT = Path("/home/lana/ALERGRAF/output/bepipred_full_1548.fasta")

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)
pool_ids = sorted(residue_embeddings.keys())
print(f"Proteini sa rezidue-embeddinzima: {len(pool_ids)}")

clean = pd.read_csv(CLEAN_ALLERGENS)
id_to_seq = dict(zip(clean["allergen_id"], clean["fasta_sequence"]))

written = 0
mismatched = 0
with open(FASTA_OUTPUT, "w") as f:
    for aid in pool_ids:
        seq = id_to_seq.get(aid)
        if not seq:
            print(f"  [upozorenje] '{aid}' nema sekvencu, preskacem")
            continue
        if len(seq) != residue_embeddings[aid].shape[0]:
            mismatched += 1
            continue
        f.write(f">{aid}\n{seq}\n")
        written += 1

print(f"Napisano {written} sekvenci u {FASTA_OUTPUT} ({mismatched} preskoceno zbog razlike u duzini)")
