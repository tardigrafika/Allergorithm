"""
Racuna povrsinske (solvent-exposed) rezidue po proteinu iz AlphaFold struktura.

Koristi biotite (Shrake-Rupley "rolling probe" algoritam) da izracuna SASA
(solvent accessible surface area) po atomu, agregira po rezidui (suma), pa
normalizuje relativnom pristupacnoscu (RSA = SASA / max_ASA_za_tip_rezidue).
Referentne max ASA vrednosti: Tien et al. 2013 (PLOS ONE), teorijske
("Theoretical") vrednosti - standardna, siroko koriscena referentna tabela
u strukturnoj bioinformatici (isti tip vrednosti koristi DSSP/NACCESS/FreeSASA
za normalizaciju).

Rezidua se smatra "povrsinskom" ako je RSA >= SURFACE_THRESHOLD (25%,
uobicajen prag u literaturi).

VAZNO: poredak rezidua iz PDB fajla (CA atomi, po res_id) mora da se poklopi
1:1 sa poretkom u residue_embeddings.pkl (koji je izveden direktno iz FASTA
sekvence bez gap-ova) - proverava se eksplicitno po proteinu, preskace se
(uz upozorenje) svaki gde se duzine ne poklapaju.

Racuna za SVE dostupne AlphaFold strukture (1240), ne samo povezani univerzum
(348) - treba nam i za KANDIDATE u retrieval evaluaciji (rangiranje protiv
punog ~1534 pool-a, isti protokol kao "Eksperiment 2"), ne samo za upite.

Ulaz:
    data/alphafold_structures/*.pdb
    embeddings/residue_embeddings.pkl (samo za duzinu-check, ne ucitava se u memoriju ovde)

Izlaz:
    output/surface_residue_masks_1548.pkl
        dict: {allergen_id: bool array (L,), True = povrsinska rezidua}
"""

import pickle
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

STRUCTURE_DIR = Path("/home/lana/ALERGRAF/data/alphafold_structures")
CONNECTED_IDS_FILE = Path("/home/lana/ALERGRAF/output/connected_universe_ids_1548.txt")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
OUTPUT = Path("/home/lana/ALERGRAF/output/surface_residue_masks_1548.pkl")

SURFACE_THRESHOLD = 0.25  # relativna SASA (RSA) >= 25% -> povrsinska rezidua

# Tien et al. 2013 (PLOS ONE), teorijske max ASA vrednosti (A^2), 3-slovni kod
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


def compute_rsa(pdb_path):
    f = pdb.PDBFile.read(str(pdb_path))
    array = pdb.get_structure(f, model=1)
    array = array[struc.filter_amino_acids(array)]

    sasa_per_atom = struc.sasa(array, vdw_radii="ProtOr")
    res_sasa = struc.apply_residue_wise(array, sasa_per_atom, np.nansum)

    ca_mask = array.atom_name == "CA"
    res_names = array.res_name[ca_mask]

    if len(res_sasa) != len(res_names):
        raise ValueError(f"residue-wise SASA count ({len(res_sasa)}) != CA atom count ({len(res_names)})")

    max_asa = np.array([MAX_ASA.get(rn, np.nan) for rn in res_names])
    rsa = res_sasa / max_asa
    return rsa


print("Loading residue embedding lengths...")
with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_emb = pickle.load(f)
emb_lengths = {aid: mat.shape[0] for aid, mat in residue_emb.items()}
del residue_emb  # done with this, free the ~2GB

pdb_files = sorted(STRUCTURE_DIR.glob("*.pdb"))
targets = pdb_files
print(f"All available structures (candidates + connected universe): {len(targets)}")

surface_masks = {}
n_mismatch = 0
n_ok = 0

for p in targets:
    aid = p.stem
    try:
        rsa = compute_rsa(p)
    except Exception as e:
        print(f"  WARNING: failed to compute SASA for {aid}: {e}")
        continue

    emb_len = emb_lengths.get(aid)
    if emb_len is None:
        print(f"  WARNING: {aid} has a structure but no residue embedding -- skipping")
        continue
    if len(rsa) != emb_len:
        print(f"  WARNING: length mismatch for {aid}: structure={len(rsa)} vs embedding={emb_len} -- skipping")
        n_mismatch += 1
        continue

    surface_masks[aid] = rsa >= SURFACE_THRESHOLD
    n_ok += 1

print(f"\nComputed surface masks: {n_ok} proteins ({n_mismatch} skipped due to length mismatch)")

frac_surface = [m.mean() for m in surface_masks.values()]
print(f"Fraction of residues classified as surface -- mean={np.mean(frac_surface):.2f}, "
      f"min={np.min(frac_surface):.2f}, max={np.max(frac_surface):.2f}")

with open(OUTPUT, "wb") as f:
    pickle.dump(surface_masks, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved: {OUTPUT}")
