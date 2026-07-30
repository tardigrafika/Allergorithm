"""
Racuna punu simetricnu TM-score matricu (strukturno poravnanje, tmtools/TM-align)
za sve proteine koji imaju AlphaFold strukturu (data/alphafold_structures/*.pdb).

Paralelizovano (multiprocessing, sva jezgra) jer je TM-align sporiji od
sequence poravnanja koje smo koristili za BLAST (~27 poravnanja/sek na 1
jezgru -> puna matrica bi trajala ~9h na 1 jezgru, ~1-1.5h na 8).

TM-score se racuna u OBA smera (normalizovano dužinom svakog proteina
posebno) i uzima se prosek (standardna konvencija za simetrican TM-score).

Ulaz:
    data/alphafold_structures/{allergen_id}.pdb

Izlaz:
    output/tmscore_matrix_1443.pkl
        dict: {"ids": [...], "tm_score_matrix": NxN float32}
"""

import pickle
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index

warnings.filterwarnings("ignore")

STRUCTURE_DIR = Path("/home/lana/ALERGRAF/data/alphafold_structures")
OUTPUT = Path("/home/lana/ALERGRAF/output/tmscore_matrix_1443.pkl")

N_WORKERS = 6  # sredina izmedju 8 (glasno) i 4 (predugo)


def load_ca(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(path))
    chain = next(iter(structure[0]))
    coords, seq = [], []
    for res in chain:
        if "CA" in res:
            coords.append(res["CA"].get_coord())
            try:
                seq.append(index_to_one(three_to_index(res.get_resname())))
            except KeyError:
                seq.append("X")
    return np.array(coords, dtype=np.float64), "".join(seq)


print("Loading structures...")
pdb_files = sorted(STRUCTURE_DIR.glob("*.pdb"))
structures = {}
for f in pdb_files:
    try:
        coords, seq = load_ca(f)
        if len(seq) >= 5:  # tm-align needs a minimal number of residues
            structures[f.stem] = (coords, seq)
    except Exception as e:
        print(f"  WARNING: failed to parse {f.name}: {e}")

ids = sorted(structures.keys())
n = len(ids)
print(f"Usable structures: {n}")

pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
print(f"Pairs to compute: {len(pairs)}")


def compute_pair(args):
    i, j = args
    from tmtools import tm_align  # imported inside worker for multiprocessing safety
    c1, s1 = structures[ids[i]]
    c2, s2 = structures[ids[j]]
    try:
        res = tm_align(c1, c2, s1, s2)
        score = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2.0
    except Exception:
        score = 0.0
    return i, j, score


if __name__ == "__main__":
    tm_score_matrix = np.zeros((n, n), dtype=np.float32)

    start = time.time()
    done = 0
    with Pool(N_WORKERS) as pool:
        for i, j, score in pool.imap_unordered(compute_pair, pairs, chunksize=200):
            tm_score_matrix[i, j] = tm_score_matrix[j, i] = score
            done += 1
            if done % 5000 == 0 or done == len(pairs):
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(pairs) - done) / rate if rate > 0 else float("inf")
                print(f"  {done}/{len(pairs)} pairs ({rate:.1f} pairs/s, "
                      f"~{remaining/60:.1f} min remaining)")

    np.fill_diagonal(tm_score_matrix, 1.0)

    print(f"\nTotal time: {(time.time()-start)/60:.1f} minutes")

    with open(OUTPUT, "wb") as f:
        pickle.dump({"ids": ids, "tm_score_matrix": tm_score_matrix}, f)

    print(f"Saved: {OUTPUT}")
    off_diag = tm_score_matrix[~np.eye(n, dtype=bool)]
    print(f"TM-score stats (off-diagonal): mean={off_diag.mean():.3f}, max={off_diag.max():.3f}")
