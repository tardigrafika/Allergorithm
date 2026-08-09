"""
Racuna TM-score matricu (strukturno poravnanje, tmtools/TM-align) -- za VM.

Optimizacija #1 u odnosu na compute_tmscore_matrix_1443.py (koji je pokusan
lokalno i ubijen posle 3.6h bez vidljivog napretka): NE racuna punu N x N
matricu za svih 1240 struktura. Racuna samo parove gde je BAR JEDAN protein
iz "povezanog univerzuma" (proteini koji se pojavljuju u bar jednom kuriranom
cross-reactivity paru, iz cross_reactive_1548.csv) -- jer "slobodni" proteini
(bez ijednog poznatog para) NIKAD nisu upit u retrieval evaluaciji, samo
kandidati, pa par slobodan<->slobodan nikad nije potreban.
To smanjuje posao sa ~768,180 na ~339,000 parova (~56% manje).

Optimizacija #2: cesto flush-ovano izvestavanje o napretku (svakih 300 parova
I najmanje svakih 30 sekundi, sta god prvo nastupi) -- lokalni pokusaj je
imao samo bafer izlaz na svakih 5000 parova sto je znacilo SATIMA bez ijednog
reda ispisa dok je sporo isao.

Optimizacija #3: periodicno cuva DELIMICAN rezultat (svakih ~10% posla) u
OUTPUT_PARTIAL, tako da ako se proces mora prekinuti, posao nije izgubljen --
moze se nastaviti (vidi RESUME_FROM_PARTIAL nize).

Ulaz (prebaciti na VM):
    data/alphafold_structures/*.pdb   (1240 fajlova, ~preuzeti prilikom
                                        prethodnog AlphaFold fetch koraka)
    output/cross_reactive_1548.csv    (definise "povezan univerzum")

Izlaz (preneti nazad sa VM):
    output/tmscore_matrix_1548.pkl
        dict: {"ids": [...], "tm_score_matrix": NxN float32 (NaN gde nije
               racunato -- slobodan x slobodan par, nikad potreban)}
"""

import pickle
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index

warnings.filterwarnings("ignore")

STRUCTURE_DIR = Path("data/alphafold_structures")
GOLD = Path("output/cross_reactive_1548.csv")
OUTPUT = Path("output/tmscore_matrix_1548.pkl")
OUTPUT_PARTIAL = Path("output/tmscore_matrix_1548_partial.pkl")

N_WORKERS = 8  # podesi prema broju jezgara na VM
CHECKPOINT_EVERY_FRAC = 0.10  # sacuvaj delimican rezultat svakih ~10% posla
PROGRESS_EVERY_N = 300
PROGRESS_EVERY_SEC = 30


# =====================================================
# CONNECTED UNIVERSE (proteini koji se pojavljuju u bar jednom kuriranom paru)
# =====================================================

print("Loading connected universe from gold standard...")
gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

parent = {}
for _, row in gold.iterrows():
    a, b = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    parent.setdefault(a, a)
    parent.setdefault(b, b)
connected_names = set(parent.keys())
print(f"Connected universe (official_name, before mapping to allergen_id): {len(connected_names)}")


# =====================================================
# LOAD STRUCTURES
# =====================================================

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

# allergen_id -> official_name mapping isn't available here (no embeddings
# parquet needed on VM) -- match by allergen_id directly since fetch script
# names files by allergen_id, and gold's allergen_id_1/_2 columns are
# official_name. We only have official_name from the gold file, so instead
# mark "connected" via a name-agnostic proxy: reuse the SAME logic as the
# main pipeline scripts would, but since VM doesn't have embeddings.parquet,
# ship a small id list instead (see NOTE in the run instructions).
CONNECTED_IDS_FILE = Path("output/connected_universe_ids_1548.txt")
if CONNECTED_IDS_FILE.exists():
    connected_ids = set(CONNECTED_IDS_FILE.read_text().split())
    print(f"Connected universe (allergen_id, from {CONNECTED_IDS_FILE}): {len(connected_ids)}")
else:
    raise SystemExit(f"Missing {CONNECTED_IDS_FILE} -- generate it locally first (see run instructions) and ship it to the VM.")

connected_with_structure = [i for i in ids if i in connected_ids]
print(f"Connected proteins with a usable structure: {len(connected_with_structure)}")

connected_idx = {i for i, aid in enumerate(ids) if aid in connected_ids}
pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if i in connected_idx or j in connected_idx]
print(f"Pairs to compute (skipping free x free): {len(pairs)}  "
      f"(vs {n * (n - 1) // 2} for the full matrix)")


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
    tm_score_matrix = np.full((n, n), np.nan, dtype=np.float32)
    total_pairs = len(pairs)
    resume_done = 0

    if OUTPUT_PARTIAL.exists():
        print(f"Found partial result at {OUTPUT_PARTIAL}, attempting to resume...", flush=True)
        with open(OUTPUT_PARTIAL, "rb") as f:
            partial = pickle.load(f)
        if partial["ids"] == ids:
            tm_score_matrix = partial["tm_score_matrix"]
            pairs = [(i, j) for (i, j) in pairs if np.isnan(tm_score_matrix[i, j])]
            resume_done = total_pairs - len(pairs)
            print(f"Resumed: {resume_done}/{total_pairs} pairs already computed, "
                  f"{len(pairs)} remaining", flush=True)
        else:
            print("WARNING: partial result's protein id list doesn't match current "
                  "structures -- ignoring partial, starting fresh", flush=True)

    checkpoint_step = max(1, int(total_pairs * CHECKPOINT_EVERY_FRAC))
    start = time.time()
    last_print = start
    done = resume_done

    with Pool(N_WORKERS) as pool:
        for i, j, score in pool.imap_unordered(compute_pair, pairs, chunksize=50):
            tm_score_matrix[i, j] = tm_score_matrix[j, i] = score
            done += 1
            now = time.time()
            if done % PROGRESS_EVERY_N == 0 or (now - last_print) >= PROGRESS_EVERY_SEC or done == total_pairs:
                elapsed = now - start
                rate = (done - resume_done) / elapsed
                remaining = (total_pairs - done) / rate if rate > 0 else float("inf")
                print(f"  {done}/{total_pairs} pairs ({rate:.2f} pairs/s, "
                      f"~{remaining/60:.1f} min remaining, {elapsed/60:.1f} min elapsed this session)", flush=True)
                last_print = now
            if done % checkpoint_step == 0:
                with open(OUTPUT_PARTIAL, "wb") as f:
                    pickle.dump({"ids": ids, "tm_score_matrix": tm_score_matrix, "done": done, "total": total_pairs}, f)
                print(f"  [checkpoint] saved partial result at {done}/{total_pairs} to {OUTPUT_PARTIAL}", flush=True)

    np.fill_diagonal(tm_score_matrix, 1.0)

    print(f"\nTotal time: {(time.time()-start)/60:.1f} minutes")

    with open(OUTPUT, "wb") as f:
        pickle.dump({"ids": ids, "tm_score_matrix": tm_score_matrix}, f)

    print(f"Saved: {OUTPUT}")
    computed = tm_score_matrix[~np.isnan(tm_score_matrix) & ~np.eye(n, dtype=bool)]
    print(f"TM-score stats (off-diagonal, computed pairs only): "
          f"mean={computed.mean():.3f}, max={computed.max():.3f}, n_computed={len(computed)}")
