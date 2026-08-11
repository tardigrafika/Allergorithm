"""
Pilot: pravi TM-align (tmtools) na 50 najgorih RRF upita (nsLTP/Profilin,
Gly m 1 hub), da proverimo da li FoldseekTM (brza aproksimacija) gubi
signal koji pravi TM-align hvata, PRE nego sto ulazemo u sate racunanja
na sirem skupu. Poredi tmtools TM-score sa vec sacuvanim FoldseekTM za
ISTE parove.

Izlaz:
    output/tmalign_pilot_worst50_1548_summary.txt
"""

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index

warnings.filterwarnings("ignore")

WORST50 = Path("/tmp/claude-1000/-home-lana-ALERGRAF/5e5b5d87-85aa-463a-8307-60af3beb2a94/scratchpad/worst50.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
STRUCTURE_DIR = Path("/home/lana/ALERGRAF/data/alphafold_structures")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/tmalign_pilot_worst50_1548_summary.txt")

worst = pd.read_csv(WORST50)
allergens = pd.read_csv(CLEAN_ALLERGENS)
name_to_id = dict(zip(allergens["official_name"], allergens["allergen_id"]))

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)


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


structure_cache = {}


def get_structure(aid):
    if aid not in structure_cache:
        path = STRUCTURE_DIR / f"{aid}.pdb"
        if not path.exists():
            structure_cache[aid] = None
        else:
            try:
                structure_cache[aid] = load_ca(path)
            except Exception:
                structure_cache[aid] = None
    return structure_cache[aid]


from tmtools import tm_align  # noqa: E402

records = []
print(f"Racunam pravi TM-align za {len(worst)} najgorih parova...")
t0 = time.time()
for i, row in worst.iterrows():
    n1, n2 = row["allergen_id_1"], row["allergen_id_2"]
    id1, id2 = name_to_id.get(n1), name_to_id.get(n2)
    fs_key = frozenset({id1, id2}) if id1 and id2 else None
    fs_score = foldseek_lookup.get(fs_key) if fs_key else None

    s1 = get_structure(id1) if id1 else None
    s2 = get_structure(id2) if id2 else None
    if s1 is None or s2 is None:
        records.append({"pair_id": row["pair_id"], "name_1": n1, "name_2": n2,
                         "foldseek_tm": fs_score, "true_tmalign": None,
                         "note": "missing structure"})
        continue

    c1, seq1 = s1
    c2, seq2 = s2
    try:
        res = tm_align(c1, c2, seq1, seq2)
        true_tm = (res.tm_norm_chain1 + res.tm_norm_chain2) / 2.0
    except Exception as e:
        true_tm = None

    records.append({"pair_id": row["pair_id"], "name_1": n1, "name_2": n2,
                     "foldseek_tm": fs_score, "true_tmalign": true_tm, "note": ""})

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(worst)}  ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

print(f"Ukupno vreme: {(time.time()-t0)/60:.1f} min")

df = pd.DataFrame(records)
valid = df[df["true_tmalign"].notna() & df["foldseek_tm"].notna()]

summary_lines = ["=" * 70, "Pilot: pravi TM-align vs FoldseekTM na 50 najgorih RRF upita", "=" * 70, "",
                  f"Ukupno parova: {len(df)}, sa oba skora dostupna: {len(valid)}",
                  f"Nedostaje struktura: {(df['true_tmalign'].isna() & (df['note']=='missing structure')).sum()}",
                  ""]

if len(valid) > 0:
    summary_lines.append(f"FoldseekTM (ovi parovi):   mean={valid['foldseek_tm'].mean():.4f}  "
                          f"max={valid['foldseek_tm'].max():.4f}")
    summary_lines.append(f"Pravi TM-align (ovi parovi): mean={valid['true_tmalign'].mean():.4f}  "
                          f"max={valid['true_tmalign'].max():.4f}")
    diff = valid["true_tmalign"] - valid["foldseek_tm"]
    summary_lines.append(f"Razlika (TM-align - Foldseek): mean={diff.mean():+.4f}  "
                          f"max={diff.max():+.4f}  n sa razlikom >0.1: {(diff>0.1).sum()}")
    summary_lines.append("")
    summary_lines.append("Detalji (sortirano po razlici, najvece prvo):")
    for _, r in valid.assign(diff=diff).sort_values("diff", ascending=False).head(15).iterrows():
        summary_lines.append(f"  {r['name_1']} <-> {r['name_2']}: Foldseek={r['foldseek_tm']:.3f}  "
                              f"TM-align={r['true_tmalign']:.3f}  diff={r['diff']:+.3f}")

missing = df[df["note"] == "missing structure"]
if len(missing) > 0:
    summary_lines.append("")
    summary_lines.append(f"Parovi bez strukture (nisu racunati): {len(missing)}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
