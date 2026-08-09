"""
Pretvara sirov Foldseek easy-search izlaz (output/foldseek_tmscore_1548.tsv)
u dict-based lookup kompatibilan sa RF feature pipeline-om (isti obrazac kao
blast_identity_matrix_1443.pkl, samo dict umesto guste matrice jer je
foldseek izlaz redak - ~230K od mogucih ~768K parova).

Simetrican skor = prosek qtmscore i ttmscore (normalizovano duzinom upita i
mete posebno - standardna konvencija, isto kao ranije tm_norm_chain1/chain2
prosek u planiranoj tmtools verziji). Ako je par pronadjen u OBA smera,
usrednjava se svih 4 vrednosti (robusnije).

Parovi koje Foldseek nije pronasao (ispod praga detekcije) dobijaju 0.0
fallback pri lookup-u - to je razumno (nema detektovane strukturne
slicnosti), isti fallback kao u originalnom planu.

Ulaz:
    output/foldseek_tmscore_1548.tsv

Izlaz:
    output/foldseek_tmscore_lookup_1548.pkl
        dict: {frozenset({id_a, id_b}): symmetric_tm_score}
"""

import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

INPUT = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_1548.tsv")
OUTPUT = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

df = pd.read_csv(INPUT, sep="\t", header=None,
                  names=["query", "target", "alntmscore", "qtmscore", "ttmscore", "fident", "alnlen", "evalue"])
print(f"Raw rows: {len(df)}")

pair_scores = defaultdict(list)
for row in df.itertuples(index=False):
    key = frozenset((row.query, row.target))
    pair_scores[key].append((row.qtmscore + row.ttmscore) / 2.0)

lookup = {k: sum(v) / len(v) for k, v in pair_scores.items()}
print(f"Unique unordered pairs: {len(lookup)}")

with open(OUTPUT, "wb") as f:
    pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved: {OUTPUT}")

vals = list(lookup.values())
import numpy as np
print(f"Score stats: mean={np.mean(vals):.3f}, median={np.median(vals):.3f}, max={np.max(vals):.3f}")
