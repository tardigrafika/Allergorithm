"""
Pitanje 3 iz PR-10 dijagnoze: da li gold-pozitivni parovi DELE ISTI (relativni)
region proteina kao najbolje poklapajuci par prozora (LSE "hot spot"), ili je
top-poklapanje rasuto/nasumicno po razlicitim delovima proteina? Ako je
konzistentno lokalizovano -> postoji deljeni epitope-regoin signal koji LSE
moze da iskoristi. Ako je rasuto -> nema lokalne diferencijacije za LSE da
nadje, sto bi objasnilo zasto LSE ne pomaze PR-10 (dodatna potvrda nalaza
iz pr10_diagnosis_1548.py: PR-10 ima gotovo NULTU varijansu unutar-familije
cosine slicnosti, std=0.0052 vs nsLTP 0.0465 / Profilin 0.0255).

Metod: za svaki gold-pozitivan par (A,B) u cilnoj familiji, nadji POZICIJU
(kao razlomak duzine proteina, 0=N-terminus, 1=C-terminus) prozora sa
NAJVECOM slicnoscu u A i u B. Izracunaj varijansu tih pozicija PREKO SVIH
parova iste familije -- niska varijansa = konzistentna lokalizacija
(deljen epitope region), visoka varijansa = rasuto/nasumicno.

Izlaz:
    output/pr10_hotspot_diagnosis_1548_summary.txt
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/pr10_hotspot_diagnosis_1548_summary.txt")

WINDOW = 20
STRIDE = 5
TARGET_FAMILIES = ["nsLTP", "Profilin", "PR-10"]

print("Loading dataset + residue embeddings...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

print("Racunam sliding-window embeddinge...")
window_vecs_per_protein = {}
window_positions_per_protein = {}  # relativna pozicija centra svakog prozora (0..1)
for aid in dataset.all_ids:
    res_emb = residue_embeddings.get(aid)
    if res_emb is None or len(res_emb) == 0:
        continue
    L = res_emb.shape[0]
    if L <= WINDOW:
        w = res_emb.mean(axis=0, keepdims=True)
        positions = np.array([0.5])
    else:
        starts = list(range(0, L - WINDOW + 1, STRIDE))
        w = np.array([res_emb[s:s + WINDOW].mean(axis=0) for s in starts])
        positions = np.array([(s + WINDOW / 2) / L for s in starts])
    window_vecs_per_protein[aid] = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
    window_positions_per_protein[aid] = positions

print(f"Gotovo za {len(window_vecs_per_protein)}/{len(dataset.all_ids)} proteina.\n")

lines = ["=" * 80, "Da li gold-pozitivni parovi dele isti (relativni) 'hot spot' region?",
         "=" * 80, ""]

for fam in TARGET_FAMILIES:
    fam_pairs = [p for p in dataset.gold_pairs if p.get("family_1") == fam]
    hot_positions_a, hot_positions_b = [], []
    for p in fam_pairs:
        wa, wb = window_vecs_per_protein.get(p["id_1"]), window_vecs_per_protein.get(p["id_2"])
        pos_a, pos_b = window_positions_per_protein.get(p["id_1"]), window_positions_per_protein.get(p["id_2"])
        if wa is None or wb is None:
            continue
        sim = wa @ wb.T  # (n_wa, n_wb)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        hot_positions_a.append(pos_a[i])
        hot_positions_b.append(pos_b[j])

    hot_positions_a = np.array(hot_positions_a)
    hot_positions_b = np.array(hot_positions_b)
    combined = np.concatenate([hot_positions_a, hot_positions_b])

    lines.append(f"{fam} (n={len(hot_positions_a)} parova, {len(combined)} pozicija ukupno):")
    lines.append(f"  Pozicija hot-spota (0=N-terminus, 1=C-terminus): mean={combined.mean():.3f}  "
                 f"std={combined.std():.3f}  median={np.median(combined):.3f}")
    hist, edges = np.histogram(combined, bins=10, range=(0, 1))
    hist_str = " ".join(f"{h:3d}" for h in hist)
    lines.append(f"  Histogram (10 binova, N-term->C-term): [{hist_str}]  "
                 f"(referentno: uniformno rasuto = ~{len(combined)/10:.0f} po binu)")
    lines.append("")

summary_text = "\n".join(lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"Saved: {SUMMARY_OUTPUT}")
