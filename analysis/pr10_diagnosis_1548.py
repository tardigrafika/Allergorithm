"""
Zasebna dijagnoza: zasto MI/LSE-pooling (mi_lse_loco_1548.py) popravlja
nsLTP (+0.0218) i Profilin (+0.0334), ali NE PR-10 (-0.0012)? Sve tri
familije su prosle isti mehanizam, isti trening protokol, ista LOCO
validacija -- razlika mora biti u samoj PR-10 familiji.

Pokriva pitanja (korisnicki definisana lista):
  1. Distribucija duzine proteina (PR-10 vs nsLTP/Profilin/Tropomiozin kontrola)
  2. Sequence identity poznatih pozitivnih PR-10 parova (visok/nizak)
  4. Raznovrsnost UNUTAR familije (prosecna medjusobna cosine slicnost clanova)
  5. Label coverage / broj pozitivnih partnera po proteinu (gustina grafa)
  6. Da li se poboljsanje gubi ravnomerno po upitu, ili je agregatni artefakt
     (mesavina velikih dobitaka i velikih gubitaka koji se ponisti)

Pitanje 3 (da li cross-reactive parovi dele ISTI region koji LSE istice)
je zaseban, racunski teziji test -- u pr10_hotspot_diagnosis_1548.py.

Izlaz:
    output/pr10_diagnosis_1548_summary.txt
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
LOCO_PER_QUERY = Path("/home/lana/ALERGRAF/output/mi_lse_loco_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/pr10_diagnosis_1548_summary.txt")

TARGET_FAMILIES = ["nsLTP", "Profilin", "PR-10"]
CONTROL_FAMILY = "Tropomyosin"  # familija gde cosine vec radi dobro, za kontrast

dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)
gold = pd.read_csv(GOLD)
allergens = pd.read_csv(ALLERGENS)
loco_per_query = pd.read_csv(LOCO_PER_QUERY)

lines = ["=" * 80, "PR-10 zasebna dijagnoza -- zasto MI/LSE ne pomaze PR-10 (za razliku od "
         "nsLTP/Profilin)", "=" * 80, ""]

# =========================================================
# Helper: skup allergen_id za svaku familiju (iz gold parova, oba smera)
# =========================================================
def family_protein_ids(fam):
    ids = set()
    for p in dataset.gold_pairs:
        if p.get("family_1") == fam:
            ids.add(p["id_1"])
        if p.get("family_2") == fam:
            ids.add(p["id_2"])
    return ids


fam_ids = {fam: family_protein_ids(fam) for fam in TARGET_FAMILIES + [CONTROL_FAMILY]}
id_to_name = {v: k for k, v in dataset.name_to_id.items()}

# =========================================================
# (1) Distribucija duzine proteina
# =========================================================
lines.append("--- (1) Distribucija duzine proteina ---")
name_to_length = dict(zip(allergens["official_name"], allergens["sequence_length"]))
for fam in TARGET_FAMILIES + [CONTROL_FAMILY]:
    lengths = [name_to_length.get(id_to_name.get(pid)) for pid in fam_ids[fam]]
    lengths = [l for l in lengths if l is not None and not pd.isna(l)]
    if lengths:
        lengths = np.array(lengths)
        lines.append(f"{fam} (n={len(lengths)}): mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  "
                      f"std={lengths.std():.1f}  min={lengths.min():.0f}  max={lengths.max():.0f}")
lines.append("")

# =========================================================
# (2) Sequence identity poznatih pozitivnih parova
# =========================================================
lines.append("--- (2) Sequence identity gold parova (uniformni BLAST) ---")
for fam in TARGET_FAMILIES + [CONTROL_FAMILY]:
    sub = gold[(gold["family_1"] == fam) | (gold["family_2"] == fam)]
    ident = sub["sequence_identity_pct"].dropna()
    if len(ident):
        lines.append(f"{fam} (n={len(ident)} parova): mean={ident.mean():.1f}%  median={ident.median():.1f}%  "
                      f"min={ident.min():.1f}%  max={ident.max():.1f}%")
lines.append("")

# =========================================================
# (4) Raznovrsnost UNUTAR familije (prosecna medjusobna cosine slicnost SVIH parova clanova,
#     ne samo gold-labeled parova -- pravo pitanje diverziteta unutar celog familijskog poola)
# =========================================================
lines.append("--- (4) Raznovrsnost unutar familije (prosecna medjusobna cosine SVIH parova clanova) ---")
for fam in TARGET_FAMILIES + [CONTROL_FAMILY]:
    ids = sorted(fam_ids[fam])
    idxs = [dataset.id_to_index[i] for i in ids]
    if len(idxs) < 2:
        continue
    sub_matrix = cosine_matrix[np.ix_(idxs, idxs)]
    iu = np.triu_indices(len(idxs), k=1)
    pairwise = sub_matrix[iu]
    lines.append(f"{fam} (n={len(ids)} proteina, {len(pairwise)} parova): mean cosine={pairwise.mean():.4f}  "
                  f"median={np.median(pairwise):.4f}  std={pairwise.std():.4f}  "
                  f"(visi mean/nizi std = manje raznovrsnosti = teze razlikovanje)")
lines.append("")

# =========================================================
# (5) Label coverage / gustina grafa (broj pozitivnih partnera po proteinu)
# =========================================================
lines.append("--- (5) Label coverage: broj poznatih pozitivnih partnera po proteinu (gustina grafa) ---")
for fam in TARGET_FAMILIES + [CONTROL_FAMILY]:
    ids = fam_ids[fam]
    degree = {i: 0 for i in ids}
    n_edges = 0
    for p in dataset.gold_pairs:
        if p["id_1"] in ids and p["id_2"] in ids:
            degree[p["id_1"]] = degree.get(p["id_1"], 0) + 1
            degree[p["id_2"]] = degree.get(p["id_2"], 0) + 1
            n_edges += 1
    n = len(ids)
    max_possible_edges = n * (n - 1) / 2
    density = n_edges / max_possible_edges if max_possible_edges > 0 else float("nan")
    degs = np.array(list(degree.values()))
    lines.append(f"{fam}: {n} proteina, {n_edges} ivica, gustina={density:.1%} "
                  f"(od max mogucih {max_possible_edges:.0f}), prosecan stepen={degs.mean():.1f}, "
                  f"medijan stepena={np.median(degs):.1f}")
lines.append("")

# =========================================================
# (6) Da li se poboljsanje gubi ravnomerno, ili je mesavina dobitaka/gubitaka
# =========================================================
lines.append("--- (6) Distribucija delta (lse_rr - cosine_rr) PO UPITU -- pravi null ili mesavina? ---")
for fam in TARGET_FAMILIES:
    sub = loco_per_query[loco_per_query["family"] == fam].copy()
    sub["delta"] = sub["lse_rr"] - sub["cosine_rr"]
    d = sub["delta"]
    n_pos = (d > 0.001).sum()
    n_neg = (d < -0.001).sum()
    n_flat = len(d) - n_pos - n_neg
    lines.append(f"{fam} (n={len(d)} upita): mean delta={d.mean():+.4f}  std={d.std():.4f}  "
                  f"poboljsano={n_pos} ({n_pos/len(d):.1%})  pogorsano={n_neg} ({n_neg/len(d):.1%})  "
                  f"nepromenjeno={n_flat} ({n_flat/len(d):.1%})")
    lines.append(f"    percentili delta: p10={d.quantile(0.1):+.4f}  p25={d.quantile(0.25):+.4f}  "
                  f"p50={d.quantile(0.5):+.4f}  p75={d.quantile(0.75):+.4f}  p90={d.quantile(0.9):+.4f}")
lines.append("")

summary_text = "\n".join(lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
