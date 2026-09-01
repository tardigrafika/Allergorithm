"""
Sliding-window ESM embedding poredjenje -- test da li lokalna rezolucija
(preklapajuci prozori od ~20 rezidua, mean-pooled) razdvaja unutar-familije
"zagusenje" bolje od whole-protein mean-pool cosine-a, KOJE JE DIJAGNOSTIKOVANO
kao pravi uzrok katastrofalnog MRR-a za nsLTP/Profilin/PR-10 (analysis
pokazala da desetine clanova iste familije score gotovo identicno -- nije
da signal nedostaje, nego da whole-protein pooling ne razdvaja koji je
TACAN partner).

RAZLIKUJE SE od ranijeg null rezultata (residue_topk_nsltp_profilin_1548.py):
taj je koristio SIROVE per-rezidua embeddinge (opciono SASA-filtrirane),
top-K NAJBOLJIH POJEDINACNIH rezidua. Ovo koristi PROZORE (lokalno mean-
pooled preko ~20 rezidua = velicina tipicnog linearnog epitopa), sto je
manje sumovito od pojedinacnih rezidua i direktno cilja "koji lokalni
region je najslicniji", ne "koji pojedinacni amino-kiselinski par".

Metod: za par (A,B), skor = MAX cosine slicnost preko SVIH parova prozora
(prozor iz A, prozor iz B) -- "best local match" umesto "global average".

Testira se SAMO na nsLTP/Profilin/PR-10 (gde je problem dijagnostikovan) --
whole-protein cosine kao referenca na ISTOM podskupu za fer poredjenje.

Izlaz:
    output/sliding_window_esm_1548_summary.txt
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
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/sliding_window_esm_1548_summary.txt")

WINDOW = 20
STRIDE = 5
TARGET_FAMILIES = ["nsLTP", "Profilin", "PR-10"]

print("Loading dataset + residue embeddings...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

print("Racunam sliding-window embeddinge za sve proteine...")
window_vecs_per_protein = {}  # allergen_id -> (n_windows, 1280) array
for aid in dataset.all_ids:
    res_emb = residue_embeddings.get(aid)
    if res_emb is None or len(res_emb) == 0:
        continue
    L = res_emb.shape[0]
    if L <= WINDOW:
        window_vecs_per_protein[aid] = res_emb.mean(axis=0, keepdims=True)
        continue
    windows = []
    for start in range(0, L - WINDOW + 1, STRIDE):
        windows.append(res_emb[start:start + WINDOW].mean(axis=0))
    window_vecs_per_protein[aid] = np.array(windows)

print(f"Prozori izracunati za {len(window_vecs_per_protein)}/{len(dataset.all_ids)} proteina.")

# Jedna VELIKA matrica svih prozora svih proteina (vektorizovano, ne petlja po kandidatima)
print("Gradim globalnu matricu svih prozora (vektorizovano)...")
all_ids_with_windows = [aid for aid in dataset.all_ids if aid in window_vecs_per_protein]
window_owner = []  # paralelan niz: koji protein (indeks u dataset.all_ids) "poseduje" svaki red
window_blocks = []
for aid in all_ids_with_windows:
    w = window_vecs_per_protein[aid]
    window_blocks.append(w)
    window_owner.extend([dataset.id_to_index[aid]] * len(w))
global_windows = np.vstack(window_blocks)
global_windows_n = global_windows / (np.linalg.norm(global_windows, axis=1, keepdims=True) + 1e-12)
window_owner = np.array(window_owner)
print(f"Ukupno prozora u globalnoj matrici: {global_windows_n.shape[0]}")

n_proteins = len(dataset.all_ids)


def sliding_window_scores_for_query(query_id):
    """Vraca (n_proteins,) niz -- za SVAKOG kandidata, MAX cosine slicnost
    izmedju BILO KOG prozora upita i BILO KOG prozora kandidata."""
    qw = window_vecs_per_protein.get(query_id)
    if qw is None:
        return None
    qw_n = qw / (np.linalg.norm(qw, axis=1, keepdims=True) + 1e-12)
    sim = qw_n @ global_windows_n.T          # (n_query_windows, total_windows)
    best_per_window = sim.max(axis=0)         # (total_windows,) -- najbolji upit-prozor za svaki globalni prozor
    scores = np.full(n_proteins, -np.inf)
    np.maximum.at(scores, window_owner, best_per_window)  # group-by-protein MAX
    return scores


# gold pairs u ciljanim familijama
target_pairs = [p for p in dataset.gold_pairs if p.get("family_1") in TARGET_FAMILIES]
print(f"\nGold parova u {TARGET_FAMILIES}: {len(target_pairs)}")

results = []
for i, p in enumerate(target_pairs, 1):
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qi = dataset.id_to_index[query_id]
        ti = dataset.id_to_index[target_id]

        cos_scores = cosine_matrix[qi].copy()
        cos_scores[qi] = -np.inf
        cos_rank = int(np.argsort(np.argsort(-cos_scores))[ti]) + 1

        sw_scores = sliding_window_scores_for_query(query_id)
        if sw_scores is None:
            continue
        sw_scores[qi] = -np.inf
        sw_rank = int(np.argsort(np.argsort(-sw_scores))[ti]) + 1

        results.append({
            "pair_id": p["pair_id"], "family": p.get("family_1"),
            "query": query_id, "target": target_id,
            "cosine_rank": cos_rank, "sliding_window_rank": sw_rank,
            "n_candidates": len(dataset.all_ids),
        })
    if i % 50 == 0:
        print(f"  {i}/{len(target_pairs)} parova obradjeno", flush=True)

results_df = pd.DataFrame(results)
results_df["cosine_rr"] = 1.0 / results_df["cosine_rank"]
results_df["sw_rr"] = 1.0 / results_df["sliding_window_rank"]

summary_lines = ["=" * 80, "Sliding-window ESM vs whole-protein cosine (nsLTP/Profilin/PR-10)", "=" * 80, "",
                  f"Window={WINDOW}, Stride={STRIDE}", f"Ukupno upita: {len(results_df)}", ""]
for fam in TARGET_FAMILIES:
    sub = results_df[results_df["family"] == fam]
    if len(sub) == 0:
        continue
    cos_mrr, sw_mrr = sub["cosine_rr"].mean(), sub["sw_rr"].mean()
    delta = sw_mrr - cos_mrr
    summary_lines.append(f"{fam} (n={len(sub)}): cosine MRR={cos_mrr:.4f}  sliding-window MRR={sw_mrr:.4f}  "
                          f"delta={delta:+.4f}")

overall_cos, overall_sw = results_df["cosine_rr"].mean(), results_df["sw_rr"].mean()
summary_lines.append(f"\nUKUPNO: cosine MRR={overall_cos:.4f}  sliding-window MRR={overall_sw:.4f}  "
                      f"delta={overall_sw - overall_cos:+.4f}")

n_improved = (results_df["sw_rr"] > results_df["cosine_rr"]).sum()
summary_lines.append(f"Broj upita gde sliding-window POBOLJŠAVA rang: {n_improved}/{len(results_df)} "
                      f"({n_improved/len(results_df)*100:.1f}%)")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
results_df.to_csv("/home/lana/ALERGRAF/output/sliding_window_esm_1548_per_query.csv", index=False)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
