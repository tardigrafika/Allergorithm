"""
Top-3 window agregacija: brz follow-up na sliding_window_esm_1548.py, koji je
pokazao null/blago negativan rezultat sa MAX agregacijom (jedan najbolji par
prozora). Hipoteza za null: panalergeni dele visoko konzervisano strukturno
jezgro, pa gotovo SVAKI clan familije ima BAR JEDAN slucajno-dobar par prozora
negde u tom jezgru -- MAX je preosetljiv na tu jednu slucajnost.

Ovde: isti prozori (window=20, stride=5), ali skor para = prosek NAJBOLJA 3
para prozora (umesto jednog) -- trazi KONZISTENTNO dobro lokalno poklapanje
na vise mesta, ne jedan mogucno-slucajan pogodak.

Implementacija: fully vektorizovano preko padded-index gather-a (za razliku
od python petlje po segmentima) -- isti globalni window-pool kao pre, ali
sa unapred izgradjenom (n_proteina x max_windows) indeks-maskom, tako da je
top-3 agregacija PO UPITU brza za racunanje (samo gather + sort, ne petlja).

Testira se SAMO na nsLTP/Profilin/PR-10, isti podskup kao MAX verzija, za
direktno poredjenje.

Izlaz:
    output/sliding_window_top3_1548_summary.txt
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
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/sliding_window_top3_1548_summary.txt")

WINDOW = 20
STRIDE = 5
TOP_K = 3
TARGET_FAMILIES = ["nsLTP", "Profilin", "PR-10"]

print("Loading dataset + residue embeddings...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

print("Racunam sliding-window embeddinge za sve proteine...")
window_vecs_per_protein = {}
for aid in dataset.all_ids:
    res_emb = residue_embeddings.get(aid)
    if res_emb is None or len(res_emb) == 0:
        continue
    L = res_emb.shape[0]
    if L <= WINDOW:
        window_vecs_per_protein[aid] = res_emb.mean(axis=0, keepdims=True)
        continue
    windows = [res_emb[start:start + WINDOW].mean(axis=0) for start in range(0, L - WINDOW + 1, STRIDE)]
    window_vecs_per_protein[aid] = np.array(windows)

print(f"Prozori izracunati za {len(window_vecs_per_protein)}/{len(dataset.all_ids)} proteina.")

print("Gradim globalnu matricu svih prozora + padded indeks-masku (vektorizovano)...")
all_ids_with_windows = [aid for aid in dataset.all_ids if aid in window_vecs_per_protein]
window_blocks = [window_vecs_per_protein[aid] for aid in all_ids_with_windows]
global_windows = np.vstack(window_blocks)
global_windows_n = global_windows / (np.linalg.norm(global_windows, axis=1, keepdims=True) + 1e-12)

n_proteins = len(dataset.all_ids)
protein_indices = [dataset.id_to_index[aid] for aid in all_ids_with_windows]
counts = [len(b) for b in window_blocks]
max_windows = max(counts)

# padded_row_idx[p, k] = globalni red-indeks k-tog prozora proteina p (ili -1 ako ne postoji)
padded_row_idx = np.full((n_proteins, max_windows), -1, dtype=np.int64)
row_cursor = 0
for pidx, cnt in zip(protein_indices, counts):
    padded_row_idx[pidx, :cnt] = np.arange(row_cursor, row_cursor + cnt)
    row_cursor += cnt
valid_mask = padded_row_idx >= 0
safe_idx = np.where(valid_mask, padded_row_idx, 0)  # 0 kao dummy za gather, maskirano posle

print(f"Ukupno prozora: {global_windows_n.shape[0]}, max prozora po proteinu: {max_windows}")


def sliding_window_top3_scores_for_query(query_id):
    qw = window_vecs_per_protein.get(query_id)
    if qw is None:
        return None
    qw_n = qw / (np.linalg.norm(qw, axis=1, keepdims=True) + 1e-12)
    sim = qw_n @ global_windows_n.T                 # (n_query_windows, total_windows)
    best_per_window = sim.max(axis=0)                 # (total_windows,)

    gathered = best_per_window[safe_idx]               # (n_proteins, max_windows)
    gathered = np.where(valid_mask, gathered, -np.inf)

    k = min(TOP_K, max_windows)
    top_k_vals = np.partition(gathered, -k, axis=1)[:, -k:]  # (n_proteins, k)
    with np.errstate(invalid="ignore"):
        n_valid = np.minimum(valid_mask.sum(axis=1), k)
        top_k_vals_sum = np.where(np.isfinite(top_k_vals), top_k_vals, 0.0).sum(axis=1)
        scores = np.divide(top_k_vals_sum, n_valid, out=np.full(n_proteins, -np.inf), where=n_valid > 0)
    return scores


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

        sw_scores = sliding_window_top3_scores_for_query(query_id)
        if sw_scores is None:
            continue
        sw_scores[qi] = -np.inf
        sw_rank = int(np.argsort(np.argsort(-sw_scores))[ti]) + 1

        results.append({
            "pair_id": p["pair_id"], "family": p.get("family_1"),
            "query": query_id, "target": target_id,
            "cosine_rank": cos_rank, "sliding_window_top3_rank": sw_rank,
            "n_candidates": len(dataset.all_ids),
        })
    if i % 50 == 0:
        print(f"  {i}/{len(target_pairs)} parova obradjeno", flush=True)

results_df = pd.DataFrame(results)
results_df["cosine_rr"] = 1.0 / results_df["cosine_rank"]
results_df["sw_rr"] = 1.0 / results_df["sliding_window_top3_rank"]

summary_lines = ["=" * 80, "Sliding-window ESM top-3 agregacija vs whole-protein cosine (nsLTP/Profilin/PR-10)",
                  "=" * 80, "", f"Window={WINDOW}, Stride={STRIDE}, Top-K={TOP_K}",
                  f"Ukupno upita: {len(results_df)}", ""]
for fam in TARGET_FAMILIES:
    sub = results_df[results_df["family"] == fam]
    if len(sub) == 0:
        continue
    cos_mrr, sw_mrr = sub["cosine_rr"].mean(), sub["sw_rr"].mean()
    delta = sw_mrr - cos_mrr
    summary_lines.append(f"{fam} (n={len(sub)}): cosine MRR={cos_mrr:.4f}  top3-window MRR={sw_mrr:.4f}  "
                          f"delta={delta:+.4f}")

overall_cos, overall_sw = results_df["cosine_rr"].mean(), results_df["sw_rr"].mean()
summary_lines.append(f"\nUKUPNO: cosine MRR={overall_cos:.4f}  top3-window MRR={overall_sw:.4f}  "
                      f"delta={overall_sw - overall_cos:+.4f}")

n_improved = (results_df["sw_rr"] > results_df["cosine_rr"]).sum()
summary_lines.append(f"Broj upita gde top3-window POBOLJŠAVA rang: {n_improved}/{len(results_df)} "
                      f"({n_improved/len(results_df)*100:.1f}%)")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
results_df.to_csv("/home/lana/ALERGRAF/output/sliding_window_top3_1548_per_query.csv", index=False)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
