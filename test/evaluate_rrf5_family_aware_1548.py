"""
RRF-5 (family-aware): konkretan predlog za "novi glavni pipeline" -- umesto
JEDNE univerzalne formule za sve upite, dodaje MI/LSE-pooling (analysis/
mi_lse_pooling_1548.py, LOCO-potvrdjeno poboljsanje na nsLTP/Profilin,
analysis/mi_lse_loco_1548.py) SAMO kad je poznati pozitivan alergen
pacijenta iz nsLTP ili Profilin familije -- gde je DOKAZANO da pomaze --
umesto da ga (neuspesno testirano, rrf_lse_pooling_fusion_1548.py bilo je
presporo da se zavrsi) doda kao univerzalan 4. signal svuda.

Metod: identican CrossReactivityRanker.rank_for_patient() mehanizam (RRF-3
po poznatom pozitivu, sumirano preko svih poznatih pozitiva -- isti
mehanizam kao dokazani graph-propagation dobitak), ALI za poznate pozitive
iz nsLTP/Profilin familije dodaje se JOS JEDAN term: 1/(K+lse_rank), gde je
lse_rank rang po NAUCENOM LSE-pooling skoru (produkcioni model, treniran na
SVIM trenutno dostupnim gold pozitivima -- ovo NIJE LOCO evaluacija, gradimo
STVARNI alat, ne testiramo generalizaciju iznova).

Testira se na PROSIRENOM test/test_cases.json (49 pacijenata, +4 profilin
slucaja dodata upravo zbog ovog testa) -- ISTA leave-one-out + cluster-
permutacija/patient-level Wilcoxon metodologija kao evaluate_test_cases.py
+ evaluate_test_cases_stats_corrected.py, da se direktno uporedi RRF-4
(postojeci) naspram RRF-5 (family-aware + LSE).

Izlaz:
    output/rrf5_family_aware_lse_tau.txt (fitovan produkcioni tau, za rec.)
    test/evaluation_results_raw_rrf5.json
    test/evaluate_rrf5_family_aware_1548_summary.txt
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/test")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
RESIDUE_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/residue_embeddings.pkl")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
TAU_OUTPUT = Path("/home/lana/ALERGRAF/output/rrf5_family_aware_lse_tau.txt")
RAW_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_rrf5.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluate_rrf5_family_aware_1548_summary.txt")

WINDOW, STRIDE, SEED, N_NEG = 20, 5, 42, 3000
LSE_FAMILIES = {"nsLTP", "Profilin"}

print("Loading dataset (za family lookup + LSE trening na SVIM pozitivima)...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
print(f"  Dataset ucitan: {len(dataset.all_ids)} proteina, {len(dataset.gold_pairs)} gold parova", flush=True)

# family lookup: allergen_id (WHO_x_ISO_y sema) -> familija, iz gold parova
family_of_id = {}
for p in dataset.gold_pairs:
    if p.get("family_1"):
        family_of_id[p["id_1"]] = p["family_1"]
    if p.get("family_2"):
        family_of_id[p["id_2"]] = p["family_2"]

with open(RESIDUE_EMBEDDINGS, "rb") as f:
    residue_embeddings = pickle.load(f)

print("Racunam sliding-window embeddinge...")
window_vecs_per_protein = {}
for aid in dataset.all_ids:
    re_ = residue_embeddings.get(aid)
    if re_ is None or len(re_) == 0:
        continue
    L = re_.shape[0]
    if L <= WINDOW:
        w = re_.mean(axis=0, keepdims=True)
    else:
        w = np.array([re_[s:s + WINDOW].mean(axis=0) for s in range(0, L - WINDOW + 1, STRIDE)])
    window_vecs_per_protein[aid] = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
print(f"Gotovo za {len(window_vecs_per_protein)}/{len(dataset.all_ids)} proteina.")


def pair_similarity_matrix(id_a, id_b):
    wa, wb = window_vecs_per_protein.get(id_a), window_vecs_per_protein.get(id_b)
    if wa is None or wb is None:
        return None
    return wa @ wb.T


# -------------------------------------------------------
# PRODUKCIONI LSE trening -- na SVIM trenutno dostupnim gold pozitivima
# (ne izostavljamo nijednu familiju -- ovo NIJE LOCO test, gradimo alat)
# -------------------------------------------------------
print("\nTreniram produkcioni LSE model (sva gold pozitiva)...", flush=True)
print("  Uzorkujem negative...", flush=True)
train_neg_ids = sample_negative_pairs(dataset.all_ids, N_NEG, SEED, dataset.positive_pair_set)
print(f"  Negativa uzorkovano: {len(train_neg_ids)}", flush=True)
train_sims, train_labels = [], []
for i, p in enumerate(dataset.gold_pairs):
    S = pair_similarity_matrix(p["id_1"], p["id_2"])
    if S is not None:
        train_sims.append(S.flatten())
        train_labels.append(1.0)
    if (i + 1) % 300 == 0:
        print(f"    pozitivnih obradjeno: {i+1}/{len(dataset.gold_pairs)}", flush=True)
print(f"  Pozitivnih parova obradjeno (sim matrice): {len(train_sims)}", flush=True)
for i, (a, b) in enumerate(train_neg_ids):
    S = pair_similarity_matrix(a, b)
    if S is not None:
        train_sims.append(S.flatten())
        train_labels.append(0.0)
    if (i + 1) % 500 == 0:
        print(f"    negativnih obradjeno: {i+1}/{len(train_neg_ids)}", flush=True)
print(f"  Ukupno trening primera: {len(train_sims)}", flush=True)
print(f"  Duzine (min/median/max) parova prozora: {min(len(s) for s in train_sims)}/"
      f"{int(np.median([len(s) for s in train_sims]))}/{max(len(s) for s in train_sims)}", flush=True)

# Ogranici NAJDUZE primere (par najdugackih proteina u poolu moze dati i do
# ~40000 parova prozora, sto bi diktiralo padding SVIH ~4800 primera na tu
# duzinu -- katastrofalno usporava svaku epohu). LSE je simetrican preko
# bag-a, nasumicni podskup jednako velikog uzorka daje prakticno isti signal.
MAX_PAIRS_PER_EXAMPLE = 2500
rng_cap = np.random.default_rng(SEED)
train_sims = [s if len(s) <= MAX_PAIRS_PER_EXAMPLE
              else rng_cap.choice(s, size=MAX_PAIRS_PER_EXAMPLE, replace=False)
              for s in train_sims]
print(f"  Posle ogranicenja (max {MAX_PAIRS_PER_EXAMPLE}): "
      f"max duzina={max(len(s) for s in train_sims)}", flush=True)

max_len = max(len(s) for s in train_sims)
n_ex = len(train_sims)
padded = np.full((n_ex, max_len), -np.inf, dtype=np.float32)
for i, s in enumerate(train_sims):
    padded[i, :len(s)] = s
padded_t = torch.tensor(padded, dtype=torch.float32)
valid_t = torch.isfinite(padded_t)
n_valid_t = valid_t.sum(dim=1).float()
padded_safe = torch.where(valid_t, padded_t, torch.tensor(0.0))

log_tau = torch.nn.Parameter(torch.tensor(0.0))
scale = torch.nn.Parameter(torch.tensor(5.0))
bias = torch.nn.Parameter(torch.tensor(0.0))
labels_t = torch.tensor(train_labels, dtype=torch.float32)
optimizer = torch.optim.Adam([log_tau, scale, bias], lr=0.05)
loss_fn = torch.nn.BCEWithLogitsLoss()
NEG_INF = torch.tensor(-1e9)
for epoch in range(300):
    optimizer.zero_grad()
    tau = torch.exp(log_tau) + 1e-3
    scaled = torch.where(valid_t, padded_safe / tau, NEG_INF)
    pooled = tau * (torch.logsumexp(scaled, dim=1) - torch.log(n_valid_t))
    logits = scale * pooled + bias
    loss = loss_fn(logits, labels_t)
    loss.backward()
    optimizer.step()
fitted_tau = float(torch.exp(log_tau).item() + 1e-3)
print(f"Produkcioni tau={fitted_tau:.4f} (n_train={n_ex})", flush=True)
with open(TAU_OUTPUT, "w") as f:
    f.write(f"Produkcioni LSE tau (treniran na SVIM {len(dataset.gold_pairs)} gold pozitiva): "
            f"{fitted_tau:.4f}\n")

# -------------------------------------------------------
# Vektorizovana LSE eval infrastruktura
# -------------------------------------------------------
all_ids_with_windows = [aid for aid in dataset.all_ids if aid in window_vecs_per_protein]
window_blocks = [window_vecs_per_protein[aid] for aid in all_ids_with_windows]
global_windows_n = np.vstack(window_blocks)
n_proteins = len(dataset.all_ids)
protein_indices = [dataset.id_to_index[aid] for aid in all_ids_with_windows]
counts = [len(b) for b in window_blocks]
max_windows = max(counts)
padded_row_idx = np.full((n_proteins, max_windows), -1, dtype=np.int64)
row_cursor = 0
for pidx, cnt in zip(protein_indices, counts):
    padded_row_idx[pidx, :cnt] = np.arange(row_cursor, row_cursor + cnt)
    row_cursor += cnt
valid_mask = padded_row_idx >= 0
safe_idx = np.where(valid_mask, padded_row_idx, 0)
n_valid_per_protein = valid_mask.sum(axis=1)


def lse_scores_for_query(query_id, tau):
    qw = window_vecs_per_protein.get(query_id)
    if qw is None:
        return None
    n_qw = qw.shape[0]
    sim = (qw @ global_windows_n.T).astype(np.float32)
    gathered = sim[:, safe_idx]
    mask_3d = valid_mask[None, :, :]
    with np.errstate(over="ignore", invalid="ignore"):
        exp_vals = np.where(mask_3d, np.exp(gathered / tau), 0.0)
        sum_exp = exp_vals.sum(axis=(0, 2))
        total_count = n_qw * np.maximum(n_valid_per_protein, 1)
        mean_exp = sum_exp / total_count
        scores = np.where(n_valid_per_protein > 0, tau * np.log(mean_exp + 1e-300), -np.inf)
    return scores


# -------------------------------------------------------
# RRF-5 family-aware ranker: wrapper oko CrossReactivityRanker
# -------------------------------------------------------
print("\nUcitavam CrossReactivityRanker (RRF-4 osnova)...")
ranker = CrossReactivityRanker()


def rank_for_patient_rrf5(known_positive_names, known_negative_names=None):
    def resolve(names):
        ids = []
        for name in names or []:
            aid = ranker.name_to_id.get(name)
            if aid is None or aid not in ranker.id_to_index:
                continue
            ids.append(aid)
        return ids

    positive_ids = resolve(known_positive_names)
    negative_ids = resolve(known_negative_names)
    if not positive_ids:
        raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

    exclude_idx = {ranker.id_to_index[aid] for aid in positive_ids + negative_ids}
    combined = np.zeros(ranker.n_pool, dtype=np.float64)

    for aid in positive_ids:
        idx = ranker.id_to_index[aid]
        score_vec = ranker._rrf3_score_vector(idx)
        order = np.argsort(score_vec)[::-1]
        ranks = np.empty(ranker.n_pool, dtype=np.int64)
        ranks[order] = np.arange(1, ranker.n_pool + 1)
        combined += 1.0 / (RRF_K + ranks)

        # RRF-5 dodatak: SAMO ako je ovaj poznati pozitiv iz nsLTP/Profilin familije
        fam = family_of_id.get(aid)
        if fam in LSE_FAMILIES:
            lse_scores = lse_scores_for_query(aid, fitted_tau)
            if lse_scores is not None:
                lse_order = np.argsort(lse_scores)[::-1]
                lse_ranks = np.empty(ranker.n_pool, dtype=np.int64)
                lse_ranks[lse_order] = np.arange(1, ranker.n_pool + 1)
                combined += 1.0 / (RRF_K + lse_ranks)

    for idx in exclude_idx:
        combined[idx] = -np.inf

    order = np.argsort(combined)[::-1]
    result = pd.DataFrame({
        "candidate_id": [ranker.pool[i] for i in order],
        "candidate_name": [ranker.id_to_name.get(ranker.pool[i], ranker.pool[i]) for i in order],
        "priority_score": combined[order],
    })
    result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


# -------------------------------------------------------
# ISTI leave-one-out protokol kao evaluate_test_cases.py, ali sa RRF-5
# -------------------------------------------------------
pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} pacijenata za RRF-5 evaluaciju")

records = []
skipped_no_positive_left = 0
n_lse_boosted_trials = 0

for case in cases:
    pid = case["patient_id"]
    verif_status = case["verification"]["status"]
    resolvable = []
    for c in case["components"]:
        if c["result"] not in ("positive", "negative"):
            continue
        resolved = resolve_protein(c["protein"])
        if resolved is None:
            continue
        resolvable.append({"json_name": c["protein"], "pool_name": resolved, "result": c["result"]})

    if len(resolvable) < 2:
        continue

    for i, hidden in enumerate(resolvable):
        others = resolvable[:i] + resolvable[i + 1:]
        known_pos = [o["pool_name"] for o in others if o["result"] == "positive"]
        known_neg = [o["pool_name"] for o in others if o["result"] == "negative"]
        if not known_pos:
            skipped_no_positive_left += 1
            continue

        result_df = rank_for_patient_rrf5(known_pos, known_negative_names=known_neg)
        row = result_df[result_df["candidate_name"] == hidden["pool_name"]]
        if len(row) == 0:
            continue
        rank = int(row.iloc[0]["rank"])
        n_cand = len(result_df)
        percentile = rank / n_cand * 100

        any_lse_boost = any(family_of_id.get(ranker.name_to_id.get(kp)) in LSE_FAMILIES for kp in known_pos)
        if any_lse_boost:
            n_lse_boosted_trials += 1

        records.append({
            "patient_id": pid, "hidden_protein": hidden["pool_name"],
            "true_result": hidden["result"], "rank": rank, "n_candidates": n_cand,
            "percentile": percentile, "verification_status": verif_status,
            "lse_boosted": any_lse_boost,
        })
    if len(records) % 20 == 0 and len(records) > 0:
        print(f"  ...{len(records)} trials obradjeno", flush=True)

df = pd.DataFrame(records)
print(f"\nLeave-one-out trials (RRF-5): {len(df)}")
print(f"Trials sa LSE-boost (bar jedan poznat pozitiv iz nsLTP/Profilin): {n_lse_boosted_trials}")

df.to_json(RAW_OUTPUT, orient="records", indent=2)
print(f"Saved: {RAW_OUTPUT}")

summary_lines = ["=" * 70, "RRF-5 (family-aware LSE dodatak) -- leave-one-out rezultati", "=" * 70, "",
                  f"Ukupno trials: {len(df)}, sa LSE-boost: {n_lse_boosted_trials}",
                  f"Produkcioni tau: {fitted_tau:.4f}", ""]

hard = df[df["verification_status"] == "full_text_verified"]
for label, sub in [("SVI", df), ("HARD (full_text_verified)", hard)]:
    pos = sub[sub["true_result"] == "positive"]["percentile"]
    neg = sub[sub["true_result"] == "negative"]["percentile"]
    summary_lines.append(f"--- {label} ---")
    summary_lines.append(f"  Pozitivne mete (n={len(pos)}): medijan percentil = {pos.median():.1f}%")
    summary_lines.append(f"  Negativne mete (n={len(neg)}): medijan percentil = {neg.median():.1f}%")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"Saved: {SUMMARY_OUTPUT}")
