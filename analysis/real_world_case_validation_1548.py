"""
Spoljna (van gold dataseta) validacija RRF-3 algoritma na stvarnim pacijent-
panelima iz literature -- ne pojedinacni citirani parovi, nego CELE tabele
onoga na sta je jedan stvaran pacijent testiran (pozitivno i negativno), da
bismo proverili: kad se algoritmu da 1 poznat pozitivan protein, da li
rangira ostatak TE ISTE tabele visoko (pozitivne) i nisko (negativne)?

VAZNO -- proverio sam svaki koristeni par protiv output/cross_reactive_1548.csv
da ne bude cirkularno (da par nije vec deo gold dataseta na kom je RRF-3
razvijen). Fel d 2 <-> Sus s 1 (pork-cat sindrom) JE vec u gold datasetu kao
"Confirmed" i NIJE ovde koriscen kao nov test.

Slucaj 1 (Limao R, Bartolome B, Cabral Duarte F. "The relevance of oral food
challenge in a patient allergic to peanut and tree nuts." Asia Pacific
Allergy. 2023;13(3):132-134.):
  12-godisnji decak, potvrdjeno oralnim izazovom (OFC):
    - kikiriki (Ara h 2) POZITIVAN
    - kesju (Ana o 1, Ana o 2) POZITIVAN
    - pistaci (nepoznat tacan protein u izvoru) POZITIVAN
    - orah (Jug r) NEGATIVAN
  NAPOMENA: Ara h 2 <-> Jug r 1 je vec u gold datasetu kao "Inferred"
  (populaciono-nivo pretpostavka) -- ovaj stvaran pacijent je NEGATIVAN,
  vredan podatak za diskusiju, ne kontradiktornost (individualna varijacija
  je ocekivana, "Inferred" ne znaci "svaki pacijent reaguje").

Slucaj 2 (Pereira C et al. "Specific sublingual immunotherapy with peach LTP
(Pru p 3). One year treatment: a case report." Cases Journal. 2009;2:6553.):
  40-godisnja pacijentkinja, potvrdjeno DBPCFC:
    - breskva/Pru p 3 (nsLTP) POZITIVAN
    - Pru p 4 (profilin, ISTO voce) NEGATIVAN specificni IgE
  Slabiji dokaz (isti plod, ne cross-food), ali koristan kao negativna
  kontrola unutar iste namirnice.

Izlaz:
    output/real_world_case_validation_1548_summary.txt
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
FROZEN_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/real_world_case_validation_1548_summary.txt")

RRF_K = 60

allergens = pd.read_csv(CLEAN_ALLERGENS)
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id

with open(FROZEN_EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}
with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)

pool = sorted(embeddings_dict.keys())
id_to_index = {aid: i for i, aid in enumerate(pool)}
n_pool = len(pool)
print(f"Candidate pool: {n_pool} proteins")

embedding_matrix = np.array([embeddings_dict[aid] for aid in pool], dtype=np.float64)
cosine_matrix = cosine_similarity(embedding_matrix)

perm = np.array([blast_id_to_index.get(aid, -1) for aid in pool])
valid = perm >= 0
blast_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

foldseek_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in id_to_index and b in id_to_index:
        i, j = id_to_index[a], id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score


def rrf_rank_all(query_name):
    """Given a query official_name, return a DataFrame of ALL candidates ranked by RRF-3."""
    qid = name_to_id.get(query_name)
    if qid is None or qid not in id_to_index:
        return None
    qidx = id_to_index[qid]

    def ranks_from_scores(scores, self_index):
        s = scores.astype(np.float64, copy=True)
        s[self_index] = -np.inf
        order = np.argsort(s)[::-1]
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[order] = np.arange(1, len(s) + 1)
        return ranks

    cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
    blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
    fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)
    rrf_scores = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)
    rrf_ranks = ranks_from_scores(rrf_scores, qidx)

    id_to_name = {v: k for k, v in name_to_id.items()}
    return pd.DataFrame({
        "candidate_id": pool,
        "candidate_name": [id_to_name.get(aid, aid) for aid in pool],
        "rrf_rank": rrf_ranks,
    }).sort_values("rrf_rank")


def report_targets(query_name, targets_by_prefix, label, expect):
    """targets_by_prefix: list of official_name PREFIXES (e.g. 'Ana o 1' matches all isoforms)."""
    ranked = rrf_rank_all(query_name)
    lines = [f"  Query: {query_name}  (izlaz: {label}, ocekivano: {expect})"]
    if ranked is None:
        lines.append("    [GRESKA: query nije nadjen u embedding pool-u]")
        return lines
    for prefix in targets_by_prefix:
        matches = ranked[ranked["candidate_name"].str.startswith(prefix)]
        if len(matches) == 0:
            lines.append(f"    {prefix}: [nije nadjen u pool-u]")
            continue
        best = matches.sort_values("rrf_rank").iloc[0]
        pct = best["rrf_rank"] / n_pool * 100
        lines.append(f"    {prefix} (najbolji od {len(matches)} izoformi): "
                      f"rang {int(best['rrf_rank'])}/{n_pool} (top {pct:.1f}%) -- {best['candidate_name']}")
    return lines


summary_lines = ["=" * 70,
                  "Spoljna validacija: stvarni pacijent-paneli iz literature (van gold dataseta)",
                  "=" * 70, ""]

summary_lines.append("--- Slucaj 1: Limao et al. 2023, Asia Pacific Allergy 13(3):132-134 ---")
summary_lines.append("12g decak, kikiriki/kesju/pistaci alergija potvrdjena OFC, orah OFC-negativan")
summary_lines += report_targets("Ara h 2.0101", ["Ana o 1", "Ana o 2"], "POZITIVNO (OFC potvrdjeno)", "visok rang")
summary_lines += report_targets("Ara h 2.0101", ["Pis v"], "POZITIVNO (OFC potvrdjeno, tacan protein nepoznat)", "visok rang")
summary_lines += report_targets("Ara h 2.0101", ["Jug r"], "NEGATIVNO (OFC negativan)", "nizak rang (ali VEC 'Inferred' u gold datasetu -- diskusija, ne cist test)")
summary_lines.append("")

summary_lines.append("--- Slucaj 2: Pereira et al. 2009, Cases Journal 2:6553 (slabiji dokaz, isti plod) ---")
summary_lines.append("40g pacijentkinja, breskva/Pru p 3 potvrdjeno DBPCFC, Pru p 4 (profilin) sIgE-negativan")
summary_lines += report_targets("Pru p 3.0101", ["Pru p 4"], "NEGATIVNO (sIgE negativan)", "nizak rang")
summary_lines.append("")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
