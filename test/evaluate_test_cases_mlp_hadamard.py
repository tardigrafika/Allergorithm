"""
Real-world test_cases.json (41 pacijenata) evaluacija za MLP(hadamard) --
isti leave-one-out protokol kao test/evaluate_test_cases.py (koji koristi
produkcioni RRF-3/RRF-4 signal), samo sa MLP(hadamard) kao osnovnim
signalom umesto RRF-a.

Zasto: LOCO (ml/loco_mlp_hadamard_1548.py) je pokazao da je MLP(hadamard)
jedini "Hadamard-porodica" model koji preziveo rigorozniju validaciju --
micro MRR TACNO izjednacen sa cosine-om (0.1209=0.1209), za razliku od
cistog Hadamard bilinear-a koji je ispao znacajno GORI. Ovo testira da li
ta interna (gold-dataset) slika drzi i na 41 real-world pacijentu iz
literature -- potpuno odvojen test od LOCO-a.

MLP(hadamard) se trenira JEDNOM na CELOM gold datasetu (produkciona logika,
ne LOCO fold) -- isto kao sto RRF-3/RRF-4 signal (cosine/BLAST/Foldseek) nije
"treniran" po foldovima nego racunat na celom poolu; ovo je fer poredjenje
istog "produkcionog rezima".

Visestruki poznati pozitivi kombinuju se REKUENDIM RANK FUSION-om (isti
mehanizam kao CrossReactivityRanker.rank_for_patient) da bi metodologija
ostala uporediva sa RRF rezultatom -- razlika je SAMO osnovni signal
(MLP score_all() umesto RRF-3 rank vektora).

Izlaz:
    test/evaluation_results_raw_mlp_hadamard.json
    test/evaluation_mlp_hadamard_summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/test")  # "test" kolidira sa Python stdlib paketom, uvozimo direktno iz fajla
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
RAW_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlp_hadamard.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_mlp_hadamard_summary.txt")

SEED = 42
NEG_PER_POS = 10
RRF_K = 20  # ista konstanta kao patient_ranking_1548.py, radi uporedivosti
N_PERM = 10000

# =====================================================
# TRENIRAJ MLP(hadamard) NA CELOM GOLD DATASETU (produkciona logika)
# =====================================================

print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
n_neg = len(dataset.gold_pairs) * NEG_PER_POS
negatives = sample_negative_pairs(set(dataset.all_ids), n_neg, SEED, dataset.positive_pair_set)

print(f"Trening MLP(hadamard) na CELOM gold datasetu ({len(dataset.gold_pairs)} pozitivnih parova)...")
MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)
clf = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED)
clf.fit(dataset.gold_pairs, negatives, dataset.embedding_matrix, dataset.id_to_index)
print(f"Trening gotov (stopped_epoch={clf.stopped_epoch})")

pool = dataset.all_ids
id_to_index = dataset.id_to_index
name_to_id = dataset.name_to_id
id_to_name = {v: k for k, v in name_to_id.items()}
n_pool = len(pool)


def mlp_score_vector(known_id):
    """MLP(hadamard) skor known_id-a naspram CELOG pool-a, kao rang vektor
    (visi rang = jaci kandidat) -- isti oblik kao CrossReactivityRanker._rrf3_score_vector,
    da fuzija preko vise poznatih pozitiva ostane metodoloski uporediva."""
    scores = clf.score_all(known_id)
    s = scores.astype(np.float64).copy()
    s[id_to_index[known_id]] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def rank_for_patient_mlp(known_positive_names, known_negative_names=None):
    def resolve(names):
        ids = []
        for name in names or []:
            aid = name_to_id.get(name)
            if aid is not None and aid in id_to_index:
                ids.append(aid)
        return ids

    positive_ids = resolve(known_positive_names)
    negative_ids = resolve(known_negative_names)
    if not positive_ids:
        raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

    exclude_idx = {id_to_index[aid] for aid in positive_ids + negative_ids}
    combined = np.zeros(n_pool, dtype=np.float64)
    for aid in positive_ids:
        ranks = mlp_score_vector(aid)
        combined += 1.0 / (RRF_K + ranks)
    for idx in exclude_idx:
        combined[idx] = -np.inf

    order = np.argsort(combined)[::-1]
    result = pd.DataFrame({
        "candidate_id": [pool[i] for i in order],
        "candidate_name": [id_to_name.get(pool[i], pool[i]) for i in order],
        "priority_score": combined[order],
    })
    result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


# =====================================================
# LEAVE-ONE-OUT preko test_cases.json (identican protokol kao evaluate_test_cases.py)
# =====================================================

pool_names = sorted(name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} pacijenata")

records = []
skipped_no_positive_left = 0
unresolved_components = set()

for case in cases:
    pid = case["patient_id"]
    verif_status = case["verification"]["status"]
    resolvable = []
    for c in case["components"]:
        if c["result"] not in ("positive", "negative"):
            continue
        resolved = resolve_protein(c["protein"])
        if resolved is None:
            unresolved_components.add(c["protein"])
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

        result_df = rank_for_patient_mlp(known_pos, known_negative_names=known_neg)
        row = result_df[result_df["candidate_name"] == hidden["pool_name"]]
        if len(row) == 0:
            continue
        rank = int(row.iloc[0]["rank"])
        n_cand = len(result_df)
        percentile = rank / n_cand * 100

        records.append({
            "patient_id": pid, "hidden_protein": hidden["pool_name"],
            "true_result": hidden["result"], "rank": rank, "n_candidates": n_cand,
            "percentile": percentile, "verification_status": verif_status,
            "n_known_positive": len(known_pos), "n_known_negative": len(known_neg),
        })

df = pd.DataFrame(records)
print(f"Leave-one-out trials: {len(df)}")
print(f"Preskoceno (nema known pozitivan posle sakrivanja): {skipped_no_positive_left}")
df.to_json(RAW_OUTPUT, orient="records", indent=2)

# =====================================================
# STATISTIKA -- ISTA ispravljena metodologija kao evaluate_test_cases_stats_corrected.py
# =====================================================


def cluster_permutation_test(sub_df, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    pos_mask = sub_df["true_result"] == "positive"
    neg_mask = sub_df["true_result"] == "negative"
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None
    observed = sub_df.loc[neg_mask, "percentile"].mean() - sub_df.loc[pos_mask, "percentile"].mean()

    labels = sub_df["true_result"].to_numpy().copy()
    patient_ids = sub_df["patient_id"].to_numpy()
    percentiles = sub_df["percentile"].to_numpy()

    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = labels.copy()
        for pid in np.unique(patient_ids):
            idx = np.where(patient_ids == pid)[0]
            perm_labels[idx] = rng.permutation(labels[idx])
        p_mask = perm_labels == "positive"
        n_mask = perm_labels == "negative"
        if p_mask.sum() == 0 or n_mask.sum() == 0:
            perm_stats[i] = np.nan
            continue
        perm_stats[i] = percentiles[n_mask].mean() - percentiles[p_mask].mean()

    valid_perm = perm_stats[~np.isnan(perm_stats)]
    p_value = (valid_perm >= observed).mean()
    return observed, p_value, len(valid_perm)


def patient_level_wilcoxon(sub_df):
    per_patient_diff = []
    for pid, g in sub_df.groupby("patient_id"):
        pos = g.loc[g["true_result"] == "positive", "percentile"]
        neg = g.loc[g["true_result"] == "negative", "percentile"]
        if len(pos) == 0 or len(neg) == 0:
            continue
        per_patient_diff.append(neg.median() - pos.median())
    per_patient_diff = np.array(per_patient_diff)
    if len(per_patient_diff) < 5:
        return per_patient_diff, None
    stat, pval = wilcoxon(per_patient_diff, alternative="greater")
    return per_patient_diff, pval


summary_lines = ["=" * 70, "MLP(hadamard) na 41 real-world pacijenta -- leave-one-out", "=" * 70, "",
                  f"Ukupno trials: {len(df)}  Preskoceno (monospecifican): {skipped_no_positive_left}",
                  f"Nerezolvovane komponente: {sorted(unresolved_components)}", ""]

for label, sub in [
    ("SVI trials (hard + soft)", df),
    ("SAMO hard (full_text_verified)", df[df["verification_status"] == "full_text_verified"]),
]:
    n_patients = sub["patient_id"].nunique()
    pos = sub[sub["true_result"] == "positive"]["percentile"]
    neg = sub[sub["true_result"] == "negative"]["percentile"]
    summary_lines.append(f"--- {label} ---")
    summary_lines.append(f"  n_trials={len(sub)} (pos={len(pos)}, neg={len(neg)}), n_pacijenata={n_patients}")
    summary_lines.append(f"  Pozitivne mete: medijan percentil = {pos.median():.1f}% (zelimo NIZAK)")
    summary_lines.append(f"  Negativne mete: medijan percentil = {neg.median():.1f}% (zelimo VISOK)")

    perm_result = cluster_permutation_test(sub)
    if perm_result is None:
        summary_lines.append("  Cluster-permutacija: nedovoljno podataka")
    else:
        observed, pval, n_valid = perm_result
        sig = "ZNACAJNO" if pval < 0.05 else "nije znacajno"
        summary_lines.append(f"  Cluster-permutacija (N={N_PERM}): observed diff={observed:+.2f}pp, "
                              f"p={pval:.4f} -- {sig}")

    per_patient_diff, wpval = patient_level_wilcoxon(sub)
    summary_lines.append(f"  Patient-level parovi (>=1 poz I >=1 neg): n={len(per_patient_diff)}")
    if wpval is not None:
        sig = "ZNACAJNO" if wpval < 0.05 else "nije znacajno"
        summary_lines.append(f"  Wilcoxon signed-rank: p={wpval:.4f} -- {sig}")
    else:
        summary_lines.append("  Wilcoxon: n<5, nije izvrsen")
    summary_lines.append("")

summary_lines.append("=" * 70)
summary_lines.append("POREDJENJE sa produkcionim RRF signalom (test/evaluation_stats_corrected_summary.txt)")
summary_lines.append("=" * 70)
summary_lines.append("Vidi taj fajl za RRF-3 brojeve na (mogucem drugacijem broju) pacijenata -- ")
summary_lines.append("uporedi medijan percentil pozitivnih/negativnih i p-vrednosti iznad.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {RAW_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
