"""
Nedostajuci deo slike: cosine-SAM (bez treninga, bez BLAST-a) kao pacijentski
single-signal ranker -- vec ranije flagovano (2026-08-30 sesija) kao "not yet
done", nikad izvrseno. Isti leave-one-out mehanizam kao test/evaluate_mlp_
only_vs_blast_only_patients_1548.py, ali BEZ treninga -- skor je prosta
cosine slicnost ESM-2 650M embeddinga (isti prostor kao MLP/BLAST poredjenje).

Racuna se SAMO na originalnih 57 pacijenata (test_cases.json pre Giuffrida
prosirenja) da bi bilo direktno uporedivo sa postojecim evaluation_results_
raw_blastonly.json / _mlponly.json (koji NISU regenerisani na 68) -- filtrira
patient_id skup da odgovara tacno onim koji se pojavljuju u tim fajlovima.

Izlaz:
    test/evaluation_results_raw_cosineonly.json
    output/evaluate_cosine_only_patients_1548_summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/test")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
EXISTING_BLAST = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_blastonly.json")
EXISTING_MLP = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_mlponly.json")
RAW_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_results_raw_cosineonly.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/evaluate_cosine_only_patients_1548_summary.txt")

SEED = 42
N_PERM = 10000
N_BOOTSTRAP = 10000

print("Loading dataset (ESM-2 650M)...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)

ranker = CrossReactivityRanker()
assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])

emb = dataset.embedding_matrix[perm_dataset_to_ranker]
norms = np.linalg.norm(emb, axis=1, keepdims=True)
emb_normed = emb / np.clip(norms, 1e-12, None)


def cosine_score_fn(aid):
    q = emb_normed[ranker.id_to_index[aid]]
    return emb_normed @ q


def rank_for_patient(known_positive_names, known_negative_names=None):
    def resolve(names):
        ids = []
        for name in names or []:
            a = ranker.name_to_id.get(name)
            if a is None or a not in ranker.id_to_index:
                continue
            ids.append(a)
        return ids

    positive_ids = resolve(known_positive_names)
    negative_ids = resolve(known_negative_names)
    if not positive_ids:
        raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

    exclude_idx = {ranker.id_to_index[aid] for aid in positive_ids + negative_ids}
    combined = np.zeros(ranker.n_pool, dtype=np.float64)

    for aid in positive_ids:
        scores = cosine_score_fn(aid)
        order = np.argsort(scores)[::-1]
        ranks = np.empty(ranker.n_pool, dtype=np.int64)
        ranks[order] = np.arange(1, ranker.n_pool + 1)
        combined += 1.0 / (RRF_K + ranks)

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


pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


existing_blast = pd.read_json(EXISTING_BLAST)
existing_mlp = pd.read_json(EXISTING_MLP)
original_57_patients = set(existing_blast["patient_id"].unique())
print(f"Originalnih 57-pacijentski skup (iz postojecih raw fajlova): {len(original_57_patients)} pacijenata",
      flush=True)

with open(TEST_CASES) as f:
    all_cases = json.load(f)
cases = [c for c in all_cases if c["patient_id"] in original_57_patients]
print(f"Filtrirano na {len(cases)} pacijenata (originalnih 57, bez Giuffrida)", flush=True)

records = []
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
            continue
        result_df = rank_for_patient(known_pos, known_negative_names=known_neg)
        row = result_df[result_df["candidate_name"] == hidden["pool_name"]]
        if len(row) == 0:
            continue
        rank = int(row.iloc[0]["rank"])
        n_cand = len(result_df)
        records.append({
            "patient_id": pid, "hidden_protein": hidden["pool_name"], "true_result": hidden["result"],
            "rank": rank, "n_candidates": n_cand, "percentile": rank / n_cand * 100,
            "verification_status": verif_status,
        })

df_cosine = pd.DataFrame(records)
df_cosine.to_json(RAW_OUTPUT, orient="records", indent=2)
print(f"\n{len(df_cosine)} trials (cosine-only)", flush=True)

# ---------------------------------------------------------------------------
# Upareni testovi: cosine vs BLAST, cosine vs MLP(hadamard) -- ista metodologija
# kao test/paired_test_mlp_vs_blast_1548.py.
# ---------------------------------------------------------------------------
df_cosine["rr"] = 1.0 / df_cosine["rank"]
existing_blast["rr"] = 1.0 / existing_blast["rank"]
existing_mlp["rr"] = 1.0 / existing_mlp["rank"]

summary_lines = ["=" * 90, "Cosine SAM vs BLAST SAM vs MLP(hadamard) SAM -- originalnih 57 pacijenata",
                  "=" * 90, ""]


def run_all_tests(sub, label, col_a, col_b, name_a, name_b):
    lines = [f"--- {label} (n={len(sub)} upita, {sub['patient_id'].nunique()} pacijenata) ---"]
    per_patient = sub.groupby("patient_id").agg(mrr_a=(col_a, "mean"), mrr_b=(col_b, "mean"))
    diffs = per_patient["mrr_a"] - per_patient["mrr_b"]
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) >= 5:
        stat, pval = wilcoxon(diffs_nonzero)
        lines.append(f"  1) Patient-level Wilcoxon (MRR_{name_a}-MRR_{name_b}, n={len(diffs_nonzero)}): "
                      f"mean diff={diffs.mean():+.4f}, p={pval:.4f} "
                      f"-- {'ZNACAJNO' if pval < 0.05 else 'nije znacajno'}")
    else:
        lines.append(f"  1) Patient-level Wilcoxon: n={len(diffs_nonzero)} < 5, nepouzdan")

    rng = np.random.default_rng(SEED)
    observed = (sub[col_a] - sub[col_b]).mean()
    sub_by_patient = {pid: g[[col_a, col_b]].to_numpy() for pid, g in sub.groupby("patient_id")}
    perm_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        total, n = 0.0, 0
        for pid, arr in sub_by_patient.items():
            flip = rng.random() < 0.5
            a = arr if not flip else arr[:, ::-1]
            total += (a[:, 0] - a[:, 1]).sum()
            n += len(a)
        perm_diffs[i] = total / n
    p_perm = (np.abs(perm_diffs) >= np.abs(observed)).mean()
    lines.append(f"  2) Cluster-permutacija (N={N_PERM}): observed mean(rr_{name_a}-rr_{name_b})={observed:+.4f}, "
                  f"p={p_perm:.4f} -- {'ZNACAJNO' if p_perm < 0.05 else 'nije znacajno'}")

    rng2 = np.random.default_rng(SEED)
    patient_ids = sub["patient_id"].unique()
    boot_diffs = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng2.choice(patient_ids, size=len(patient_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on="patient_id", right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled[col_a], weights=w) - np.average(resampled[col_b], weights=w)
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    sig_boot = (ci_lo > 0) or (ci_hi < 0)
    lines.append(f"  3) Patient-level bootstrap (N={N_BOOTSTRAP}): mean diff={boot_diffs.mean():+.4f}, "
                  f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {'ZNACAJNO' if sig_boot else 'nije znacajno'}")
    lines.append("")
    return lines


for comp_label, other_df, name_other in [("Cosine vs BLAST", existing_blast, "blast"),
                                           ("Cosine vs MLP(hadamard)", existing_mlp, "mlp")]:
    merged = df_cosine[["patient_id", "hidden_protein", "true_result", "verification_status", "rr"]].merge(
        other_df[["patient_id", "hidden_protein", "rr"]],
        on=["patient_id", "hidden_protein"], suffixes=("_cosine", f"_{name_other}"), how="inner")
    n_dropped = len(df_cosine) - len(merged)
    if n_dropped:
        summary_lines.append(
            f"NAPOMENA: {n_dropped} cosine trial-ova ({comp_label}) izbaceno iz poredjenja -- nema "
            f"para u starijem raw fajlu ({name_other}). Uzrok: protein_resolution.py popravka od "
            f"2026-09-02 (Pen a1/Pen m1 aliasi) retroaktivno otkljucava dodatne probe kod "
            f"uklejasokolowska2021 pacijenata koje NISU postojale kad je stariji raw fajl racunat -- "
            f"ne bug u ovom skriptu, ocekivana posledica. Poredjenje ide na presek (n={len(merged)}), "
            f"identican skup kao vec objavljeni BLAST/MLP brojevi.")

    summary_lines.append(f"### {comp_label} ###")
    summary_lines.append(f"Ukupno uparenih upita: {len(merged)}, pacijenata: {merged['patient_id'].nunique()}")
    summary_lines.append("")
    for label, sub in [("SVI upiti", merged),
                        ("SAMO hard (full_text_verified)",
                         merged[merged["verification_status"] == "full_text_verified"])]:
        summary_lines.extend(run_all_tests(sub, label, "rr_cosine", f"rr_{name_other}", "cosine", name_other))
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {RAW_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
