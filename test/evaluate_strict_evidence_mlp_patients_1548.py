"""
Opcija 1 iz korisnicke diskusije (2026-09-02): umesto tezinskog obaranja
"Suspected*" parova (test/evaluate_weighted_evidence_mlp_patients_1548.py --
NEGATIVAN rezultat, weighted bio GORI od baseline-a), ovde ih potpuno
IZBACUJEMO iz treninga -- trening SAMO na "Strong evidence*" i "Confirmed*"
tier-ovima (spojeno u jedan strogi skup, korisnicki predlog "mozda i strong
i confirmed da spojim"). Manji trening skup (~514 vs ~825 training_eligible
parova) -- direktan test da li je manji-ali-cistiji skup bolji od
veceg-ali-mesovitog, suprotan pravac od weighted eksperimenta.

BLAST i baseline MLP(hadamard)-650M REKORISCENI iz vec postojeceg
output/weighted_evidence_mlp_1548_per_trial.csv (isti 68-pacijentski
leave-one-out, ista disciplina reuse-a svuda u sesiji) -- trenira se i
evaluira SAMO novi "strict" model.

Izlaz:
    output/strict_evidence_mlp_1548_per_trial.csv
    output/strict_evidence_mlp_1548_summary.txt
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
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402
from ml.patient_ranking_1548 import CrossReactivityRanker, RRF_K  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
EXISTING_BASELINE = Path("/home/lana/ALERGRAF/output/weighted_evidence_mlp_1548_per_trial.csv")
PER_TRIAL_OUTPUT = Path("/home/lana/ALERGRAF/output/strict_evidence_mlp_1548_per_trial.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/strict_evidence_mlp_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
N_PERM = 10000
N_BOOTSTRAP = 10000

CROWDED_FAMILIES = {"nsLTP", "profilin", "PR-10"}

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset (ESM-2 650M)...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)

strict_pairs = [p for p in dataset.gold_pairs
                if str(p.get("evidence_level", "")).startswith("Strong evidence")
                or str(p.get("evidence_level", "")).startswith("Confirmed")]
print(f"  Strict trening skup (Strong evidence* + Confirmed* SAMO): {len(strict_pairs)} "
      f"(vs ~825 training_eligible_pairs baseline)", flush=True)

train_negatives = sample_negative_pairs(dataset.all_ids, len(strict_pairs) * NEG_PER_POS, SEED,
                                          dataset.positive_pair_set)

print("\nTreniram STRICT MLP(hadamard)-650M...", flush=True)
mlp_strict = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED)
mlp_strict.fit(strict_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
print(f"  Gotovo. best_val_auc={mlp_strict.best_val_auc:.4f}", flush=True)

ranker = CrossReactivityRanker()
assert set(dataset.all_ids) == set(ranker.pool), "Razlicit skup proteina!"
perm_dataset_to_ranker = np.array([dataset.id_to_index[pid] for pid in ranker.pool])


def mlp_score_fn(aid):
    return mlp_strict.score_all(aid)[perm_dataset_to_ranker]


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
        scores = mlp_score_fn(aid)
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


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"\nUcitano {len(cases)} pacijenata", flush=True)

records = []
for case in cases:
    pid = case["patient_id"]
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
        records.append({"patient_id": pid, "hidden_protein": hidden["pool_name"], "true_result": hidden["result"],
                         "rank": rank, "n_candidates": n_cand, "percentile": rank / n_cand * 100})

df_strict = pd.DataFrame(records)
df_strict["rr"] = 1.0 / df_strict["rank"]
print(f"\n{len(df_strict)} trials (strict)", flush=True)

# ---------------------------------------------------------------------------
# Spoji sa POSTOJECIM baseline/blast rezultatima (weighted_evidence_mlp_1548_per_trial.csv)
# -- ne racunaju se ponovo.
# ---------------------------------------------------------------------------
existing = pd.read_csv(EXISTING_BASELINE)
merged = df_strict[["patient_id", "hidden_protein", "true_result", "rr", "percentile"]].rename(
    columns={"rr": "rr_strict", "percentile": "percentile_strict"})
merged = merged.merge(
    existing[["patient_id", "hidden_protein", "verification_status", "rr_baseline", "percentile_baseline",
              "rr_blast", "percentile_blast", "organism", "protein_family", "family_crowding"]],
    on=["patient_id", "hidden_protein"])
assert len(merged) == len(df_strict), "Spajanje sa postojecim baseline/blast nije 1:1 -- proveri kljuceve"
merged.to_csv(PER_TRIAL_OUTPUT, index=False)

# ---------------------------------------------------------------------------
# Per-family breakdown
# ---------------------------------------------------------------------------
summary_lines = ["=" * 100,
                  f"STRICT-evidence MLP(hadamard)-650M (Strong evidence*+Confirmed* SAMO, n_train={len(strict_pairs)}) "
                  f"vs baseline vs BLAST -- svih {len(cases)} pacijenata ({len(merged)} trial-ova)",
                  "=" * 100, "",
                  f"best_val_auc: strict={mlp_strict.best_val_auc:.4f}", ""]

for fam in sorted(merged["protein_family"].dropna().unique()):
    sub = merged[merged["protein_family"] == fam]
    tag = " [CROWDED]" if fam in CROWDED_FAMILIES else ""
    line = [f"--- {fam}{tag} (n={len(sub)}) ---"]
    for res in ("positive", "negative"):
        rsub = sub[sub["true_result"] == res]
        if len(rsub) == 0:
            continue
        line.append(f"  {res:8s} (n={len(rsub):3d}): baseline={rsub['percentile_baseline'].median():5.1f}%  "
                     f"strict={rsub['percentile_strict'].median():5.1f}%  "
                     f"blast={rsub['percentile_blast'].median():5.1f}%")
    summary_lines.extend(line)
summary_lines.append("")


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


for comp_label, col_a, name_a, col_b, name_b in [
    ("Strict vs Baseline (SVI trial-ovi)", "rr_strict", "strict", "rr_baseline", "baseline"),
    ("Strict vs BLAST (SVI trial-ovi)", "rr_strict", "strict", "rr_blast", "blast"),
]:
    summary_lines.append(f"### {comp_label} ###")
    summary_lines.extend(run_all_tests(merged, comp_label, col_a, col_b, name_a, name_b))

crowded = merged[merged["family_crowding"]]
summary_lines.append(f"### Strict vs Baseline, SAMO crowded familije (n={len(crowded)}) ###")
summary_lines.extend(run_all_tests(crowded, "crowded", "rr_strict", "rr_baseline", "strict", "baseline"))

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {PER_TRIAL_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
