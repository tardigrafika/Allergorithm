"""
LOCO POTVRDA za retunovane MLP(hadamard) config-e na ESM-2 3B embeddinzima
-- nastavak ml/loco_esm2_3b_vs_esm2_650m_1548.py, koji je koristio ISTE
hiperparametre kao 650M (standardize=False, h32) BEZ retuninga i dobio
katastrofalan kolaps (MRR=0.0395). Korisnica je ispravno primetila da to
nije fer poredjenje. Brzi single-split sweep (analysis/mlp_hadamard_esm2_3b_
sensitivity_1548.py) pokazao da standardize=True (ili 10x manji LR) potpuno
resava kolaps -- ovaj skript POTVRDJUJE dva najbolja kandidata iz tog
sweep-a pravim 40-fold LOCO-om (single-split se NIKAD ne veruje direktno,
ista disciplina kao svuda u projektu).

Dva config-a:
  1. h32_stdTrue -- MINIMALNA izmena, SAMO standardize=False->True, hidden_dims
     identican 650M config-u (32) -- najcistije izolovana promena, bez
     dodatnog confound-a od menjanja i sirine sloja.
  2. h64_stdTrue -- najbolji MRR u screening sweep-u (sirina + standardize
     zajedno promenjeni).

BLAST i ESM-2 650M MLP strana PONOVO KORISCENI (ista optimizacija/disciplina
kao ESM-1b i 3B-original eksperimenti -- deterministicki identicno vec
potvrdjenim brojevima).

Preduslov:
    output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv (vec postoji)

Izlaz:
    output/loco_esm2_3b_retuned_mlp_hadamard_1548_per_query.csv
    output/loco_esm2_3b_retuned_mlp_hadamard_1548_summary.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.pkl")
METADATA_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
EXISTING_650M_PER_QUERY = Path("/home/lana/ALERGRAF/output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm2_3b_retuned_mlp_hadamard_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm2_3b_retuned_mlp_hadamard_1548_summary.txt")

for f in (EMBEDDINGS_3B, METADATA_3B, EXISTING_650M_PER_QUERY):
    if not f.exists():
        raise FileNotFoundError(f"{f} ne postoji.")

SEED = 42
NEG_PER_POS = 10
N_BOOTSTRAP = 2000

BASE = dict(input_encoding="hadamard", dropout=[0.3], learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3,
             batch_size=64, max_epochs=300, patience=20, val_fraction=0.15)
CONFIGS = [
    ("esm2_3b_h32_stdTrue", {**BASE, "hidden_dims": [32], "standardize": True}),
    ("esm2_3b_h64_stdTrue", {**BASE, "hidden_dims": [64], "standardize": True}),
]

print("Loading ESM-2 3B dataset...", flush=True)
dataset_3b = load_dataset(EMBEDDINGS_3B, METADATA_3B, GOLD)

folds = loco_folds(dataset_3b.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}", flush=True)


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


existing_650m = pd.read_csv(EXISTING_650M_PER_QUERY)
existing_650m["direction"] = existing_650m.groupby(["fold", "pair_id"]).cumcount()
blast_lookup = {(r.fold, r.pair_id, r.direction): (r.blast_rank, r.blast_rr) for r in existing_650m.itertuples()}
print(f"Ponovo koriscen BLAST/650M rezultat: {len(existing_650m)} redova "
      f"(650M MLP MRR micro={existing_650m['mlp_rr'].mean():.4f} -- treba da se poklopi sa 0.1259)", flush=True)

all_records = []
overall_start = time.time()

for config_name, mlp_params in CONFIGS:
    print(f"\n{'='*70}\nCONFIG = {config_name}\n{'='*70}", flush=True)
    config_start = time.time()

    for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
        train_pairs_clean = training_eligible_pairs(train_pairs)
        train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
        train_ids |= {pid for pid in dataset_3b.all_ids if pid not in test_ids and pid not in train_ids}
        n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
        train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED + fold_idx,
                                                  dataset_3b.positive_pair_set)

        if len(train_pairs_clean) < 5:
            mlp = None
        else:
            mlp = MLPPairClassifier(params=mlp_params, seed=SEED + fold_idx)
            mlp.fit(train_pairs_clean, train_negatives, dataset_3b.embedding_matrix, dataset_3b.id_to_index)

        for p in test_pairs:
            for direction, (query_id, target_id) in enumerate([(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]):
                blast_rank, blast_rr = blast_lookup.get((fold_idx, p["pair_id"], direction), (None, None))
                qi = dataset_3b.id_to_index[query_id]
                ti = dataset_3b.id_to_index[target_id]

                if mlp is not None:
                    mlp_scores = mlp.score_all(query_id)
                    mlp_rank = ranks_from_scores(mlp_scores, qi)
                    mlp_final_rank = int(mlp_rank[ti])
                else:
                    mlp_final_rank = None

                all_records.append({
                    "config": config_name, "fold": fold_idx, "pair_id": p["pair_id"], "direction": direction,
                    "blast_rank": blast_rank, "mlp_rank": mlp_final_rank, "blast_rr": blast_rr,
                    "mlp_rr": (1.0 / mlp_final_rank) if mlp_final_rank is not None else None,
                })

        elapsed = time.time() - config_start
        fold_df_tmp = pd.DataFrame([r for r in all_records if r["fold"] == fold_idx and r["config"] == config_name])
        mlp_mrr_str = f"{fold_df_tmp['mlp_rr'].mean():.4f}" if fold_df_tmp["mlp_rr"].notna().any() else "N/A"
        print(f"  fold {fold_idx + 1}/{K_FOLDS} -- mlp={mlp_mrr_str} ({elapsed/60:.1f} min)", flush=True)

df = pd.DataFrame(all_records)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - overall_start
print(f"Sve konfiguracije x {K_FOLDS} LOCO folda gotovo za {total_elapsed/60:.1f} min", flush=True)


def paired_bootstrap(sub, group_col, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    groups = sub[group_col].dropna().unique()
    deltas = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = sub.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        d = np.average(resampled["mlp_rr"], weights=w) - np.average(resampled["blast_rr"], weights=w)
        deltas.append(d)
    return np.array(deltas)


summary_lines = ["=" * 80, f"LOCO ({K_FOLDS} folds) -- RETUNOVAN MLP(hadamard) na ESM-2 3B vs BLAST vs 650M",
                  "=" * 80, "", f"Ukupno runtime: {total_elapsed/60:.1f} min",
                  f"Referenca -- 650M (isti config, LOCO): MRR=0.1259",
                  f"Referenca -- 3B BEZ retuninga (LOCO): MRR=0.0395 (kolaps)", ""]

for config_name, _ in CONFIGS:
    sub = df[df["config"] == config_name]
    sub_valid = sub.dropna(subset=["mlp_rr"]).copy()
    n_skipped = len(sub) - len(sub_valid)
    summary_lines.append(f"--- {config_name} (n={len(sub)} upita, izbaceno {n_skipped}) ---")
    summary_lines.append(f"  BLAST MRR (micro): {sub_valid['blast_rr'].mean():.4f}")
    summary_lines.append(f"  MLP(hadamard) MRR (micro): {sub_valid['mlp_rr'].mean():.4f}")
    summary_lines.append(f"  Delta vs BLAST: {sub_valid['mlp_rr'].mean() - sub_valid['blast_rr'].mean():+.4f}")
    for label, group_col in [("PO PARU", "pair_id"), ("PO IZVORU", "reference")]:
        deltas = paired_bootstrap(sub_valid, group_col, N_BOOTSTRAP, SEED)
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        significant = (ci_lo > 0) or (ci_hi < 0)
        verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
        summary_lines.append(f"    {label}: mean delta={deltas.mean():+.4f}, "
                              f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")
    summary_lines.append("")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
