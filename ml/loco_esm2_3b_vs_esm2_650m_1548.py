"""
ESM-2 3B naspram ESM-2 650M (trenutni glavni embeddings.pkl) naspram
BLAST-a -- test da li VECI model U ISTOJ arhitekturnoj familiji (isti
training recipe, ~4.6x vise parametara, dim 1280->2560) probija
"representation ceiling" nalaz koji drzi kroz ceo projekat. Direktan
nastavak ESM-1b eksperimenta (drugaciji model, izgubio) -- ovde je model
ISTI, samo veci.

Dva dela, ISTA "ne racunaj ono sto vec imas" disciplina naucena iz ESM-1b
eksperimenta (korisnica ispravno primetila da je prva verzija tog skripta
nepotrebno retrenirala ESM-2 650M baseline):

  1. COSINE (bez treninga, pa bez LOCO-a potrebnog uopste -- cosine nema
     leakage rizik) -- direktno racunanje na CELOM gold datasetu, isti
     protokol kao README "Cosine baseline" sekcija (cosine_similarity na
     celom embedding_matrix, MRR/Hits@K na SVIM gold parovima). Racuna se
     SVEZE za oba embedding izvora (jeftino, sekunde, nema smisla cuvati/
     ponovo koristiti kad je racunanje ionako trivijalno brzo).

  2. MLP(hadamard) LOCO (40 folda) -- BLAST i ESM-2 650M MLP strana
     PONOVO KORISCENI iz output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv
     (deterministicki identicno ranije potvrdjenim brojevima, MRR=0.1259).
     SAMO ESM-2 3B se trenira sveze.

Preduslov:
    embeddings/embeddings_esm2_3b.pkl, embeddings/embeddings_esm2_3b.parquet
    output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv (vec postoji)

Izlaz:
    output/loco_esm2_3b_vs_esm2_650m_1548_per_query.csv  (MLP deo)
    output/loco_esm2_3b_vs_esm2_650m_1548_summary.txt     (cosine + MLP)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS_650M = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA_650M = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
EMBEDDINGS_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.pkl")
METADATA_3B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm2_3b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
EXISTING_650M_PER_QUERY = Path("/home/lana/ALERGRAF/output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm2_3b_vs_esm2_650m_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm2_3b_vs_esm2_650m_1548_summary.txt")

for f in (EMBEDDINGS_3B, METADATA_3B, EXISTING_650M_PER_QUERY):
    if not f.exists():
        raise FileNotFoundError(f"{f} ne postoji.")

SEED = 42
NEG_PER_POS = 10
N_BOOTSTRAP = 2000

MLP_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                    learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                    max_epochs=300, patience=20, val_fraction=0.15)

print("Loading ESM-2 650M dataset...", flush=True)
dataset_650m = load_dataset(EMBEDDINGS_650M, METADATA_650M, GOLD)

print("Loading ESM-2 3B dataset...", flush=True)
dataset_3b = load_dataset(EMBEDDINGS_3B, METADATA_3B, GOLD)

overlap = set(dataset_650m.all_ids) & set(dataset_3b.all_ids)
print(f"  650M: {len(dataset_650m.all_ids)} proteina, 3B: {len(dataset_3b.all_ids)} proteina, "
      f"preklapanje: {len(overlap)}", flush=True)
missing = set(dataset_650m.all_ids) - overlap
if missing:
    print(f"  UPOZORENJE: {len(missing)} proteina nedostaje u 3B setu", flush=True)


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


# ---------------------------------------------------------------------------
# 1) COSINE -- bez treninga, celi dataset, oba embedding izvora sveze
#    (jeftino -- sekunde, nema potrebe za reuse-om ovde)
# ---------------------------------------------------------------------------
def evaluate_cosine(dataset, label):
    cosine_matrix = cosine_similarity(dataset.embedding_matrix)
    records = []
    for p in dataset.gold_pairs:
        for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi = dataset.id_to_index[query_id]
            ti = dataset.id_to_index[target_id]
            rank = ranks_from_scores(cosine_matrix[qi], qi)
            records.append({"pair_id": p["pair_id"], "rank": int(rank[ti]), "rr": 1.0 / int(rank[ti])})
    df = pd.DataFrame(records)
    hits1 = (df["rank"] <= 1).mean()
    hits5 = (df["rank"] <= 5).mean()
    hits10 = (df["rank"] <= 10).mean()
    print(f"  {label}: n={len(df)}, MRR={df['rr'].mean():.4f}, Hits@1={hits1:.4f}, "
          f"Hits@5={hits5:.4f}, Hits@10={hits10:.4f}", flush=True)
    return df, dict(mrr=df["rr"].mean(), hits1=hits1, hits5=hits5, hits10=hits10, n=len(df))


print("\n" + "=" * 70, flush=True)
print("DEO 1: COSINE (ceo dataset, bez treninga)", flush=True)
print("=" * 70, flush=True)
cos_650m_df, cos_650m_stats = evaluate_cosine(dataset_650m, "ESM-2 650M cosine")
cos_3b_df, cos_3b_stats = evaluate_cosine(dataset_3b, "ESM-2 3B cosine  ")

# Upareno poredjenje cosine 3B vs 650M (isti pair_id+direction, bootstrap po paru).
# NAPOMENA: 2 reda po pair_id (oba smera) -- eksplicitan direction indeks PRE merge-a,
# inace bi merge na golom pair_id dao 4 (pogresne) kombinacije po paru.
cos_650m_df["direction"] = cos_650m_df.groupby("pair_id").cumcount()
cos_3b_df["direction"] = cos_3b_df.groupby("pair_id").cumcount()
cos_merged = cos_650m_df.merge(cos_3b_df, on=["pair_id", "direction"], suffixes=("_650m", "_3b"))
cos_delta = cos_merged["rr_3b"] - cos_merged["rr_650m"]

rng = np.random.default_rng(SEED)
groups = cos_merged["pair_id"].dropna().unique()
cos_boot = []
for _ in range(N_BOOTSTRAP):
    sampled = rng.choice(groups, size=len(groups), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = cos_merged.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = resampled["w"].to_numpy()
    d = np.average(resampled["rr_3b"], weights=w) - np.average(resampled["rr_650m"], weights=w)
    cos_boot.append(d)
cos_boot = np.array(cos_boot)
cos_ci_lo, cos_ci_hi = np.percentile(cos_boot, [2.5, 97.5])
cos_significant = (cos_ci_lo > 0) or (cos_ci_hi < 0)

# ---------------------------------------------------------------------------
# 2) MLP(hadamard) LOCO -- 650M/BLAST PONOVO KORISCENI, SAMO 3B sveze
# ---------------------------------------------------------------------------
print("\n" + "=" * 70, flush=True)
print("DEO 2: MLP(hadamard) LOCO (40 folda) -- SAMO 3B se trenira sveze", flush=True)
print("=" * 70, flush=True)

folds = loco_folds(dataset_650m.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds: {K_FOLDS}", flush=True)

existing_650m = pd.read_csv(EXISTING_650M_PER_QUERY)
existing_650m["direction"] = existing_650m.groupby(["fold", "pair_id"]).cumcount()
existing_650m["config"] = "esm2_650m"
existing_650m["skipped_missing_embedding"] = False
print(f"Ponovo koriscen ESM-2 650M rezultat: {len(existing_650m)} redova "
      f"(MRR micro={existing_650m['mlp_rr'].mean():.4f} -- treba da se poklopi sa 0.1259)", flush=True)

blast_lookup = {(r.fold, r.pair_id, r.direction): (r.blast_rank, r.blast_rr) for r in existing_650m.itertuples()}

config_start = time.time()
records_3b = []

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_pairs_clean = training_eligible_pairs(train_pairs)
    train_pairs_clean = [p for p in train_pairs_clean
                          if p["id_1"] in dataset_3b.id_to_index and p["id_2"] in dataset_3b.id_to_index]
    train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset_3b.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
    train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED + fold_idx,
                                              dataset_3b.positive_pair_set)

    if len(train_pairs_clean) < 5:
        mlp = None
    else:
        mlp = MLPPairClassifier(params=MLP_PARAMS, seed=SEED + fold_idx)
        mlp.fit(train_pairs_clean, train_negatives, dataset_3b.embedding_matrix, dataset_3b.id_to_index)

    for p in test_pairs:
        for direction, (query_id, target_id) in enumerate([(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]):
            blast_rank, blast_rr = blast_lookup.get((fold_idx, p["pair_id"], direction), (None, None))

            if query_id not in dataset_3b.id_to_index or target_id not in dataset_3b.id_to_index:
                records_3b.append({"config": "esm2_3b", "fold": fold_idx, "pair_id": p["pair_id"],
                                     "direction": direction, "blast_rank": blast_rank, "mlp_rank": None,
                                     "blast_rr": blast_rr, "mlp_rr": None, "skipped_missing_embedding": True})
                continue

            if mlp is not None:
                mlp_scores = mlp.score_all(query_id)
                qi = dataset_3b.id_to_index[query_id]
                ti = dataset_3b.id_to_index[target_id]
                mlp_rank = ranks_from_scores(mlp_scores, qi)
                mlp_final_rank = int(mlp_rank[ti])
            else:
                mlp_final_rank = None

            records_3b.append({
                "config": "esm2_3b", "fold": fold_idx, "pair_id": p["pair_id"], "direction": direction,
                "blast_rank": blast_rank, "mlp_rank": mlp_final_rank, "blast_rr": blast_rr,
                "mlp_rr": (1.0 / mlp_final_rank) if mlp_final_rank is not None else None,
                "skipped_missing_embedding": False,
            })

    elapsed = time.time() - config_start
    fold_df_tmp = pd.DataFrame([r for r in records_3b if r["fold"] == fold_idx])
    mlp_mrr_str = f"{fold_df_tmp['mlp_rr'].mean():.4f}" if fold_df_tmp["mlp_rr"].notna().any() else "N/A"
    print(f"  fold {fold_idx + 1}/{K_FOLDS} -- mlp(esm2_3b)={mlp_mrr_str} ({elapsed/60:.1f} min)", flush=True)

df_3b = pd.DataFrame(records_3b)
existing_650m_out = existing_650m[["config", "fold", "pair_id", "direction", "blast_rank", "mlp_rank",
                                     "blast_rr", "mlp_rr", "skipped_missing_embedding"]]
df = pd.concat([existing_650m_out, df_3b], ignore_index=True)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - config_start


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


summary_lines = ["=" * 80, "ESM-2 3B naspram ESM-2 650M naspram BLAST -- cosine + MLP(hadamard) LOCO",
                  "=" * 80, "", "--- DEO 1: COSINE (ceo dataset, bez treninga) ---", ""]
for label, stats in [("ESM-2 650M", cos_650m_stats), ("ESM-2 3B  ", cos_3b_stats)]:
    summary_lines.append(f"  {label}: MRR={stats['mrr']:.4f}  Hits@1={stats['hits1']:.4f}  "
                          f"Hits@5={stats['hits5']:.4f}  Hits@10={stats['hits10']:.4f}  (n={stats['n']})")
verdict_cos = "ZNACAJNO" if cos_significant else "nije znacajno (CI ukljucuje 0)"
summary_lines.append(f"\n  ESM-2 3B vs 650M (upareno, po paru): mean delta={cos_delta.mean():+.4f}, "
                      f"95% CI [{cos_ci_lo:+.4f}, {cos_ci_hi:+.4f}] -- {verdict_cos}")

summary_lines += ["", "--- DEO 2: MLP(hadamard) LOCO (40 folda) ---",
                   f"ESM-2 3B runtime: {total_elapsed/60:.1f} min (650M/BLAST strana ponovo koriscena)", ""]

for config_name in ["esm2_650m", "esm2_3b"]:
    sub = df[df["config"] == config_name]
    n_skipped_missing = int(sub["skipped_missing_embedding"].sum())
    sub_valid = sub.dropna(subset=["mlp_rr"]).copy()
    n_skipped_train = len(sub) - n_skipped_missing - len(sub_valid)
    summary_lines.append(f"--- {config_name} (n={len(sub)} upita, {n_skipped_missing} izbaceno, "
                          f"{n_skipped_train} zbog praznog fold treninga) ---")
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

e650 = df[(df["config"] == "esm2_650m")].dropna(subset=["mlp_rr"])
e3b = df[(df["config"] == "esm2_3b")].dropna(subset=["mlp_rr"])
merged_cfg = e650.merge(e3b, on=["fold", "pair_id", "direction"], suffixes=("_650m", "_3b"))
delta_cfg = merged_cfg["mlp_rr_3b"] - merged_cfg["mlp_rr_650m"]
summary_lines.append(f"--- ESM-2 3B vs 650M, MLP(hadamard) (direktno, n={len(merged_cfg)}) ---")
summary_lines.append(f"  Mean MRR delta (3b - 650m): {delta_cfg.mean():+.4f}")

rng2 = np.random.default_rng(SEED)
groups2 = merged_cfg["pair_id"].dropna().unique()
deltas_cfg = []
for _ in range(N_BOOTSTRAP):
    sampled = rng2.choice(groups2, size=len(groups2), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = merged_cfg.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = resampled["w"].to_numpy()
    d = np.average(resampled["mlp_rr_3b"], weights=w) - np.average(resampled["mlp_rr_650m"], weights=w)
    deltas_cfg.append(d)
deltas_cfg = np.array(deltas_cfg)
ci_lo2, ci_hi2 = np.percentile(deltas_cfg, [2.5, 97.5])
significant2 = (ci_lo2 > 0) or (ci_hi2 < 0)
verdict2 = "ZNACAJNO" if significant2 else "nije znacajno (CI ukljucuje 0)"
summary_lines.append(f"  Bootstrap (po paru, N={N_BOOTSTRAP}): 95% CI [{ci_lo2:+.4f}, {ci_hi2:+.4f}] -- {verdict2}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
