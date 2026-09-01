"""
MLP(hadamard) na ESM-1b embeddinzima naspram MLP(hadamard) na ESM-2
embeddinzima (trenutni glavni embeddings.pkl) naspram BLAST-a -- test da li
NEZAVISNO trenirana proteinska reprezentacija (ESM-1b, 2019, drugaciji
training recipe/UniRef filter od ESM-2, ali ISTI red velicine parametara,
650M) probija "representation ceiling" nalaz koji drzi kroz ceo projekat.

VAZNO: ne pravi se nov RRF voter niti fuzija -- na eksplicitan zahtev
korisnice, ESM-1b se koristi SAMO za MLP(hadamard), direktno uporedjen sa
istim modelom na ESM-2 i sa BLAST-om, ista LOCO disciplina kao svuda.

OPTIMIZACIJA (korisnica ispravno primetila da je prvobitna verzija ovog
skripta nepotrebno RETRENIRALA ceo ESM-2 baseline iznova -- vec imamo taj
rezultat, DVA PUTA nezavisno reprodukovan na tacno istu vrednost, MRR=0.1259,
u ml/loco_blast_vs_mlp_hadamard_only_1548.py i u ml/loco_mlp_hadamard_
layernorm_ablation_1548.py "baseline" config-u -- deterministicno, isti seed/
foldovi/params). Ovde se ESM-2/BLAST strana PONOVO KORISTI direktno iz
output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv, SAMO ESM-1b se
racuna sveze. BLAST rank/rr se takodje preuzima iz tog fajla za oba config-a
(BLAST ne zavisi od embedding izvora, isti rezultat u oba slucaja -- nema
potrebe ni za blast_identity_matrix_1443.pkl u ovom skriptu).

Poravnanje reused (esm2) i sveze (esm1b) strane: oba koriste ISTI
deterministicki loco_folds(dataset.gold_pairs) poziv (ista GOLD putanja) i
ISTU [(id_1,id_2),(id_2,id_1)] iteraciju smera po paru -- dodat eksplicitan
"direction" (0/1, cumcount unutar fold+pair_id grupe) da spajanje bude
sigurno indeksirano, ne oslonjeno na implicitan redosled reda.

Preduslov (mora postojati pre pokretanja):
    embeddings/embeddings_esm1b.pkl, embeddings/embeddings_esm1b.parquet
    output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv (vec postoji)

Izlaz:
    output/loco_esm1b_vs_esm2_mlp_hadamard_1548_per_query.csv
    output/loco_esm1b_vs_esm2_mlp_hadamard_1548_summary.txt
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

EMBEDDINGS_ESM2 = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA_ESM2 = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
EMBEDDINGS_ESM1B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm1b.pkl")
METADATA_ESM1B = Path("/home/lana/ALERGRAF/embeddings/embeddings_esm1b.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
EXISTING_ESM2_PER_QUERY = Path("/home/lana/ALERGRAF/output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm1b_vs_esm2_mlp_hadamard_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_esm1b_vs_esm2_mlp_hadamard_1548_summary.txt")

for f in (EMBEDDINGS_ESM1B, METADATA_ESM1B, EXISTING_ESM2_PER_QUERY):
    if not f.exists():
        raise FileNotFoundError(f"{f} ne postoji.")

SEED = 42
NEG_PER_POS = 10
N_BOOTSTRAP = 2000

MLP_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                    learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                    max_epochs=300, patience=20, val_fraction=0.15)

print("Loading ESM-2 dataset (SAMO da definise foldove -- MLP se NE trenira ponovo)...", flush=True)
dataset_esm2 = load_dataset(EMBEDDINGS_ESM2, METADATA_ESM2, GOLD)

print("Loading ESM-1b dataset...", flush=True)
dataset_esm1b = load_dataset(EMBEDDINGS_ESM1B, METADATA_ESM1B, GOLD)

overlap = set(dataset_esm2.all_ids) & set(dataset_esm1b.all_ids)
print(f"  ESM-2: {len(dataset_esm2.all_ids)} proteina, ESM-1b: {len(dataset_esm1b.all_ids)} proteina, "
      f"preklapanje: {len(overlap)}", flush=True)
missing = set(dataset_esm2.all_ids) - overlap
if missing:
    print(f"  UPOZORENJE: {len(missing)} proteina iz ESM-2 seta nedostaje u ESM-1b setu "
          f"(upiti koji ih dodiruju bice izbaceni, ne tiho ignorisani)", flush=True)


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


folds = loco_folds(dataset_esm2.gold_pairs)
K_FOLDS = len(folds)
print(f"LOCO folds (iz ESM-2 gold grafa, deterministicno -- isto sto je koristio i original run): {K_FOLDS}",
      flush=True)

# ---------------------------------------------------------------------------
# 1) ESM-2 strana: PONOVNO KORISCENJE postojeceg rezultata, bez retreniranja.
# ---------------------------------------------------------------------------
esm2_reused = pd.read_csv(EXISTING_ESM2_PER_QUERY)
esm2_reused["direction"] = esm2_reused.groupby(["fold", "pair_id"]).cumcount()
esm2_reused["config"] = "esm2"
esm2_reused["skipped_missing_embedding"] = False
print(f"\nPonovo koriscen ESM-2 rezultat: {len(esm2_reused)} redova "
      f"(MRR micro={esm2_reused['mlp_rr'].mean():.4f} -- treba da se poklopi sa 0.1259)", flush=True)

blast_lookup = {(r.fold, r.pair_id, r.direction): (r.blast_rank, r.blast_rr) for r in esm2_reused.itertuples()}

# ---------------------------------------------------------------------------
# 2) ESM-1b strana: sveze treniranje, BLAST preuzet iz esm2 lookup-a (BLAST
#    ne zavisi od embedding izvora -- isti rezultat, nema smisla racunati 2x).
# ---------------------------------------------------------------------------
print(f"\n{'='*70}\nEMBEDDING SOURCE = esm1b (sveze racunanje)\n{'='*70}", flush=True)
config_start = time.time()
esm1b_records = []

for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
    train_pairs_clean = training_eligible_pairs(train_pairs)
    train_pairs_clean = [p for p in train_pairs_clean
                          if p["id_1"] in dataset_esm1b.id_to_index and p["id_2"] in dataset_esm1b.id_to_index]
    train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
    train_ids |= {pid for pid in dataset_esm1b.all_ids if pid not in test_ids and pid not in train_ids}
    n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
    train_negatives = sample_negative_pairs(sorted(train_ids), n_train_neg, SEED + fold_idx,
                                              dataset_esm1b.positive_pair_set)

    if len(train_pairs_clean) < 5:
        mlp = None
    else:
        mlp = MLPPairClassifier(params=MLP_PARAMS, seed=SEED + fold_idx)
        mlp.fit(train_pairs_clean, train_negatives, dataset_esm1b.embedding_matrix, dataset_esm1b.id_to_index)

    for p in test_pairs:
        for direction, (query_id, target_id) in enumerate([(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]):
            blast_rank, blast_rr = blast_lookup.get((fold_idx, p["pair_id"], direction), (None, None))

            if query_id not in dataset_esm1b.id_to_index or target_id not in dataset_esm1b.id_to_index:
                esm1b_records.append({"config": "esm1b", "fold": fold_idx, "pair_id": p["pair_id"],
                                        "direction": direction, "blast_rank": blast_rank, "mlp_rank": None,
                                        "blast_rr": blast_rr, "mlp_rr": None, "skipped_missing_embedding": True})
                continue

            if mlp is not None:
                mlp_scores = mlp.score_all(query_id)
                qi = dataset_esm1b.id_to_index[query_id]
                ti = dataset_esm1b.id_to_index[target_id]
                mlp_rank = ranks_from_scores(mlp_scores, qi)
                mlp_final_rank = int(mlp_rank[ti])
            else:
                mlp_final_rank = None

            esm1b_records.append({
                "config": "esm1b", "fold": fold_idx, "pair_id": p["pair_id"], "direction": direction,
                "blast_rank": blast_rank, "mlp_rank": mlp_final_rank,
                "blast_rr": blast_rr,
                "mlp_rr": (1.0 / mlp_final_rank) if mlp_final_rank is not None else None,
                "skipped_missing_embedding": False,
            })

    elapsed = time.time() - config_start
    fold_df_tmp = pd.DataFrame([r for r in esm1b_records if r["fold"] == fold_idx])
    mlp_mrr_str = f"{fold_df_tmp['mlp_rr'].mean():.4f}" if fold_df_tmp["mlp_rr"].notna().any() else "N/A"
    print(f"  fold {fold_idx + 1}/{K_FOLDS} -- mlp(esm1b)={mlp_mrr_str} ({elapsed/60:.1f} min)", flush=True)

esm1b_df = pd.DataFrame(esm1b_records)

# ---------------------------------------------------------------------------
# 3) Spoji i sacuvaj
# ---------------------------------------------------------------------------
esm2_reused_out = esm2_reused[["config", "fold", "pair_id", "direction", "blast_rank", "mlp_rank",
                                 "blast_rr", "mlp_rr", "skipped_missing_embedding"]]
df = pd.concat([esm2_reused_out, esm1b_df], ignore_index=True)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - config_start
print(f"ESM-1b racunanje gotovo za {total_elapsed/60:.1f} min (ESM-2 strana ponovo koriscena, ne racunata)",
      flush=True)


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


summary_lines = ["=" * 80, f"LOCO ({K_FOLDS} folds) -- MLP(hadamard) ESM-1b (sveze) vs ESM-2 (ponovo koriscen) "
                  "vs BLAST", "=" * 80, "",
                  f"ESM-1b runtime: {total_elapsed/60:.1f} min (ESM-2 strana NIJE ponovo racunata)", ""]

for config_name in ["esm2", "esm1b"]:
    sub = df[df["config"] == config_name]
    n_skipped_missing = int(sub["skipped_missing_embedding"].sum())
    sub_valid = sub.dropna(subset=["mlp_rr"]).copy()
    n_skipped_train = len(sub) - n_skipped_missing - len(sub_valid)
    summary_lines.append(f"--- {config_name} (n={len(sub)} upita, {n_skipped_missing} izbaceno zbog "
                          f"nedostajuceg embedding-a, {n_skipped_train} zbog praznog fold treninga) ---")
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

e2 = df[(df["config"] == "esm2")].dropna(subset=["mlp_rr"])
e1b = df[(df["config"] == "esm1b")].dropna(subset=["mlp_rr"])
merged_cfg = e2.merge(e1b, on=["fold", "pair_id", "direction"], suffixes=("_esm2", "_esm1b"))
delta_cfg = merged_cfg["mlp_rr_esm1b"] - merged_cfg["mlp_rr_esm2"]
summary_lines.append(f"--- ESM-1b vs ESM-2 (direktno, isti upiti gde oba validna, n={len(merged_cfg)}) ---")
summary_lines.append(f"  Mean MRR delta (esm1b - esm2): {delta_cfg.mean():+.4f}")

rng = np.random.default_rng(SEED)
groups = merged_cfg["pair_id"].dropna().unique()
deltas_cfg = []
for _ in range(N_BOOTSTRAP):
    sampled = rng.choice(groups, size=len(groups), replace=True)
    counts = pd.Series(sampled).value_counts()
    resampled = merged_cfg.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = resampled["w"].to_numpy()
    d = np.average(resampled["mlp_rr_esm1b"], weights=w) - np.average(resampled["mlp_rr_esm2"], weights=w)
    deltas_cfg.append(d)
deltas_cfg = np.array(deltas_cfg)
ci_lo, ci_hi = np.percentile(deltas_cfg, [2.5, 97.5])
significant = (ci_lo > 0) or (ci_hi < 0)
verdict = "ZNACAJNO" if significant else "nije znacajno (CI ukljucuje 0)"
summary_lines.append(f"  Bootstrap (po paru, N={N_BOOTSTRAP}): 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] -- {verdict}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
