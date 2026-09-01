"""
Targeted hard-negative eksperiment za MLP(hadamard) -- testira da li ciljano
mesanje "istog organizma/food source-a, RAZLICITA porodica, NIJE poznat
pozitivan par" negativa (umesto cisto uniformnog nasumicnog uzorkovanja)
popravlja dijagnostikovanu slabost sa 57-pacijentskog testa: MLP ne potiskuje
negative (npr. Pru p 1 kad je Pru p 3 poznat pozitivan) skoro kao dobro
kao BLAST.

VAZNO (eksplicitan zahtev korisnika): NE koristi se test/test_cases.json
(pacijentski CRD podaci) NIGDE u ovom fajlu -- ni za trening ni za odabir
kandidata. Kandidati dolaze ISKLJUCIVO iz vec postojecih training resursa:
output/clean_allergens.csv (organism/source_food kolone) + gold-standard
family_1/family_2 labele iz output/cross_reactive_1548.csv (da se zna koja
je porodica svakog proteina) -- ne izmisljaju se novi parovi, samo se
IZDVAJA podskup vec postojeceg "sve sto nije poznat pozitivan" negativnog
prostora koji ima jednu dodatnu, proverljivu osobinu (isti organizam,
razlicita porodica).

Kandidat pravila (konzervativno, "ne izmisljaj"):
  1) Oba proteina moraju imati TACNO JEDNU, konzistentnu family labelu
     preko svih gold parova u kojima se pojavljuju (32/477 proteina ima
     nekonzistentne/varijantne family stringove u sirovim podacima --
     ti se PRESKACU, ne pokusava se fuzzy-normalizacija).
  2) Isti 'organism' (naucno ime, 1536/1536 popunjeno u clean_allergens.csv --
     pouzdanije od 'source_food', koje ima 27 praznih vrednosti).
  3) Family_a != family_b.
  4) Par NIJE u gold-standard positive_pair_set (ni u jednom smeru).
  Rezultat: 930 kandidata, 76 organizama, 192 razlicite family-par kombinacije
  (nije dominirano jednim family-parom) -- dovoljno da se probni eksperiment
  opravda. (Pru p 1, Pru p 3) je tacno 1 od ovih 930 -- ciljni slucaj postoji
  u kandidatskom skupu bez ikakvog specijalnog tretmana.

Eksperiment: 4 varijante (0% = baseline identican
ml/loco_blast_vs_mlp_hadamard_only_1548.py, 5%, 10%, 20% od negativnog
budzeta po foldu zamenjeno ciljanim kandidatima), SVE na ISTIM LOCO foldovima
(loco_folds je deterministicna, bez seed-a), ista MLP_HADAMARD_PARAMS/
NEG_PER_POS/SEED disciplina kao original -- jedina razlika je KOJI negativi
ulaze u trening.

Po foldu, kandidati koji diraju test_ids (held-out komponentu) se ISKLJUCUJU
iz ciljanog skupa za taj fold (leakage guard) -- ako posle toga ima manje
kandidata nego trazeni budzet, popunjava se OSTATAK uniformnim nasumicnim
negativima (ne duplira se agresivno vestacki isti mali skup).

Nakon LOCO-a: dodatna, NE-pacijentska provera na PRODUKCIONOM modelu (trening
na svih 785 training_eligible_pairs() + puni negativni budzet po istoj ratio
logici, bez fold-restrikcije) -- rank(Pru p 1 | query=Pru p 3) i rang par
poznatih pravih Pru p 3 partnera (Cor a 8, Ara h 9, Jug r 3), za svaku
varijantu.

Izlaz:
    output/loco_targeted_hardneg_mlp_hadamard_1548_per_query.csv
    output/loco_targeted_hardneg_mlp_hadamard_1548_summary.txt
"""

import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset, training_eligible_pairs  # noqa: E402
from ml.pipeline.common.features import load_blast_matrices  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import loco_folds  # noqa: E402
from ml.pipeline.models.classifiers.mlp import MLPPairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
BLAST_MATRIX = "/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl"
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_targeted_hardneg_mlp_hadamard_1548_per_query.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/loco_targeted_hardneg_mlp_hadamard_1548_summary.txt")

SEED = 42
NEG_PER_POS = 10
N_BOOTSTRAP = 2000
RATIOS = [0.0, 0.05, 0.10, 0.20]

MLP_HADAMARD_PARAMS = dict(input_encoding="hadamard", standardize=False, hidden_dims=[32], dropout=[0.3],
                             learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, batch_size=64,
                             max_epochs=300, patience=20, val_fraction=0.15)

print("Loading dataset...", flush=True)
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
blast = load_blast_matrices(BLAST_MATRIX)

perm = np.array([blast["id_to_index"].get(aid, -1) for aid in dataset.all_ids])
valid_idx = np.where(perm >= 0)[0]
blast_score_matrix_full = np.zeros((len(dataset.all_ids), len(dataset.all_ids)), dtype=np.float32)
blast_score_matrix_full[np.ix_(valid_idx, valid_idx)] = blast["score_matrix"][np.ix_(perm[valid_idx], perm[valid_idx])]


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


# ---------------------------------------------------------------------------
# 1) Ciljani kandidatski skup: isti organizam, razlicita (konzistentna)
#    family labela, nije poznat pozitivan par. Gradi se ISKLJUCIVO iz
#    clean_allergens.csv + cross_reactive_1548.csv (production trening
#    resursi) -- test_cases.json se ovde NIGDE ne cita.
# ---------------------------------------------------------------------------
print("\nGradim ciljani kandidatski skup (isti organizam, razlicita porodica)...", flush=True)
allergens_df = pd.read_csv(ALLERGENS)
gold_raw = pd.read_csv(GOLD)

fam_votes = defaultdict(set)
for _, r in gold_raw.iterrows():
    fam_votes[r["allergen_id_1"]].add(r["family_1"])
    fam_votes[r["allergen_id_2"]].add(r["family_2"])
fam_single = {k: next(iter(v)) for k, v in fam_votes.items() if len(v) == 1}

positive_pairs_names = set()
for _, r in gold_raw.iterrows():
    positive_pairs_names.add((r["allergen_id_1"], r["allergen_id_2"]))
    positive_pairs_names.add((r["allergen_id_2"], r["allergen_id_1"]))

org_groups = allergens_df.groupby("organism")["official_name"].apply(list)
multi_org = org_groups[org_groups.apply(len) > 1]

candidate_names = []
for org, names in multi_org.items():
    names_fam = [n for n in names if n in fam_single]
    for a, b in itertools.combinations(sorted(set(names_fam)), 2):
        if fam_single[a] == fam_single[b]:
            continue
        if (a, b) in positive_pairs_names:
            continue
        candidate_names.append((a, b))

# official_name -> allergen_id (dataset.name_to_id, isti mapping kao svuda u
# pipeline-u), zadrzi samo parove gde OBA proteina imaju embedding.
candidate_ids = set()
for a, b in candidate_names:
    ia = dataset.name_to_id.get(a)
    ib = dataset.name_to_id.get(b)
    if ia is None or ib is None or ia not in dataset.id_to_index or ib not in dataset.id_to_index:
        continue
    candidate_ids.add(tuple(sorted((ia, ib))))
candidate_ids = sorted(candidate_ids)
print(f"  Kandidata (allergen_id prostor, sa embeddinzima): {len(candidate_ids)}", flush=True)

pru_p3_id = dataset.name_to_id.get("Pru p 3.0101")
pru_p1_id = dataset.name_to_id.get("Pru p 1.0101")
assert tuple(sorted((pru_p3_id, pru_p1_id))) in set(candidate_ids), "Pru p1/Pru p3 nije u kandidatskom skupu!"
print(f"  Potvrdjeno: (Pru p 1, Pru p 3) je u kandidatskom skupu.", flush=True)


def hard_negatives_for_pool(candidates, forbidden_ids, positive_pair_set, n_target, seed):
    """forbidden_ids: proteini koji se ne smeju pojaviti (npr. LOCO test_ids)."""
    avail = [p for p in candidates
             if p[0] not in forbidden_ids and p[1] not in forbidden_ids and p not in positive_pair_set]
    rng = np.random.default_rng(seed)
    if len(avail) <= n_target:
        return avail
    idx = rng.choice(len(avail), size=n_target, replace=False)
    return [avail[i] for i in sorted(idx)]


def build_train_negatives(train_ids, train_pairs_clean, candidates, positive_pair_set, ratio, seed):
    n_train_neg = max(len(train_pairs_clean) * NEG_PER_POS, 50)
    n_hard_target = int(round(n_train_neg * ratio))
    hard_negs = hard_negatives_for_pool(candidates, set(dataset.all_ids) - set(train_ids),
                                          positive_pair_set, n_hard_target, seed)
    n_remaining = n_train_neg - len(hard_negs)
    exclude = positive_pair_set | set(hard_negs)
    uniform_negs = sample_negative_pairs(sorted(train_ids), n_remaining, seed + 5000, exclude)
    return sorted(set(hard_negs) | set(uniform_negs)), len(hard_negs), n_hard_target


# ---------------------------------------------------------------------------
# 2) LOCO, iste folds za sve 4 varijante (loco_folds je deterministicna).
# ---------------------------------------------------------------------------
folds = loco_folds(dataset.gold_pairs)
K_FOLDS = len(folds)
print(f"\nLOCO folds: {K_FOLDS}", flush=True)

all_records = []
overall_start = time.time()

for ratio in RATIOS:
    print(f"\n{'='*70}\nRATIO = {ratio:.0%}\n{'='*70}", flush=True)
    ratio_start = time.time()
    fold_hard_frac = []

    for fold_idx, (train_pairs, test_pairs, test_ids) in enumerate(folds):
        train_pairs_clean = training_eligible_pairs(train_pairs)
        train_ids = {pid for p in train_pairs_clean for pid in (p["id_1"], p["id_2"])}
        train_ids |= {pid for pid in dataset.all_ids if pid not in test_ids and pid not in train_ids}

        if len(train_pairs_clean) < 5:
            mlp = None
            achieved_hard, target_hard = 0, 0
        else:
            train_negatives, achieved_hard, target_hard = build_train_negatives(
                train_ids, train_pairs_clean, candidate_ids, dataset.positive_pair_set,
                ratio, SEED + fold_idx)
            mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED + fold_idx)
            mlp.fit(train_pairs_clean, train_negatives, dataset.embedding_matrix, dataset.id_to_index)
        fold_hard_frac.append((achieved_hard, target_hard))

        for p in test_pairs:
            for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
                qi = dataset.id_to_index[query_id]
                ti = dataset.id_to_index[target_id]

                blast_rank = ranks_from_scores(blast_score_matrix_full[qi], qi)
                blast_final_rank = int(blast_rank[ti])

                if mlp is not None:
                    mlp_scores = mlp.score_all(query_id)
                    mlp_rank = ranks_from_scores(mlp_scores, qi)
                    mlp_final_rank = int(mlp_rank[ti])
                else:
                    mlp_final_rank = None

                all_records.append({
                    "ratio": ratio, "fold": fold_idx, "pair_id": p["pair_id"],
                    "blast_rank": blast_final_rank, "mlp_rank": mlp_final_rank,
                    "blast_rr": 1.0 / blast_final_rank,
                    "mlp_rr": (1.0 / mlp_final_rank) if mlp_final_rank is not None else None,
                })

        elapsed = time.time() - ratio_start
        fold_df_tmp = pd.DataFrame([r for r in all_records if r["fold"] == fold_idx and r["ratio"] == ratio])
        mlp_mrr_str = f"{fold_df_tmp['mlp_rr'].mean():.4f}" if fold_df_tmp["mlp_rr"].notna().any() else "N/A"
        print(f"  fold {fold_idx + 1}/{K_FOLDS} (hard_neg={achieved_hard}/{target_hard}) -- "
              f"blast={fold_df_tmp['blast_rr'].mean():.4f} mlp={mlp_mrr_str} ({elapsed/60:.1f} min)", flush=True)

    total_target = sum(t for _, t in fold_hard_frac)
    total_achieved = sum(a for a, _ in fold_hard_frac)
    if total_target > 0:
        print(f"  Ukupno ovaj ratio: hard negativi dostignuto {total_achieved}/{total_target} "
              f"({total_achieved/total_target:.1%} od cilja)", flush=True)

df = pd.DataFrame(all_records)
gold_ref = pd.read_csv(GOLD)[["pair_id", "reference"]].drop_duplicates(subset="pair_id")
df = df.merge(gold_ref, on="pair_id", how="left")
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"\nSaved: {PER_QUERY_OUTPUT}", flush=True)

total_elapsed = time.time() - overall_start
print(f"Sve 4 varijante x {K_FOLDS} LOCO folda gotovo za {total_elapsed/60:.1f} min", flush=True)


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


summary_lines = ["=" * 80,
                  f"LOCO ({K_FOLDS} folds) -- targeted hard-negative eksperiment, "
                  "MLP(hadamard) vs BLAST", "=" * 80, "",
                  f"Ukupno runtime: {total_elapsed/60:.1f} min",
                  f"Kandidatski skup (isti organizam, razlicita porodica, nije poznat pozitiv): "
                  f"{len(candidate_ids)} parova, potvrdjeno da (Pru p 1, Pru p 3) pripada skupu.", ""]

for ratio in RATIOS:
    sub = df[df["ratio"] == ratio]
    sub_valid = sub.dropna(subset=["mlp_rr"]).copy()
    n_skipped = len(sub) - len(sub_valid)
    summary_lines.append(f"--- RATIO={ratio:.0%} (n={len(sub)} upita, izbaceno {n_skipped}) ---")
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

baseline_mrr = df[(df["ratio"] == 0.0)].dropna(subset=["mlp_rr"])["mlp_rr"].mean()
summary_lines.append("--- Delta vs 0% (interni) baseline, po ratio-u ---")
for ratio in RATIOS:
    if ratio == 0.0:
        continue
    sub_valid = df[df["ratio"] == ratio].dropna(subset=["mlp_rr"])
    d = sub_valid["mlp_rr"].mean() - baseline_mrr
    summary_lines.append(f"  {ratio:.0%}: MRR delta vs 0% = {d:+.4f}")
summary_lines.append("")

# ---------------------------------------------------------------------------
# 3) Produkcioni model po ratio-u (trening na SVE eligible parove, ceo pool,
#    bez LOCO fold-restrikcije) -- direktna, NE-pacijentska provera:
#    rank(Pru p 1 | query=Pru p 3) i rank par poznatih pravih partnera.
# ---------------------------------------------------------------------------
summary_lines.append("=" * 80)
summary_lines.append("Produkcioni model (svi training_eligible_pairs, ceo pool) -- "
                      "Pru p 1 potiskivanje vs Pru p 3 pravi partneri")
summary_lines.append("(NE koristi pacijentske CRD podatke -- direktan upit na trenirani model)")
summary_lines.append("=" * 80)
summary_lines.append("")

train_pairs_prod = training_eligible_pairs(dataset.gold_pairs)
known_partners = ["Cor a 8.0101", "Ara h 9.0101", "Jug r 3.0101"]
print("\nProdukcioni modeli po ratio-u (Pru p1/Pru p3 provera)...", flush=True)

for ratio in RATIOS:
    train_negatives, achieved_hard, target_hard = build_train_negatives(
        dataset.all_ids, train_pairs_prod, candidate_ids, dataset.positive_pair_set, ratio, SEED)
    mlp = MLPPairClassifier(params=MLP_HADAMARD_PARAMS, seed=SEED)
    mlp.fit(train_pairs_prod, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    scores = mlp.score_all(pru_p3_id)
    qi = dataset.id_to_index[pru_p3_id]
    ranks = ranks_from_scores(scores, qi)
    pru_p1_rank = int(ranks[dataset.id_to_index[pru_p1_id]])
    pru_p1_pct = pru_p1_rank / len(dataset.all_ids) * 100

    summary_lines.append(f"--- RATIO={ratio:.0%} (hard_neg u produkcionom treningu: "
                          f"{achieved_hard}/{target_hard}) ---")
    summary_lines.append(f"  Query=Pru p 3, rank(Pru p 1) = {pru_p1_rank} ({pru_p1_pct:.2f} percentil) "
                          f"-- veci broj/percentil = bolje potisnuto")
    for partner_name in known_partners:
        pid = dataset.name_to_id.get(partner_name)
        if pid is None or pid not in dataset.id_to_index:
            summary_lines.append(f"  Query=Pru p 3, rank({partner_name}) = N/A (nije u pool-u)")
            continue
        r = int(ranks[dataset.id_to_index[pid]])
        pct = r / len(dataset.all_ids) * 100
        summary_lines.append(f"  Query=Pru p 3, rank({partner_name}) = {r} ({pct:.2f} percentil) "
                              f"-- manji broj/percentil = bolje (pravi pozitiv treba da ostane visoko)")
    summary_lines.append("")
    print(f"  ratio={ratio:.0%}: rank(Pru p1)={pru_p1_rank}, done", flush=True)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
