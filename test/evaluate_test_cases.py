"""
Evaluator za test/test_cases.json (33 pacijenta, sve necirkularno spram
gold dataseta -- proveriti otom nije uradjeno OVDE, uradjeno je u ranijim
real_world_case_validation skriptama za manji podskup istih izvora).

FEATURE 1 (rank_for_patient): leave-one-out po pacijentu.
  Za pacijenta sa >=2 resolvable komponente: za svaku komponentu K,
  otkrij OSTALE (kao known_positive/known_negative), sakrij K, pokreni
  ranker.rank_for_patient(), zabelezi gde K zavrsi.
  - Ako iza uklanjanja K ne ostane nijedan known POZITIVAN, taj trial se
    preskace (rank_for_patient zahteva >=1 pozitivan) -- ne bag, ogranicenje
    dizajna kad je pacijent monospecifican (npr. vando2005_pt04 sa hidden=
    Gad c 1, jedini pozitivan).
  - Generic/whole-extract/band komponente (npr. "tuna (extract)", "CCD",
    "~35kDa band") se ne mogu resolve-ovati na nas protein pool -- automatski
    izostavljene (nema match-a), ne posebno kodirano.

Glavna metrika: percentile rank sakrivene komponente, odvojeno za POZITIVNE
(zelimo NIZAK percentil = visok prioritet) i NEGATIVNE (zelimo VISOK
percentil = ne-prioritet) mete -- odvojeno za "hard" (full_text_verified)
i "soft" (abstract_snippet_only/partially_verified) verifikaciju, PER
UPUTSTVU: soft slucajevi se ne koriste za pass/fail odluku.

Posebno istaknuti slucajevi (per napomenama):
  - vando2005_pt04: monospecifican, hard test
  - penas2014_case: EXPECTED-FAILURE case (beta-enolaza, ne parvalbumin) --
    karakteriseGmo failure mode, ne pass/fail
  - uklejasokolowska2021_pt04 vs _pt18: inverzni par, side-by-side

FEATURE 2 (graduated_introduction_path): samo pereira2009_case ima
resolvable sequential_protocol cilj (Pru p 3) -- porcaro2016_case koristi
samo whole-extract nazive (salmon/cod), nije resolvable na protein nivou.
Kvalitativan face-validity izvestaj SAMO, eksplicitno ne "validirano".

Izlaz:
    test/evaluation_results.json
    test/evaluation_summary.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, "/home/lana/ALERGRAF")
sys.path.insert(0, "/home/lana/ALERGRAF/test")  # "test" kolidira sa Python stdlib paketom, uvozimo direktno iz fajla
from ml.patient_ranking_1548 import CrossReactivityRanker  # noqa: E402
from protein_resolution import resolve_protein as _resolve_protein  # noqa: E402

TEST_CASES = Path("/home/lana/ALERGRAF/test/test_cases.json")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
JSON_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_results.json")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/test/evaluation_summary.txt")

print("Loading ranker (ovo ucitava sve matrice, potraje par sekundi)...")
ranker = CrossReactivityRanker()
n_pool = ranker.n_pool

clean = pd.read_csv(CLEAN_ALLERGENS)
pool_names = sorted(ranker.name_to_id.keys())


def resolve_protein(json_name):
    return _resolve_protein(json_name, pool_names)


with open(TEST_CASES) as f:
    cases = json.load(f)
print(f"Ucitano {len(cases)} pacijenata")

# =====================================================
# FEATURE 1: leave-one-out
# =====================================================

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
        resolvable.append({"json_name": c["protein"], "pool_name": resolved, "result": c["result"],
                            "confirmed_by_challenge": c.get("confirmed_by_challenge", False)})

    if len(resolvable) < 2:
        continue

    for i, hidden in enumerate(resolvable):
        others = resolvable[:i] + resolvable[i + 1:]
        known_pos = [o["pool_name"] for o in others if o["result"] == "positive"]
        known_neg = [o["pool_name"] for o in others if o["result"] == "negative"]
        if not known_pos:
            skipped_no_positive_left += 1
            continue

        result_df = ranker.rank_for_patient(known_pos, known_negative_names=known_neg)
        row = result_df[result_df["candidate_name"] == hidden["pool_name"]]
        if len(row) == 0:
            # hidden protein je vec medju known_neg iskljucenim -- ne bi trebalo
            # da se desi jer je hidden uklonjen iz others, ali proveravamo
            continue
        rank = int(row.iloc[0]["rank"])
        n_cand = len(result_df)
        percentile = rank / n_cand * 100

        records.append({
            "patient_id": pid, "hidden_protein": hidden["pool_name"],
            "true_result": hidden["result"], "rank": rank, "n_candidates": n_cand,
            "percentile": percentile, "verification_status": verif_status,
            "confirmed_by_challenge": hidden["confirmed_by_challenge"],
            "n_known_positive": len(known_pos), "n_known_negative": len(known_neg),
        })

df = pd.DataFrame(records)
print(f"\nLeave-one-out trials: {len(df)}")
print(f"Preskoceno (nema known pozitivan posle sakrivanja): {skipped_no_positive_left}")
print(f"Nerezolvovane komponente (whole-extract/band/generic, ocekivano): {sorted(unresolved_components)}")

df.to_json("/home/lana/ALERGRAF/test/evaluation_results_raw.json", orient="records", indent=2)

# =====================================================
# AGGREGATE -- odvojeno hard (full_text_verified) vs soft
# =====================================================

hard = df[df["verification_status"] == "full_text_verified"]
soft = df[df["verification_status"] != "full_text_verified"]

summary_lines = ["=" * 70, "Test-suite evaluacija: Feature 1 (rank_for_patient), 33 pacijenta",
                  "=" * 70, "",
                  f"Ukupno leave-one-out trials: {len(df)} (hard/full_text_verified: {len(hard)}, "
                  f"soft/ostalo: {len(soft)})",
                  f"Preskoceno (monospecifican, nema known pozitivan): {skipped_no_positive_left}",
                  ""]

for label, sub in [("HARD (full_text_verified) -- ovo je osnova za pass/fail", hard),
                    ("soft (abstract_snippet_only/partially_verified) -- samo informativno", soft)]:
    summary_lines.append(f"--- {label} ---")
    pos = sub[sub["true_result"] == "positive"]["percentile"]
    neg = sub[sub["true_result"] == "negative"]["percentile"]
    summary_lines.append(f"  Pozitivne mete (n={len(pos)}): medijan percentil = {pos.median():.1f}% "
                          f"(zelimo NIZAK)")
    summary_lines.append(f"  Negativne mete (n={len(neg)}): medijan percentil = {neg.median():.1f}% "
                          f"(zelimo VISOK)")
    if len(pos) >= 3 and len(neg) >= 3:
        stat, pval = mannwhitneyu(pos, neg, alternative="less")
        summary_lines.append(f"  Mann-Whitney U (pozitivne < negativne po percentilu): p={pval:.4f} "
                              f"-- {'ZNACAJNO' if pval < 0.05 else 'nije znacajno'}")
    summary_lines.append("")

# =====================================================
# Posebno istaknuti slucajevi
# =====================================================

summary_lines.append("--- Istaknuti slucajevi ---")

vando4 = df[df["patient_id"] == "vando2005_pt04"]
summary_lines.append(f"\nvando2005_pt04 (monospecifican Gad c 1+, sve ostalo -): {len(vando4)} trials")
for _, r in vando4.iterrows():
    summary_lines.append(f"    sakriveno={r['hidden_protein']} (stvarno {r['true_result']}): "
                          f"rang {r['rank']}/{r['n_candidates']} ({r['percentile']:.1f}%)")

penas = df[df["patient_id"] == "penas2014_case"]
summary_lines.append(f"\npenas2014_case (EXPECTED-FAILURE test -- beta-enolaza, ne parvalbumin, "
                      f"NE tretirati kao pass/fail): {len(penas)} trials")
for _, r in penas.iterrows():
    summary_lines.append(f"    sakriveno={r['hidden_protein']} (stvarno {r['true_result']}): "
                          f"rang {r['rank']}/{r['n_candidates']} ({r['percentile']:.1f}%)")

summary_lines.append("\nuklejasokolowska2021_pt04 vs _pt18 (inverzni par -- mite+/tropomiozin- naspram mite-/tropomiozin+):")
for pid in ["uklejasokolowska2021_pt04", "uklejasokolowska2021_pt18"]:
    sub = df[df["patient_id"] == pid]
    summary_lines.append(f"  {pid}:")
    for _, r in sub.iterrows():
        summary_lines.append(f"    sakriveno={r['hidden_protein']} (stvarno {r['true_result']}): "
                              f"rang {r['rank']}/{r['n_candidates']} ({r['percentile']:.1f}%)")

# =====================================================
# FEATURE 2: kvalitativan face-validity (samo pereira2009_case)
# =====================================================

summary_lines.append("\n" + "=" * 70)
summary_lines.append("Feature 2 (graduated_introduction_path) -- KVALITATIVNO, NE validirano")
summary_lines.append("=" * 70)
summary_lines.append("Samo pereira2009_case ima resolvable cilj (Pru p 3); porcaro2016_case koristi")
summary_lines.append("samo whole-extract nazive (salmon/cod), nije testabilno na protein nivou.")
summary_lines.append("Realan protokol: Pru p 3 SLIT, DIREKTNO na cilju (ne postepeni put ka njemu),")
summary_lines.append("pa nema stvarnog 'redosleda koraka' da se uporedi sa nasim graduated path-om.")
summary_lines.append("ZAKLJUCAK: Feature 2 se NE MOZE smisleno evaluirati na ovom datasetu (n=2, oba")
summary_lines.append("bez uporedivog step-by-step protokola) -- ne pokusavati dalje bez novih podataka.")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")

with open(JSON_OUTPUT, "w") as f:
    json.dump({"n_trials": len(df), "skipped_no_positive_left": skipped_no_positive_left,
                "unresolved_components": sorted(unresolved_components),
                "records": records}, f, indent=2)

print(f"\nSaved: {SUMMARY_OUTPUT}")
print(f"Saved: {JSON_OUTPUT}")
