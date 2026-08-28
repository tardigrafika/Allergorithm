"""
SEKUNDARNI, eksplicitno obelezen "best-guess" run za penas2014_case --
GLAVNI evaluator (test/evaluate_test_cases.py) je ispravno preskocio ovaj
slucaj jer test_cases.json koristi generalne nazive komponenti ("Sal s",
"Gad m", bez broja izoforme, tacno kako original ISAC panel izvestava),
koji se ne mogu automatski resolve-ovati na konkretan protein.

Ovde RUCNO pretpostavljamo najverovatniju konkretnu izoformu (isto sto je
ranije u sesiji radjeno ad hoc) -- OVO JE PRETPOSTAVKA, NE VERIFIKOVANO IZ
IZVORA. Rezultat se NE racuna u glavnu evaluaciju, prijavljuje se odvojeno.

Mapiranje (best-guess, obrazlozenje u komentaru po redu):
  "Onc m (rainbow trout parvalbumin marker)" -> Onc m 1.0101  (jedini Onc m u pool-u)
  "Sal s (salmon parvalbumin marker)"        -> Sal s 1.0101  (dominantan losos parvalbumin)
  "Gad m (cod parvalbumin marker)"           -> Gad m 1.0101  (dominantan bakalar parvalbumin)
  "Thu a (tuna parvalbumin marker)"          -> Thu a 1.0101  (jedini Thu a u pool-u)
  "Sol so (sole parvalbumin marker)"         -> NEMA u pool-u cak ni sa best-guess, izostavljeno
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.patient_ranking_1548 import CrossReactivityRanker  # noqa: E402

ranker = CrossReactivityRanker()

known_pos = ["Onc m 1.0101", "Sal s 1.0101"]
known_neg = ["Gad m 1.0101", "Thu a 1.0101"]

print("=" * 70)
print("penas2014_case -- BEST-GUESS mapping (nije verifikovano, pretpostavka)")
print("=" * 70)
print(f"Known pozitivni (best-guess): {known_pos}")
print(f"Known negativni (best-guess): {known_neg}")
print()

for hidden_known_pos, hidden_known_neg, hidden_name, true_result in [
    (["Sal s 1.0101"], known_neg, "Onc m 1.0101", "positive"),
    (["Onc m 1.0101"], known_neg, "Sal s 1.0101", "positive"),
    (known_pos, ["Thu a 1.0101"], "Gad m 1.0101", "negative"),
    (known_pos, ["Gad m 1.0101"], "Thu a 1.0101", "negative"),
]:
    result_df = ranker.rank_for_patient(hidden_known_pos, known_negative_names=hidden_known_neg)
    row = result_df[result_df["candidate_name"] == hidden_name]
    if len(row) == 0:
        print(f"  sakriveno={hidden_name}: nije nadjen u izlazu (neocekivano)")
        continue
    rank = int(row.iloc[0]["rank"])
    n = len(result_df)
    pct = rank / n * 100
    flag = "OCEKIVANO (dobro rangiran)" if (true_result == "positive") == (pct < 50) else \
           "NEOCEKIVANO (isti obrazac kao Jug r1/vando2005 -- molekularna slicnost visoka uprkos kliniсki negativnom)"
    print(f"  sakriveno={hidden_name} (stvarno {true_result}): rang {rank}/{n} ({pct:.1f}%) -- {flag}")

print()
print("NAPOMENA: ovo je best-guess mapiranje generickih ISAC marker naziva na")
print("konkretne izoforme -- NIJE verifikovano iz originalnog izvora (Penas i sar.")
print("2014 ne navode tacan broj komponente). Ne racunati u glavnu evaluaciju.")
