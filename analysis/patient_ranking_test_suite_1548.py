"""
Konsolidovan, ponovljiv test-skript za SVIH 5 potvrdjenih real-world
pacijent-slucajeva (necirkularni, van gold dataseta) protiv TRENUTNE verzije
alata (ml/patient_ranking_1548.py -- ukljucujuci negativne unose). Zamenjuje
razbacane jednokratne provere iz prethodnih koraka sesije jednim skriptom
koji se moze ponovo pokrenuti kad god se alat promeni.

Slucajevi:
  1) Mothes-Luksch 2017 -- Pru p 3 (nsLTP) vs Bet v 1 (PR-10) vs Phl p 12
     (profilin), populaciona konkordanca, nema pojedinacnog pacijent-nivoa
     pozitivno/negativno para -- testira se kao pattern-consistency, ne
     strogo tacno/netacno.
  2) Ukleja-Sokolowska 2021 -- Der p 10 (grinja tropomiozin) vs krustacejski
     tropomiozini (proxy za Pen m 1, koji nije u datasetu) vs Der p 1/Der p 2
     (ista grinja, druga familija).
  3) Profilin desenzitizacija (PMC5806761) -- Pru p 4 (profilin, DBPCFC+)
     vs Pru p 3 (nsLTP)/Pru p 1 (PR-10), nema potvrdjenog ishoda za target-e
     (pattern-consistency, isto kao slucaj 1).
  4) Limao 2023 -- Ara h 2 POZ (kikiriki), Ana o 1/Ana o 2/Pis v POZ (kesju/
     pistaci), Jug r NEG (orah) -- koristi NOVI known_negative_names.
  5) Pereira 2009 -- Pru p 3 POZ (breskva, DBPCFC), Pru p 4 NEG (profilin) --
     koristi known_negative_names.
  6) Penas 2014 -- Sal s 1 POZ (losos), Onc m 1 POZ (pastrmka), Gad m 1 NEG
     (bakalar, IgE cilja beta-enolazu, ne parvalbumin) -- known_negative_names.

Izlaz:
    output/patient_ranking_test_suite_1548_summary.txt
"""

from ml.patient_ranking_1548 import CrossReactivityRanker

ranker = CrossReactivityRanker()
n_pool = ranker.n_pool

summary_lines = ["=" * 70, "Test-suite: patient_ranking_1548 na 6 real-world slucajeva (necirkularno)",
                  "=" * 70, ""]


def best_rank(df, exact_name=None, prefix=None):
    if exact_name is not None:
        m = df[df["candidate_name"] == exact_name]
    else:
        m = df[df["candidate_name"].str.startswith(prefix)]
    if len(m) == 0:
        return None
    return m.iloc[0]


def report(lines, label, row, n, expect):
    if row is None:
        lines.append(f"    {label}: [nije nadjen u pool-u]")
        return
    pct = row["rank"] / n * 100
    lines.append(f"    {label}: rang {int(row['rank'])}/{n} (top {pct:.1f}%) -- ocekivano: {expect}")


# =====================================================
# Slucaj 1: Mothes-Luksch 2017 (pattern-consistency)
# =====================================================
summary_lines.append("--- 1) Mothes-Luksch 2017: nsLTP vs PR-10 vs profilin (populaciona konkordanca 46%) ---")
for known, targets in [("Pru p 3.0101", [("Bet v 1", "prefix", "NISKO")]),
                        ("Pru p 4.0101", [("Pru p 3", "prefix", "NISKO"), ("Pru p 1", "prefix", "umereno")])]:
    r = ranker.rank_for_patient([known])
    summary_lines.append(f"  Upit: {known}")
    for target, kind, expect in targets:
        row = best_rank(r, prefix=target) if kind == "prefix" else best_rank(r, exact_name=target)
        report(summary_lines, target, row, n_pool, expect)
summary_lines.append("")

# =====================================================
# Slucaj 2: Ukleja-Sokolowska 2021
# =====================================================
summary_lines.append("--- 2) Ukleja-Sokolowska 2021: Der p 10 vs krustacejski tropomiozini vs Der p 1/2 ---")
r = ranker.rank_for_patient(["Der p 10.0101"])
for target, expect in [("Pan b 1.0101", "VISOKO (krustacejski tropomiozin)"),
                        ("Lit v 1.0101", "VISOKO"), ("Cra c 1.0101", "VISOKO"), ("Met e 1.0101", "VISOKO"),
                        ("Der p 1.0101", "median (ista grinja, druga familija)"),
                        ("Der p 2.0101", "median")]:
    row = best_rank(r, exact_name=target)
    report(summary_lines, target, row, n_pool, expect)
summary_lines.append("")

# =====================================================
# Slucaj 4: Limao 2023 -- KORISTI known_negative_names
# =====================================================
summary_lines.append("--- 4) Limao 2023: Ara h 2 POZ, Jug r 1 NEG (koristi negativni unos) ---")
r = ranker.rank_for_patient(["Ara h 2.0101"], known_negative_names=["Jug r 1.0101"])
jugr_present = "Jug r 1.0101" in r["candidate_name"].values
summary_lines.append(f"    Jug r 1 iskljucen iz predloga (vec testiran): "
                      f"{'DA, ispravno' if not jugr_present else 'NE, GRESKA'}")
for target, expect in [("Ana o 1", "VISOKO (POZ, OFC potvrdjeno)"), ("Ana o 2", "umereno-visoko (POZ)"),
                        ("Pis v", "VISOKO (POZ)")]:
    row = best_rank(r, prefix=target)
    report(summary_lines, target, row, len(r), expect)
summary_lines.append("")

# =====================================================
# Slucaj 5: Pereira 2009 -- KORISTI known_negative_names
# =====================================================
summary_lines.append("--- 5) Pereira 2009: Pru p 3 POZ (DBPCFC), Pru p 4 NEG (koristi negativni unos) ---")
r = ranker.rank_for_patient(["Pru p 3.0101"], known_negative_names=["Pru p 4.0101"])
prup4_present = "Pru p 4.0101" in r["candidate_name"].values
summary_lines.append(f"    Pru p 4 iskljucen iz predloga (vec testiran): "
                      f"{'DA, ispravno' if not prup4_present else 'NE, GRESKA'}")
summary_lines.append("")

# =====================================================
# Slucaj 6: Penas 2014 -- KORISTI known_negative_names
# =====================================================
summary_lines.append("--- 6) Penas 2014: Sal s 1 POZ, Onc m 1 POZ, Gad m 1 NEG (koristi negativni unos) ---")
r = ranker.rank_for_patient(["Sal s 1.0101"], known_negative_names=["Gad m 1.0101"])
gadm_present = "Gad m 1.0101" in r["candidate_name"].values
summary_lines.append(f"    Gad m 1 iskljucen iz predloga (vec testiran): "
                      f"{'DA, ispravno' if not gadm_present else 'NE, GRESKA'}")
row = best_rank(r, exact_name="Onc m 1.0101")
report(summary_lines, "Onc m 1.0101", row, len(r), "VISOKO (POZ, drugi salmonid)")
summary_lines.append("")
summary_lines.append("NAPOMENA: Gad m 1 je NEGATIVAN kod ovog pacijenta jer je njegov IgE ciljao")
summary_lines.append("beta-enolazu, ne parvalbumin -- alat to ne moze da zna iz sekvence (vidi")
summary_lines.append("real_world_case_validation diskusiju o molekularnom vs klinickom signalu).")
summary_lines.append("")

summary_text = "\n".join(summary_lines)
print(summary_text)
with open("/home/lana/ALERGRAF/output/patient_ranking_test_suite_1548_summary.txt", "w") as f:
    f.write(summary_text + "\n")
print("\nSaved: output/patient_ranking_test_suite_1548_summary.txt")
