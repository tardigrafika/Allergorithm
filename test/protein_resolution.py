"""
Rezolucija naziva proteina iz test_cases.json (slobodan tekst iz literature)
na sluzbeni official_name u nasem pool-u. Izdvojeno iz evaluate_test_cases.py
(sada deljeno sa evaluate_test_cases_mlp_hadamard.py i buducim skriptama)
posle provere STVARNOG konteksta svakog nerezolvovanog naziva (nista nije
mapirano nagadjanjem -- videti obrazlozenje kod svakog pravila ispod).

Provereno preko test_cases.json + clean_allergens.csv (avgust 2026):

1) "r"/"n" prefiks (rOle e 1, nBet v 1...) -- standardna literaturna notacija
   za rekombinantni/prirodni alergen, NIJE deo sluzbenog WHO/IUIS imena.
   Bezbedno skidanje prefiksa PRE poredjenja sa pool-om.

2) Eksplicitni sinonimi -- SAMO za slucajeve gde je sluzbeno WHO/IUIS ime
   NEDVOSMISLENO poznato i izvor doslovno opisuje TACNO taj protein:
     - "omega-5 gliadin" = Tri a 19 (WHO/IUIS sluzbeno registrovano ime za
       ovaj protein JE "omega-5 gliadin" -- ovo nije sinonim koji nagadjamo,
       to je definicija alergena u bazi)
     - "Sus s (pork albumin)" = Sus s 1 -- u nasem pool-u POSTOJI SAMO JEDAN
       "Sus s" protein (Sus s 1.0101, svinjski serumski albumin), pa je
       oznaka nedvosmislena, za razliku od "Gad m"/"Sal s"/"Thu a" (ispod).

3) NAMERNO OSTAJU NEREZOLVOVANI (proveril i potvrdjeno ambiguozni ili
   nepostojeci u pool-u, NE regex bug):
     - "Gad m (cod parvalbumin marker)", "Sal s (salmon parvalbumin marker)",
       "Thu a (tuna parvalbumin marker)" -- svaki od ovih genus+species
       prefiksa odgovara VISE RAZLICITIH proteina u pool-u (npr. "Sal s"
       pokriva 10 razlicitih Sal s X.Y proteina, ne samo parvalbumin Sal s 1)
       -- ISAC panel marker ne govori KOJI je tacno testiran. Ovo je bas
       razlog zasto postoji odvojen evaluate_penas_bestguess.py.
     - "Onc m (rainbow trout parvalbumin marker)" -- pool ima Onc m 1.0101
       i Onc m 1.0201 (2 alelne varijante ISTOG proteina), i dalje dvosmisleno
       koja tacno, ne diramo.
     - "Sol so (sole parvalbumin marker)" -- STVARNA RUPA U PODACIMA, sole
       parvalbumin uopste NIJE u pool-u (0 poklapanja).
     - "Pen m 1" -- STVARNA RUPA U PODACIMA: sirom (shrimp) tropomiozin
       "Pen m 1" NIJE u clean_allergens.csv (pool ima samo Pen m 13/14,
       razlicite proteine) -- pogadja 5 pacijenata (uklejasokolowska2021
       pt01/04/05/09/18). Ovo NIJE regex bug, treba dodati protein u dataset
       (van obima ovog fajla).
     - "rPhl p 1 + rPhl p 5b" -- KOMBINOVAN test dva proteina odjednom,
       jedan rezultat za oba -- ne moze se cisto svesti na JEDAN protein bez
       promene JSON strukture (deljenje u 2 zapisa), namerno ne diramo ovde.
     - Sve whole-extract/band/panel/CCD/alpha-gal oznake (nisu proteini u
       nasem smislu, ili izvor eksplicitno kaze "component not identified").
"""

import re

SYNONYMS = {
    "omega-5 gliadin": "Tri a 19",
    "Sus s (pork albumin)": "Sus s 1",
}

_R_N_PREFIX = re.compile(r"^[rn]([A-Z][a-z]{1,3} [a-z] )")
_TRAILING_PAREN = re.compile(r"\s*\(([^()]+)\)\s*$")  # "... (X)" na kraju stringa -> X


def _match_pool(name: str, pool_names: list[str]) -> str | None:
    """Tacno poklapanje (npr. json_name je vec potpuno ime kao 'Hev b 6.02')
    PA TEK ONDA prefiks+tacka poklapanje (json_name je genus+species+broj
    bez konkretne izoforme, npr. 'Sal s 1' -> 'Sal s 1.0101')."""
    if name in pool_names:
        return name
    pattern = re.compile(r"^" + re.escape(name) + r"\.")
    matches = [n for n in pool_names if pattern.match(n)]
    if not matches:
        return None
    return sorted(matches)[0]  # najniza isoform oznaka pobedjuje (deterministicki izbor)


def resolve_protein(json_name: str, pool_names: list[str]) -> str | None:
    """Vraca najbolje poklapanje official_name u pool_names, ili None.
    pool_names mora biti sortirana lista (za deterministicki izbor izmedju
    izoformi kad ih ima vise -- najniza oznaka pobedjuje, isto svuda u sesiji)."""
    name = json_name.strip()

    if name in SYNONYMS:
        name = SYNONYMS[name]
    else:
        stripped = _R_N_PREFIX.match(name)
        if stripped:
            name = name[1:]  # skini "r"/"n" prefiks, ostatak vec izgleda kao sluzbeno ime

    direct = _match_pool(name, pool_names)
    if direct is not None:
        return direct

    # Zagrada na kraju -- moze ici u OBA smera, izvor pise ili "opisno ime
    # (sluzbeni kod)" (npr. "rainbow trout parvalbumin (Onc m 1)") ili
    # "sluzbeni kod (sinonim)" (npr. "Tri a 19 (omega-5 gliadin)") -- probaj
    # OBA dela, nikad ne izmisljamo kod koji izvor vec nije naveo negde u stringu.
    paren_match = _TRAILING_PAREN.search(name)
    if paren_match:
        inside = _match_pool(paren_match.group(1).strip(), pool_names)
        if inside is not None:
            return inside
        prefix = name[:paren_match.start()].strip()
        prefix_result = _match_pool(prefix, pool_names)
        if prefix_result is not None:
            return prefix_result

    return None
