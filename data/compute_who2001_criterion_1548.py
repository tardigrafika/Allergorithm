"""
WHO (2001) kriterijum za potencijalnu cross-reaktivnost, po mentorkinom
predlogu: protein je potencijalni cross-reaktant ako pokazuje
  (a) >35% amino-acid identity preko kliznog prozora od 80 rezidua
      (FASTA/BLAST lokalno poravnanje), ILI
  (b) egzaktno poklapanje od >=6 (originalna) / >=8 (aktuelna praksa)
      kontinualnih amino kiselina.

Racuna se preko istog lokalnog poravnanja (Biopython PairwiseAligner,
BLOSUM62) kao postojeca BLAST-identity matrica (data/compute_blast_identity_1443.py),
ali OVDE se koristi PUNI aligned string (ne samo skalarni % identity) da bi
se moglo kliziti prozorom od 80 pozicija i traziti najduzi egzaktan
kontinualni match -- ta dva broja se NE mogu izvesti iz vec sacuvane
identity_matrix.pkl (ona cuva samo jedan globalni % po paru).

Racuna se za SVIH 1548 parova (ne samo Strong/Suspected) -- daje merljiv,
reproducibilan kriterijum koji mentorka trazi da zameni subjektivnu
"jaka homologija" formulaciju.

Izlaz: dodaje kolone u output/cross_reactive_1548.csv:
    who2001_best_window_identity_pct   -- max % identity u bilo kom 80-rezidua prozoru
    who2001_longest_exact_match        -- duzina najdužeg egzaktnog kontinualnog poklapanja
    who2001_pass                       -- True ako (a) ILI (b) uslov ispunjen
"""

from pathlib import Path

import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

WINDOW = 80
IDENTITY_THRESHOLD = 0.35
EXACT_MATCH_THRESHOLD = 8  # aktuelna praksa (originalno 6, cuvamo i taj broj za referencu)

clean = pd.read_csv(CLEAN_ALLERGENS)
clean = clean[clean["fasta_sequence"].notna() & (clean["fasta_sequence"] != "")]
name_to_seq = dict(zip(clean["official_name"], clean["fasta_sequence"]))

aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.mode = "local"
aligner.open_gap_score = -11
aligner.extend_gap_score = -1


def best_window_identity(seq_a_aligned: str, seq_b_aligned: str, window: int) -> float:
    """Klizi prozor od `window` pozicija preko poravnatih stringova (mogu sadrzati '-'
    za gap-ove), racuna % identicnih pozicija (bez gap-ova u imeniocu) po prozoru,
    vraca maksimum. Ako je poravnanje krace od window, koristi ceo aligned deo."""
    n = len(seq_a_aligned)
    if n == 0:
        return 0.0
    w = min(window, n)
    best = 0.0
    for start in range(0, n - w + 1):
        a_win = seq_a_aligned[start:start + w]
        b_win = seq_b_aligned[start:start + w]
        non_gap = sum(1 for x, y in zip(a_win, b_win) if x != "-" and y != "-")
        if non_gap == 0:
            continue
        identical = sum(1 for x, y in zip(a_win, b_win) if x == y and x != "-")
        pct = identical / non_gap
        best = max(best, pct)
    return best


def longest_exact_match(seq_a_aligned: str, seq_b_aligned: str) -> int:
    """Najduzi kontinualni niz pozicija sa identicnim amino kiselinama (bez gap-ova)."""
    longest = 0
    current = 0
    for x, y in zip(seq_a_aligned, seq_b_aligned):
        if x == y and x != "-":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


df = pd.read_csv(GOLD)
best_window_pcts, longest_matches, who_pass = [], [], []

for i, row in enumerate(df.itertuples(index=False), 1):
    seq1 = name_to_seq.get(str(row.allergen_id_1).strip())
    seq2 = name_to_seq.get(str(row.allergen_id_2).strip())
    if not seq1 or not seq2:
        best_window_pcts.append(None)
        longest_matches.append(None)
        who_pass.append(None)
        continue

    aln = aligner.align(seq1, seq2)[0]
    aligned_str = str(aln).split("\n")
    # Bio.Align formatted string: linije [target, match_line, query] ponavljaju se u blokovima --
    # koristimo aln.aligned indekse umesto parsiranja stringa (pouzdanije).
    target_aligned, query_aligned = "", ""
    for (t_start, t_end), (q_start, q_end) in zip(aln.aligned[0], aln.aligned[1]):
        target_aligned += seq1[t_start:t_end]
        query_aligned += seq2[q_start:q_end]

    win_pct = best_window_identity(target_aligned, query_aligned, WINDOW)
    exact_len = longest_exact_match(target_aligned, query_aligned)
    passes = (win_pct > IDENTITY_THRESHOLD) or (exact_len >= EXACT_MATCH_THRESHOLD)

    best_window_pcts.append(round(win_pct * 100, 1))
    longest_matches.append(exact_len)
    who_pass.append(bool(passes))

    if i % 200 == 0:
        print(f"  {i}/{len(df)}", flush=True)

df["who2001_best_window_identity_pct"] = best_window_pcts
df["who2001_longest_exact_match"] = longest_matches
df["who2001_pass"] = who_pass
df.to_csv(GOLD, index=False)

print(f"\nGotovo. {sum(1 for x in who_pass if x is True)}/{sum(1 for x in who_pass if x is not None)} parova prolazi WHO(2001) kriterijum.")
print("\nPo evidence_level tier-u, % koji NE prolazi (kandidati za preispitivanje):")
tmp = df[df["who2001_pass"].notna()]
for tier, group in tmp.groupby("evidence_level"):
    if len(group) < 3:
        continue
    fail_pct = (~group["who2001_pass"]).mean() * 100
    print(f"  {tier:60s} n={len(group):4d}  ne-prolazi={fail_pct:5.1f}%")
