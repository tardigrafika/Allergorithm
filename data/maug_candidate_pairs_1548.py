"""
Kandidat-parovi izvuceni iz EAACI Molecular Allergology User's Guide 2.0,
Sekcija C (Cross-Reactive Molecules, C01-C11) + appendix tabela -- ekstrakcija
odradjena preko 5 agenata (1 opsti pregled + 4 chunk-by-chunk prolaza sa
preklapanjem, plus 1 poseban prolaz kroz appendix tabelu), svaki citat je
DOSLOVNO preuzet iz PDF-a (data/... nije cuvano u repo-u, izvor:
EAACI MAUG 2.0, https://eaaci.org/books/molecular-allergology-users-guide-2-0/,
preuzeto avgust 2026).

SAMO najjace potkrepljeni kandidati ukljuceni ovde (IgE inhibicija/cross-
inhibicija, eksplicitna klinicka cross-reaktivnost, ili eksplicitna
NEGACIJA cross-reaktivnosti) -- slabiji nalazi (samo ko-senzitizacija,
referenca-samo-naslov, homologija bez potvrdjene reaktivnosti) namerno
izostavljeni iz ovog prolaza, mogu se dodati kasnije ako zatreba.

Poznate ispravke OCR/tipografskih gresaka u izvoru (obrazlozeno, ne
izmisljeno):
  - "Bla o 1" -> Bla g 1 (recenica sama kaze "from the German Cockroach",
    a Bla g = Blattella germanica prefiks; Bla o 1 ne postoji u pool-u)
  - C04 (Serum albumins) tabela pogresno navodi "Equ c 1" za konjski serum
    albumin -- Equ c 1 je u stvari LIPOKALIN (potvrdjeno u C07 poglavlju
    istog dokumenta), prava skracenica za konjski albumin je Equ c 3
    (koriscena dosledno u telu teksta/slucajevima iste C04 sekcije) --
    koristimo Equ c 3 za sve C04 (serum albumin) parove, Equ c 1 SAMO za
    C07 (lipokalin) parove.
  - "Per a 10" (Ani s 3 partner) NIJE pouzdano identifikovan kao Per a 7
    (tropomiozin) -- postoje oba imena u pool-u kao razliciti proteini,
    par IZOSTAVLJEN iz ovog prolaza (nesigurno).
  - "Sol so 1" -> Sole s 1 (verovatno formatting razlika, isti pool str)

Struktura: lista dict-ova sa id_1, id_2 (KAKO SU NAPISANI u izvoru, pre
resolucije), evidence_level (predlog), reference, note (citat), i
is_negative (True za eksplicitne "ne cross-reaguje" nalaze -- kandidati
za crossreactivity.csv NEGATIVE dataset, ne pozitivan gold set).
"""

MAUG_CITATION = "EAACI Molecular Allergology User's Guide 2.0 (Hoffmann-Sommergruber et al 2023, Allergy)"

POSITIVE_CANDIDATES = [
    # C02 -- PR-10
    {"id_1": "Bet v 1", "id_2": "Gly m 4", "evidence": "Strong evidence",
     "note": "70% of Bet v 1-allergic patients serologically cross-reactive with Gly m 4 (soy); most prevalent soy allergy mechanism in Central/Northern Europe."},
    {"id_1": "Bet v 1", "id_2": "Ara h 8", "evidence": "Strong evidence",
     "note": "High sensitisation rates to peanut in Central Europe due to cross-reactive natural Bet v 1-homologue Ara h 8."},
    {"id_1": "Bet v 1", "id_2": "Pru p 1", "evidence": "Strong evidence",
     "note": "Positive specific IgE to Bet v 1-homologue Pru p 1 (peach) demonstrates allergic sensitisation; cross-reactivity between birch pollen and peach."},
    {"id_1": "Bet v 1", "id_2": "Aln g 1", "evidence": "Strong evidence",
     "note": "Alder major allergen Aln g 1 has 'high degree of sequence homology' with Bet v 1 (appendix table)."},
    {"id_1": "Cor a 1", "id_2": "Ara h 8", "evidence": "Strong evidence",
     "note": "High correlation between sensitisation to Cor a 1 and Ara h 8, indicating PR-10 cross-reactivity as major cause of hazelnut/peanut co-sensitisation."},
    {"id_1": "Cor a 9", "id_2": "Cor a 1", "evidence": "Suspected (homology-based)",
     "note": "Most Cor a 9 sensitizations caused secondarily by the birch-pollen-homologue Cor a 1 (i.e. Cor a 9 reactivity often reflects PR-10 cross-reactivity, not primary Cor a 9 sensitisation)."},
    {"id_1": "Cor a 14", "id_2": "Cor a 1", "evidence": "Suspected (homology-based)",
     "note": "Same mechanism as Cor a 9/Cor a 1 above (appendix table)."},

    # C03 -- nsLTP
    {"id_1": "Pru p 3", "id_2": "Art v 3", "evidence": "Strong evidence",
     "note": "\"A typical example is the cross-reactivity between Pru p 3 and Art v 3 in China.\""},
    {"id_1": "Ole e 7", "id_2": "Pru p 3", "evidence": "Suspected (homology-based)",
     "note": "Cross-reactivity observed in some allergic patients despite <20% sequence identity; tertiary structure of both nsLTP rather similar."},
    {"id_1": "Pla a 3", "id_2": "Pru p 3", "evidence": "Suspected (homology-based)",
     "note": "Art v 3 and Pla a 3 display partial cross-reactivity with Pru p 3."},
    {"id_1": "Ara h 9", "id_2": "Pru p 3", "evidence": "Strong evidence",
     "note": "Peanut LTP (Ara h 9) concomitant allergy; Pru p 3 immunotherapy showed clinical benefit extending to peanut, regulatory response to both."},
    {"id_1": "Cor a 8", "id_2": "Pru p 3", "evidence": "Suspected (homology-based)",
     "note": "Cross-sensitisation to Cor a 8 (hazelnut) may occur in patients primarily sensitised to Pru p 3 (LTP syndrome)."},
    {"id_1": "Jug r 3", "id_2": "Pru p 3", "evidence": "Suspected (homology-based)",
     "note": "Cross-sensitisation to walnut Jug r 3 in LTP-syndrome patients primarily sensitised to Pru p 3."},
    {"id_1": "Mal d 3", "id_2": "Pru p 3", "evidence": "Suspected (homology-based)",
     "note": "Cross-sensitisation to apple Mal d 3 nsLTP in LTP-syndrome patients."},

    # C04 -- Serum albumins (Equ c 1 corrected to Equ c 3, see docstring)
    {"id_1": "Fel d 2", "id_2": "Sus s 1", "evidence": "Confirmed",
     "note": "Clinical case: IgE to Sus s 1 totally inhibited by prior incubation with Fel d 2 -- pork-cat syndrome, IgE inhibition assay."},
    {"id_1": "Can f 3", "id_2": "Equ c 3", "evidence": "Confirmed",
     "note": "Clinical case: inhibition experiments confirmed primary sensitisation to Can f 3, IgE-cross-reactivity to Equ c 3 (horse-meat allergy mediated by dog-allergy)."},
    {"id_1": "Fel d 2", "id_2": "Gal d 5", "evidence": "Suspected (homology-based; reduced cross-reactivity)",
     "note": "Clinical cross-reactivity mammal(Fel d 2)-to-bird(Gal d 5) documented but RARE, unidirectional (mammal to bird only)."},
    {"id_1": "Sus s 1", "id_2": "Gal d 5", "evidence": "Suspected (homology-based; reduced cross-reactivity)",
     "note": "Same rare, unidirectional mammal-to-bird cross-reactivity as Fel d 2-Gal d 5."},
    {"id_1": "Fel d 2", "id_2": "Bos d 6", "evidence": "Strong evidence",
     "note": "Clinical case: cross-reactivity between cat and dog serum albumins and cattle (Bos d 6) serum albumin causing milk allergy in dog/cat sensitised patient."},
    {"id_1": "Can f 3", "id_2": "Bos d 6", "evidence": "Strong evidence",
     "note": "Same clinical case as Fel d 2-Bos d 6 above."},

    # C05 -- Tropomyosins
    {"id_1": "Der p 10", "id_2": "Pen a 1", "evidence": "Strong evidence",
     "note": "Primary sensitiser (inhaled Der p 10) vs Pen a 1 (shrimp) affects downstream tolerance level to crustaceans/molluscs/cephalopods -- established tropomyosin cross-reactivity axis."},
    {"id_1": "Blo t 10", "id_2": "Pen a 1", "evidence": "Strong evidence",
     "note": "Same tropomyosin cross-reactivity axis as Der p 10-Pen a 1."},
    {"id_1": "Bla g 7", "id_2": "Pen a 1", "evidence": "Strong evidence",
     "note": "Same tropomyosin cross-reactivity axis (cockroach vs shrimp)."},

    # C06 -- Polcalcins
    {"id_1": "Phl p 7", "id_2": "Bet v 4", "evidence": "Suspected (homology-based)",
     "note": "IgE inhibition assay showed differential (not identical) inhibition patterns between rBet v 4 and rPhl p 7, but both usable diagnostically for polcalcin sensitisation (family-level cross-reactivity, ~77% avg identity)."},

    # C07 -- Lipocalins (Equ c 1 correct here, this IS the lipocalin)
    {"id_1": "Equ c 1", "id_2": "Fel d 4", "evidence": "Strong evidence",
     "note": "IgE inhibition studies, cross-react at low doses; up to 67% sequence identity within this group."},
    {"id_1": "Equ c 1", "id_2": "Can f 6", "evidence": "Strong evidence",
     "note": "Same cross-reactive lipocalin group as Equ c 1-Fel d 4, IgE inhibition."},
    {"id_1": "Fel d 4", "id_2": "Can f 6", "evidence": "Strong evidence",
     "note": "Same cross-reactive lipocalin group, IgE inhibition; clinical case: Can f 6 IgE completely inhibited by Fel d 4."},
    {"id_1": "Cav p 6", "id_2": "Fel d 4", "evidence": "Strong evidence",
     "note": "Cav p 6 found cross-reactive with Fel d 4 (and Can f 6)."},
    {"id_1": "Cav p 6", "id_2": "Can f 6", "evidence": "Strong evidence",
     "note": "Cav p 6 found cross-reactive with Can f 6 (and Fel d 4)."},
    {"id_1": "Can f 1", "id_2": "Fel d 7", "evidence": "Strong evidence",
     "note": "62% sequence identity, IgE cross-reactivity confirmed in polysensitised patients."},
    {"id_1": "Fel d 4", "id_2": "Can f 2", "evidence": "Suspected (homology-based; low sequence identity)",
     "note": "Only 25% overall identity, but short high-identity epitope stretches lead to patient-dependent IgE cross-reactivity."},
    {"id_1": "Mus m 1", "id_2": "Rat n 1", "evidence": "Suspected (homology-based)",
     "note": "\"Mus m 1 may cross-react with Rat n 1.\""},
    {"id_1": "Equ c 1", "id_2": "Equ c 3", "evidence": "Strong evidence",
     "note": "\"Horse allergens Equ c 1 and Equ c 3 are both cross-reactive.\" (Note: distinct proteins, lipocalin vs albumin, cross-reactivity here is as stated in source, unusual cross-family finding.)"},

    # C08 -- Seed storage proteins
    {"id_1": "Ara h 2", "id_2": "Ber e 1", "evidence": "Suspected (homology-based)",
     "note": "In vitro cross-reactivity shown between peanut Ara h 2 and Brazil nut Ber e 1 2S albumin; clinical relevance still debated."},
    {"id_1": "Jug r 6", "id_2": "Cor a 11", "evidence": "Strong evidence",
     "note": "Remarkable in vitro cross-reactivity between walnut Jug r 6 and hazelnut 7S globulin Cor a 11."},
    {"id_1": "Jug r 6", "id_2": "Pis v 3", "evidence": "Strong evidence",
     "note": "Remarkable in vitro cross-reactivity, walnut Jug r 6 vs pistachio Pis v 3."},
    {"id_1": "Jug r 6", "id_2": "Ses i 3", "evidence": "Strong evidence",
     "note": "Remarkable in vitro cross-reactivity, walnut Jug r 6 vs sesame Ses i 3."},
    {"id_1": "Pis s 1", "id_2": "Ara h 1", "evidence": "Strong evidence",
     "note": "Pea anaphylaxis + peanut allergy explained by cross-reactive IgE between Pis s 1 and Ara h 1."},
    {"id_1": "Pis s 1", "id_2": "Len c 1", "evidence": "Confirmed",
     "note": "IgE binding \"completely cross-reactive\" in vitro between pea Pis s 1 and lentil Len c 1 (clinical relevance not yet confirmed)."},
    {"id_1": "Cor a 9", "id_2": "Jug r 4", "evidence": "Strong evidence",
     "note": "IgE to 11S globulins hazelnut Cor a 9 and walnut Jug r 4 cross-reacted in allergic patients."},
    {"id_1": "Cor a 9", "id_2": "Ara h 3", "evidence": "Suspected (homology-based)",
     "note": "\"A certain degree of cross-reactivity\" shown between hazelnut Cor a 9 and peanut Ara h 3."},
    {"id_1": "Ara h 1", "id_2": "Ara h 3", "evidence": "Strong evidence",
     "note": "IgE cross-reactive to non-homologous peanut allergens Ara h 1 (vicilin) and Ara h 3 (legumin) confirmed via high-affinity antibody analysis."},
    {"id_1": "Ara h 1", "id_2": "Ara h 2", "evidence": "Strong evidence",
     "note": "IgE cross-reactive to non-homologous peanut allergens Ara h 1 and Ara h 2 confirmed via high-affinity antibody analysis."},
    {"id_1": "Ara h 2", "id_2": "Ara h 3", "evidence": "Strong evidence",
     "note": "IgE cross-reactive to non-homologous peanut allergens Ara h 2 and Ara h 3 confirmed via high-affinity antibody analysis."},
    {"id_1": "Ara h 2", "id_2": "Pru du 6", "evidence": "Strong evidence",
     "note": "Cross-reactivity demonstrated between peanut Ara h 2 (2S albumin) and almond legumin Pru du 6 (non-homologous)."},
    {"id_1": "Ara h 2", "id_2": "Jug r 2", "evidence": "Strong evidence",
     "note": "Cross-reactivity demonstrated between peanut Ara h 2 and walnut vicilin Jug r 2 (non-homologous)."},
    {"id_1": "Pru du 6", "id_2": "Jug r 2", "evidence": "Strong evidence",
     "note": "Cross-reactivity demonstrated between almond legumin Pru du 6 and walnut vicilin Jug r 2 (non-homologous)."},
    {"id_1": "Gly m 5", "id_2": "Gly m 6", "evidence": "Strong evidence",
     "note": "IgE cross-reactivity between non-related soy vicilin Gly m 5 and soy legumin Gly m 6; explains soy formula reactions in cow's-milk allergic patients."},

    # C09 -- Gibberellin-regulated proteins
    {"id_1": "Pru p 7", "id_2": "Cit s 7", "evidence": "Strong evidence",
     "note": "PFAS between peach/citrus and cypress pollen explained by IgE cross-reactivity; clinical patient reactive to Cap a 7/Pru p 7/Cit s 7/Cry j 7."},
    {"id_1": "Cap a 7", "id_2": "Pru p 7", "evidence": "Strong evidence",
     "note": "Single patient allergic to all 4 named GRPs (Cap a 7, Pru p 7, Cit s 7, Cry j 7), demonstrating clinical cross-reactivity relevance."},
    {"id_1": "Cap a 7", "id_2": "Cit s 7", "evidence": "Strong evidence",
     "note": "Same patient/clinical cross-reactivity as above."},
    {"id_1": "Cap a 7", "id_2": "Cry j 7", "evidence": "Confirmed",
     "note": "IgE inhibition assay (competitive immunoblot): reactivities inhibited by both Cry j 7 and Cap a 7."},
    {"id_1": "Pru p 7", "id_2": "Cry j 7", "evidence": "Confirmed",
     "note": "IgE inhibition assay (competitive immunoblot): reactivity inhibited by Cry j 7, Cap a 7 or Pru p 7."},
    {"id_1": "Cit s 7", "id_2": "Cry j 7", "evidence": "Strong evidence",
     "note": "Same single-patient clinical cross-reactivity as Cap a 7/Pru p 7/Cit s 7/Cry j 7."},
    {"id_1": "Cup s 7", "id_2": "Cry j 7", "evidence": "Strong evidence",
     "note": "Homologous GRP allergens with \"similar fruit cross-reactivities\"; 94% sequence identity (Table 3)."},
    {"id_1": "Cup s 7", "id_2": "Pru p 7", "evidence": "Confirmed",
     "note": "Basophil activation test positive for both Cup s 7 and Pru p 7 in same patient (in contrast to negative snakin-1)."},
    {"id_1": "Pru p 7", "id_2": "Pru p 3", "evidence": "Suspected (homology-based)",
     "note": "Clinical case: dual reactivity to Pru p 7 and Pru p 3 demonstrated in same patient over time (co-sensitisation, not confirmed molecular cross-reactivity)."},

    # C11 -- Parvalbumins
    {"id_1": "Gad m 1", "id_2": "Sal s 1", "evidence": "Confirmed (patient-dependent, not universal)",
     "note": "Explicitly labeled \"cross-reactive parvalbumins\" (Fig 2); clinical Case 1: polysensitization confirmed via cross-reactive IgE to homologue parvalbumins Gad m 1/Sal s 1/Thu a 1. CAVEAT: same source's Case 2 shows a DIFFERENT (salmonid-monosensitized) patient where Sal s 1 did NOT cross-react with Gad m 1 -- patient-dependent, not universal."},
    {"id_1": "Gad m 1", "id_2": "Thu a 1", "evidence": "Confirmed",
     "note": "Same clinical case + Fig 2/6 cross-reactive parvalbumin labeling as Gad m 1-Sal s 1."},
    {"id_1": "Sal s 1", "id_2": "Thu a 1", "evidence": "Confirmed",
     "note": "Same clinical case (Case 1): cross-reactive IgE antibodies to homologue parvalbumins from cod/salmon/tuna."},
    {"id_1": "Sal s 1", "id_2": "Onc m 1", "evidence": "Strong evidence",
     "note": "Clinical case: salmonid-specific shared epitope, co-positive IgE, salmon/trout monospecific sensitisation cluster (distinguished from cod/carp/tuna, see negative candidates)."},
    {"id_1": "Gad c 1", "id_2": "Cro p 1", "evidence": "Confirmed",
     "note": "Clinical case: anaphylaxis to crocodile meat explained by primary fish-parvalbumin sensitisation with cross-reactivity to crocodile homologue Cro p 1."},
]

NEGATIVE_CANDIDATES = [
    # C04 -- Serum albumins
    {"id_1": "Gal d 5", "id_2": "Gal d 2", "note": "\"Chicken serum albumin Gal d 5 does not share any sequence identity with ovalbumin Gal d 2.\" -- explicit, clean negative."},

    # C06 -- Polcalcins (Phl p 7 vs 3-EF-hand and 4-EF-hand family members)
    {"id_1": "Phl p 7", "id_2": "Bet v 3", "note": "\"Phl p 7 and related two EF-hand allergens do not share epitopes with other 3-EF-hand calcium binding proteins (e.g., Bet v 3...)\" -- explicit no shared epitopes."},
    {"id_1": "Phl p 7", "id_2": "Ole e 8", "note": "Same source: Phl p 7 does not share epitopes with 4-EF-hand allergens (Ole e 8, Jun o 4, Amb a 10)."},

    # C07 -- Lipocalins (contrasting weak-inhibition case)
    # (kept out of negative set -- case showed WEAK not absent inhibition, ambiguous, skip)

    # C08 -- Seed storage
    {"id_1": "Cor a 14", "id_2": "Ara h 2", "note": "\"Negligible in vitro cross-reactivity was shown for IgE to hazelnut Cor a 14 and peanut Ara h 2.\""},
    {"id_1": "Jug r 6", "id_2": "Jug r 2", "note": "Explicit contrast: Jug r 6 shows remarkable cross-reactivity with other 7S globulins, \"in contrast to characteristics of IgE to the walnut 7S globulin Jug r 2\" (i.e. Jug r 2 does NOT show the same broad cross-reactivity)."},

    # C10/mite -- explicit phylogenetic-but-not-cross-reactive
    {"id_1": "Blo t 21", "id_2": "Blo t 5", "note": "\"Phylogenetically related to Blo t 5, but not cross-reactive with this allergen\" (appendix table) -- explicit negative despite relatedness."},

    # C11 -- Parvalbumins, salmonid-specific monosensitisation (explicit non-cross-reactive)
    # NOTE: Sal s 1 x Gad m 1 intentionally NOT listed here as negative -- same source's
    # Case 1 shows this pair CAN cross-react (patient-dependent); see positive list, added
    # there with an explicit caveat instead of a contradictory pair of rows.
    {"id_1": "Sal s 1", "id_2": "Cyp c 1", "note": "Case 2: IgE negative to carp despite salmon/trout positive (species-specific salmonid epitope)."},
    {"id_1": "Onc m 1", "id_2": "Gad m 1", "note": "Fig 3: explicitly \"non cross-reactive parvalbumins\", cod vs trout, salmonid-monosensitized patients."},
    {"id_1": "Onc m 1", "id_2": "Cyp c 1", "note": "Case 2: trout IgE negative for carp cross-reactivity."},
]
