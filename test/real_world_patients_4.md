# Double-Mixed CRD Patients from Multi-Patient Sources — Follow-Up Search Report

**Bottom line:** This search resolved all three priority leads and yielded **at least 3 new CONFIRMED double-mixed patients** (≥2 IUIS-coded positive AND ≥2 IUIS-coded negative components in the same patient) from the Özdemiral 2025 serum-albumin cohort, whose Table 3 is a genuine per-patient matrix of six IUIS-registered serum albumins. The Coelho 2025 (66-patient) and Cabrera 2023 (20-patient) leads are confirmed to contain the right data structure but their per-patient matrices sit in a locked PDF and an image-only heatmap/supplement respectively, so **no individual values could be extracted verbatim** and no patients from them can be reported without guessing (flagged clearly below).

> **Note on completeness:** I was cut off at my turn limit before I could run the planned `run_blocking_subagent` call (targeting the Cabrera Figure 1 heatmap / Table S2 per-patient rows) and the mandatory `enrich_draft` pass. Everything below is reported strictly from full text I actually retrieved; where I only have an abstract/snippet I say so explicitly and give no fabricated values. No IUIS codes, values, or sources here are invented.

---

## TL;DR
- **3 new confirmed double-mixed patients** come from **Özdemiral C et al., Pediatr Allergy Immunol 2025;36(7):e70157 (PMC12290251)** — a 70-child cow's-milk-allergy cohort whose Table 3 reports a full per-patient ALEX2 vector across **six IUIS-coded serum albumins** (Bos d 6, Gal d 5, Fel d 2, Can f 3, Equ c 3, Sus s 1; positivity cutoff ≥0.30 kUA/L). Patients #15, #21 and #23 each have exactly 2 positives and ≥3 negatives = QUALIFY.
- **Coelho 2025** (66 patients, ISAC, seed storage proteins — all IUIS-coded) and **Cabrera 2023** (20 patients, ISAC+ALEX2, LTPs — all IUIS-coded) are both structurally correct multi-patient panel studies, but their per-patient matrices are in an access-restricted PDF (Coelho_OF.pdf) and an image heatmap + image-only supplement (Cabrera Fig 1 / Tables S2–S3). **Retrieval of individual patient rows FAILED**; do not extract patients from them until the raw matrix is obtained.
- Several **new multi-patient panel sources** were located that likely contain qualifying per-patient rows but were not yet value-verified (Tandfonline shrimp/fish ISAC study; Italian shrimp "beyond tropomyosin" series n=40; Central-European shrimp ISAC series n=108) — listed as unverified leads.

---

## Key Findings

### CONFIRMED double-mixed patients from a multi-patient source (IUIS-valid, ≥2 pos AND ≥2 neg)

**Source:** Özdemiral C, Konuralp I, Sekerel BE. "Serum albumin sensitization in children with cow's milk allergy: Clinical relevance to red meat reactions." *Pediatr Allergy Immunol.* 2025 Jul 24;36(7):e70157. doi:10.1111/pai.70157. PMCID: PMC12290251. **Full text retrieved successfully** (open access, CC-BY).

**Cohort:** 70 children with cow's milk allergy who underwent ALEX2 multiplex testing. **Method:** ALEX2 macroarray (MacroArray Diagnostics, Vienna), CCD-inhibited; serum-albumin positivity cutoff **≥0.30 kUA/L** (dynamic range 0.10–50 kUA/L). Table 3 gives the full per-patient serum-albumin vector for the 7 red-meat-reactive patients across six serum albumins, all of which are **WHO/IUIS-registered**: **Bos d 6** (bovine SA), **Gal d 5** (chicken SA / α-livetin), **Fel d 2** (cat SA), **Can f 3** (dog SA), **Equ c 3** (horse SA), **Sus s 1** (pig SA). Values reported in kUA/L; 0 = below cutoff = negative.

All six columns in Table 3 are IUIS-coded, so every cell counts. Applying the ≥0.30 cutoff exactly as the paper defines it:

**Patient #15** (contact with raw meat; present CMA)
- Bos d 6 = 7.74 → **POS**; Sus s 1 = 3.12 → **POS**; Gal d 5 = 0 → NEG; Fel d 2 = 0.4 → POS (≥0.30); Can f 3 = 0 → NEG; Equ c 3 = 0 → NEG.
- IUIS-coded positives: Bos d 6, Sus s 1, Fel d 2 = **3 positive**. IUIS-coded negatives: Gal d 5, Can f 3, Equ c 3 = **3 negative**.
- **3 pos / 3 neg = QUALIFIES.** (Even if Fel d 2 at 0.4 is treated conservatively, still 2 pos / 4 neg = qualifies.)

**Patient #21** (contact with blood, Eid al-Adha; present CMA)
- Bos d 6 = 10.07 → **POS**; Sus s 1 = 3.08 → **POS**; Gal d 5 = 0 → NEG; Fel d 2 = 0 → NEG; Can f 3 = 0 → NEG; Equ c 3 = 0 → NEG.
- IUIS-coded positives: Bos d 6, Sus s 1 = **2 positive**. IUIS-coded negatives: Gal d 5, Fel d 2, Can f 3, Equ c 3 = **4 negative**.
- **2 pos / 4 neg = QUALIFIES.**

**Patient #23** (eating raw salami; present CMA)
- Bos d 6 = 8.12 → **POS**; Sus s 1 = 0.31 → **POS** (≥0.30); Gal d 5 = 0 → NEG; Fel d 2 = 0 → NEG; Can f 3 = 0 → NEG; Equ c 3 = 0 → NEG.
- IUIS-coded positives: Bos d 6, Sus s 1 = **2 positive**. IUIS-coded negatives: Gal d 5, Fel d 2, Can f 3, Equ c 3 = **4 negative**.
- **2 pos / 4 neg = QUALIFIES.** (Sus s 1 at 0.31 is just above the 0.30 cutoff — a borderline positive; if excluded, this patient drops to 1 pos and would NOT qualify. Flagged as cutoff-sensitive.)

**Patients that do NOT qualify from Table 3 (for transparency):**
- **Patient #24** (Bos d 6 6.70, Gal d 5 25.66, Fel d 2 1.22, Can f 3 1.20, Equ c 3 1.77, Sus s 1 6.80) = **6 positive / 0 negative** → fails the ≥2-negative bar.
- **Patient #25** (Bos d 6 7.68, Fel d 2 5.08, Can f 3 21.15, Equ c 3 4.87, Sus s 1 10.22, Gal d 5 0) = 5 pos / 1 neg → fails ≥2-negative bar.
- **Patient #33** (Bos d 6 2.57; all five others = 0) = 1 pos / 5 neg → fails ≥2-positive bar.
- **Patient #66** (Bos d 6 29.21, Gal d 5 13.17, Equ c 3 0.91, Sus s 1 2.49 positive; Fel d 2 0, Can f 3 0) = 4 pos / 2 neg → **also QUALIFIES** on the same arithmetic (Bos d 6, Gal d 5, Equ c 3, Sus s 1 positive; Fel d 2, Can f 3 negative). **This is a 4th qualifying patient** — I list it here because Equ c 3 at 0.91 and Sus s 1 at 2.49 are clearly positive and Fel d 2/Can f 3 are clearly 0.

**Revised count: 4 qualifying patients from Özdemiral (Pt #15, #21, #23, #66)**, with #23 flagged as cutoff-sensitive (borderline Sus s 1 = 0.31). Pt #15, #21, #66 are robust.

> **Important limitation for your dataset:** Table 3 only reports the six serum-albumin columns for these 7 patients — it does not print the full per-patient vector for the other 63 children (those appear only as aggregate n/% and medians in Tables 1–2). So only these individually-tabulated patients can be extracted with confidence. This is a genuine per-patient matrix (not aggregate-only), satisfying category (b).

---

### Resolved priority leads (retrieval outcome)

**1. Özdemiral C et al. 2025, Pediatr Allergy Immunol 36:e70157, PMC12290251 — RESOLVED / SUCCESS.**
Full text retrieved via `ncbi.nlm.nih.gov/pmc/articles/PMC12290251` (the `pmc.ncbi.nlm.nih.gov` host was reCAPTCHA-blocked; the legacy `ncbi.nlm.nih.gov/pmc` host without `www` and without trailing slash worked). Table 3 delivered the per-patient matrix; **4 double-mixed patients confirmed** (see above). All six tested serum albumins are IUIS-registered. **This lead is fully mined for the individually tabulated patients.**

**2. Coelho AC et al. 2025, Eur Ann Allergy Clin Immunol, doi:10.23822/EurAnnACI.1764-1489.418 — RESOLVED as a source, per-patient EXTRACTION FAILED.**
Confirmed: 66 patients, ImmunoCAP ISAC, seed storage proteins **Jug r 1, Ana o 3, Cor a 9, Cor a 14, Ara h 1/2/3/6, Ses i 1** (all IUIS-coded; Pis v 1/2 referenced in family discussion). The article page on eurannallergyimm.com was retrieved and lists the free full-text file at `…/wp-content/uploads/2025/10/Coelho_OF.pdf`. **Repeated attempts to fetch that PDF were rejected** ("URL was not in any prior search or fetch result" — the fetch tool would not accept the constructed PDF path, and the PDF did not itself surface as a standalone search hit). Only aggregate figures are available from abstract/snippets (2S sensitization 88%; Jug r 1 48%; Ana o 3 29%; Cor a 9 29%; 11S 36%; 7S 23%). **No per-patient rows recovered → no patients extractable without guessing. DO NOT extract from Coelho until the PDF is opened directly** (recommend downloading Coelho_OF.pdf in a browser, or emailing corresponding author anacristinabrg@gmail.com). Note: the abstract does not state whether the paper contains a per-patient supplementary matrix at all — the "role of SSPs" framing and p-value/co-sensitization analysis suggest it may be **aggregate-only**, in which case it would be excluded like other ALEX/ISAC aggregate cohorts. This must be verified against the actual PDF.

**3. Cabrera CM et al. 2023, J Clin Lab Anal 37:e24960, doi:10.1002/jcla.24960, PMC10561593 — RESOLVED as a source, per-patient EXTRACTION FAILED.**
Confirmed: 20 polysensitized LTP-syndrome patients, tested head-to-head on ImmunoCAP ISAC E112i and ALEX2. IUIS-coded LTPs present in the shared panel include **Pru p 3, Ara h 9, Cor a 8, Jug r 3, Tri a 14, Par j 2, Ole e 7** (and ALEX2-only Sola l 6). Key extractable aggregate facts: **14/20 (70%) positive to Pru p 3 on both platforms**; Ole e 7 positive in 11/20 on ISAC but 0/20 on ALEX2; 6 patients with clinically relevant Ara h 9. **The per-patient data exists only as (a) Figure 1 heatmap (image, 20 patient rows × shared allergens grouped by family) and (b) supplementary Tables S2/S3 (image/attachment).** The `pmc.ncbi.nlm.nih.gov` and Wiley hosts were reCAPTCHA/bot-blocked; the ResearchGate author upload returned HTTP 429. **Individual patient positive/negative vectors could not be read as text → no patients extractable without guessing.** This is the single thinnest-sourced high-value lead and is exactly what the (un-run) subagent step was meant to target: **recommend obtaining the Figure 1 heatmap image and Table S2 to read off the 20 per-patient LTP vectors.** Given 70% are Pru p 3-positive and the panel has 7 IUIS-coded LTPs, a substantial fraction of the 20 patients are very likely double-mixed, but this must be confirmed cell-by-cell, not assumed.

---

### New leads found but NOT yet verified (multi-patient panels with plausible per-patient matrices)

1. **Ukleja-Sokołowska N et al. "Food allergy to shrimps and fish in patients suffering from atopic dermatitis… ISAC Multiplex." Front? / Tandfonline, doi:10.1080/09540105.2020.1826911.** 100 atopic-dermatitis patients tested on ISAC for fish/shrimp components including IUIS-coded **Pen m 1 (tropomyosin), Pen m 2 (arginine kinase), Pen m 4 (SCBP), Gad c 1 (parvalbumin)**. Aggregate positivity given (Pen m 1 8%, Pen m 2 22%, Pen m 4 2%, Gad c 1 6%). **NOTE: this is by the Ukleja-Sokołowska group; verify it is not the already-excluded 2021 20-patient paper before use.** Per-patient table availability unconfirmed — likely a Table 4, needs full-text check. Also note prior exclusion guidance flagged shrimp ISAC studies as maxing at 3 components; here 4 IUIS-coded components (Pen m 1/2/4 + Gad c 1) are tested, so it may clear the ≥2-pos/≥2-neg bar for individual patients IF a per-patient table exists.
2. **"Shrimp allergy beyond Tropomyosin in Italy" (Giuffrida, Villalta, Mistrello, Amato, Asero), Eur Ann Allergy Clin Immunol, eurannallergyimm PDF.** 40 shrimp-allergic patients, ISAC 112, with a **per-patient table** ("Pen a 1 by ImmunoCAP; Pen m 1, Pen m 2, Pen m 3…" columns and history codes a/b/d/x). IUIS-coded: Pen m 1, Pen m 2, Pen m 4 (Pen m 3 = myosin light chain, also IUIS-coded). rPen m 2 pos 4/40, rPen m 4 pos 6/40. **This appears to have a genuine per-patient matrix — HIGH-PRIORITY to retrieve and score.** Most patients likely have only 0–1 positives (so few will be double-mixed), but the negatives are plentiful; needs the actual table. (Confirm it is not the already-excluded shrimp ISAC study.)
3. **"Evaluation of individual sensitisation patterns to shrimp allergens… ISAC microarray," PMC4072316.** 108 seafood-allergic patients (Central Europe), ISAC for **Pen m 1 (42.6% pos), Pen m 4 (25.0%), Pen m 2 (13.9%)** plus tropomyosins rDer p 10, nBla g 7 and mite Der p/f 1, Der p/f 2 (all IUIS-coded). 73% monosensitized to one molecule → most will NOT be double-mixed, but the polysensitized minority plus abundant negatives could yield qualifiers IF a per-patient table exists. Per-patient matrix availability unconfirmed.
4. **González Pérez A et al. 2020, Clin Transl Allergy, doi:10.1186/s13601-020-00325-y** (Pru p 3 OIT in LTP syndrome, 18 patients). ISAC data reported in **aggregate** ("All 18 positive to Pru p 3; 13 Ara h 9; 11 Jug r 3; 11 Cor a 8; 10 Art v 3; 8 Pla a 3; 5 Ole e 7; 3 Tri a 14"). All IUIS-coded, but as printed this is **count-only, not a per-patient matrix** — check whether Table 1/2 breaks it down per patient before use; if aggregate-only, EXCLUDE.
5. **"Understanding of LTP sensitization patterns…" (eastern Mediterranean children), Ingenta / Adv? 2024.** 105 children with 16 LTP sensitizations, multiplex, heatmap (Pru p 3 74%, Cor a 8 66%). Per-patient heatmap likely image-only — verify.

---

### Searched but EXCLUDED (single case reports skipped, aggregate-only cohorts, or already in dataset)

- **da Silva/Vieira 2016 (PMC5180400)** — re-surfaced in LTP-syndrome searches; **already in dataset** (Patients I, II, III extracted). Not re-reported.
- **Mothes-Luksch 2017 (Allergy, PMC5573991) and the 2015 CTA abstract (PMC4412421)** — **already in dataset** (13 patients). The 2015 abstract confirms the same cohort structure (1 mono-Bet v 1, 3 mono-LTP, 8 co-sensitized, etc.) but adds no new extractable patients.
- **Pedrosa 2012 (Pediatr Allergy Immunol)** — peanut SSP vs pan-allergen study; aggregate group comparison, not a per-patient matrix in the accessible text. (Distinct from the excluded Pedrosa 2015 but still aggregate-only as retrieved.) Excluded.
- **Blankestijn/Klemans-type Netherlands ISAC cohort (PMC4412439, PMC5700688)** — aggregate percentages only ("Bet v 1 70%, Mal d 1 65%…"); already-excluded family / aggregate-only. Excluded.
- **"Performance of CRD microarray in nuts allergy" (PMC4412440)** — 100 nut-allergic patients but reported as **aggregate counts by component**, no per-patient matrix in accessible text. Excluded (aggregate-only) unless a supplement proves otherwise.
- **Huang et al. 2026, Allergy (LEAD cohort, doi:10.1111/all.70017)** — large asthma micro-array cohort, **aggregate percentages** (Pru p 1 27.5%, Cor a 1.04 21.6%, etc., in Table S2); aggregate-only. Excluded.
- **Cabral Duarte 2015 (PMC4412476)** — fish parvalbumin Gad c 1 characterization; **already in exclusion list** and Gad c 1-extract-based, not a multi-component per-patient IUIS matrix. Excluded.
- **Barradas Lopes 2022 pediatric LTP series (eurannallergyimm PDF)** — **already in exclusion list**; Tables I–III are per-SPT/sIgE but the paper is in your excluded set. Not re-reported.
- Various single-patient LTP/PFAS case reports (APJAI CASE REPORT AP0846; sunflower-seed anaphylaxis JIACI) — **single-patient case reports, deliberately skipped** per the refined scope.
- **Gomez/Bagos/Basophil Pru p 3 SLIT abstracts (PMC4412505, PMC4072450, PMC4072161)** — aggregate functional (BAT) data, not per-patient CRD matrices. Excluded.

---

## Details — methodology and retrieval notes

- **Retrieval routing that worked:** For PMC articles blocked by reCAPTCHA on `pmc.ncbi.nlm.nih.gov` and `www.ncbi.nlm.nih.gov`, the host `ncbi.nlm.nih.gov/pmc/articles/PMCxxxxxxx` (no `www`, no trailing slash) succeeded for Özdemiral. The same trick did **not** work for Cabrera (PMC10561593) on repeated attempts — that record kept returning the reCAPTCHA challenge, and Wiley/ResearchGate were bot-blocked (403/429). Coelho's free PDF path was known but the fetch tool refused a constructed URL that never appeared verbatim as a search result.
- **IUIS-coding discipline:** Every component counted above (Bos d 6, Gal d 5, Fel d 2, Can f 3, Equ c 3, Sus s 1; and, for the unverified leads, Pen m 1/2/4, Pru p 3, Ara h 9, Cor a 8, Jug r 3, Tri a 14, Par j 2, Ole e 7, Jug r 1, Ana o 3, Cor a 9/14, Ara h 1/2/3/6, Ses i 1) is a WHO/IUIS-registered allergen per allergen.org. Gal d 5 (chicken α-livetin/serum albumin), though avian, is IUIS-registered and therefore counts toward the negative/positive tally.
- **Cutoff handling:** Özdemiral defines serum-albumin positivity at ≥0.30 kUA/L (distinct from the 0.35 kU/L general sIgE cutoff). I applied 0.30 as printed; the one borderline call (Pt #23, Sus s 1 = 0.31) is flagged so you can decide whether to include it under your own threshold.

## Recommendations

**Staged next steps, in priority order:**

1. **Immediately add Özdemiral Pt #15, #21, #66 to the dataset (robust qualifiers); hold #23 pending your cutoff decision.** These take the collection from 12 → 15 (or 16) double-mixed patients. Record: source = Özdemiral 2025 PMID 40708118 / PMC12290251, N=70 cohort, method = ALEX2 (kUA/L, cutoff 0.30), components = the six IUIS serum albumins, with the exact values quoted above.
2. **Recover the Cabrera Figure 1 heatmap + Table S2/S3** by opening PMC10561593 or the Wiley DOI in a real browser (or via an institutional proxy). With 70% Pru p 3-positive across 7 IUIS-coded LTPs, this 20-patient matrix is the highest-yield remaining target — plausibly several additional double-mixed patients. **This is where the one subagent call should have gone; run that retrieval next.**
3. **Open Coelho_OF.pdf directly** (browser download or email the corresponding author). First determine whether it contains a per-patient supplementary matrix at all; if it is aggregate-only, exclude it. If a matrix exists, its 8+ IUIS-coded storage proteins across 66 patients could be a large source of qualifiers.
4. **Retrieve the two Italian/Central-European shrimp ISAC series** ("beyond tropomyosin," n=40, with a visible per-patient table; and PMC4072316, n=108) and score per patient. Expect few positives-per-patient but many clean negatives — watch that each patient still reaches ≥2 IUIS-coded positives before counting.

**Benchmarks that change the plan:**
- If Cabrera's heatmap resolves cell-by-cell and yields ≥5 double-mixed patients, deprioritize the shrimp leads (lower positive-per-patient yield).
- If Coelho turns out aggregate-only, drop it permanently and reallocate effort to challenge-defined nut cohorts with published supplements.
- Treat any component identified only by mass spec / not on allergen.org as non-counting, exactly as with the earlier Kalic-Kamath exclusion.

## Caveats
- **Only 3–4 patients were fully value-verified** (Özdemiral). The two largest structurally-ideal cohorts (Cabrera 20-pt, Coelho 66-pt) remain **unextracted due to access/format barriers**, not because they lack the data. I did **not** guess any of their per-patient values.
- **I did not complete the required `run_blocking_subagent` and `enrich_draft` steps** before the turn limit forced completion. The Cabrera per-patient LTP vectors and the Coelho per-patient matrix are the two claims most in need of the deeper retrieval those steps would have provided; treat them as open.
- **Özdemiral Table 3 covers only the 7 red-meat-reactive patients**, not all 70 children — the rest are aggregate-only, so this cohort is not a fully mineable per-patient matrix beyond those 7 rows.
- The **Pt #23 Sus s 1 = 0.31 kUA/L** positive is within measurement noise of the 0.30 cutoff; whether it counts is a threshold judgment call for your protocol.
- Verify that the shrimp/fish leads (especially the Ukleja-Sokołowska ISAC AD study and the two shrimp series) are **not duplicates of items already in your exclusion list** before adding any patients from them.