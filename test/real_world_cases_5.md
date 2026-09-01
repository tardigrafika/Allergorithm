# Double-Mixed CRD Patient Dataset — Continuation: Priority 1 Lead Resolution + New Candidates

## TL;DR
- The single biggest win: the **Giuffrida et al. 2014** shrimp ISAC series (Eur Ann Allergy Clin Immunol 46(5):172–7, PMID 25224947) yielded a **complete, machine-readable per-patient Table 1** for 40 patients across four IUIS-registered components (Pen a 1, Pen m 1, Pen m 2, Pen m 4), producing **11 new IUIS-valid double-mixed patients** — enough to lift the dataset from ~15–16 to ~26–27.
- **Cabrera 2023** (PMC10561593) and **Coelho 2025** (PMID 41165422) could NOT be extracted: Cabrera's per-patient data sits in a Figure 1 heatmap image plus restricted supplementary tables; Coelho's open-access PDF is not retrievable through the available tools and its abstract is aggregate-only.
- The 108-patient Central European shrimp study (**PMC4072316**) is confirmed to be an **aggregate-only conference abstract** with no per-patient matrix — a definitive dead end.

## Key Findings

### CONFIRMED double-mixed patients (IUIS-valid, ≥2 pos AND ≥2 neg)

All from **Giuffrida MG, Villalta D, Mistrello G, Amato S, Asero R. "Shrimp allergy beyond Tropomyosin in Italy: clinical relevance of Arginine Kinase, Sarcoplasmic calcium binding protein and Hemocyanin." Eur Ann Allergy Clin Immunol 2014;46(5):172–177. PMID 25224947.** Full text (per-patient Table 1) retrieved verbatim from the publisher PDF (eurannallergyimm.com/wp-content/uploads/2014/09/volume-shrimp-allergy-beyond-tropomyosin-italy-866allasp1.pdf). Method: ImmunoCAP ISAC 112 microarray for Pen m 1, Pen m 2, Pen m 4 (positivity >0.3 ISU/l); ImmunoCAP singleplex for Pen a 1 (positivity >0.1 kU/l). Aggregate context: IgE to rPen m 2 in 4/40 (10%) and rPen m 4 in 6/40 (15%), two sera to both.

IUIS/WHO registration check — all four are officially registered (confirmed against allergen.org): **Pen a 1** (tropomyosin, *Penaeus aztecus*), **Pen m 1** (tropomyosin, *Penaeus monodon*), **Pen m 2** (arginine kinase), **Pen m 4** (sarcoplasmic calcium-binding protein).

| Pt | History | Pen a 1 (CAP) | Pen m 1 | Pen m 2 | Pen m 4 | Arithmetic |
|----|---------|---------------|---------|---------|---------|------------|
| 1  | rhinitis/asthma | 3.92 (+) | 10 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 2  | urticaria/OAS | 0.63 (+) | 4.3 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 3  | anaphylaxis | Neg | Neg | 4 (+) | 4.2 (+) | 2 pos / 2 neg = QUALIFIES |
| 4  | urticaria/OAS | 0.24 (+) | 0.7 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 6  | urticaria | 0.28 (+) | 0.6 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 15 | urticaria | 0.75 (+) | 2.6 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 16 | OAS | 0.55 (+) | 3 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 20 | urticaria | 56.2 (+) | 77 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 21 | OAS | 50.6 (+) | 86 (+) | Neg | Neg | 2 pos / 2 neg = QUALIFIES |
| 24 | urticaria | Neg | Neg | 1.1 (+) | 1.3 (+) | 2 pos / 2 neg = QUALIFIES |
| 35 | OAS | 0.14 (+) | Neg | Neg | 0.6 (+) | 2 pos / 2 neg = QUALIFIES |

**11 qualifying patients.** Verification status: FULL TEXT, exact values transcribed from the publisher PDF. History codes per the paper: a = oral allergy syndrome; b = urticaria/angioedema; d = rhinitis and/or asthma; x = anaphylaxis.

**CRITICAL INTERPRETIVE CAVEAT (tropomyosin redundancy):** Pen a 1 and Pen m 1 are both tropomyosins from different shrimp species and are near-identical / highly cross-reactive proteins, though each carries its own IUIS registration. Under the strict literal IUIS-registration rule all 11 patients qualify. But if the validation requires two *distinct, non-redundant* positive proteins, the eight patients whose two positives are BOTH tropomyosins (#1, 2, 4, 6, 15, 16, 20, 21) collapse to a single independent positive protein. The three most robust "distinct-protein" profiles are:
- **#3 and #24**: arginine kinase (Pen m 2) + SCBP (Pen m 4) positive; both tropomyosins negative — two distinct positive proteins, two distinct negatives.
- **#35**: tropomyosin (Pen a 1) + SCBP (Pen m 4) positive; tropomyosin (Pen m 1) + arginine kinase (Pen m 2) negative — genuinely mixed distinct proteins on both sides.

### Priority 1 lead resolution status

**1. Cabrera et al. 2023 (J Clin Lab Anal 37:e24960, PMC10561593) — PARTIAL / EXTRACTION FAILED.** Full narrative text successfully retrieved via ncbi.nlm.nih.gov/pmc/articles/PMC10561593 (the no-www no-trailing-slash trick worked). However, the per-patient sensitization data exists ONLY as a Figure 1 heatmap (a rendered image — axis labels and color intensities are not machine-readable through the tools) and in supplementary Tables S2/S3 bundled into a single "Appendix S1" file that is not separately fetchable; the article's data-availability statement says raw data are "available on request... not publicly available due to privacy or ethical restrictions." Wiley full text (onlinelibrary.wiley.com/doi/10.1002/jcla.24960) was blocked by bot detection; pmc.ncbi.nlm.nih.gov returned a reCAPTCHA wall. Routes tried: (a) PMC no-www — got narrative text but no per-patient values; (b) Wiley — blocked; (c) pmc.ncbi — CAPTCHA; (d) targeted searches for the supplementary tables — none exposed patient rows. Recoverable aggregate facts confirmed: 20 polysensitized LTP-syndrome patients (mean age 29, range 6–54) tested on BOTH ISAC E112i and ALEX2; 14/20 (70%) Pru p 3-positive on both platforms; 11/20 (55%) Ole e 7-positive on ISAC but 0/20 on ALEX2; 4 patients Sola l 6-positive on ALEX2; 10/20 had nut symptoms. All panel components (Pru p 3, Ara h 9, Cor a 8, Jug r 3, Tri a 14, Par j 2, Ole e 7, Sola l 6) are IUIS-registered. **Verdict: per-patient matrix is theoretically present but NOT extractable without the heatmap image or the restricted supplement.**

**2. Coelho et al. 2025 (Eur Ann Allergy Clin Immunol, doi:10.23822/EurAnnACI.1764-1489.418, PMID 41165422) — EXTRACTION FAILED; per-patient matrix existence UNCONFIRMED.** The journal landing page and abstract were retrieved, but the open-access PDF (eurannallergyimm.com/wp-content/uploads/2025/10/Coelho_OF.pdf) consistently returned a PERMISSIONS_ERROR because the URL only ever appeared embedded in page-body text, never as a standalone fetchable search-result link. My dedicated subagent hit the identical tool limitation across ~8 query variants and alternate routes (Europe PMC, Google Scholar, PubMed). The abstract is entirely aggregate: 66 patients (median age 10, range 1–67, IQR 15); walnut caused 41% of reactions, hazelnut and peanut 21% each; sensitization 88% to 2S albumins (Jug r 1 48%, Ana o 3 29%), 36% to 11S (Cor a 9 29%), 23% to 7S; Jug r 1, Cor a 14, Ana o 3, Ara h 1/2/6 clinically relevant (p<0.05); severe reactions correlated with nsLTP co-sensitization (p=0.05). Whether a per-patient matrix exists in the full text or supplement remains OPEN and requires a manual PDF download. All listed components (Jug r 1, Ana o 3, Cor a 9, Cor a 14, Ara h 1/2/3/6, Ses i 1) are IUIS-registered, so if a matrix is obtained it would be high-value.

**3a. Giuffrida shrimp series — FULL SUCCESS.** See CONFIRMED section above. 11 qualifying patients extracted from the complete Table 1.

**3b. PMC4072316 (Hemmer, Wöhrl, Sesztak-Greinecker, Jarisch, Wantke — 108-patient Central European seafood ISAC study, Clin Transl Allergy 2014;4(Suppl 2):P41) — CONFIRMED DEAD END.** Full text retrieved via ncbi.nlm.nih.gov/pmc/articles/PMC4072316 (no-www trick worked). This is a 2014 WAO Congress meeting abstract, structured Background/Methods/Results/Conclusions with NO tables and NO per-patient matrix — only aggregate percentages (Pen m 1 42.6%, Pen m 4 25.0%, Pen m 2 13.9%; 67/108 = 62% microarray-positive; Der p/f 1 or 2 positive in 34.3% of the 67 microarray-positive sera). No individual patient rows exist in the published record. **Not usable.**

### New leads found but not yet verified

- **Grilo J, Vollmann U, Aumayr M, Sturm GJ, Bohle B. "Tropomyosin is no accurate marker allergen for diagnosis of shrimp allergy in Central Europe." Allergy 2022;77(6):1921–1923. PMID 35293628; PMC9321988.** 79 individuals with allergic reactions to shrimp and shrimp-specific IgE on ImmunoCAP, tested on Allergy Explorer ALEX2 (ethics approval EK 1344/2018, Medical University of Vienna) for Pen m 1, Pen m 2, Pen m 3, Pen m 4 and Cra c 6 (all IUIS-registered) plus immunoblotting to *Litopenaeus vannamei* extract. Whole-cohort aggregate (verbatim): "42% of the patients displayed IgE to Pen m 1 (TM, 19% exclusively), 20% to Pen m 2 (AK, 3.7% exclusively), 10% to Pen m 3 (MLC, 2.5% exclusively), and 11% to Cra c 6 (TC)"; separately, "30% of the patients recognized Pen m 4 and 16% showed exclusive IgE reactivity to this SCBP." Immunoblot additionally showed 10% recognized ~70 kDa (hemocyanin) and 9% recognized 5–7 kDa (ubiquitin); all recombinant allergens combined achieved 68% sensitivity. A per-patient matrix DOES exist as Figure 1 ("IgE reactivity to chip-spotted allergens represented as dark boxes," 79 individuals) and in supplementary Tables S1/S2 — but Figure 1 is an image and the per-patient values were not machine-readable in this pass. **HIGH PRIORITY** for a targeted supplementary-table retrieval: this five-component panel is richer than Giuffrida and could yield more robust distinct-protein double-mixed shrimp patients.
- **Molina/Valbuena et al. "Storage Proteins Are Driving Pediatric Hazelnut Allergy in a Lipid Transfer Protein-Rich Area." Foods 2021;10(10):2463 (PMC8535272; doi:10.3390/foods10102463).** 22 challenge-proven pediatric hazelnut patients tested on ImmunoCAP AND ALEX2 for Cor a 1, Cor a 8, Cor a 9, Cor a 11, Cor a 14 (all IUIS-registered). Full text retrieved. However, the published Table S1 is an aggregate count ("number of patients with positive results to the different in vitro hazelnut allergens"), NOT a per-patient matrix; raw data is "available on request." Aggregate sensitization: storage proteins dominant (nCor a 9 / nCor a 11 / rCor a 14 positive in 19–20/22 by ALEX; Cor a 11 positive in 18/22), Cor a 1 in 3–4, Cor a 8 in 7. Would qualify as a per-patient source only if the raw dataset is obtained from the authors.

### Excluded candidates (checked this pass, not usable)
- **PMC4072316** (Hemmer 108-patient) — aggregate-only meeting abstract (detailed above).
- **Molina/Valbuena Foods 2021** hazelnut — published tables aggregate-only (detailed above).
- **Dutch PR-10 co-sensitization cohort** (PMC5700688, 305 patients) and **Netherlands ISAC cohort** (PMC4412439) — aggregate-only (already on exclude list; re-confirmed).
- **French Allergen Chip PR-10 database** (Allergy 2025 abstract, all.70135, 4,271 patients) — aggregate-only.
- **Scala LTP cohorts** (Allergy 2015 PMID 25903791; ALEX2 PMID 37712443; Eur Ann 2023 "Scala.pdf") — aggregate percentages / group comparisons, no per-patient matrix accessible.
- **González Pérez Pru p 3 OIT** (PMC7278159, 18 patients) — aggregate ISAC counts only, no per-patient matrix.
- **Spanish nut microarray study** (PMC4412440, 100 patients) — reports aggregate component counts by nut, no per-patient rows.

## Details
The core methodological lesson from this pass is that qualifying "double-mixed" patients require an *accessible* per-patient matrix, and this requirement fails far more often than the underlying data is absent. Three of the strongest leads (Cabrera, Grilo, Molina) all encode their per-patient data as heatmap figures or count-only supplementary tables rather than transcribable matrices. Only Giuffrida 2014 published its individual data as a plain text table — which is precisely why it was the only fully extractable source.

On the Giuffrida cohort: the paper studied 40 randomly selected sera from a larger 116-patient Italian multicentre cohort. Of the 40, 9 were tropomyosin-positive (Pen a 1 >0.1 kU/l) and 8 were positive to arginine kinase and/or SCBP (4 Pen m 2, 6 Pen m 4, 2 to both). The 11 double-mixed patients are those with at least two positive and at least two negative results across the four registered components. 26 of the 40 were negative to all four, which is why the qualifying yield is 11.

The tropomyosin-redundancy issue matters for downstream statistics: because Pen a 1 and Pen m 1 are effectively the same protein measured on two platforms, the eight patients positive to both-and-only-both tropomyosins provide within-patient paired data that is really tropomyosin(+)/tropomyosin(+)/arginine-kinase(−)/SCBP(−). For a cross-reactivity prediction tool this remains legitimate (the tool would correctly predict Pen a 1↔Pen m 1 concordance, and the two negatives are genuinely distinct proteins), but patients #3, #24 and #35 are the cleanest for paired testing across non-homologous proteins.

## Recommendations
1. **Immediately bank the 11 Giuffrida shrimp patients** (#1, 2, 3, 4, 6, 15, 16, 20, 21, 24, 35), tagging each with the tropomyosin-redundancy flag so the analyst can select the strict-literal count (all 11) or the distinct-protein subset (#3, #24, #35 strongest). This alone lifts the dataset from ~15–16 to ~26–27 patients.
2. **Obtain the Coelho 2025 PDF by manual download** (open access at eurannallergyimm.com/wp-content/uploads/2025/10/Coelho_OF.pdf) and inspect for a per-patient or supplementary matrix. This is the highest-value unresolved lead — all components are IUIS-registered storage proteins in a fresh 66-patient cohort. Threshold to act: if the full text/supplement contains any patient-level table, extract; if confirmed aggregate-only, close the lead.
3. **Pursue the Grilo 2022 supplementary Tables S1/S2** (PMC9321988) for machine-readable per-patient ALEX2 values across Pen m 1/2/3/4 + Cra c 6; if only Figure 1 is available, request the underlying data from the authors or attempt OCR of the heatmap. This five-component panel could yield more distinct-protein double-mixed shrimp patients than Giuffrida.
4. **For Cabrera and Molina, email the corresponding authors** for the raw per-patient matrices (both papers explicitly state data are available on request; Cabrera's is Carmen Maria Cabrera, Ciudad Real). These are the only viable routes, since the published artifacts are a heatmap image and count-only tables respectively.
5. **Benchmark that changes the plan:** if manual retrieval of the Coelho and Grilo supplements yields per-patient matrices, prioritize extracting those over further web searching — each could add roughly 5–20 qualifying patients, far exceeding the yield of additional lead-hunting.

## Caveats
- No values were guessed. Every Giuffrida value is transcribed from the publisher's full-text PDF; every failure is documented with the exact routes attempted.
- The Pen a 1 positivity threshold (>0.1 kU/l) is the authors' own stated cutoff and reproduces their reported count of 9 tropomyosin-positive patients, so the low positives (#4 = 0.24, #6 = 0.28, #35 = 0.14) are genuinely positive per the source, not interpolated.
- Cabrera and Coelho remain unresolved due to hard tool limitations (heatmap image; non-fetchable PDF), NOT because the data was searched and found absent — both may still contain usable matrices retrievable by a human.
- The tropomyosin-redundancy caveat is an interpretive judgment about cross-reactivity validation, not a defect in IUIS registration; under the strict literal rule stated in the task, all 11 Giuffrida patients qualify.
- Grilo and Molina are reported as new leads, not confirmed patients, because their per-patient data was not machine-readable in this pass.
- PMC4072316 is definitively closed: it is a meeting abstract with no tables, so no per-patient matrix can ever be recovered from the published record.