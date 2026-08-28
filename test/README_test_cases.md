# Allergen Cross-Reactivity Validation Dataset — README

## Files
- `test_cases.json` — structured per-patient case data extracted from published literature
- `README_test_cases.md` — this file

## Source & provenance
Every case was transcribed from a specific cited publication during two research passes.
Nothing here is simulated. Verification status is recorded per-case in the `verification`
field — read it before trusting a value. Where a source explicitly could not be fully
verified (paywalled, bot-blocked, abstract-only), that is stated in `verification.notes`.

## Schema

```
{
  "patient_id": "unique string, e.g. 'vando2005_pt04'",
  "source": {
    "citation": "full citation string",
    "doi": "string or null",
    "pmid": "string or null",
    "url": "string or null"
  },
  "protein_family": "nsLTP | profilin | PR-10 | storage_protein_2S | storage_protein_7S |
                      storage_protein_11S | tropomyosin | parvalbumin | serum_albumin",
  "patient_description": "verbatim or near-verbatim from source (age, sex, history)",
  "components": [
    {
      "protein": "e.g. 'Pru p 3'",
      "protein_family": "nsLTP",  // per-component family, since some patients span families
      "source_food_or_organism": "e.g. 'peach'",
      "method": "OFC | DBPCFC | sIgE | SPT | immunoblot | ISAC | ALEX2",
      "value": "numeric kU/L, ISU, or wheal-mm, or null if only pos/neg given",
      "value_unit": "kU/L | ISU | mm | null",
      "result": "positive | negative | borderline",
      "confirmed_by_challenge": true/false   // true only if OFC/DBPCFC, not sIgE/SPT alone
    }
  ],
  "sequential_protocol": {
    // present ONLY for the handful of cases with an actual graded
    // introduction/immunotherapy protocol and documented outcome per step.
    // null/absent for all sensitization-only cases.
    "type": "SLIT | OIT | graded_OFC",
    "steps": [
      {"step": 1, "item": "string", "dose_or_note": "string", "outcome": "tolerated | reaction | not_reported"}
    ]
  } | null,
  "verification": {
    "status": "full_text_verified | abstract_snippet_only | partially_verified",
    "notes": "explicit caveats, e.g. 'full text bot-blocked, values from abstract'"
  }
}
```

## How to use this for your two features

### Feature 1 — "given known pos/neg, what should be tested next"
Leave-one-out per patient:
1. Pick a patient with ≥3 components.
2. Reveal N-1 components to your tool as "known", hide 1 (ideally a positive/negative pair,
   hide one of each across repeated runs).
3. Run your tool's suggestion logic on the revealed subset.
4. Check whether the hidden component appears in the output, and whether its predicted
   priority/probability direction (test soon vs. deprioritize) matches its true `result`.
5. Aggregate across all patients: this is your test set for ranking metrics
   (precision@k, or a simple "correctly ranked positive above negative" count).

Best patients for this (explicit within-patient positive/negative pairs on structurally
distinct components):
- `motheslucksch2017_pt02`, `_pt04`, `_pt10` (nsLTP+ / PR-10– / profilin–)
- `motheslucksch2017_pt03`, `_pt06` (nsLTP– / PR-10+)
- `vando2005_pt04` (Gad c 1+ / Sal s 1– / all extracts–)
- `vando2005_pt01`, `_pt12` (Gad c 1+, Sal s 1+, but NOT inhibited by Sal s 1 — partial cross-reactivity, a harder case)
- `penas2014_case` (Sal s/Onc m parvalbumin marker+, Gad m/Sol so/Thu a–, AND the true driver is enolase not parvalbumin — use this as a stress test / expected-failure case)
- `pereira2009_case` (Pru p 3+ / Pru p 4– / Bet v 2– / CCD–)
- `kim2020_case`, `shiratsuki2020_case` (Fel d 2+/Sus s 1+ / alpha-gal– / chicken or wheat–)

### Feature 2 — sequential introduction / immunotherapy-style ordering
Only cases with a non-null `sequential_protocol` field are usable here:
- `pereira2009_case` (Pru p 3 SLIT, 1 patient, 1-year outcome)
- `porcaro2016_case` (fish OIT, salmon+cod)
Sample size is far too small for statistics. Use these only as qualitative/face-validity
checks: does your tool's suggested step order and step count move in the same direction
as the real protocol (e.g., lower cross-reactivity first, more steps for higher sIgE)?
Do not report this as "validated" — report it as "did not contradict the two available
real protocols."

## Known limitations of this dataset
- Total N is small (~40 patients across all families) and heavily skewed toward
  sensitization-level data (sIgE/SPT/immunoblot) rather than OFC/DBPCFC-confirmed allergy.
  Treat `confirmed_by_challenge: false` cases as weaker ground truth.
- Geographic bias: nsLTP/profilin data mostly Austrian/Spanish; tropomyosin Polish;
  serum-albumin Korean/Japanese/Portuguese; fish parvalbumin Norwegian.
  Do not assume generalization across populations.
- Some cases (Cortellini 2011, Barradas Lopes 2022, Hasegawa 2009) have
  `verification.status: "abstract_snippet_only"` — do not use these for hard
  pass/fail scoring until you've pulled the full text yourself, only for
  directional sanity checks.
- `vando2005` patients numbered 1,2,3,4,5,7,8,9,11,12 — numbers 6 and 10 do not exist
  (2 of 12 recruited patients were excluded from the original study for non-IgE-mediated
  reactions); this gap is intentional, not a data-entry error.
