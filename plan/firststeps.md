# Critical Review & Implementation Roadmap
## "Prediction of Cross-Reactive Food Allergies Using Protein Language Models" (V2, Lana Lejić)

*Reviewed as a research supervisor would review a graduate thesis proposal, before any code is written.*

---

## PART 1 — Project Analysis

### 1.1 What the project actually tries to accomplish
You want to replace/augment sequence-alignment-based cross-reactivity assessment (BLAST/FASTA, manual allergen families) with **learned protein embeddings (ESM-2)** to (a) predict whether two food allergens are immunologically cross-reactive, and (b) rank, for a specific patient with known sensitizations, which untested allergens are most likely to be risky.

### 1.2 The scientific hypothesis
Deep protein-language-model embeddings encode structural/functional features (relevant to IgE epitope recognition) that are **not captured by sequence identity alone**, and therefore a similarity or learned metric in embedding space will correlate better with documented cross-reactivity than alignment-based similarity.

This is a legitimate, testable hypothesis — but note it is a **correlational** hypothesis about a *representation*, not a causal immunological claim. The project should be scoped as "does embedding similarity predict documented cross-reactivity better than a BLAST baseline," not "this model explains why cross-reactivity happens."

### 1.3 What the ML problem really is
It decomposes into three nested problems, and the proposal should say this explicitly (it currently blurs them together):

1. **Metric/representation learning** — learn or select an embedding space where cross-reactive protein pairs are close.
2. **Pair classification** — binary prediction p(cross-reactive | S1, S2).
3. **Ranking/retrieval** — given a patient's known allergen set, rank candidate allergens by risk (this is a retrieval/recommendation problem, not classification).

These require different loss functions, different evaluation metrics, and — critically — **different data splitting strategies**. Treating them as one pipeline with one dataset split is the single biggest structural risk in the proposal (see Part 4).

### 1.4 Inputs / outputs
- **Input (baseline/pair models):** two FASTA amino-acid sequences.
- **Input (personalized ranking):** a set of "known positive" and "known negative" allergens for one simulated patient + full candidate pool.
- **Output (pair task):** probability or similarity score.
- **Output (ranking task):** ordered list of candidate allergens with a risk score.

### 1.5 What is already well-defined
- Embedding generation pipeline (ESM-2, layer choice, truncation to 1022 aa).
- Baseline cosine-similarity experiment.
- PCA exploration step.
- The general architecture diagram.

### 1.6 What is vague or needs clarification (this is where most of the real work is)
- **How exactly the 45 "gold standard" pairs were/will be verified** — from what source, what counts as "experimentally verified" (skin-prick test? IgE-binding assay? clinical challenge? case report?).
- **How negative pairs are defined.** Absence of a documented positive report is *not* the same as a confirmed negative — this is the classic missing-negatives problem in bioinformatics.
- **Whether "family membership" is used as a proxy label.** If so, the project risks circularity: ESM embeddings are known to cluster by evolutionary family, so a model "predicting cross-reactivity" from family-correlated embeddings may just be rediscovering family membership, not new immunological signal.
- **What "personalized" actually means here.** There is no real patient dataset — this will be a **simulated** patient (a subset of the 80 allergens marked as "known positive/negative"). This must be labeled clearly as a simulation study, not a clinical validation, both in your writing and to any professor evaluating it.
- **Whether structural validation (AlphaFold) is validating full protein folds or epitope-level structure.** IgE cross-reactivity is driven by small conformational epitopes (5–15 residues), and whole-protein structural similarity is a fairly loose proxy for epitope similarity.

### 1.7 Biggest technical risks (ranked)
1. **Dataset size**: ~80 proteins / 45 positive pairs is very small for any learned model (MLP transformation, pair classifier). High variance, easy to overfit, hard to get statistically meaningful metrics.
2. **Negative pair validity**: constructing negatives from "no documented reaction" conflates *true negative* with *untested*.
3. **Family-driven circularity**: risk that the model just learns family clusters, which BLAST would already give you almost for free.
4. **Data leakage from family structure**: random pair splitting will leak information (same protein appears in train and test).
5. **Overclaiming personalization**: no real patient data exists; must be framed as simulated / proof-of-concept.
6. **AlphaFold at full-protein scale** answering an epitope-level immunological question is a partial mismatch in resolution.

### 1.8 Feasibility per stage (summary — see Part 7 for full table)

| Stage | Verdict |
|---|---|
| Embedding generation (ESM-2) | **Definitely feasible** |
| Cosine-similarity baseline | **Definitely feasible** |
| PCA analysis | **Definitely feasible** |
| Learned MLP transformation / metric learning | **Feasible with modifications** (needs regularization, small-data protocol, family-level CV) |
| Pair classification (Approach A) | **Feasible with modifications** |
| Embedding-to-embedding regression (Approach B) | **Risky** — underspecified target, easy to produce degenerate solutions with 45 pairs |
| Multi-model comparison (300M/1B/3B) | **Feasible**, compute is not the bottleneck here |
| AlphaFold structural validation | **Risky** — feasible only if scoped to epitope regions or a small validation subset, not full proteome-wide structural comparison |
| Personalized ranking (simulated patient) | **Feasible with modifications**, but must be explicitly framed as simulated, not clinical |

---

## PART 2 — Project Architecture

Your diagram is fundamentally right; it needs two additions: an explicit **label-construction stage** (separate from raw data cleaning) and a **leakage-safe splitting stage** placed *before* model training, not folded into "training dataset construction."

```
Protein sequence (FASTA, AllergenOnline)
        ↓
[1] Cleaning & normalization
   - dedup sequences, strip non-standard residues, truncate >1022 aa
   - attach metadata: family, source food, allergen ID
        ↓
[2] Embedding generation (frozen ESM-2)
   - mean-pool or CLS-token final hidden layer → 1280-d vector (650M model)
   - cache to disk (parquet/HDF5), keyed by allergen ID
        ↓
[3] Label construction  (NEW — separate from cleaning)
   - positive pairs: curated from literature (gold-standard set)
   - negative pairs: explicit sampling strategy + documented assumptions
   - family / cross-family tags attached to every pair
        ↓
[4] Leakage-safe splitting (NEW — before any similarity/training work)
   - split by PROTEIN FAMILY, not by pair
   - hold out entire families for test
        ↓
[5] Similarity computation (baseline)
   - cosine sim on raw ESM embeddings
   - compare to BLAST/FASTA alignment score baseline
        ↓
[6] Dimensionality reduction (PCA)
   - explore whether structure is clearer in reduced space
        ↓
[7] Learned metric (MLP transformation) / pair classifier
   - train only on train-family pairs
   - validate on held-out family pairs
        ↓
[8] Evaluation
   - ROC-AUC, Precision@k, MAP, NDCG on held-out families
   - compare against BLAST baseline (this comparison is your core result)
        ↓
[9] Structural validation (optional, scoped)
   - AlphaFold / existing PDB structures for a small, well-studied subset
        ↓
[10] Personalized risk ranking (simulated)
   - simulate patient profiles from held-out allergens
   - Risk(Sj) = max similarity to patient's known-positive set
   - output ranked list + Precision@k / MAP evaluation
```

Each stage's library/model recommendation:

| Stage | Tooling |
|---|---|
| Cleaning | biopython, pandas |
| Embedding | `fair-esm` / HuggingFace `facebook/esm2_t33_650M_UR50D` |
| Storage | parquet or HDF5 (100 proteins × 1280 floats is trivial, <1MB) |
| Baseline similarity | numpy / scipy cosine |
| PCA | scikit-learn |
| MLP / classifier | PyTorch (small nets, <100k params) |
| Evaluation | scikit-learn metrics, `pytrec_eval` or custom NDCG/MAP |
| Structural | AlphaFold DB (precomputed, not run yourself), Biotite/PyMOL for structure comparison |

---

## PART 3 — Dataset Verification (the most important part)

### AllergenOnline (v18.0)
- **Contains:** curated allergen protein sequences, source organism/food, some taxonomic/family grouping, links to literature, IgE-binding evidence flags for allergenicity (not cross-reactivity between specific pairs).
- **Does NOT contain:** a structured, pairwise cross-reactivity label table. It is built for allergenicity risk assessment of novel proteins (e.g., GMO safety screening), not for cross-reactivity pair prediction. This is a genuine mismatch you must address explicitly in your methodology section.
- **Usable for this project?** Yes, as the sequence/metadata source — but the "gold standard 45 pairs" cannot come from AllergenOnline itself; they must be **manually curated from clinical/immunology literature** (component-resolved diagnostics papers, review articles like Radauer & Breiteneder 2018, WHO/IUIS allergen family literature, case reports of oral allergy syndrome).
- **Licensing/access:** Free for academic/research use with registration; no blocking issue.
- **Preprocessing required:** sequence dedup, removal of signal peptides/propeptides if annotated, consistent FASTA formatting, family label harmonization (AllergenOnline family names vs WHO/IUIS nomenclature can differ).
- **Storage:** simple relational table (SQLite or CSV) keyed by allergen ID: sequence, family, source food, links to literature ID(s) supporting any cross-reactivity claim.
- **Additional labels required:** yes — you need an explicit, separately documented cross-reactivity table with a citation per positive pair. Build this as its own curated CSV, versioned, with a citation column. This is the actual scientific contribution of the "gold standard" and deserves its own methodology subsection.
- **Dataset size:** 80 proteins / 45 positive pairs is **small but usable for a proof-of-concept thesis**, provided you are honest about statistical power. It is not enough for a deep model with many parameters; it is enough for a well-regularized small MLP and rigorous baseline comparison.
- **Class imbalance:** severe. With 80 proteins, the full pair space is C(80,2) = 3,160 pairs. If only 45 are documented positive, and even fewer confirmed negatives exist, you have **at best a 1:9 to 1:70 imbalance** depending on how negatives are constructed — this must be handled explicitly (see Part 4).
- **Missing annotations:** degree/strength of cross-reactivity (most literature reports are binary yes/no, not graded); epitope-level annotations exist for only a handful of extremely well-studied allergens (e.g., Bet v 1 family, LTPs, profilins) — do not assume epitope data is available project-wide.

### Recommended supplementary sources (staying close to your idea)
- **SDAP (Structural Database of Allergenic Proteins)** — adds structural/epitope annotations for a subset of allergens, useful for Part 8 (structural validation) without needing full AlphaFold runs.
- **WHO/IUIS Allergen Nomenclature database** — for authoritative family/isoform naming, reducing label noise.
- **AlphaFold DB (not local AlphaFold runs)** — precomputed structures already exist for the vast majority of allergen UniProt entries; you do not need to run structure prediction yourself, only fetch and compare.
- **Published cross-reactivity review tables** (e.g., component-resolved diagnostics literature) as the actual source for your 45-pair gold standard, cited pair-by-pair.

### How to generate missing information
Where AllergenOnline gives you sequence/family but no cross-reactivity pair label, you generate it by **manual literature curation**: for each of the 8 families, search for published IgE cross-inhibition or clinical co-sensitization studies, extract the specific allergen pairs and citation, and record confidence (e.g., "confirmed by IgE inhibition assay" vs. "co-sensitization observed, mechanism unconfirmed"). This curation table is itself a deliverable worth documenting carefully — reviewers will scrutinize it.

---

## PART 4 — Training Data Construction

**Step by step, from raw sequence to training example:**

1. **Raw sequences** → 80 allergens, 8 families, cleaned FASTA + metadata table.
2. **Positive pairs**: every literature-confirmed cross-reactive pair (target ~45, but expect this to shrink after your own quality filtering — some published pairs may be weakly supported).
3. **Negative pairs** — three defensible strategies, use more than one and report sensitivity to the choice:
   - *Cross-family negatives*: pairs from families with no documented cross-reactivity in the literature — lowest risk of being false negatives, but arguably "easy" negatives (this is exactly the circularity risk from Part 1).
   - *Within-family non-reported negatives*: pairs from the same family without a specific documented report — harder, more informative, but riskier (may include real but unstudied cross-reactivity).
   - *Random pairs excluding known positives*: simplest, most biased toward the "easy negative" problem.
   
   Report results **separately** for each negative-sampling strategy — this is more scientifically honest than picking one and hiding the sensitivity.

4. **Pair balancing**: with severe imbalance, use either (a) class-weighted BCE loss, or (b) 1:3–1:5 negative:positive subsampling per epoch rather than full 1:70 — full imbalance will make ROC-AUC misleadingly high and useless.

5. **Splitting — the most important methodological fix versus your current draft**: split by **protein family**, not by pair or by protein alone.
   - Hold out 2 of the 8 families entirely for test.
   - Hold out 1–2 more for validation.
   - Train on the remaining families' pairs (both within-family positives and cross-family negatives involving only train-side families).
   - **Why:** if you split by pair, the *same protein* can appear in both a train pair and a test pair — the model can memorize that protein's embedding rather than learning a generalizable similarity function. This is the textbook "protein family leakage" problem in bioinformatics ML, and reviewers will ask about it directly.

6. **Example:**
   - Train families: PR-10, profilins, Cupin, tropomyosins, LTPs, storage proteins.
   - Held-out test families: 2S albumins, one additional family.
   - Test pairs: only pairs where **both** proteins come from the held-out families (cross-family test pairs involving a train-family protein are a softer, secondary test condition — report both).

7. **Label assignment**: binary (1 = documented cross-reactive, 0 = negative per chosen strategy). Do not silently mix graded confidence into a binary label — if you want confidence levels, model them explicitly (e.g., as sample weights) rather than hiding them.

8. **What should be predicted**: for the classifier, p(cross-reactive); for the ranking task, a scalar risk score used only for ordering, not calibrated probability (unless you explicitly calibrate it, which with 45 positives is not advisable to over-interpret).

---

## PART 5 — Protein Language Models

| Model | Params | Embedding dim | Relative quality | GPU memory (inference) | Notes |
|---|---|---|---|---|---|
| ESM-2 300M | 300M | 640 | Good baseline | ~2–3 GB | Fast, good first pass |
| ESM-2 650M | 650M | 1280 | Strong general-purpose choice | ~4–6 GB | **Recommended primary model** — your choice is sound |
| ESM-2 3B | 3B | 2560 | Marginal gains over 650M for most tasks, much heavier | ~14–16 GB | Use only as ablation if GPU available (e.g., a single A100/T4-class Colab GPU can handle it with batch size 1) |
| ProtBERT | 420M | 1024 | Older, generally underperforms ESM-2 on structure-related tasks | ~3–4 GB | Fine as a secondary comparison, as you already planned |

**Key point for your project size:** you only have ~80–100 sequences. This is **not a compute-bound problem** — a full forward pass over your entire dataset with ESM-2 650M takes well under a minute on any modern GPU, and is feasible even on CPU (slower, but tractable given the tiny N). Runtime and memory are **not a real risk** for this project; do not over-invest planning time here. Even the 3B model is feasible for one-off embedding extraction given your dataset size — the earlier warnings about ESM-2 3B being heavy apply to large-scale corpora, not to 80 proteins.

**Precompute vs. on-the-fly:** precompute all embeddings once and cache them (they are frozen — you are not fine-tuning the language model itself, only training small downstream MLPs). This decouples your experiments from GPU dependency almost entirely after the initial extraction step.

**Recommendation:** start with 650M (as you planned) as primary, run 300M and ProtBERT as secondary comparisons since they're nearly free given the dataset size, and treat 3B as an optional ablation, not a required part of the core pipeline.

---

## PART 6 — Implementation Roadmap

| Phase | Goal | Expected output | Verification | Common mistakes | Gate to continue |
|---|---|---|---|---|---|
| **0. Environment setup** | Working Python env with ESM-2, PyTorch | Reproducible env file | `import esm; import torch` works, GPU detected | Version mismatches between `fair-esm` and torch | Env runs a toy embedding extraction |
| **1. Dataset exploration** | Understand AllergenOnline export, families, counts | Cleaned metadata table, family counts, sequence length histogram | Manual spot-check of 5–10 entries against AllergenOnline website | Trusting family labels without cross-checking WHO/IUIS nomenclature | Metadata table reviewed and family counts make sense |
| **2. Data cleaning** | Deduplicate, normalize sequences | Clean FASTA + ID table | No duplicate IDs, all sequences valid amino acids | Silently dropping unusual sequences without logging | Clean dataset checksum matches expected count |
| **3. Gold-standard curation** | Build cited positive-pair table | CSV: pair, family, citation, confidence | Every row traceable to a source | Copying pairs from secondary sources without citation | At least 30–45 citable pairs collected |
| **4. Embedding generation** | Compute ESM-2 embeddings for all proteins | Cached embedding matrix (N×1280) | Spot-check cosine similarity of two obviously similar proteins is high | Forgetting to truncate/pad consistently | Embeddings cached, shapes correct |
| **5. Baseline (cosine + BLAST)** | Compare raw embedding similarity vs. sequence alignment | ROC-AUC/Precision@k for both baselines | Numbers computed on the *same* pair set | Comparing baselines on different pair subsets | Baseline clearly beats or loses to random; both numbers reported honestly |
| **6. Family-safe splitting** | Implement leakage-safe split | Train/val/test family assignment file | No protein appears in both train and test | Splitting by pair instead of family | Split checked programmatically for leakage |
| **7. PCA exploration** | Check if reduced space separates classes better | 2D/3D plots, explained variance | Visual + quantitative silhouette check | Over-reading noisy PCA plots with 80 points | Decide whether PCA space feeds later stages or is descriptive only |
| **8. Learned metric / classifier (first NN)** | Train small MLP on train-family pairs | Trained model + val metrics | Val metrics computed on held-out families | Random pair split "leaking" performance | Val performance beats baseline non-trivially, or you report honestly that it doesn't |
| **9. Ranking model** | Turn pairwise scores into ranking output | MAP/NDCG on held-out families | Compare ranking metrics to baseline similarity ranking | Using train-family data in ranking test | Ranking metric computed only on test-family candidates |
| **10. Personalized recommendation (simulated)** | Simulate patient profiles, generate recommendations | Ranked candidate lists per simulated patient | Sanity check: known cross-reactive allergens rank near top | Presenting simulated results as clinically validated | Clear write-up distinguishing simulation from clinical claim |
| **11. Structural validation (optional/scoped)** | Compare embedding similarity to structural similarity for a small well-studied subset | Correlation plot, small case-study table | Structures pulled from AlphaFold DB, not self-run | Trying to run AlphaFold at scale unnecessarily | Case study written up regardless of correlation strength |
| **12. Final evaluation & writeup** | Consolidate all metrics, compare against BLAST throughout | Final results table + limitations section | All numbers reproducible from cached artifacts | Cherry-picking best run without reporting variance | Thesis draft complete |

---

## PART 7 — Feasibility Check (per planned experiment)

| Experiment | Can be implemented? | Data exists? | Correct inputs? | Evaluation sound? | Leakage risk | Enough data? | Statistically meaningful? | Fix needed |
|---|---|---|---|---|---|---|---|---|
| ESM-2 embedding extraction | Yes | Yes | Yes | N/A | None | Yes | N/A | None |
| Cosine similarity baseline | Yes | Yes | Yes | Yes | Low | Marginal (45 pos.) | Report CIs / bootstrap | Bootstrap confidence intervals on ROC-AUC |
| PCA | Yes | Yes | Yes | Descriptive only | None | Marginal for meaningful clusters | Low (80 points) | Treat as exploratory, not confirmatory |
| MLP metric learning | Yes, with care | Yes | Yes | Yes, if split correctly | **High if split by pair** | Marginal — needs strong regularization | Report variance across seeds/folds | Family-level CV, small model, dropout/weight decay |
| Pair classifier (Approach A) | Yes | Yes | Yes | Yes | High if split by pair | Marginal | Report variance | Family-level split, class weighting |
| Embedding regression (Approach B) | Risky | Yes | Ambiguous target definition | Weak (MSE to another protein's embedding is a strange training signal) | High | Low | Low | Consider dropping or treating as secondary ablation only |
| Multi-model comparison (300M/650M/1B/3B) | Yes | Yes | Yes | Yes | None (same splits reused) | Yes (compute is cheap here) | Yes | None |
| AlphaFold structural validation | Yes, if scoped | Partial (structures exist in AlphaFold DB) | Yes for full-protein fold; imperfect for epitope-level claims | Needs care in interpretation | Low | Low (small case-study scale) | Low — treat as qualitative | Scope to a handful of well-studied allergens, not full dataset |
| Personalized ranking (simulated) | Yes | Simulated only | Yes | Yes (MAP/NDCG well-suited) | Must exclude test-family leakage into "known" set | Low | Low, frame as proof-of-concept | Explicit "simulated patient" framing in writeup |

---

## PART 8 — Evaluation Strategy

| Task type | Appropriate metrics | Why |
|---|---|---|
| **Pair classification** | ROC-AUC, Precision/Recall, Confusion matrix, PR-AUC (important given imbalance — PR-AUC is often more informative than ROC-AUC here) | Standard binary classification metrics; PR-AUC specifically robust under class imbalance |
| **Similarity/metric learning** | ROC-AUC on similarity threshold, Precision@k | Tests whether the learned space separates classes, without forcing a hard probability calibration |
| **Ranking** | MAP, NDCG, Precision@k (k=5,10) | Rewards correct ordering of top candidates, matches the actual clinical use case ("which allergen to test next") better than plain classification metrics |
| **Personalized recommendation** | Precision@k, NDCG, plus qualitative case studies | With one/few simulated patients, aggregate statistics are less meaningful than a few well-explained worked examples |

**Cross-validation:** use family-level k-fold (leave-one/two-families-out), not random k-fold, for the same leakage reasons as Part 4.

**Train/val/test split:** with only 8 families, consider **nested leave-family-out cross-validation** rather than a single fixed split — this gives you a distribution of metric values across folds rather than one fragile number, which is far more defensible with this data size.

**Statistical significance:** with 45 positive pairs, report **bootstrap confidence intervals** on your headline metrics (ROC-AUC, MAP) rather than point estimates alone, and be explicit that the sample size limits how strong any significance claim can be. A professor will specifically look for whether you acknowledge this rather than present a single ROC-AUC number as definitive.

---

## PART 9 — Final Project Review (as a thesis committee would score it)

**Strengths**
- Clear, well-motivated real-world problem with genuine clinical relevance.
- Sound core hypothesis, testable, with an appropriate baseline (BLAST) already built into the design.
- Architecture is coherent and appropriately staged (baseline → learned metric → ranking → personalization).
- Good instinct to include both a classification approach and an embedding-transformation approach as alternatives.
- Multi-model comparison (ESM-2 sizes, ProtBERT) adds useful robustness analysis at near-zero extra compute cost.

**Weaknesses**
- Dataset is small (80 proteins, 45 pairs); as written, splitting/leakage strategy is not yet specified and is the single most important fix.
- Negative-pair construction is undefined — this needs its own explicit methodology.
- "Personalized" component currently implies real patient validation; it is actually a simulation and must be reframed accordingly.
- AlphaFold structural validation, as scoped, risks a resolution mismatch between whole-protein structure and epitope-level immunological cross-reactivity.
- Gold-standard curation methodology (how the 45 pairs were verified) is not yet documented as its own reproducible artifact.

**Scientific novelty:** Moderate. The idea of comparing learned embeddings against alignment-based similarity for allergen cross-reactivity is a reasonable, publishable-scale contribution particularly if the family-leakage-safe evaluation is done rigorously — most existing work in this space either doesn't compare against a strong baseline or doesn't control for family leakage.

**Engineering difficulty:** Low-to-moderate. The compute burden is genuinely small; the difficulty is almost entirely in careful data curation and correct experimental design, not in engineering complexity.

**Expected bottlenecks**
- **Computational bottleneck:** minimal — this is not a compute-bound project.
- **Dataset bottleneck:** the real bottleneck — literature curation of correctly cited, verifiable positive pairs, and a defensible negative-sampling strategy.

**Potential publication value:** Reasonable as a workshop paper or thesis-level contribution if the leakage-safe evaluation and honest small-sample reporting are done properly; unlikely to be venue-competitive as a top-tier ML paper given dataset scale, but that is expected and appropriate for this project's scope.

**Overall feasibility score:** 7/10 (core pipeline is entirely feasible; personalization and structural validation components need explicit rescoping).

**Likelihood of successful implementation:** 8/10 (engineering risk is low; the main risk is methodological rigor, which is addressable with the fixes above).

**Risk level:** Medium — driven by dataset size and leakage/negative-sampling design choices, not by technical feasibility of the models themselves.