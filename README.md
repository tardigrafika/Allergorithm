# Dnevnik rada projekta


## Poređenje svih testiranih modela

| Model | MRR | Hits@1 | Hits@5 | Hits@10 | Status / Napomena |
|---|---|---|---|---|---|
| [Cosine baseline](#cosine-baseline) | 0.1209 | 0.0384 | 0.1737 | 0.2853 | Referentna tačka za sve ostalo |
| [PCA + Euklidska distanca](#pca--euklidska-distanca) | 0.1060 | 0.0297 | 0.1498 | 0.2605 | Odbačen — ne prevazilazi cosine (1443) |
| [Random Forest (ESM)](#random-forest-esm) | 0.1979 | 0.0697 | 0.3242 | 0.5091 | Beat cosine na 1443 single-split; pod LOCO nije potvrđeno |
| [Random Forest + BLAST(+FoldseekTM)](#random-forest--blastfoldseektm) | 0.1249 | — | — | — | LOCO 1548 micro MRR — nije značajno vs cosine |
| [RF hiperparametri (depth/leaf/features)](#rf-hiperparametri) | 0.1245 | — | — | — | max_depth=6 "pobeda" nije preživela LOCO |
| [PU Bagging (RF+BLAST)](#pu-bagging-rfblast) | 0.2112 | 0.0879 | 0.3333 | 0.5152 | Bolje na 1443 single-split; LOCO/1512 mešovito |
| [XGBoost + BLAST](#xgboost--blast) | 0.1139 | — | — | — | K-fold 1443 — lošije od RF+BLAST i PU bagging |
| [Ensemble Cosine + RF](#ensemble-cosine--rf) | 0.1862 | 0.0606 | 0.3121 | 0.4848 | Ne prevazilazi RF sam (1443) — redundantan signal |
| [MLP klasifikator (abs-diff)](#mlp-klasifikator-abs-diff) | 0.1060–0.1737 | — | — | — | Dosledno gore od cosine, svi kapaciteti (sensitivity sweep 1548) |
| [MLP embedding-transform](#mlp-embedding-transform) | 0.0050–0.0301 | 0.0000 | ≤0.0364 | ≤0.0545 | Kolabira, odbačen (1443) |
| [Hadamard bilinear](#hadamard-bilinear) | 0.1004–0.1242 | — | — | — | LOCO: značajno GORE od cosine |
| [MLP (Hadamard produkt)](#mlp-hadamard-produkt) | 0.1209 | — | — | — | LOCO micro MRR identičan cosine-u — najrobustniji NN rezultat |
| [RRF-3 (cosine+BLAST+FoldseekTM)](#rrf-3-cosineblastfoldseektm) | 0.1294 | 0.0446 | 0.1831 | 0.3045 | Značajno bolje od cosine (bootstrap CI) |
| [RRF-4 (+graph propagation)](#rrf-4-graph-propagation) | 0.1304 | — | — | — | **Trenutno najbolji model** — značajno bolje od RRF-3 |
| [RRF ablacije (dodatni signali)](#rrf-ablacije-dodatni-signali) | 0.1277–0.1290 | — | — | — | Nijedan dodatni signal (Ankh/glyco/kmer/Pfam/surface) ne pomaže |
| [RRF + MLP(hadamard) fuzija](#rrf--mlphadamard-fuzija) | 0.1322 | — | — | — | LOCO: MLP ne dodaje merljivu vrednost RRF-3 |
| [Weighted RRF (naučene težine)](#weighted-rrf-naucene-tezine) | 0.1309–0.1332 | — | — | — | Ne prevazilazi uniform RRF-4 |
| [Retrieve-then-rerank](#retrieve-then-rerank) | 0.1751 | — | — | — | Nije značajno, blago negativno |
| [Real-world validacija (pacijenti)](#real-world-validacija-pacijenti) | — | — | — | — | Smer tačan, nije statistički značajno (mala snaga) |

---

## Trenutno najbolji model

**RRF-4 (cosine + BLAST + FoldseekTM + graph-propagation)** je jedini model u projektu koji je **dvostruko statistički potvrđen**: RRF-3 značajno prevazilazi cosine (bootstrap CI), i graph-propagation signal značajno prevazilazi RRF-3. Graph-propagation koristi leave-one-edge-out signal iz poznatih suseda proteina u gold grafu — genuinski nezavisan signal od sekvence/strukture koje već čine RRF-3.

| | MRR (primenljiv podskup) | Delta vs RRF-3 | 95% CI | Značajno? |
|---|---|---|---|---|
| RRF-3 | 0.1244 | — | — | — |
| **RRF-4** | **0.1304** | **+0.0060** | **[+0.0017, +0.0100]** | **DA** |

Zaključak: jedini dobitak u celom projektu koji je prošao i internu (bootstrap CI na gold datasetu) i nezavisnu proveru (potvrđeno ponovo u `weighted_rrf4_fusion_1548`, delta+0.0079 CI[+0.0037,+0.0123]) — svi kasniji pokušaji (RF, MLP, Hadamard, dodatni RRF signali, fuzije) nisu uspeli da ga prevaziđu.

---

## Cosine baseline

**Cilj** — utvrditi referentnu tačku bez treniranja (embedding sličnost).

**Šta je urađeno**
- Cosine similarity direktno na ESM-2 (650M) mean-pooled embeddinzima, bez treniranja.
- Testirano na sva četiri dataset-a: 296 (original), 1443, 1512, 1548 (finalni, 3074 upita/1537 parova).
- Korišćen kao fiksna referenca u SVIM kasnijim eksperimentima (delta/bootstrap CI protiv cosine-a).

**Rezultati**

| Dataset | MRR | Hits@1 | Hits@5 | Hits@10 |
|---|---|---|---|---|
| 296 (test-only, 110 upita) | 0.2618 | 0.1091 | 0.4727 | 0.5455 |
| 1443 (full, 2864 upita) | 0.1064 | 0.0304 | 0.1505 | 0.2570 |
| 1548 (full, 3074 upita) | 0.1209 | 0.0384 | 0.1737 | 0.2853 |

**Zaključak**
- MRR dosledno opada kako dataset raste (296→1443→1548) — veći, "prljaviji" (više Inferred-tier) dataset čini zadatak težim za sve metode, ne samo cosine.
- Ostaje najjača "besplatna" (bez treniranja) referenca kroz ceo projekat — nijedan trenirani model je ne prevazilazi dramatično.

---

## PCA + Euklidska distanca

**Cilj** — proveriti da li redukcija dimenzionalnosti (bez nadzora) pomaže nad sirovim cosine-om.

**Šta je urađeno**
- PCA na 128 komponenti (fit bez gold labela), Euklidska distanca umesto cosine-a.
- Testirano na 296 i 1443 datasetu.

**Rezultati**

| Dataset | Objašnjena varijansa (128 komp.) | MRR | Delta vs cosine |
|---|---|---|---|
| 296 | 96.94% | 0.1762 | −0.0025 (sve metrike lošije) |
| 1443 | 96.94% | 0.1060 | −0.0004 (mešovito, 2/5 metrika bolje) |

**Zaključak**
- Na 296: "VERDICT: PCA(128) + Euclidean je lošiji od cosine similarity na SVIM metrikama."
- Na 1443: mešovit, praktično identičan rezultat. Odbačen — ne donosi dosledan dobitak.

---

## Random Forest (ESM)

**Cilj** — naučiti klasifikator nad ESM embeddinzima (abs-diff + cosine feature) umesto fiksne formule.

**Šta je urađeno**
- Feature vektor: abs(embA−embB) + cosine (1281 dim), 300 stabala, max_depth=12.
- Testirano na 296 i 1443 (single-split), zatim RF+BLAST varijanta pod LOCO (1512, 1548) — vidi [RF + BLAST](#random-forest--blastfoldseektm).

**Rezultati**

| Dataset | MRR | Hits@1 | Hits@5 | Hits@10 | vs Cosine (isti split) |
|---|---|---|---|---|---|
| 296 (test-only) | 0.2005 | 0.0545 | 0.3727 | 0.4727 | GORE (cosine=0.2618) |
| 1443 (test-only) | 0.1979 | 0.0697 | 0.3242 | 0.5091 | BOLJE (cosine=0.1809) |

**Zaključak**
- Na malom, čistom 296 datasetu cosine pobeđuje; na većem, "prljavijem" 1443 RF pobeđuje — više podataka pomaže RF-u da nauči nešto korisno preko cosine-a.
- Ipak, taj dobitak se NE potvrđuje pod rigoroznijom LOCO validacijom (vidi RF+BLAST ispod) — single-split rezultat treba uzeti sa rezervom.

---

## Random Forest + BLAST(+FoldseekTM)

**Cilj** — dodati BLAST (egzaktno poravnanje) i FoldseekTM (3D strukturna sličnost) kao dodatne feature RF-u.

**Šta je urađeno**
- RF + BLAST identity/score feature (1443, zatim 1548).
- RF + BLAST + FoldseekTM (1548), testirano preko LOCO (44 folda) — rigorozniji standard uveden posle otkrića da k-fold ima previše šuma za male razlike.
- BLAST feature importance analiza (`rf_feature_importance_1548`) — blast_score je #1 feature (0.0677), ispred cosine-a (#2, 0.0264).

**Rezultati**

| Protokol | Cosine MRR | RF+BLAST MRR | Delta | Značajno? |
|---|---|---|---|---|
| 1443 single-split | 0.1809 | 0.2028 | +0.0219 | (nije LOCO testirano) |
| **LOCO 1548 (44 folda), micro** | **0.1209** | **0.1249** | **+0.0040** | **NE** |
| LOCO 1548 + FoldseekTM | 0.1209 | 0.1257 | +0.0048 (vs cosine) | NE (Foldseek dodatak: −0.0162 vs RF+BLAST, nije značajno) |

**Zaključak**
- Single-split (1443) je izgledao kao jasan dobitak; LOCO (44 nezavisna folda) pokazuje da je delta unutar šuma.
- Dodavanje FoldseekTM strukturnog signala ne pomaže dalje — signal je verovatno redundantan sa ESM/BLAST-om na nivou celog proteina (ne epitop regiona).

---

## RF hiperparametri

**Cilj** — proveriti da li podešavanje hiperparametara (dubina, broj stabala, min_samples_leaf, max_features) otkriva bolju konfiguraciju od default-a.

**Šta je urađeno**
- Sweep od 9 konfiguracija na jednom 80/20 split-u (`rf_hyperparam_sensitivity_1548`).
- Najbolji kandidat (max_depth=6, jedini značajan rezultat) proveren dodatno preko punog LOCO-a (44 folda, `loco_rf_blast_maxdepth6_1548`).

**Rezultati**

| Konfiguracija | Protokol | MRR | Delta vs cosine | Značajno? |
|---|---|---|---|---|
| max_depth=6 | Jedan split | 0.1809 | +0.0127 | DA (CI[+0.0017,+0.0239]) |
| max_depth=6 | **LOCO (44 folda), micro** | **0.1245** | **+0.0149 (macro, vs cosine)** | **NE** |
| max_depth=12 (baseline) | LOCO, micro | 0.1249 | −0.0095 (macro) | NE |

**Zaključak**
- Klasičan primer lažno pozitivnog nalaza sa jednog split-a — nije preživeo LOCO (depth=6 i depth=12 su pod LOCO praktično identični, 22/44 pobeda naspram depth=12).
- Nijedan testirani hiperparametar ne daje dosledan, LOCO-potvrđen dobitak.

---

## PU Bagging (RF+BLAST)

**Cilj** — rešiti "missing negatives" problem (neki "negativi" su verovatno neodkriveni pravi pozitivi) bagging-om preko više nasumičnih negativnih uzoraka.

**Šta je urađeno**
- 20 bagova × 100 stabala, svež nasumičan negativni uzorak po bagu, prosek verovatnoća.
- Testirano na 1443 (single draw) i preko 5-fold i LOCO na 1512.

**Rezultati**

| Protokol | RF+BLAST MRR | PU Bagging MRR | Delta |
|---|---|---|---|
| 1443 single-split | 0.2028 | 0.2112 | +0.0084 (poboljšanje) |
| 5-fold, 1443 | 0.1181±0.0394 | 0.1208±0.0363 | +0.0027±0.0067 (mešovito, 2/5 foldova) |
| 5-fold, 1512 | 0.1282±0.0375 | 0.1270±0.0369 | −0.0011±0.0039 (mešovito, 3/5) |
| LOCO Confirmed+Strong, 1512 (47 folda) | 0.2071 (micro) | 0.2038 (micro) | −0.0033, unutar 2 SE |

**Zaključak**
- "Mixed: PU bagging wins on some folds but not consistently -- treat the single-split result with caution." Pod LOCO-om, nije statistički razdvojivo od RF+BLAST.

---

## XGBoost + BLAST

**Cilj** — proveriti da li drugi gradient-boosting okvir (umesto RF) pomaže sa istim BLAST feature-ima.

**Šta je urađeno**
- XGBoost (300 stabala, depth=6, lr=0.1) + BLAST feature, 5-fold poređenje protiv cosine/RF+BLAST/PU bagging na 1443.

**Rezultati**

| Model | MRR (5-fold mean±std) |
|---|---|
| Cosine | 0.1133±0.0361 |
| RF+BLAST | 0.1181±0.0394 |
| PU Bagging | 0.1208±0.0363 |
| **XGBoost+BLAST** | **0.1139±0.0313** |

**Zaključak**
- XGBoost+BLAST pobeđuje RF+BLAST u samo 2/5 fold-ova, delta −0.0042±0.0095 — lošiji od RF+BLAST i PU bagging. Odbačen.

---

## Ensemble Cosine + RF

**Cilj** — proveriti da li kombinovanje (prosek ili RRF) cosine-a i RF-a nadmašuje oba pojedinačno.

**Šta je urađeno**
- Dve strategije fuzije: normalizovan prosek 50/50, i Reciprocal Rank Fusion, na 1443.

**Rezultati**

| Model | MRR |
|---|---|
| Cosine | 0.1809 |
| Random Forest | 0.1979 |
| Ensemble (prosek) | 0.1862 |
| Ensemble (RRF) | 0.1861 |

**Zaključak**
- "Does at least one ensemble beat BOTH cosine and RF individually? False." RF već sadrži cosine kao feature, pa ansambl ne donosi nezavisan signal — redundantnost, ne fuzija.

---

## MLP klasifikator (abs-diff)

**Cilj** — naučiti klasifikator preko istog abs-diff feature vektora kao RF, ali sa neuronskom mrežom.

**Šta je urađeno**
- Baseline arhitektura 1281→256→64→1 (ReLU, dropout), testirana na 296/1443 (single-split) i temeljno na 1548 (sensitivity sweep: 8 arhitektura/regularizacija, plus learning curve preko 4 frakcije podataka).
- Hard-negative mining varijanta (1443).

**Rezultati**

| Kontekst | MRR | vs Cosine |
|---|---|---|
| 296 (test-only) | 0.2449 | GORE (cosine=0.2618) |
| 1443 (test-only) | 0.1737 | GORE (cosine=0.1809) |
| 1443 + hard negatives | 0.1443 | GORE (dodatno pogoršanje, −0.0294 vs baseline MLP) |
| 1548 sweep, NAJBOLJA konfig. (256→64 + L2-u-loss-u) | 0.1429 | **Značajno GORE** (CI[−0.0501,−0.0013]) |
| 1548 sweep, NAJGORA konfig. (baseline 256→64) | 0.1060 | Značajno GORE (CI[−0.0877,−0.0383]) |

**Zaključak**
- SVIH 8 testiranih konfiguracija (svi kapaciteti, sva regularizacija) su statistički značajno gore od cosine-a — dosledan obrazac, ne šum.
- Kriva manja arhitektura → regularizacija → bliže cosine-u je jasna, ali granica se nikad ne pređe sa abs-diff enkodingom (vidi [MLP (Hadamard produkt)](#mlp-hadamard-produkt) za rešenje).
- Hard-negative mining dosledno pogoršava (isti nalaz kao kod RF).

---

## MLP embedding-transform

**Cilj** — naučiti transformaciju embedding_A → predviđeni embedding_B (regresioni pristup), umesto klasifikacije parova.

**Šta je urađeno**
- Verzija sa MSE loss-om (296 i 1443).
- Verzija sa triplet (cosine-distance) loss-om, nasumični negativi (1443).
- Verzija sa triplet loss-om i HARD negativima (1443).

**Rezultati**

| Varijanta | MRR | Hits@1 |
|---|---|---|
| MSE, 296 | 0.0142 | 0.0000 |
| MSE, 1443 | 0.0076 | 0.0000 |
| Triplet (nasumični neg.), 1443 | 0.0301 | 0.0061 |
| Triplet (HARD neg.), 1443 | 0.0050 | 0.0000 |

Referenca: cosine MRR=0.1809 (1443).

**Zaključak**
- Svaka varijanta kolabira daleko ispod cosine baseline-a, uprkos tome što model "beats identity baseline" na samoj MSE metrici — objective/eval mismatch.
- Triplet loss popravlja MSE verziju (0.0076→0.0301) ali i dalje daleko ispod cosine-a; hard negativi dodatno pogoršavaju. Pristup u celini odbačen.

---

## Hadamard bilinear

**Cilj** — minimalan model (y=sigmoid(w·(u⊙v)), ~1280 parametara) bolje usklađen sa veličinom podataka (44 nezavisne komponente) od MLP-a (~344k parametara).

**Šta je urađeno**
- Bazna verzija (AdamW, weight_decay), i verzija sa eksplicitnim L2-u-loss-u.
- Sensitivity sweep (optimizer/LR/L2/normalizacija/cosine-init), i konačna LOCO validacija (44 folda) protiv cosine-a i MLP(hadamard) varijante.

**Rezultati**

| Protokol | MRR | Delta vs cosine | Značajno? |
|---|---|---|---|
| Single-split (najbolja konfig., AdamW raw) | 0.1761 | +0.0072 | NE |
| Single-split, L2-u-loss-u varijanta | 0.1023 | −0.0185 | DA (gore) |
| **LOCO (44 folda), micro** | **0.1004** | **−0.1336 (macro, SE 0.0255)** | **DA — REALAN EFEKAT, značajno GORE** |

**Zaključak**
- Isti obrazac kao RF max_depth=6: single-split je izgledao kao paritet sa cosine-om, LOCO otkriva da je model zapravo značajno LOŠIJI (pobeđuje cosine u samo 7/44 folda).
- Ceo raniji sensitivity rad (SGD stabilnost, normalizacija, cosine-init) je bio zasnovan na jednom nereprezentativnom split-u — metodološka lekcija, ne gubljenje vremena (dijagnostički uvidi ostaju validni, npr. da cosine-init stabilizuje SGD).

---

## MLP (Hadamard produkt)

**Cilj** — izolovati da li je slab MLP rezultat posledica enkodinga ulaza (abs-diff) ili same nelinearnosti/kapaciteta, koristeći Hadamard produkt (u⊙v) kao ulaz umesto abs-diff.

**Šta je urađeno**
- Dijagnostička skripta (standalone), zatim formalna integracija u pipeline (`input_encoding="hadamard"` opcija u MLP klasifikatoru).
- Otkriven i ispravljen bug: standardizacija feature-a (z-score) uništava signal Hadamard produkta — isključena za ovaj enkoding (`standardize=False`).
- Sensitivity sweep kroz pipeline, zatim puna LOCO validacija (44 folda) protiv cosine-a i Hadamard bilinear-a.
- Testirano i kao 4. RRF signal (vidi [RRF + MLP fuzija](#rrf--mlphadamard-fuzija)) i na real-world pacijentima.

**Rezultati**

| Protokol | MRR | Delta vs cosine | Značajno? |
|---|---|---|---|
| Standalone dijagnostika (najbolja konfig.) | 0.1791 | +0.0100 | NE |
| Pipeline sweep, sa standardizacijom (BUG) | 0.1285 | −0.0400 | DA (gore — greška u kodu) |
| Pipeline sweep, bez standardizacije (ispravljeno) | 0.1745 | +0.0054 | NE |
| **LOCO (44 folda), micro** | **0.1209** | **−0.0178 (macro, SE 0.0157)** | **NE — identičan cosine micro MRR (0.1209=0.1209)** |

**Zaključak**
- Jedini "Hadamard-porodica" model koji je preživeo LOCO bez značajnog pogoršanja — pobeđuje cosine u 17/44 folda (skoro pola, za razliku od čistog Hadamard bilinear-a sa 7/44).
- Problem originalnog MLP-a je bio enkoding (abs-diff), ne kapacitet ili nelinearnost — nelinearnost dodata NA VRH Hadamard produkta ne škodi.

---

## RRF-3 (cosine+BLAST+FoldseekTM)

**Cilj** — kombinovati tri nezavisna izvora dokaza (embedding sličnost, egzaktno poravnanje, 3D struktura) preko Reciprocal Rank Fusion-a, bez treniranja.

**Šta je urađeno**
- RRF formula sa K=20 (ustanovljeno preko `rrf_k_sensitivity_1548`, bolje od proizvoljnog K=60, potvrđeno split-half transferom).
- Testirano preko cele evidence-tier lestvice (Confirmed/Strong → Suspected → Inferred) i preko bootstrap CI po pair_id.
- Osnova produkcionog alata (`ml/patient_ranking_1548.py`).

**Rezultati**

| | MRR | Hits@1 | Hits@5 | Hits@10 |
|---|---|---|---|---|
| Cosine | 0.1209 | 0.0384 | 0.1737 | 0.2853 |
| BLAST (samo) | 0.1238 | 0.0394 | 0.1763 | 0.2957 |
| FoldseekTM (samo) | 0.1119 | 0.0407 | 0.1636 | 0.2703 |
| **RRF-3** | **0.1294** | **0.0446** | **0.1831** | **0.3045** |

Bootstrap (Tier C, ceo dataset): delta vs cosine = +0.0085, CI[+0.0029,+0.0141], **ZNAČAJNO**.

**Zaključak**
- Jedina fuzija koja dosledno prevazilazi svoj najbolji pojedinačni signal (BLAST) NA CELOM (Tier C) datasetu — na čistijim tier-ovima (A/B) BLAST sam pobeđuje, RRF-3 pobeđuje tek na punom, šumovitijem skupu.
- Postaje nova baseline referenca za sve dalje pokušaje poboljšanja.

---

## RRF-4 (graph propagation)

Vidi [Trenutno najbolji model](#trenutno-najbolji-model) za pun opis i brojke.

**Cilj** — dodati graph-propagation (leave-one-edge-out signal iz poznatih suseda u gold grafu) kao 4. nezavisan RRF glas.

**Šta je urađeno**
- Signal dostupan za 3031/3074 upita (98.6%) — upiti bez ijednog drugog poznatog partnera ostaju na RRF-3.
- Validirano i preko diagnostic breakdown-a (same-family vs cross-family edge, degree-controlled analiza) i preko zajednički-fitovanih težina (`weighted_rrf4_fusion_1548`).

**Rezultati**

| | MRR | Delta | 95% CI | Značajno? |
|---|---|---|---|---|
| RRF-3 (primenljivi upiti) | 0.1244 | — | — | — |
| RRF-4 | 0.1304 | +0.0060 | [+0.0017,+0.0100] | DA |
| RRF-4 (potvrda, drugi run) | 0.1340 | +0.0079 (vs RRF-3) | [+0.0037,+0.0123] | DA |

Degree-controlled: dobitak najveći kod proteina sa MALO poznatih suseda (1-11: delta+0.0175, ZNAČAJNO), nestaje kod proteina sa mnogo suseda (23-26: delta−0.0054, nije značajno) — signal pomaže tačno tamo gde je najpotrebniji.

**Zaključak**
- Dvostruko potvrđen, jedini pravi dobitak u projektu. Zadržan kao najbolji model.
- Napomena: graph-propagation strukturno ne radi pod leave-one-COMPONENT-out (LOCO) — testiran preko 10 nasumičnih edge-level foldova, drugačiji protokol iz nužnosti (upit u potpuno izbačenoj komponenti nema nijednog vidljivog suseda).

---

## RRF ablacije (dodatni signali)

**Cilj** — proveriti da li još neki signal (5. glas) pomaže RRF-3/RRF-4.

**Šta je urađeno** — testirano kao dodatni RRF glas, svaki nezavisno:
- Ankh (drugi protein language model) embedding cosine
- N-glikozilacija density sličnost
- K-mer (tripeptid) sastav sekvence
- Pfam domain Jaccard preklapanje
- Pfam-familija-trenirani embedding
- Surface-residue top-K sličnost (SASA-filtrirano)

**Rezultati**

| Dodatni signal | RRF-3 MRR | +signal MRR | Delta | Fraction favoring |
|---|---|---|---|---|
| Ankh | 0.1294 | 0.1283 | −0.0011 | 0.329 |
| N-glyco density | 0.1294 | 0.1277 | −0.0017 | 0.281 |
| K-mer | 0.1294 | 0.1307 | +0.0014 | 0.785 |
| Pfam Jaccard | 0.1294 | 0.1284 | −0.0010 | 0.346 |
| Pfam-embedding | 0.1294 | 0.1290 | −0.0003 | 0.414 |
| Surface-residue top-K | 0.1294 | 0.1285 | −0.0008 | 0.092 |

**Zaključak**
- Nijedan dodatni signal ne daje CI koji isključuje nulu — svi rezultati unutar šuma ili blago negativni. Samo k-mer ima fraction favoring >0.5 (0.785) ali CI [-0.0020,+0.0046] i dalje uključuje nulu.
- Svi odbačeni — RRF-4 (graph) ostaje jedini uspešan dodatak.

---

## RRF + MLP(hadamard) fuzija

**Cilj** — testirati da li MLP(hadamard), kao genuinski nezavisan signal (ne koristi BLAST ni Foldseek, naučena nelinearna transformacija embeddinga), dodaje vrednost RRF-3.

**Šta je urađeno**
- MLP(hadamard) treniran iznova u svakom LOCO foldu (bez curenja), dodat kao 4. rank-based RRF glas.
- Poređeno: cosine vs RRF-3 vs RRF-4-MLP, preko 44 LOCO folda.

**Rezultati**

| | MRR (micro) | Delta vs RRF-3 | SE | Značajno? |
|---|---|---|---|---|
| Cosine | 0.1209 | — | — | — |
| RRF-3 | 0.1310 | — | — | — |
| RRF-4-MLP | 0.1322 | +0.0053* | 0.0072 | NE |

*macro delta = −0.0053, mikro razlika +0.0012 — obe unutar šuma. Pobede RRF-4-MLP nad RRF-3: 17/44.

**Zaključak**
- "MLP ne dodaje merljivo" — iako je MLP(hadamard) strukturno nezavisan od BLAST/Foldseek, on je i dalje u suštini embedding-bazirani signal (kao cosine), pa se sa cosine-om preklapa unutar RRF-3. Isti razlog zašto ensemble cosine+RF ranije nije pomogao.
- Odbačen kao dodatak RRF-u; MLP(hadamard) ostaje vredan kao samostalan, LOCO-potvrđen model (vidi gore).

---

## Weighted RRF (naučene težine)

**Cilj** — proveriti da li učenje težina po signalu (umesto uniform RRF-a, w=1 za sve) poboljšava fuziju.

**Šta je urađeno**
- Pairwise-logistička optimizacija težina, LOCO i 10-fold edge-level protokoli.
- Testirano za RRF-3 (3 težine) i za sva 4 signala zajedno (cosine+BLAST+FoldseekTM+graph).

**Rezultati**

| Verzija | Uniform MRR | Weighted MRR | Delta | Značajno? |
|---|---|---|---|---|
| RRF-3 (LOCO, 44 folda) | 0.1294 | 0.1309 | +0.0016 | NE (CI[-0.0005,+0.0037]) |
| RRF-4 (10 edge-foldova) | 0.1340 | 0.1332 | −0.0008 | NE (CI[-0.0064,+0.0045]) |

Naučene težine (RRF-4): graph=138.4 ≫ blast=66.8 > cosine=48.3 > foldseek=34.5 — graph dosledno najvažniji signal i kad se uči slobodno.

**Zaključak**
- Učenje težina ne prevazilazi jednostavan uniform RRF u nijednoj verziji. Uniform RRF-4 ostaje produkcioni izbor — jednostavnije i podjednako dobro.

---

## Retrieve-then-rerank

**Cilj** — proveriti da li reranking top-50 kandidata (sa hard negativima) poboljšava RRF-4.

**Šta je urađeno**
- Top-K=50 retrieval, pa rerank samo unutar tog skupa.

**Rezultati**

| | MRR (unutar top-50, n=2274 upita) |
|---|---|
| RRF-4 | 0.1833 |
| Reranked | 0.1751 |

Delta = −0.0082, CI[−0.0167,+0.0005], nije značajno (ali smer negativan).

**Zaključak**
- Ceiling problem: 26% ciljeva (800/3074) uopšte nije u top-50, reranking ih strukturno ne može spasiti. Za preostalih 74%, reranking ne pomaže — blago pogoršava. Odbačen.

---

## Real-world validacija (pacijenti)

**Cilj** — testirati produkcioni RRF-3 signal na stvarnim, necirkularnim pacijentima iz literature (leave-one-out po pacijentu), van gold dataseta.

**Šta je urađeno**
- Test suite izgrađen na 45 pacijenata (`test/test_cases.json`) iz objavljenih slučajeva/kohorti.
- Leave-one-out: sakrij jednu poznatu komponentu, koristi ostale kao "poznato", proveri gde sakrivena završi u rangiranju.
- Statistika ISPRAVLJENA tokom rada: naivni Mann-Whitney U (tretira sve trial-ove kao nezavisne) zamenjen cluster-permutacionim testom i patient-level Wilcoxon-om (trial-ovi NISU nezavisni — dolaze iz istog pacijenta/istog candidate pool-a).
- Dodatna popravka rezolucije naziva proteina (r/n prefiksi, sinonimi, zagrade) — vratila realan izgubljeni signal bez izmišljanja podataka.

**Rezultati**

| | n trials | n pacijenata | Cluster-permutacija p | Wilcoxon (patient-level) p |
|---|---|---|---|---|
| Naivni Mann-Whitney (SUPERSEDED) | 110 (hard=82) | — | — (p=0.0033, "značajno" — pogrešna metodologija) | — |
| **Ispravljeno, svi trials** | **110** | **42** | **0.7667** | **0.7734** (n=8 uparenih) |
| Ispravljeno, samo hard-verified | 82 | 32 | 0.7671 | 0.7734 |

**Zaključak**
- Nakon statističke ispravke: smer je dosledno tačan (pozitivne mete rangirane bolje od negativnih), ali NIJE statistički značajno — efektivna snaga ograničena na ~8 pacijenata koji imaju i poznat pozitivan i negativan hidden trial.
- Odvojeno od ovoga: interni graph-propagation nalaz (RRF-4 na gold datasetu) OSTAJE značajan — ova korekcija se odnosi samo na eksterni test na pacijentima, ne menja status RRF-4.
- Glavno usko grlo za dalji napredak: broj pacijenata sa i pozitivnim i negativnim resolvable komponentama, ne količina podataka za trening modela.
