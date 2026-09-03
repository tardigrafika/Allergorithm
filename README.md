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
| [Sliding-window ESM (nsLTP/Profilin/PR-10)](#sliding-window-esm-nsltpprofilinpr-10) | 0.0524 | — | — | — | Null/blago negativno — ne rešava unutar-familije zagušenje |
| [MI/hypergraph LSE-pooling (nsLTP/Profilin/PR-10)](#mihypergraph-lse-pooling-nsltpprofilinpr-10) | 0.0537–0.0843 | — | — | — | **Prvo pravo poboljšanje na 2/3 familije, LOCO-potvrđeno** (nsLTP/Profilin značajno, PR-10 ne) |
| [Attention-MIL](#attention-mil-eskalacija-lse-poolinga) | — | — | — | — | Ne prevazilazi jednostavni LSE — zatvoren pravac |
| [PR-10 dijagnoza](#pr-10-dijagnoza--zašto-mi-lse-ne-pomaže-baš-ovoj-familiji) | — | — | — | — | Strukturno bez lokalnog signala, treba drugu vrstu podataka |
| [Error taxonomy + Inferred tier odluka](#error-taxonomy--inferred-tier-odluka) | — | — | — | — | 96% grešaka objašnjeno; Inferred isključen iz treninga (ne iz dataseta) |
| [RRF-5 (family-aware LSE, pacijenti)](#rrf-5-family-aware-lse--test-na-pacijentima) | — | — | — | — | **Pogoršao pacijentski rezultat** — interno dobar signal ne prenosi se klinički |
| [RRF-6 (MLP-hadamard, pacijenti)](#rrf-6-mlphadamard--test-na-pacijentima) | — | — | — | — | Bolji od RRF-4, ali **RRF fuzija sama pokazala se lošijom od pojedinačnih signala** (videti niže) |
| [BLAST SAM vs MLP(hadamard) SAM — gold LOCO](#blast-sam-vs-mlphadamard-sam--gold-loco) | BLAST 0.1243 / MLP 0.1259 | — | — | — | Izjednačeni, delta zanemarljiv, nije značajno (ni pair ni study-level) |
| [BLAST SAM vs MLP(hadamard) SAM — pacijenti (upareni test)](#blast-sam-vs-mlphadamard-sam--pacijenti-ispravan-upareni-test) | — | — | — | — | **MLP(hadamard) značajno bolji od BLAST-a na pacijentima** — glavni nalaz sesije |
| [OuterProductBilinear (mentorov predlog)](#outerproductbilinear-mentorov-predlog--odbačeno) | ~0.005–0.05 (gold) / 0.083 (pac.) | — | — | — | Kolabira i na gold datasetu i na pacijentima — odbačeno |
| [57-pacijentski suite + ojačan nalaz](#57-pacijentski-test-suite--ojačan-blast-vs-mlphadamard-nalaz) | — | — | — | — | Sva 3 testa sada značajna na oba podskupa (bilo granično na 54 pac.) |
| [Failure analysis MLP vs BLAST (57 pac.)](#failure-analysis-gde-mlphadamard-greši-naspram-blast-a-57-pacijenata) | — | — | — | — | 5 empirijskih nalaza — MLP bolji na PR-10/nsLTP pozitivima, lošiji na crowded negativima |
| [Targeted hard-negative eksperiment](#targeted-hard-negative-eksperiment--pokušaj-popravke-negativan-rezultat) | 0.0804–0.1267 (gold LOCO, 4 ratio-a) | — | — | — | Bez efekta na LOCO, POGORŠAO flagship probu — odbačeno, veto odbijen |
| [LSE-pooling kao primarni ranker](#lse-pooling-kao-primarni-ranker--feasibility-check-odustalo-se-pre-punog-pokretanja) | — | — | — | — | ~3h projektovana cena — odloženo pre punog LOCO-a |
| [LayerNorm ablacija](#layernorm-ablacija-za-mlphadamard--čist-značajan-negativan-rezultat) | 0.0804 | — | — | — | Značajno GORE od baseline-a (−0.0455, CI isključuje 0) — odbačeno |
| [ESM-1b naspram ESM-2 backbone](#esm-1b-naspram-esm-2-backbone--mlphadamard-značajan-negativan-rezultat) | 0.1065 (ESM-1b) | — | — | — | Značajno GORE od ESM-2 (−0.0195, CI isključuje 0) — ne koristiti |
| [ESM-2 3B naspram 650M backbone](#esm-2-3b-naspram-esm-2-650m-backbone--mlphadamard-fer-poredjenje-posle-uhvaćenog-confound-a) | 0.1131–0.1136 (3B, retunovano) | — | — | — | Retuning resio kolaps, i dalje značajno GORE od BLAST-a i 650M-a — ne koristiti |

---
Trenutni broj gold-cross-reactive parova: 1922
## Trenutno najbolji model

**Najjača, najpouzdanije potvrđena tvrdnja projekta (2026-08-30, ojačana 2026-08-31 na 57 pacijenata): MLP(hadamard) SAM (bez RRF fuzije) statistički značajno prevazilazi čist BLAST na pravim pacijentima**, pravim uparenim testom na istim upitima — sada sva tri testa (Wilcoxon, cluster-permutacija, bootstrap) značajna na OBA podskupa (svi upiti i "hard"), ne samo na hard podskupu kao ranije. Videti [punu sekciju](#blast-sam-vs-mlphadamard-sam--pacijenti-ispravan-upareni-test) i [57-pacijentski update](#57-pacijentski-test-suite--ojačan-blast-vs-mlphadamard-nalaz) — ovo je headline tvrdnja za tezu, ne RRF-4/RRF-6.

**Svaki pokušaj poboljšanja MLP(hadamard)-a preko ovog nalaza (2026-08-30/31) vratio se negativan ili nepraktičan**: targeted hard-negative trening (bez efekta, pogoršao flagship probu), inference-time veto (eksplicitno odbijen — slabi naučnu interpretaciju), LSE-pooling kao primarni ranker (odloženo, ~3h projektovana cena), LayerNorm (značajno gore, CI isključuje 0). Detaljna [failure analysis](#failure-analysis-gde-mlphadamard-greši-naspram-blast-a-57-pacijenata) pokazuje TAČNO gde i zašto: MLP dobija na true pozitivima u PR-10/nsLTP, gubi na potiskivanju negativa unutar crowded familija — ovo ostaje dokumentovano, nerešeno ograničenje trenutnog modela, ne patch-ovano pravilom.

**RRF-4 (cosine + BLAST + FoldseekTM + graph-propagation)** ostaje produkcioni pipeline za praktičnu upotrebu, ALI sa važnom ogradom dodatom 2026-08-29 (videti [Error taxonomy + Inferred tier odluka](#error-taxonomy--inferred-tier-odluka)): njegove interne statističke tvrdnje su **slabije nego što je ranije prijavljeno**, otkriveno kroz study-level bootstrap (resampling po IZVORU citata, ne po pojedinačnom paru — mnogi parovi dele isti izvor, pa pair-level bootstrap potcenjuje nesigurnost). Dodatno, 2026-08-30 je otkriveno da **RRF fuzija (RRF-4/RRF-6) dosledno gubi od pojedinačnih signala (BLAST sam, MLP sam) na pacijentima** — kompleksnija fuzija nije uvek bolja.

| | MRR (primenljiv podskup) | Delta | Pair-level 95% CI | **Study-level 95% CI** | Značajno (study-level)? |
|---|---|---|---|---|---|
| RRF-3 vs cosine | — | +0.0085 (pair) / +0.0113 (study) | [+0.0028,+0.0142] | [+0.0032,+0.0301] | DA (ali NE na ne-Inferred podskupu: CI[−0.0004,+0.0338]) |
| RRF-3 vs BLAST | — | +0.0057 | [+0.0001,+0.0116] | [−0.0139,+0.0113] | **NE** |
| RRF-4 vs RRF-3 (graph-prop) | — | +0.0060 | [+0.0016,+0.0101] | [−0.0015,+0.0172] | **NE** |

Tačka procene ostaje pozitivna svuda — ovo NIJE dokaz da je RRF-4 pogrešan, znači da je deo dataseta (57.5% je jedan blanket-citat, Inferred tier) manje nezavisan dokaz nego što je pair-level bootstrap pretpostavljao.

**Novi kandidat, još ne potvrđen dovoljno podataka: RRF-6 (RRF-4 + MLP(hadamard))** — prva dopuna koja NE kvari, nego poboljšava rezultat na pravim pacijentima (videti [RRF-6 sekciju](#rrf-6-mlphadamard--test-na-pacijentima)). Nije još zamenio RRF-4 kao default — glavni pacijentski test i dalje nije statistički značajan, treba više podataka.

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

---

## Sliding-window ESM (nsLTP/Profilin/PR-10)

**Cilj** — testirati da li lokalna rezolucija (preklapajući prozori od ~20 rezidua, veličina tipičnog linearnog epitopa, mean-pooled) razdvaja unutar-familije "zagušenje" bolje od whole-protein mean-pool cosine-a. Zagušenje je dijagnostikovano zasebno (`mrr_by_family_1548.py` + prosečan broj kandidata unutar 0.01 cosine-a od tačnog cilja: nsLTP=322.9, Profilin=161.5, PR-10=122.0 vs Tropomiozin=91.8) kao verovatan mehanizam iza katastrofalno niskog MRR-a baš ovih familija.

**Šta je urađeno**
- Prozor=20 rezidua, korak=5, mean-pooled iz postojećih per-residue ESM embeddinga.
- Skor para = MAX cosine sličnost preko svih parova prozora (najbolje lokalno poklapanje), ne prosek.
- Testirano SAMO na nsLTP/Profilin/PR-10 (gde je problem dijagnostikovan), whole-protein cosine na ISTOM podskupu kao referenca.
- Razlikuje se od ranijeg null nalaza (`residue_topk_nsltp_profilin_1548.py`, top-15 pojedinačnih rezidua bez pool-ovanja) — ovo koristi lokalno pool-ovane prozore, manje šumovito.

**Rezultati**

| Familija | cosine MRR | sliding-window MRR | delta |
|---|---|---|---|
| nsLTP (n=798) | 0.0559 | 0.0561 | +0.0002 |
| Profilin (n=648) | 0.0509 | 0.0451 | −0.0058 |
| PR-10 (n=370) | 0.0549 | 0.0570 | +0.0021 |
| **Ukupno (n=1816)** | **0.0539** | **0.0524** | **−0.0015** |

Samo 39.8% upita (723/1816) je poboljšalo rang.

**Zaključak**
- Null/blago negativan rezultat — sliding-window MAX-pooling ne rešava zagušenje, na Profilinu ga pogoršava. Verovatno objašnjenje: panalergeni dele visoko konzervisano strukturno jezgro, pa gotovo svaki član familije ima BAR JEDAN par prozora koji se skoro savršeno poklapa negde u tom jezgru — MAX agregacija tu sličnost pojačava za skoro sve kandidate podjednako, umesto da razblaži zagušenje kako whole-protein prosek delimično radi. Odbačen u ovom obliku (MAX preko sliding-window prozora).
- Brz follow-up (top-3 prosek umesto MAX-1, `sliding_window_top3_1548.py`) daje praktično identičan rezultat (ukupno delta=-0.0014, isti obrazac po familijama, 39.2% upita poboljšano) — problem nije u načinu agregacije prozora, potvrđuje da je hipoteza (konzervisano jezgro dominira signal) tačna nezavisno od agregacije. Zatvoren pravac.

---

## MI/hypergraph LSE-pooling (nsLTP/Profilin/PR-10)

**Cilj** — testirati mentorovu multiple-instance/hypergraph ideju: umesto FIKSNE agregacije preko parova prozora (mean=cosine baseline, MAX=sliding-window, oba null), NAUČITI agregaciju. Koristi se Log-Sum-Exp ("smooth-max") pooling sa jednim trenabilnim parametrom temperature τ, koji glatko interpolira između mean-a (τ→∞) i max-a (τ→0) — principijelan srednji put između dve već testirane i odbačene krajnosti.

**Šta je urađeno**
- LSE_τ(S) = τ·log(mean_i(exp(S_i/τ))) preko cele matrice sličnosti svih parova prozora (window=20, stride=5, isti prozori kao sliding-window eksperimenti).
- Trening: logistička regresija sa 3 parametra (τ, scale, bias), fit gradient descent-om (PyTorch, ~sekunde) na pozitivima + nasumičnim negativima iz celog poola (ista bezbedna metoda kao svuda u projektu — **ne** in-family hard-negative mining, ta ideja je razmotrena i odbačena kao biološki nepouzdana, videti niže).
- **Prava LOCO validacija** (`mi_lse_loco_1548.py`): 3 odvojena fold-a, svaki izdvaja TAČNO JEDNU od tri ciljne familije iz treninga (druge dve ostaju) — moguće jer je svaka familija tačno jedna connected komponenta u gold grafu (potvrđeno u `bridge_protein_analysis_1548.py`). Rezultat identičan pilot verziji (koja je izdvojila sve tri odjednom) do 4. decimale — nezavisno od toga da li model vidi ostale dve teške familije u treningu.
- **Šum-ablacija testirana i odbačena kao neefikasna** (ne kao pogrešna ideja): ~140 Inferred-tier parova ne prolazi WHO2001 kriterijum, ali WebSearch verifikacija pokazala je da su nsLTP (69)/2S albumin (33)/Lipocalin (11) sve literaturom potvrđene "nizak identitet ALI pravi fold-konzervisan cross-reactivity" panalergen familije — isključivanje bi uklonilo pravi signal, ne šum. Samo Mite group 5/7/21 (11 parova, svi grupa-7-naspram-5/21) ima literaturnu potvrdu za slabu/nepotvrđenu reaktivnost. Isključivanje baš tih 11 parova iz treninga nije promenilo NIŠTA merljivo (11 od ~3970 primera je premalo za 3-parametarski model).

**Rezultati (LOCO, bootstrap CI 2000 resample po pair_id)**

| Familija | cosine MRR | LSE-pooling MRR | delta | 95% CI | Značajno? |
|---|---|---|---|---|---|
| nsLTP (n=798) | 0.0559 | 0.0777 | +0.0218 | [+0.0116, +0.0329] | **DA** |
| Profilin (n=648) | 0.0509 | 0.0843 | +0.0334 | [+0.0183, +0.0487] | **DA** |
| PR-10 (n=370) | 0.0549 | 0.0537 | −0.0012 | [−0.0181, +0.0131] | Ne |

**Zaključak**
- Prvi mehanizam u celom ovom dijagnostičkom nizu (sliding-window MAX, top-3, konzerviranost, BepiPred, površinski top-K — sve null) koji pokazuje realan, LOCO-potvrđen pozitivan signal. Prvi NN-adjacent model od MLP(hadamard) naovamo koji prolazi punu LOCO disciplinu — i prvi koji to radi sa pravim poboljšanjem, ne samo izjednačenjem sa cosine-om.
- **Nije "rešen problem"** — apsolutne MRR vrednosti (0.078, 0.084) ostaju daleko ispod dobro-ponašanih familija (Tropomiozin/Troponin C ~0.6-0.7). Ovo je "manje katastrofalno", ne "dobro".
- PR-10 potpuno nepromenjen — mehanizam koji pomaže nsLTP/Profilin ne rešava šta god je specifično pokvareno kod PR-10; verovatno drugačiji uzrok.
- Sledeći koraci (nije još urađeno): puna attention-MIL verzija (naučena po-prozor relevantnost, ne samo jedan globalni τ), per-familija τ (moguće objašnjenje zašto PR-10 ne reaguje), dijagnoza PR-10 specifično.

---

## PR-10 dijagnoza — zašto MI/LSE ne pomaže baš ovoj familiji

**Cilj** — pre bilo kakvog daljeg ulaganja (per-familija τ, attention-MIL), razumeti STRUKTURNI razlog zašto isti mehanizam popravlja nsLTP/Profilin ali ne PR-10. Korisnički definisan skup od 6 provera.

**Šta je urađeno** (`analysis/pr10_diagnosis_1548.py`, `analysis/pr10_hotspot_diagnosis_1548.py`)
1. Distribucija dužine proteina po familiji
2. Sequence identity gold parova po familiji
3. Da li gold-pozitivni parovi dele isti relativni "hot spot" region (najbolji par prozora)
4. Raznovrsnost unutar familije (prosečna međusobna cosine sličnost SVIH parova članova, ne samo gold)
5. Label coverage / gustina grafa (broj poznatih partnera po proteinu)
6. Distribucija delta po upitu (pravi null ili mešavina dobitaka/gubitaka koja se poništi)

**Rezultati**

| Familija | Dužina proteina (std) | Unutar-familijska cosine (mean / **std**) | Gustina grafa | Prosečan stepen |
|---|---|---|---|---|
| nsLTP | 28.1 aa | 0.9430 / **0.0465** | 46.3% | 19.0 |
| Profilin | 4.7 aa | 0.9915 / **0.0255** | 46.1% | 17.1 |
| **PR-10** | **1.8 aa** | 0.9918 / **0.0052** | 39.8% | 11.9 |
| Tropomiozin (kontrola) | 35.2 aa | 0.9907 / 0.0209 | 67.3% | 11.4 |

Hot-spot provera: PR-10 hot-spotovi su DOSLEDNO lokalizovani blizu C-terminusa (66% u poslednja 3/10 binova) — ali to nije partner-specifično; deluje kao univerzalno-konzervisan strukturni landmark cele familije. Distribucija delta po upitu (6): PR-10 IMA mešavinu velikih dobitaka/gubitaka (std=0.134, slično nsLTP/Profilin) — nije trivijalan "upiti nisu pogođeni" ishod, efekat se samo poništi na nuli.

**Zaključak**
- PR-10 (Bet v 1 fold) je izuzetno rigidna, strukturno uniformna familija (dužina proteina std=1.8 rezidua — gotovo identična dužina svih članova). Ovo se prevodi u skoro NULTU varijansu unutar-familijske embedding sličnosti (std=0.0052, 5-9x niže od ostalih familija) — nema diferencijacije koju BILO KOJI mehanizam agregacije može otkriti, jer ne postoji u embeddinzima.
- Hot-spot koji LSE nalazi za PR-10 je verovatno univerzalno-konzervisan strukturni region, ne partner-specifičan epitop — pali podjednako jako za pravog partnera i bilo kog drugog člana familije, pa ne pomaže diskriminaciji.
- Sekundaran faktor: PR-10 ima ređi gold graf (gustina 39.8% vs 46%+ kod ostalih, stepen 11.9 vs 17-19) — manje trening signala specifičnog za ovu familiju.
- **Implikacija**: per-familija τ verovatno NE bi pomogao PR-10 — problem nije pogrešna temperatura agregacije, nego odsustvo diskriminativnog signala u samoj reprezentaciji, koje nijedan skalar hiperparametar ne može proizvesti. PR-10 verovatno zahteva kvalitativno drugačiju informaciju (prava 3D struktura, ili prošireno label coverage) — ne dalje variranje agregacije nad istim ESM embeddinzima. **Odlučeno: PR-10 ostaje dokumentovan negativan rezultat, dalji rad (attention-MIL) fokusiran samo na nsLTP/Profilin.**

---

## Attention-MIL (eskalacija LSE-poolinga)

**Cilj** — testirati da li NAUČENA nelinearna funkcija (mala MLP, 1→8→1, primenjena na svaku sličnost para prozora pre softmax-attention agregacije) prevazilazi jednostavni LSE-pooling (jedan skalarni parametar τ). LSE je specijalan slučaj ove opštije forme (fiksna linearna f(s)=s/τ) — attention-MIL bi trebalo da bude STROGO izražajniji.

**Šta je urađeno** (`analysis/attention_mil_1548.py`) — ista LOCO disciplina (2 fold-a, nsLTP i Profilin odvojeno izdvojeni, PR-10 namerno izostavljen kao cilj posle dijagnoze), isti bootstrap CI protokol.

**Rezultati**

| Familija | LSE-pooling (3 parametra) | Attention-MIL (~25 parametara) |
|---|---|---|
| nsLTP | delta=+0.0218, CI[+0.0116,+0.0329] **značajno** | delta=+0.0075, CI[−0.0048,+0.0200] **NIJE značajno** |
| Profilin | delta=+0.0334, CI[+0.0183,+0.0487] značajno | delta=+0.0341, CI[+0.0189,+0.0502] značajno (praktično identično) |

**Zaključak**
- Dodatna fleksibilnost NIJE pomogla — na Profilinu praktično identičan rezultat kao jednostavni LSE, na nsLTP CAK GORE (izgubljena statistička značajnost). Verovatno blago overfitovanje na trening raspodelu koje se lošije generalizuje na held-out familiju nego ograničenija LSE forma.
- **LSE-pooling (jedan naučen τ) ostaje najbolji, robusniji nalaz u ovom pravcu.** Odbačeno dalje širenje modela u ovom smeru — jednostavnije je ovde bilo bolje. Zatvoren pravac za MI/hypergraph agregacione eksperimente na trenutnom nivou infrastrukture (window-pooling preko ESM embeddinga).

---

## Error taxonomy + Inferred tier odluka

**Cilj** — mentorov predlog: umesto nagađanja familija-po-familija (kako je rađeno za PR-10), sistematski kategorisati SVE loše rangirane RRF-4 upite po verovatnom uzroku, da se vidi gde stvarno vredi ulagati dalje.

**Šta je urađeno** (`analysis/error_taxonomy_1548.py`) — svaki upit sa rangom > 100 (211/3074 = 6.9%) označen po kategoriji: cold_start, ccd_driven, inferred_tier, who2001_borderline_expected (familije gde je nizak identitet OČEKIVAN — nsLTP/2S albumin/Lipocalin), who2001_fail_suspect, low_identity, directionality_known.

**Rezultati**

| Kategorija | Udeo loših upita |
|---|---|
| Inferred tier | 90% (3.5× veći rizik neuspeha nego ne-Inferred: 8.5% vs 2.5% stopa) |
| WHO2001 pada, očekivano (fold-conserved familije) | 14.7% |
| Nizak identitet (<30%) | 14.2% |
| WHO2001 pada, sumnjivo | 8.5% |
| CCD-vođeno | 2.4% |
| **NEOBJAŠNJENO** | **3.8%** (8/211 upita) |

**96.2% loših upita ima bar jednu poznatu, već dijagnostikovanu kategoriju** — samo 8 upita je prava misterija.

**Zaključak i odluka (2026-08-29)**
- Ono što izgleda kao "loš MRR" je uglavnom posledica kvaliteta/pouzdanosti Inferred tier-a (57.5% dataseta, jedan blanket citat), ne skrivenog reprezentacijskog problema — konvergira sa study-level bootstrap nalazom (ista tier je i statistički najmanje pouzdana).
- **Literatura-nadogradnja**: primenjeno 13/17 već pronađenih kandidata (Scala et al. 2011, n=3113 pacijenata, samo Profilin — nedvosmisleno pokriveno) na `evidence_level="Strong evidence (population correlation...)"`. 4 sumnjiva kandidata (3 PR-10, 1 Oleosin) namerno OSTAVLJENA kao Inferred — izvor ih ne pokriva nedvosmisleno.
- **Za ostatak (~1093 para) koji se ne može pojedinačno nadograditi**: korisnica je eksplicitno odbacila potpuno brisanje (87% prolazi WHO2001, nekoliko familija ima nizak identitet iz pravih bioloških razloga — brisanje bi verovatno uklonilo pretežno tačne podatke, kosi se sa "nikad ne brišemo informacije" principom). Umesto toga: **novi `training_eligible_pairs()` u `ml/pipeline/common/data.py`** isključuje preostale Inferred parove SAMO iz treninga supervizovanih modela — ostaju puni deo dataseta i evaluacije. Aditivna, bezbedna izmena (ne menja ponašanje nijednog postojećeg skripta dok se eksplicitno ne pozove).

---

## RRF-5 (family-aware LSE) — test na pacijentima

**Cilj** — konkretan predlog za "novi glavni pipeline": umesto univerzalne formule, dodati LSE-pooling (dokazano poboljšanje na nsLTP/Profilin, LOCO-potvrđeno) SAMO za upite iz tih familija, testirati direktno na proširenom pacijentskom test suite-u (49 pacijenata, 4 nova profilin slučaja dodata baš za ovaj test).

**Šta je urađeno** (`test/evaluate_rrf5_family_aware_1548.py`) — identičan RRF-4 mehanizam (`CrossReactivityRanker`), plus LSE-pooling termin za poznate pozitive iz nsLTP/Profilin. Testirano i sa punim gold treningom i sa `training_eligible_pairs()` (bez preostalih Inferred) — **oba varijante daju praktično identičan (loš) rezultat**.

**Rezultati** (cluster-permutacija + patient-level Wilcoxon, ista metodologija kao real-world validacija)

| Test | RRF-4 (bazni) | RRF-5 original | RRF-5 čist trening |
|---|---|---|---|
| Svi pacijenti, cluster-perm p | 0.517 | 0.555 | 0.555 |
| Bez Mothes-Luksch, cluster-perm p | **0.044 (značajno)** | 0.131 (nije) | 0.131 (nije) |
| Bez Mothes-Luksch, Wilcoxon p | **0.047 (značajno)** | 0.156 (nije) | 0.156 (nije) |

**Zaključak**
- Dodavanje INTERNO potvrđenog signala (LSE-pooling) **pogoršalo je** rezultat na pravim pacijentima — nestao je jedini značajan nalaz koji je RRF-4 imao. Čišćenje trening podataka (bez Inferred) NIJE popravilo problem — identičan rezultat, isključuje hipotezu "nepouzdan trening je kriv".
- Verovatno objašnjenje: LSE-pooling optimizuje uzak sekvencijalni/strukturni signal (validiran na molekularnom gold datasetu) koji se ne prenosi na stvarnu kliničku IgE reaktivnost — isti gap koji alat već priznaje u svom disclaimer-u.
- RRF-5 u ovom obliku NE zamenjuje RRF-4.

---

## RRF-6 (MLP-hadamard) — test na pacijentima

**Cilj** — revizitirati MLP(hadamard) (interno tačno izjednačio cosine pod LOCO — ni pobeda ni poraz) na pravim pacijentima, sa istom disciplinom kao RRF-5 ali GLOBALNO dodato (ne familijski-ograničeno — MLP nema dokazanu familijsku prednost, za razliku od LSE-a).

**Šta je urađeno** (`test/evaluate_rrf6_mlp_hadamard_1548.py`) — MLP(hadamard) treniran od početka na `training_eligible_pairs()` (785 čistih parova, bez Inferred), dodat kao dodatni RRF term za SVAKOG poznatog pozitiva, svih familija. Uhvaćen i ispravljen potencijalni bug PRE pokretanja: `CrossReactivityRanker.pool` i `dataset.all_ids` imaju različit redosled istog skupa proteina — bez eksplicitne permutacije, MLP skorovi bi se tiho pogrešno poravnali sa pogrešnim proteinima.

**Rezultati**

| Test | RRF-4 (bazni) | RRF-5 (LSE) | **RRF-6 (MLP-hadamard)** |
|---|---|---|---|
| Svi pacijenti, cluster-perm p | 0.517 | 0.555 | **0.168** |
| Bez Mothes-Luksch, cluster-perm p | 0.044 (značajno) | 0.131 (nije) | **0.009 (značajno, 5× jače)** |
| Bez Mothes-Luksch, Wilcoxon p | 0.047 (značajno) | 0.156 (nije) | **0.047 (značajno, isto)** |

**Zaključak**
- **Prva dopuna u celoj sesiji koja ne kvari, nego poboljšava rezultat na pravim pacijentima** — glavni test se pomerio u obećavajućem smeru (iako još nije značajan), sensitivity provera je ojačala skoro 5×.
- Interno-vs-klinički gap NIJE univerzalan — specifičan je za LSE-pooling (ili lokalne/familijski-ograničene signale generalno), ne za "bilo koji ML dodatak". MLP(hadamard), uprkos internoj redundantnosti sa cosine-om, hvata nešto na pravim pacijentima što RRF-3/RRF-4 sami ne hvataju.
- **Vodeći kandidat za bolji alat od RRF-4** — još nije dovoljno potvrđen (glavni test nije značajan, treba više pacijentskih podataka) da zameni produkcioni default, ali prvi kredibilan pravac napretka na metrici koja je stvarno bitna.
- **AŽURIRANO 2026-08-30**: sledeći koraci (niže) pokazuju da čak i RRF-6 fuzija gubi od SAMOG MLP(hadamard) signala bez ikakve RRF fuzije — ova sekcija ostaje kao istorijski zapis prvog pozitivnog pomaka, ali headline tvrdnja projekta je sada jednostavnija (videti niže).

---

## BLAST SAM vs MLP(hadamard) SAM — gold LOCO

**Cilj** — vratiti se na jezgro naučnog pitanja (korisnički zahtev 2026-08-30): da li embedding-based metoda SAMA (bez RRF fuzije koja već sadrži BLAST kao sastojak) statistički značajno prevazilazi čist BLAST — na CELOM gold datasetu, punom LOCO metodologijom.

**Šta je urađeno** (`ml/loco_blast_vs_mlp_hadamard_only_1548.py`) — 40-fold LOCO, MLP(hadamard) treniran ISPOČETKA svaki fold na `training_eligible_pairs()` (cist trening), BLAST rang računat naspram CELOG pool-a od 1535 proteina za svaki upit. Bootstrap CI na oba nivoa (pair i study).

**Rezultati** (3756 upita, ceo dataset)

| | MRR (micro) | Delta | Pair-level 95% CI | Study-level 95% CI |
|---|---|---|---|---|
| BLAST | 0.1243 | — | — | — |
| MLP(hadamard) | 0.1259 | +0.0016 | [−0.0061,+0.0091] nije značajno | [−0.0333,+0.0123] nije značajno |

**Zaključak**
- Na molekularnom gold datasetu, sam MLP(hadamard) i sam BLAST su statistički izjednačeni — nema dokaza da embedding-based klasifikator sam po sebi prevazilazi sekvencijalno poravnanje na OVOJ metrici. Ovo se poklapa sa ranijim LOCO nalazom da MLP(hadamard) tačno izjednačuje cosine.

---

## BLAST SAM vs MLP(hadamard) SAM — pacijenti (ispravan upareni test)

**Cilj** — isti test kao gore, ali na metrici koja je klinički najbitnija: pravi pacijenti (54, leave-one-out). **VAŽNA METODOLOŠKA NAPOMENA**: prvi pokušaj ovog poređenja (cluster-permutacija posebno za MLP i posebno za BLAST, svaki protiv sopstvene permutovane nulte hipoteze) bio je **pogrešan** — dokazuje samo da svaki model pojedinačno nosi signal iznad slučajnosti, ne da je jedan bolji od drugog. Ispravljeno na zahtev korisnice/mentora: pravi upareni test na ISTIM upitima.

**Šta je urađeno**
- `test/evaluate_mlp_only_vs_blast_only_patients_1548.py` — dva rankera, SVAKI sa SAMO JEDNIM signalom (ne RRF-3+X), isti "sabiranje 1/(K+rang) preko poznatih pozitiva" mehanizam kao `CrossReactivityRanker`. MLP(hadamard) treniran na `training_eligible_pairs()`.
- `test/paired_test_mlp_vs_blast_1548.py` — ISPRAVAN upareni test, tri metode, sve uparene po (patient_id, hidden_protein) — spajanje verifikovano 1:1, bez duplikata:
  1. Patient-level Wilcoxon signed-rank na MRR(MLP)−MRR(BLAST) po pacijentu
  2. Cluster-permutacija koja permutuje OZNAKU MODELA (ne ishod) unutar svakog pacijenta
  3. Patient-level bootstrap CI, resample po pacijentu (ne po upitu)

**Rezultati**

| Test | Svi upiti (n=152, 51 pac.) | Samo "hard"/verifikovano (n=124, 41 pac.) |
|---|---|---|
| Patient-level Wilcoxon | p=0.030 **značajno** | p=0.012 **značajno** |
| Cluster-permutacija (permutuj MLP/BLAST oznaku) | p=0.079 nije značajno | p=0.033 **značajno** |
| Patient-level bootstrap CI | [+0.0000,+0.0246] **značajno** (jedva) | [+0.0029,+0.0318] **značajno** |

**Zaključak**
- **Na "hard" (najpouzdanijem) podskupu, sva tri nezavisna testa se slažu: MLP(hadamard) SAM statistički značajno prevazilazi BLAST SAM**, pravim uparenim poređenjem na istim pacijentima. Prosečna prednost ≈ +0.015 do +0.019 MRR.
- Na punom skupu (uključujući manje pouzdano verifikovane slučajeve) dva od tri testa su značajna; cluster-permutacija je granična (p=0.079) — pošteno naznačeno kao slabija karika, ne prećutano.
- **Ovo je NAJJAČA, metodološki najispravnija tvrdnja cele sesije za centralno naučno pitanje** ("da li embedding-based metoda prevazilazi baseline bez embeddinga") — suprotstavljeno gold-LOCO rezultatu iznad (izjednačeno na gold datasetu), što samo po sebi govori da gold-standard molekularne labele i stvarna klinička reaktivnost nisu ista stvar — MLP hvata nešto klinički relevantno što BLAST ne hvata, iako oba podjednako dobro pogađaju gold-dataset labele.
- **Opšta lekcija**: dva odvojena testa "iznad slučajnosti" NIKAD nisu zamena za jedan pravi upareni test — ova greška je lako napraviti kad se izveštavaju paralelni rezultati značajnosti jedan pored drugog, i treba je eksplicitno proveravati pre svake "X bolji od Y" tvrdnje.

---

## Cosine SAM na pacijentima — nedostajuća tačka poređenja, popunjena (2026-09-02)

**Cilj** — od 2026-08-30 flagovano kao "not yet done": jedini signal koji NIKAD nije testiran kao samostalan pacijentski ranker bio je cosine (BLAST-sam i MLP-sam jesu). Bez ovoga, ne može se razlikovati da li "MLP(hadamard) bolji od BLAST-a" znači "ma koji embedding-signal bolji od BLAST-a" ili je specifično NAUČENA transformacija (Hadamard MLP) ta koja nosi prednost. `test/evaluate_cosine_only_patients_1548.py` — isti leave-one-out mehanizam, ESM-2 650M cosine sličnost bez ikakvog treninga, na identičnom (n=176, 54 pac.) skupu kao već objavljeni BLAST/MLP brojevi.

**Napomena o metodologiji**: `protein_resolution.py` popravka istog dana (Pen a1/Pen m1 aliasi) retroaktivno otključava 6 dodatnih proba kod uklejasokolowska2021 pacijenata koje nisu postojale kad su stariji BLAST/MLP raw fajlovi računati — te probe su eksplicitno izbačene iz ovog poređenja (presek, ne unija) da ostane fer, isti-skup poređenje sa već objavljenim brojevima.

**Rezultati**

| Test | Cosine vs BLAST — svi upiti | Cosine vs BLAST — hard | Cosine vs MLP — svi upiti | Cosine vs MLP — hard |
|---|---|---|---|---|
| Patient-level Wilcoxon | p=0.7994 n.z. | p=0.4692 n.z. | **p=0.0172 ZNAČAJNO** | **p=0.0009 ZNAČAJNO** |
| Cluster-permutacija | **p=0.0205 ZNAČAJNO** | **p=0.0057 ZNAČAJNO** | **p=0.0026 ZNAČAJNO** | **p=0.0002 ZNAČAJNO** |
| Bootstrap CI | **[-0.0921,-0.0121] ZNAČAJNO** | **[-0.1105,-0.0166] ZNAČAJNO** | **[-0.1042,-0.0214] ZNAČAJNO** | **[-0.1248,-0.0301] ZNAČAJNO** |

Sve razlike su u istom (negativnom za cosine) smeru — cosine je najslabiji od sva tri signala na pacijentima, značajno gori od MLP-a (sva 3 testa, oba podskupa) i značajno gori od BLAST-a (2/3 testa, oba podskupa, jedva ne-značajno na Wilcoxon-u).

**Zaključak — precizira headline nalaz**: MLP(hadamard)-ova prednost nad BLAST-om na pacijentima NIJE prosto "embedding sličnost > sekvencijalno poravnanje" — sirov cosine u ISTOM embedding prostoru gubi od BLAST-a. Prednost dolazi SPECIFIČNO od naučene, nelinearne Hadamard-MLP transformacije trenirane na gold cross-reaktivne parove, ne od same reprezentacije. Kompletna rang lista na pacijentima: **MLP(hadamard) > BLAST > cosine**, sa MLP i BLAST oba značajno iznad cosine, i MLP značajno iznad BLAST-a.

---

## OuterProductBilinear 
**Cilj** — mentorova ideja: umesto Hadamard produkta (u⊙v, hvata SAMO interakcije iste dimenzije), koristiti outer product (u⊗v, hvata SVE parove dimenzija u_i·v_j) pomnožen matricom težina, pa MLP — "dobićeš delove koji su bitni za reakciju".

**Šta je urađeno** (`test/evaluate_bilinear_outer_1548.py`, `ml/loco_blast_vs_bilinear_1548.py`) — low-rank varijanta (pun 1280×1280 outer product bi dao ~1.6M parametara naspram samo 785 čistih trening parova — zagarantovano overfitovanje): deljena naučena projekcija A (1280→64), simetričan outer product (a⊗b + b⊗a, garantuje score(u,v)=score(v,u)), pa mali MLP. Testirano i na gold LOCO (40 folda) i na pacijentima (isti upareni test kao MLP/BLAST iznad).

**Rezultati**
- Gold LOCO (prvih 5/40 folda, obrazac već nedvosmislen, run prekinut): MRR 0.005–0.05 naspram BLAST-ovih 0.05–0.15 — **5-20× gore po foldu**, isti obrazac kolapsa kao raniji odbačeni "MLP embedding-transform" model.
- Pacijenti: MRR=0.083 (znatno niže od MLP-hadamard/BLAST), **značajno GORE od BLAST-a** na "svi upiti" podskupu (patient-Wilcoxon p=0.0136, u pogrešnom smeru), dosledno gore (nije značajno) i od MLP(hadamard)-a.

**Zaključak**
- Odbačeno u ovom obliku. Verovatno objašnjenje: čak i low-rank (r=64) bilinearna forma je previše izražajna za ~785 čistih trening primera — dodatna izražajnost ovde šteti, ne pomaže, ista lekcija kao attention-MIL (25 parametara) koji je izgubio od LSE-poolinga (3 parametra) ranije istog dana. Puna (bez low-rank redukcije) verzija bi imala još više parametara — očekivano još gore, ne bolje, pa nije ni testirana. Hadamard produkt (dijagonalne interakcije) ostaje bolje-prilagođen induktivni bias za ovaj problem sa ovoliko malo podataka.

---

## 57-pacijentski test suite + ojačan BLAST vs MLP(hadamard) nalaz

**Cilj** — dodati nove pacijente u test suite i proveriti da li glavni nalaz sesije (MLP(hadamard) SAM značajno bolji od BLAST-a SAMOG na pacijentima) ostaje stabilan/jača se kad se doda više podataka, ne samo jednom potvrđen na 54 pacijenta.

**Šta je urađeno** — 3 nova pacijenta iz da Silva et al. 2016 (`test/real_world_cases_6.json`, food-dependent exercise-induced urticaria/anaphylaxis, nsLTP-pozitivni ISAC profili) konvertovana u `test/test_cases.json` šemu (54→57 pacijenata). Svih 9 referenciranih proteina (Pru p 3, Cor a 8, Ara h 9, Jug r 3, Pla a 3, Art v 3, Tri a 14, Mal d 1, Pru p 1) se čisto rezolvuju u pool-u — nema novog data gap-a (za razliku od otvorenog Pen a1/m1/m4 nedostatka koji i dalje blokira 11 Giuffrida pacijenata). `test/evaluate_mlp_only_vs_blast_only_patients_1548.py` + `test/paired_test_mlp_vs_blast_1548.py` ponovo pokrenuti nepromenjeni (152→176 trial-ova).

**Rezultati**

| Test | Svi upiti — staro (n=152, 51 pac.) | Svi upiti — novo (n=176, 54 pac.) | Hard — staro (n=124, 41 pac.) | Hard — novo (n=148, 44 pac.) |
|---|---|---|---|---|
| Patient-level Wilcoxon | p=0.030 | **p=0.0116** | p=0.0124 | **p=0.0042** |
| Cluster-permutacija | p=0.079 (nije značajno) | **p=0.0304 (ZNAČAJNO)** | p=0.0331 | **p=0.0080** |
| Bootstrap CI | [+0.0000,+0.0246] | **[+0.0019,+0.0237]** | [+0.0029,+0.0318] | **[+0.0043,+0.0295]** |

**Zaključak**
- Dodavanjem 3 nova pacijenta, **sva tri testa su sada značajna na OBA podskupa** — pre toga je cluster-permutacija na "svim upitima" bila granična/neznačajna (p=0.079). Ovo zatvara jedinu slabu kariku iz prethodne METODOLOŠKE ISPRAVKE — glavni nalaz sesije je sada robustniji, ne samo ponovljen.

---

## Pen a1/Pen m1/Pen m4 gap zatvoren — 11 Giuffrida 2014 pacijenata dodato (57→68), 2026-09-02

**Otkriće**: Pen a 1, Pen m 1, Pen m 4 NISU nedostajali iz sirovih WHO/IUIS podataka — `data/jointable.csv` (izvor PRE čišćenja) ih ima sa punim sekvencama, potvrđeno unakrsno protiv allergen.org (aid 474/490/688) i UniProt-a (Q3Y8M6/A1KYZ2/E7CGC4). `data/finalclean_whoiuis.py` ih je ispravno izbacio kao duplikate sekvence — Pen a 1.0101 i Pen m 1.0101 su bit-za-bit identični već postojećem **Lit v 1.0101**, Pen m 4.0101 identičan **Lit v 4.0101** (biološki očekivano — blisko srodne vrste škampa, tropomiozin/SCBP izrazito konzervisani). Rešeno dodavanjem 3 alias-a u `test/protein_resolution.py` (isti mehanizam kao postojeći "Sus s (pork albumin)"→"Sus s 1"), NE dodavanjem novih redova u dataset (izbegava veštačko duplikovanje embedding-a). Bonus: ovo automatski otključava i 5 već postojećih uklejasokolowska2021 pacijenata čiji Pen m 1 do sada nije bio rezolvovan.

11 Giuffrida et al. 2014 pacijenata (Eur Ann Allergy Clin Immunol 46(5):172–7, PMID 25224947, pun tekst već transkribovan u `test/real_world_cases_5.md`) dodato u `test/test_cases.json` (57→68). **Napomena o leave-one-out mehanici**: 8/11 pacijenata ima OBA pozitivna kao tropomiozin (Pen a1+Pen m1) — pošto se oba mapiraju na isti pool-zapis, te dve probe se automatski i bezbedno preskaču (sakriveni protein bi bio identičan poznatom); validne probe ostaju za Pen m2/Pen m4 komponente. `test/evaluate_giuffrida2014_patients_1548.py`: **22 validne probe** (5 pozitivnih, 17 negativnih).

| Model | Pozitivni (n=5), medijan percentil | Negativni (n=17), medijan percentil |
|---|---|---|
| MLP(hadamard)-650M | 16.2% | 24.9% |
| BLAST | 15.9% | 17.4% |

**Zaključak (preliminaran, mali n)**: modeli su uporedivi na rangiranju pravih pozitiva (~16%), ali **BLAST dosledno bolje potiskuje negativne** (17.4% vs 24.9%) — isti kvalitativni obrazac kao već dokumentovana MLP negative-suppression slabost (Pru p1/Pru p3, Phl p12 iz failure analysis-a ispod). Agregatna statistika (Wilcoxon/cluster-permutacija/bootstrap) za ceo 68-pacijentski suite NIJE još regenerisana — ovo su rezultati samo za novih 11 pacijenata, ne izmena glavnog headline nalaza (koji i dalje stoji na n=57 dok se ne uradi puni re-run).

---

## Weighted-evidence trening (Suspected*=0.5) — pokušaj popravke negative-suppression slabosti, negativan rezultat (2026-09-02)

**Hipoteza korisnice**: MLP-ova slabost potiskivanja negativa u crowded familijama (nsLTP/Profilin/PR-10, vidi Failure analysis ispod) možda potiče od "Suspected*" evidence_level parova u treningu (~267/1922, homology/izoforma-izvedeni, NISU "Inferred" pa `training_eligible_pairs()` ih zadržava) — treniranje na njima možda uči model da prejako uopštava "ista familija = pozitivno". Test: umesto brisanja (rizik od data-starvation, presedan iz RRF-5 clean-train eksperimenta da čišćenje tier-a "barely moves anything"), **težinsko obaranje** — Suspected* parovi dobijaju težinu 0.5 u loss-u, ostalo (Strong/Confirmed) 1.0, isti pozitivni/negativni parovi kao baseline. Dodata `positive_weights` opcija u `MLPPairClassifier.fit()` (`ml/pipeline/models/classifiers/mlp.py`) — čisto aditivno, `None` (podrazumevano) matematički identično starom ponašanju.

`test/evaluate_weighted_evidence_mlp_patients_1548.py` — baseline (težina 1.0 svuda) vs weighted (Suspected*=0.5) vs BLAST, leave-one-out na **svih 68 pacijenata** (prva puna regeneracija posle Giuffrida proširenja). best_val_auc gotovo identičan (0.9831 vs 0.9835) — nije problem treninga.

**Rezultat: SUPROTNO od hipoteze.** Weighted varijanta potiskuje negative GORE, ne bolje, u sve tri ciljne familije:

| Familija | Negativi: baseline → weighted | BLAST negativi |
|---|---|---|
| PR-10 | 48.7% → **31.3%** (gore) | 84.8% |
| nsLTP | 23.7% → **19.1%** (gore) | 49.0% |
| Profilin | 34.3% → **32.7%** (blago gore) | 66.9% |

Agregatno (68 pacijenata, 204 trial-a): weighted **značajno gori od baseline-a** (cluster-permutacija p=0.0022, bootstrap CI[-0.0393,-0.0054], oba isključuju nulu) i **gubi značajnu prednost nad BLAST-om** koju baseline ima (CI[-0.0320,+0.0093], uključuje nulu).

**Zaključak**: pridružuje se već postojećem obrascu (hard-negative trening avgust 2026, LayerNorm ablacija) — pokušaji da se ova specifična slabost popravi treniranjem su do sada UVEK dali negativan ili nulti rezultat. Nepotvrđena spekulacija zašto: obaranje težine Suspected* parova efektivno pomera trening masu ka Strong/Confirmed parovima, koji možda nisu ravnomerno raspoređeni baš u regionima gde je regularizacija najpotrebnija — Suspected parovi možda daju koristan regularizacioni signal uprkos slabijoj pojedinačnoj pouzdanosti. **Ne preporučuje se za produkciju.**

---

## Failure analysis: gde MLP(hadamard) greši naspram BLAST-a (57 pacijenata)

**Cilj** — na zahtev korisnice: sistematski analizirati SVIH 176 trial-ova (patient/candidate/organism/protein_family/MLP i BLAST rank-percentil/pobednik), ne samo agregatni MRR — čisto analitički rad, bez treniranja novog modela ili heuristike.

**Šta je urađeno** (`analysis/mlp_blast_crd_failure_analysis_1548.py`) — spojeni MLP/BLAST raw rezultati sa `organism` (`clean_allergens.csv`) i `protein_family` (component-level polje iz `test_cases.json`, 280/280 popunjeno). Kategorije grešaka (preklapaju se): isti-organizam/različita-familija, family crowding (nsLTP/profilin/PR-10), profilin posebno, storage proteini. Za svaku kategoriju/familiju: medijan percentil (ne samo mean — mean se pokazao zaveden duplim pacijentima sa identičnim profilom u istoj koorti).

**Rezultati**

| Familija | Pozitivi (medijan %) | Negativi (medijan %, potiskivanje) |
|---|---|---|
| PR-10 (n=23) | MLP 37.0 vs BLAST 73.9 — **MLP bolji, 12/12 trial-ova jednosmerno** | MLP 48.7 vs BLAST 84.8 — MLP lošiji |
| nsLTP (n=34) | MLP 2.3 vs BLAST 2.7 — **MLP bolji, 26/28 trial-ova** | MLP 23.8 vs BLAST 49.0 — MLP lošiji |
| profilin (n=29) | MLP 5.5 vs BLAST 1.7 — **BLAST bolji na 10/12 tipičnih slučajeva** (mean je bio zaveden sa 2 duplirana Phl p12 slučaja) | MLP 34.3 vs BLAST 66.9 — MLP lošiji |
| Storage proteini (n=32) | — | MLP 67.8 vs BLAST 66.0 — **praktično izjednačeno** (za razliku od crowded familija) |

**Zaključak — 5 empirijskih nalaza**
1. MLP robusno bolji od BLAST-a na pravim pozitivima u PR-10 (jednosmerno, 12/12) i nsLTP (26/28) — najjači nalaz u celom skupu.
2. Profilin je izuzetak: tipičan slučaj (medijan) favorizuje BLAST na pozitivima, suprotno od PR-10/nsLTP.
3. MLP-ova slabost potiskivanja negativa je koncentrisana u crowded familijama (−37.7pp), NE univerzalna — kod storage proteina razlike praktično nema.
4. **Pru p1/Pru p3 je deo prepoznatljivog ali ASIMETRIČNOG obrasca**: od 12 distinktnih isti-organizam/različita-familija parova u stvarnim trial-ovima, 7/12 ide u MLP-ovu štetu, 5/12 (svi mite Der p parovi) u njegovu korist — ali dva apsolutno najveća promašaja u celom 176-trial datasetu (Pru p3→Pru p1: 61.5pp; Phl p5→Phl p12: 59.1pp) su baš u ovoj kategoriji.
5. Family-crowding win-count je tačno 43:43 po trial-u — MLP dobija na senzitivnosti (pozitivi), gubi na specifičnosti (negativi), neto izjednačeno po broju iako su mehanizmi suprotni.

---

## Targeted hard-negative eksperiment — pokušaj popravke, negativan rezultat

**Cilj** — testirati da li ciljano mešanje "isti organizam, različita familija, nije poznat pozitivan par" negativa (umesto čisto uniformnog nasumičnog uzorkovanja) tokom MLP(hadamard) treninga popravlja dijagnostikovanu slabost potiskivanja (npr. Pru p1 kad je Pru p3 poznat pozitivan). **VAŽNO ograničenje korisnice**: nema izmišljanja negativnih parova, nema korišćenja pacijentskih CRD podataka u treningu.

**Šta je urađeno** (`ml/loco_targeted_hardneg_mlp_hadamard_1548.py`) — kandidati mineni ISKLJUČIVO iz `clean_allergens.csv` (`organism` kolona, 1536/1536 popunjena) + `cross_reactive_1548.csv` (`family_1`/`family_2`), zahtevajući DOSLEDNU family labelu za oba proteina (32/477 proteina sa nekonzistentnim stringovima konzervativno isključeno, ne fuzzy-normalizovano). Rezultat: **930 pouzdanih kandidata, 76 organizama, 192 različite family-par kombinacije** — (Pru p1, Pru p3) je tačno 1 od njih, potvrđujući da flagship slučaj nije specijalno tretiran. Testirano 0%/5%/10%/20% zamene negativnog budžeta po foldu, isti 40 LOCO folda, per-fold leakage guard.

**Rezultati**

| Ratio | LOCO MRR delta vs BLAST | Znacajno? | Produkcioni probe: rank(Pru p1 \| query=Pru p3) |
|---|---|---|---|
| 0% (baseline) | −0.0003 | ne | 561 (36.6 percentil) |
| 5% | +0.0025 | ne | 556 (36.2 percentil) |
| 10% | −0.0010 | ne | 347 (22.6 percentil) |
| 20% | +0.0004 | ne | **175 (11.4 percentil)** |

**Zaključak**
- Nema dose-response efekta na agregatnoj LOCO metrici (sve CI uključuju nulu). **Na flagship dijagnostičkom slučaju rezultat je SUPROTAN od željenog** — Pru p1 se pomera KA VRHU (rank 561→175) kako raste udeo ciljanih negativa, dok pravi pozitivi (Cor a8, Ara h9, Jug r3) ostaju stabilni. Ovo NIJE slučaj "nema dovoljno podataka" (930 kandidata je bilo dovoljno, popunjenost 100% cilja na 5%/10%). **Odbačeno za produkciju.** Speculativna, nepotvrđena hipoteza: forsiranje separacije na ovom uskom obrascu možda pomera Hadamard-prostor decision boundary tako da POVEĆA sličnost za ovaj par umesto da je smanji — trebalo bi dalje dijagnostikovati pre nego što se tretira kao objašnjenje.
- **Odluka korisnice**: predložen inference-time veto/rule (organism/family penalizacija pri rangiranju) kao sledeći korak — EKSPLICITNO ODBIJENO. Rule-based patch na predikcije učenog modela slabi naučnu interpretaciju teze (zamagljuje "šta model uči" vs "šta ručno pravilo radi"). Kad training-side popravka ne uspe, ostaje se na dokumentovanoj analizi, ne na patch-u. Veto bi mogao biti implementiran KASNIJE samo kao eksplicitno odvojen appendix/error-mitigation baseline, nikad uduvan u glavni model.

---

## LSE-pooling kao primarni ranker — feasibility check, odustalo se pre punog pokretanja

**Cilj** — MI/hypergraph LSE-pooling je jedini mehanizam u projektu sa pravim LOCO-potvrđenim dobitkom baš na dijagnostikovanom obrascu (nsLTP/Profilin crowding). Pre punog 40-fold LOCO pokretanja kao samostalnog full-pool rankera (nikad ranije testiran ovako — ranije samo protiv 3 ciljne familije), izmereno vreme na 1-2 foldu da se proceni realna cena.

**Šta je urađeno** — direktno merenje umesto pretpostavke. Full-pool scoring (postojeći padded-gather vektorizovani skorer iz `analysis/mi_lse_pooling_1548.py`) ispao je jeftin (~3s jednokratni setup + ~0.09s/upit). **Pravo usko grlo: tau/scale/bias refit po foldu — izmereno end-to-end na medijan-veličine foldu, 289.8s (~4.8 min) za 300 epoha**, zbog nevektorizovane Python petlje koja poziva `torch.logsumexp` posebno po primeru (~5585 primera/fold), ne zbog residue podataka samih.

**Rezultati**
- Projekcija za pun 40-fold LOCO: **~3+ sata**, naspram 87 min za ceo 4-ratio MLP hard-negative eksperiment.

**Zaključak**
- **Korisnička odluka: nije vredno toga trenutno** — pravac je ODLOŽEN (shelved), NE oborен/odbačen kao pogrešan. Ako se ikad revizituje, fit petlju bi trebalo prvo vektorizovati/batch-ovati (isti padded pattern već korišćen za scorer) pre nego što pun LOCO postane dovoljno jeftin da se opravda.

---

## LayerNorm ablacija za MLP(hadamard) — čist, značajan negativan rezultat

**Cilj** — jeftinija alternativa LSE-poolingu: LayerNorm na skrivenim aktivacijama (Linear→LayerNorm→ReLU→Dropout), MEHANIČKI drugačija intervencija od već utvrđene `standardize=False` odluke (ona je globalna z-score standardizacija SIROVIH ulaznih Hadamard feature-a, LayerNorm je per-primer normalizacija SKRIVENIH aktivacija posle prvog linearnog sloja).

**Šta je urađeno** — nov, podrazumevano-isključen `use_layernorm` parametar dodat u `PairMLP`/`MLPPairClassifier` (`ml/pipeline/models/classifiers/mlp.py`) — aditivna izmena, ne menja ponašanje nijednog postojećeg skripta. Pun 40-fold LOCO, oba config-a na identičnim foldovima (`ml/loco_mlp_hadamard_layernorm_ablation_1548.py`).

**Rezultati**

| Config | MRR | Delta vs BLAST | Značajno? |
|---|---|---|---|
| baseline | 0.1259 | +0.0016 | ne |
| +LayerNorm | 0.0804 | **−0.0439** | **DA, značajno GORE** |

Direktno poređenje (isti upiti, upareni bootstrap): LayerNorm vs baseline delta = **−0.0455**, 95% CI [−0.0523,−0.0389] — jasno isključuje nulu.

**Zaključak**
- Baseline broj se poklapa TAČNO sa ranije dokumentovanim rezultatom (0.1259) — potvrđuje da je rerun veran, ne artefakt seed-a. LayerNorm ubrzava konvergenciju (rano zaustavljanje, npr. epoha 21 naspram 86 na probnom foldu) ali ka ZNAČAJNO lošijem rešenju (val AUC 0.69 naspram 0.99 na istom probnom foldu) — verovatno zato što per-primer normalizacija skrivenih aktivacija uništava deo Hadamard-produkt skale koja, kao i kod `standardize=False` odluke, sama nosi signal, iako je mehanički drugačiji sloj. **Odbačeno za produkciju.**
- **Ovim je zatvoren "jeftin arhitektonski trik" pravac za sada** — kombinovano sa neuspelim hard-negative eksperimentom, odbijenim veto-om, i odloženim LSE-pravcem, svaki pokušaj poboljšanja MLP(hadamard)-a u ovoj sesiji je vraćen negativan ili nepraktičan. **Trenutno stanje: plain MLP(hadamard) (bez LayerNorm-a, standardize=False, hidden_dims=[32]) ostaje najbolja potvrđena konfiguracija** — dijagnostikovana slabost potiskivanja negativa u crowded familijama ostaje dokumentovano, nerešeno ograničenje, ne patch-ovano.

---

## ESM-1b naspram ESM-2 backbone — MLP(hadamard), značajan negativan rezultat

**Cilj** — testirati da li NEZAVISNO trenirana proteinska reprezentacija (ESM-1b, 2019, drugačiji training recipe/UniRef filter od ESM-2, ali isti red veličine, 650M) menja nešto za MLP(hadamard) — na eksplicitan zahtev korisnice, SAMO za ovaj model, ne nova RRF fuzija.

**Šta je urađeno** — ESM-1b embeddinzi generisani na klasteru (`hpclab/generate_esm1b_embeddings.py`, mean-pooling, ista konvencija kao glavni `embeddings.pkl`) — čist run, 1535/1535 proteina, 0 NaN, 100% ID preklapanje sa ESM-2 setom. Poređeno preko `ml/loco_esm1b_vs_esm2_mlp_hadamard_1548.py`, isti 40 LOCO folda. **Efikasnost**: prva verzija skripta je nepotrebno RETRENIRALA ESM-2 baseline iznova — korisnica je to odmah primetila; ispravljeno ponovnom upotrebom već postojećeg `output/loco_blast_vs_mlp_hadamard_only_1548_per_query.csv` (BLAST i ESM-2 rezultat su deterministički identični ranije dobijenim, ne treba ih računati dvaput) — runtime pao sa projektovanih ~40-60 min na 11 min.

**Rezultati**

| | MRR | Delta vs BLAST | Značajno? |
|---|---|---|---|
| MLP(hadamard) na ESM-2 | 0.1259 | +0.0016 | ne |
| MLP(hadamard) na ESM-1b | **0.1065** | **−0.0178** | **DA, značajno GORE** |

Direktno poređenje (isti upiti): ESM-1b vs ESM-2 delta = **−0.0195**, 95% CI [−0.0256,−0.0132] — značajno, oba nivoa (pair i study).

**Zaključak**
- Prelazak na ESM-1b NE probija representation ceiling — čini ga gorim, ne izjednačenim. Očekivano s obzirom da je ESM-1b (2019) stariji/slabije treniran model od ESM-2 (2022), ali sada empirijski zatvoreno, ne pretpostavljeno. **Ne koristiti ESM-1b kao backbone.** AlphaFold-bazirani embeddinzi/feature-i eksplicitno odloženi za kasniju sesiju, nisu još probani.

---

## ESM-2 3B naspram ESM-2 650M backbone — MLP(hadamard), fer poredjenje posle uhvaćenog confound-a

**Cilj** — skalirati NAGORE unutar ISTE ESM-2 familije (`facebook/esm2_t36_3B_UR50D`, dim 2560 vs 1280) — principijelnije od prelaska na ESM-1b, budući da arhitektura/training recipe ostaju isti, menja se samo skala.

**Šta je urađeno** — prva generacija 3B embeddinga na klasteru je OOM-ovala (GPU samo 3.94 GiB VRAM, potvrđeno pravom `torch.OutOfMemoryError` porukom — 3B težine same traže ~6GB u FP16; nema zvaničnog ESM-2 checkpoint-a između 650M i 3B, potvrđeno direktno na HF hub-u). Posle veće GPU alokacije, čist run: 1535/1535 proteina, 2560-dim, 0 NaN.

**Prvo poređenje je bilo METODOLOŠKI POGREŠNO, korisnica je to odmah primetila**: cosine (bez treninga) je pokazao 3B i 650M ISTINSKI izjednačene (delta −0.0007, CI uključuje 0) — ali MLP(hadamard) je koristio ISTE hiperparametre tunovane za 1280-dim ulaz (h32, lr=1e-2, standardize=False) na 2560-dim ulazu bez ikakvog retuninga, dajući katastrofalan kolaps (MRR=0.0395, delta −0.0848 vs BLAST). Brz single-split sweep (`analysis/mlp_hadamard_esm2_3b_sensitivity_1548.py`, 7 config-a) je pokazao uzrok: SVAKI `standardize=False` config na netunovanom LR kolabira; `standardize=True` (ili 10x manji LR) to potpuno rešava.

**Rezultati (puna 40-fold LOCO potvrda dva najbolja retunovana kandidata)**

| Config | MRR | Delta vs BLAST | Delta vs 650M |
|---|---|---|---|
| 650M (uspostavljen baseline) | 0.1259 | +0.0016 (n.z.) | — |
| 3B, nefer (bez retuninga) | 0.0395 | **−0.0848 (kolaps)** | — |
| 3B, h32+standardize=True | 0.1131 | **−0.0112, CI[−0.0183,−0.0040] ZNAČAJNO** | **−0.0128, CI[−0.0198,−0.0054] ZNAČAJNO** |
| 3B, h64+standardize=True | 0.1136 | **−0.0107, CI[−0.0185,−0.0027] ZNAČAJNO** | **−0.0123, CI[−0.0191,−0.0051] ZNAČAJNO** |

**Zaključak**
- Retuning je potpuno rešio kolaps (0.0395→0.113), ali fer poređenje i dalje pokazuje da je 3B blago ali GENUINSKI značajno lošiji od BLAST-a I od 650M-a, dosledno kroz oba retunovana config-a (ne slučajnost jedne konfiguracije). **Skaliranje ESM-2 backbone-a nagore NE probija representation ceiling** — isti zaključak kao ESM-1b, ali sada potvrđen bez confound-a, pravim retuning-om, ne artefaktom neuskladjene arhitekture. Zajedno sa cosine nalazom (potpuno izjednačeno), ovo zatvara "veći embedding backbone" pravac čisto: ni promena ESM familije ni skaliranje unutar iste familije ne pomaže — plafon izgleda kao prava osobina onoga što ove sequence-only reprezentacije mogu da uhvate za ovaj zadatak, ne popravljiv izborom modela. **Ne koristiti ni ESM-2 3B kao backbone.**

**Dodatak — isti retunovan 3B config testiran na 57 pacijenata** (`test/evaluate_mlp3b_patients_1548.py`, ista metodologija kao BLAST-vs-MLP(hadamard) pacijentski test): **MLP-3B značajno bolji od BLAST-a, sva 3 testa, oba podskupa** (mean diff +0.015 do +0.018, p<0.01 svuda) — čak nešto jače od 650M rezultata. Ali **MLP-3B naspram MLP-650M je mešovito, VEĆINOM neznačajno** (samo jedan od šest test/podskup kombinacija značajan) — 3B NIJE pouzdano bolji od 650M na pacijentima. Zaključak: model koji je gori od BLAST-a na gold-LOCO metrici i dalje pokazuje pravu prednost nad BLAST-om na pravim pacijentima — isti interno-vs-klinički gap kao za 650M, sada reprodukovan sa drugim modelom, ojačava taj nalaz kao pravi fenomen. Nije razlog za prelazak na 3B u produkciji — nema jasnu prednost nad 650M, a mnogo je skuplji.

## Ablaciona studija: MLP(hadamard) 650M vs 3B — konsolidacija na pacijentima (2026-09-02)

**Odluka o obimu (korisnica, 2026-09-02): cela ablaciona studija za MLP(hadamard)/MLP(richconcat) porodicu modela ide ISKLJUČIVO protiv 57-pacijentskog real-world suite-a, ne protiv gold-dataset LOCO-a.** Ovo je svesno odstupanje od uobičajene discipline projekta ("LOCO prvo, pacijenti tek za LOCO-potvrđene kandidate") — opravdano već dvaput dokumentovanim interno-vs-klinički gap-om (650M i 3B su oba gora/tie na gold-LOCO ali oba pobeđuju BLAST na pacijentima; ispod, richconcat pokazuje suprotan smer istog fenomena) — LOCO rangiranje kandidata se pokazalo kao nepouzdan prediktor pacijentskog ishoda za ovu porodicu modela, pa se dalja selekcija radi direktno na metrici koja je bitna za tezu.

**Metodologija** — identična već uspostavljenom upareno-testiranom protokolu (`test/paired_test_mlp_vs_blast_1548.py` stil): za svaki kandidat, leave-one-out rangiranje na svih 57 pacijenata (176 upita / 54 pacijenta sa ≥1 pozitivnim i negativnim skrivenim proteinom), pa 3 uparena testa na identičnim (patient_id, hidden_protein) parovima — (1) patient-level Wilcoxon signed-rank, (2) cluster-permutacija (permutuje oznaku modela unutar pacijenta), (3) patient-level bootstrap CI (resample po pacijentu) — na "svi upiti" i "samo hard (full_text_verified)" podskupu. BLAST/650M-baseline rezultati REciklirani iz postojećih raw JSON fajlova gde god su isti trials, ne računati ponovo.

### Konsolidovana tabela (samo već izvršeni pacijentski testovi, bez novog treninga)

| Kandidat | Poređenje | Svi upiti (Wilcoxon / cluster-perm / bootstrap) | Hard subset (Wilcoxon / cluster-perm / bootstrap) |
|---|---|---|---|
| **MLP(hadamard)-650M** (baseline, standardize=False, h32) | vs BLAST | p=0.0116 ✓ / p=0.0304 ✓ / CI[+0.0019,+0.0237] ✓ — sve pozitivno, sve značajno | p=0.0042 ✓ / p=0.0080 ✓ / CI[+0.0043,+0.0295] ✓ — sve pozitivno, sve značajno |
| **MLP(hadamard)-3B** (retuned, standardize=True, h32) | vs BLAST | p=0.0001 ✓ / p=0.0030 ✓ / CI[+0.0076,+0.0299] ✓ — sve pozitivno, sve značajno | p=0.0005 ✓ / p=0.0061 ✓ / CI[+0.0066,+0.0311] ✓ — sve pozitivno, sve značajno |
| **MLP(hadamard)-3B** | vs MLP(hadamard)-650M | p=0.0046 ✓(+0.0015) / p=0.3413 ✗ / CI[−0.0074,+0.0192] ✗ — mešovito, 3B NIJE pouzdano bolji | p=0.1066 ✗ / p=0.7276 ✗ / CI[−0.0135,+0.0166] ✗ — sve neznačajno, praktično izjednačeno |
| **MLP(richconcat)-3B, preL2=True** | vs BLAST | p=0.1373 ✗ / p=0.0751 ✗ / CI[−0.0743,−0.0051] ✓(negativno) | p=0.7704 ✗ / p=0.0111 ✓(negativno) / CI[−0.0912,−0.0101] ✓(negativno) |
| **MLP(richconcat)-3B, preL2=True** | vs MLP(hadamard)-650M | p=0.7598 ✗ / p=0.0082 ✓(negativno) / CI[−0.0863,−0.0146] ✓(negativno) | p=0.4205 ✗ / p=0.0006 ✓(negativno) / CI[−0.1069,−0.0235] ✓(negativno) |
| **MLP(richconcat)-3B, preL2=False** | vs BLAST | p=0.1026 ✗ / p=0.0780 ✗ / CI[−0.0663,−0.0016] ✓(negativno) | p=0.6573 ✗ / p=0.0018 ✓(negativno) / CI[−0.0828,−0.0100] ✓(negativno) |
| **MLP(richconcat)-3B, preL2=False** | vs MLP(hadamard)-650M | p=0.4306 ✗ / p=0.0137 ✓(negativno) / CI[−0.0781,−0.0105] ✓(negativno) | p=0.0411 ✓(negativno) / p=0.0002 ✓(negativno) / CI[−0.0980,−0.0236] ✓(negativno) |

Izvori: `output/paired_test_mlp_vs_blast_1548_summary.txt`, `output/evaluate_mlp3b_patients_1548_summary.txt`, `output/evaluate_mlp3b_richconcat_patients_1548_summary.txt`.

### Zaključci (bez novog treninga)

1. **Plain hadamard je bolji izbor encoding-a od richconcat-a na OBA nivoa** — ovo je nova informacija, richconcat nije bio pre ovoga uporedjen sa plain hadamard-om na pacijentima. Richconcat-3B je konzistentno GORI i od BLAST-a i od MLP(hadamard)-650M, sa najjačim signalom baš na hard (najpouzdanije verifikovanom) podskupu — obrnut smer od gold-LOCO nalaza, gde je richconcat brojčano BLIŽI BLAST-u (deltа −0.003/−0.004) nego plain hadamard-3B (delta −0.011). **Ovo je treći dokumentovan slučaj gde gold-LOCO rangiranje kandidata direktno pogrešno predviđa pacijentski ishod** — dodatna potvrda odluke da se ablacija vodi na pacijentima.
2. **3B ne daje pouzdan upgrade nad 650M ni sa plain hadamard-om** (mešovito/tie) **ni sa richconcat-om** (isključivo gori) — trenutno nema nijedne konfiguracije u ovoj porodici gde 3B robustno pobeđuje 650M na pacijentima.
3. **Trenutni najbolji kandidat ostaje plain MLP(hadamard)-650M** — jedini sa svih 6 test-podskup kombinacija značajnim i u pravom smeru; 3B plain hadamard je blizu drugi (isto tako sve značajno vs BLAST, ali bez dokazane prednosti nad 650M, i uz mnogo veći trošak).

### Faktori BEZ pacijentskih podataka (postoji samo gold-LOCO/single-split rezultat — ne mogu ući u gornju tabelu bez novog treninga)

| Faktor | Šta postoji | Nedostaje |
|---|---|---|
| richconcat na **650M** | ništa — richconcat testiran samo na 3B | trening + pacijentski test na 650M, za fer poređenje encoding-a nezavisno od backbone-a |
| absdiff encoding | samo gold-LOCO (značajno gori od cosine, svaka arhitektura) | pacijentski test |
| LayerNorm ablacija | samo gold-LOCO (značajno gori) | pacijentski test |
| Targeted hard-negative trening | samo gold-LOCO (bez efekta) + direktna proba na Pru p 3 (gore) | pacijentski test |
| Naučena projekcija (256/512, `projected_mlp.py`) | samo single-split screening (negativno) | LOCO ili direktno pacijentski test |
| PCA redukcija (256/512) | samo single-split screening (negativno) | pacijentski test |
| ESM-1b backbone | samo gold-LOCO (značajno gori) | pacijentski test |
| Regularizacija (dropout/weight_decay/l2_lambda sweep) | nigde sistematski varirano | i trening i test |
| Clean-trained (`training_eligible_pairs()`, Inferred isključen) — 3B varijanta | urađeno za 650M (kao deo RRF-6), NE za samostalni 3B single-signal | trening + pacijentski test |
