# Ablacioni test: MLP(hadamard) na 57 pacijenata

**Obim**: tačno originalnih **57 pacijenata** (`test/test_cases.json`, literature-verified suite, isti skup koji `rad/RAD.md` citira kao headline n=57 — bez 11 dodatih Giuffrida et al. 2014 pacijenata). Sve tabele ispod koriste identičan skup od **176 proba / 54 pacijenta sa validnim uparenim trial-ovima**, definisan kroz `test/evaluation_results_raw_blastonly.json` kao autoritativni skup ključeva (patient_id, hidden_protein).

**Metodologija** (ista svuda): leave-one-patient-out, single-signal ranker (RRF-K suma preko poznatih pozitiva istog pacijenta, K=20), tri uparena testa na identičnim upitima — (1) patient-level Wilcoxon signed-rank, (2) cluster-permutacija (permutuje oznaku modela unutar pacijenta, N=10000), (3) patient-level bootstrap CI (resample po pacijentu, N=10000). Izveštava se na "svi upiti" i "hard/full_text_verified" podskupu gde je primenljivo.

---

## Deo C — plan: komponentna ablacija arhitekture (leave-one-component-out)

**Cilj**: za razliku od Dela A/B (koji variraju TRENING PODATKE — evidence tier, encoding, backbone), ovo testira koja ARHITEKTONSKA komponenta MLP(hadamard) modela najviše doprinosi rezultatu — svaka komponenta se **nezavisno** zamenjuje trivijalnijom alternativom (ostale komponente ostaju nepromenjene), ne kumulativno, da se efekat ne zamagli.

**Obim**: **cela studija ide na 57-pacijentski kanonicni skup** (isti kao Deo A/B), ne gold LOCO — odluka korisnice, 2026-09-02, konzistentno sa ostatkom ovog fajla.

| Komponenta | Pun model (baseline) | Trivijalna zamena | Status |
|---|---|---|---|
| **C.1 — Reprezentacija proteina** | ESM-2 embedding (650M) | aminokiselinski sastav (20-dim frekvencijski vektor, bez proteinskog jezičkog modela) | TODO — nova, nikad testirano |
| **C.2 — Kombinovanje para** | Hadamard produkt ($u\odot v$) | apsolutna razlika ($\lvert u-v\rvert$) | TODO — trening/eval na pacijentima nikad rađen (LOCO postoji: absdiff dosledno gori od cosine) |
| **C.3 — Model** | MLP (nelinearan, skriveni slojevi) | logistička regresija (linearan model, isti Hadamard ulaz, bez skrivenih slojeva) | TODO — nova, nikad testirano |
| *(krajnja trivijalna tačka, referenca)* | — | cosine sličnost (bez ičega naučenog) | već postoji (Deo A.1) |

Svaka od tri linije poredi se protiv istog produkcionog baseline-a (MLP-hadamard-650M, pooled) na identičnom 176-trial/54-pacijent skupu, istom uparenom metodologijom (3 testa) kao Deo A. Cilj: videti koja zamena najviše obara performanse — to je komponenta koja najviše doprinosi.

### Deo C — rezultati

`test/evaluate_component_ablation_patients_1548.py`. `best_val_auc` (sopstveni trening/validacioni skup, ne pacijenti): baseline=0.9831, C.1=0.7333, C.2=0.9945, C.3=0.9827.

| Zamenjena komponenta | Δ MRR (bootstrap 95% CI) | Wilcoxon p | Cluster-perm. p | Značajno? | Uticaj |
|---|---|---:|---:|---|---|
| **C.1: ESM-2 → AA-sastav (20-dim)** | **-0.1588 [-0.2746,-0.0593]** | 0.0101 | **0.0000** | DA, sva 3 testa | **OGROMAN** |
| **C.2: Hadamard → absdiff** | -0.0552 [-0.0994,-0.0163] | 0.7207 | **0.0128** | DA, 2/3 testa (Wilcoxon granično n.z.) | umeren |
| **C.3: MLP → linearni model (logist. regresija)** | -0.0088 [-0.0200,+0.0013] | 0.6818 | 0.1096 | NE, nijedan test | zanemarljiv |

**Zaključak**: doprinos komponenti je izrazito neravnomeran. **ESM-2 reprezentacija nosi skoro sav signal** — zamena trivijalnim aminokiselinskim sastavom (bez rasporeda/motiva) uništava većinu performansi (Δ≈-0.16 MRR na skali gde je ceo model ~0.12-0.13 MRR), i to se vidi već na sopstvenom trening skupu (val AUC pada sa 0.98 na 0.73 — model se bori da nauči i sopstvene trening podatke sa ovako siromašnom reprezentacijom, nije samo problem generalizacije). Izbor kombinovanja (Hadamard vs absdiff) doprinosi umereno, potvrđuje već poznat LOCO nalaz sada i na pacijentima. **Nelinearnost MLP-a doprinosi zanemarljivo** — linearni model na istom Hadamard ulazu radi statistički identično. Redosled važnosti: **reprezentacija ≫ kombinovanje > nelinearnost modela** — direktno podržava tvrdnju iz poglavlja 1.4 rada da doprinos nije u arhitekturnoj složenosti, već u izboru i korišćenju reprezentacije.

---

## Deo A — faktori sa već postojećim podacima

### A.1 Pojedinačni signali (referenca, bez ponovnog računanja)

| Poređenje | Podskup | Wilcoxon p | Cluster-perm. p | Bootstrap 95% CI |
|---|---|---:|---:|---|
| MLP(hadamard)-650M vs BLAST | svi | **0.0116** | **0.0304** | **[+0.0019,+0.0237]** |
| MLP(hadamard)-650M vs BLAST | hard | **0.0042** | **0.0080** | **[+0.0043,+0.0295]** |
| MLP(hadamard)-3B vs BLAST | svi | **0.0001** | **0.0030** | **[+0.0076,+0.0299]** |
| MLP(hadamard)-3B vs BLAST | hard | **0.0005** | **0.0061** | **[+0.0066,+0.0311]** |
| MLP(hadamard)-3B vs MLP-650M | svi | 0.0046* | 0.3413 | [-0.0074,+0.0192] |
| MLP(hadamard)-3B vs MLP-650M | hard | 0.1066 | 0.7276 | [-0.0135,+0.0166] |
| Cosine vs BLAST | svi | 0.7994 | **0.0205** | **[-0.0921,-0.0121]** |
| Cosine vs BLAST | hard | 0.4692 | **0.0057** | **[-0.1105,-0.0166]** |
| Cosine vs MLP-650M | svi | **0.0172** | **0.0026** | **[-0.1042,-0.0214]** |
| Cosine vs MLP-650M | hard | **0.0009** | **0.0002** | **[-0.1248,-0.0301]** |
| MLP(richconcat)-3B (preL2=True) vs BLAST | svi | 0.1373 | 0.0751 | **[-0.0743,-0.0051]** |
| MLP(richconcat)-3B (preL2=True) vs BLAST | hard | 0.7704 | **0.0111** | **[-0.0912,-0.0101]** |
| MLP(richconcat)-3B (preL2=True) vs MLP-650M | svi | 0.7598 | **0.0082** | **[-0.0863,-0.0146]** |
| MLP(richconcat)-3B (preL2=True) vs MLP-650M | hard | **0.0411** | **0.0006** | **[-0.1069,-0.0235]** |

*\*jedini značajan test od tri; smatra se neznačajnim rezultatom u celini (2/3 testa n.z.).*

**Zaključak A.1**: rangiranje pojedinačnih signala na pacijentima je **MLP(hadamard)-650M ≈ MLP(hadamard)-3B > BLAST > cosine ≫ MLP(richconcat)-3B**. 3B ne daje pouzdan upgrade nad 650M. Richconcat encoding je jedini kandidat koji gubi i od BLAST-a i od baseline-a (suprotno od gold-LOCO nalaza gde je bio bliži tie-u — videti README).

### A.2 Trening-podaci varijante — preračunato na 57 pacijenata

Dve varijante iz tekuće diskusije o popravci negative-suppression slabosti (README "Weighted-evidence trening", "Strict-evidence"), originalno računate na 68 pacijenata (posle Giuffrida proširenja) — ovde **filtrirano na tačan 57-pacijentski/176-trial skup** (nema ponovnog treninga, isti trenirani modeli, samo restrikcija test-skupa).

- **weighted**: `training_eligible_pairs()` (Inferred isključen), + "Suspected*" evidence tier dobija težinu 0.5 u loss-u
- **strict**: trening SAMO na "Strong evidence*" + "Confirmed*" tier-ovima (511 parova, ~62% manje od baseline-a)
- **baseline**: produkcioni MLP(hadamard)-650M, `training_eligible_pairs()`, sve težine 1.0

| Poređenje | Wilcoxon p | Cluster-perm. p | Bootstrap 95% CI |
|---|---:|---:|---|
| weighted vs baseline | 0.2132 | **0.0011** | **[-0.0447,-0.0064]** |
| weighted vs BLAST | 0.1804 | 0.3454 | [-0.0366,+0.0109] |
| strict vs baseline | 0.0553 | 0.8545 | [-0.0063,+0.0154] |
| strict vs BLAST | **0.0012** | **0.0310** | **[+0.0022,+0.0295]** |

**Zaključak A.2**: **weighted je značajno gori od baseline-a** i gubi značajnu prednost nad BLAST-om — potvrđuje raniji nalaz na 68 pacijenata, sad i na tačnom 57-headline skupu. **strict nije značajno različit od baseline-a ukupno, ali ZADRŽAVA (čak blago jača) značajnu prednost nad BLAST-om** (p=0.0012/0.0310/CI iznad nule) — konzistentno sa nalazom na 68 pacijenata.

### A.3 Per-familija raspad (crowded familije — cilj intervencije)

Medijan percentil (pozitivi: manje=bolje; negativi: veće=bolje potiskivanje). Napomena: nijedna Giuffrida proba nije u ove tri familije, pa su brojevi identični onima na 68-pacijentskom skupu.

| Familija | Pozitivi: baseline→weighted→strict | Negativi: baseline→weighted→strict | BLAST negativi |
|---|---|---|---|
| PR-10 (n=23) | 37.0→23.8→24.4 | 48.7→31.3→**24.4 (gore)** | 84.8 |
| nsLTP (n=34) | 2.3→2.2→2.3 | 23.7→19.1→**40.2 (bolje)** | 49.0 |
| Profilin (n=29) | 5.5→6.4→5.5 | 34.3→32.7→**66.2 (skoro = BLAST!)** | 66.9 |

| Poređenje (SAMO crowded, n=86/26 pac.) | Wilcoxon p | Cluster-perm. p | Bootstrap 95% CI |
|---|---:|---:|---|
| weighted vs baseline | 0.0317* | 0.6212 | [-0.0007,+0.0004] |
| strict vs baseline | 0.0861 | **0.0184** | **[+0.0010,+0.0047]** |

*\*značajan ali zanemarljive veličine (+0.0001); ostala 2 testa nisu značajna.*

**Zaključak A.3**: **strict je jedina varijanta sa robustnim, statistički značajnim poboljšanjem baš u crowded familijama** (2/3 testa značajna, pozitivan smer) — nošeno gotovo potpunim popravljanjem profilina i umerenim poboljšanjem nsLTP-a, uprkos pogoršanju kod PR-10. Weighted ne pokazuje robustno poboljšanje nigde.

---

## Deo A — ukupan zaključak

| Kandidat | vs Baseline (ukupno) | vs BLAST (ukupno) | Crowded familije |
|---|---|---|---|
| **baseline (produkcioni)** | — | značajno bolji | referenca |
| MLP-3B | ≈ tie | značajno bolji | nije testirano posebno |
| cosine | značajno gori | značajno gori | nije testirano posebno |
| richconcat-3B | značajno gori | značajno gori/n.z. | nije testirano posebno |
| weighted (Suspected=0.5) | **značajno gori** | gubi značajnost | bez robustnog efekta |
| **strict (Strong+Confirmed)** | bez razlike | **zadržava značajnost** | **jedini robustan dobitak** |

**Trenutno najbolji kandidat ostaje baseline MLP(hadamard)-650M**, sa **strict-evidence trening kao jedinim ozbiljnim izazivačem** — ne pobeđuje baseline ukupno, ali popravlja baš dijagnostikovanu slabost (negative-suppression u nsLTP/Profilin) bez merljive štete, uz zadržanu prednost nad BLAST-om. Vredelo bi dalje istražiti zašto se PR-10 ponaša suprotno (Deo B).

---

## Deo B — faktori bez pacijentskih podataka (TODO)

*Nijedan od ovih nije još testiran na pacijentima — samo LOCO/single-split screening postoji. Popuniti posle dogovora o prioritetu.*

- [ ] richconcat encoding na **650M** backbone-u (do sad testiran samo na 3B)
- [ ] absdiff encoding
- [ ] LayerNorm ablacija
- [ ] Targeted hard-negative trening
- [ ] ESM-1b backbone
- [ ] Regularizacija (dropout / weight_decay / l2_lambda sweep)
