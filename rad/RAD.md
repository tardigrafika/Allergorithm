---
bibliography: references.bib
---

# Predikcija unakrsne reaktivnosti proteinskih alergena korišćenjem proteinskih jezičkih modela i mašinskog učenja

Lana Lejić            
mentor: Stefan Nožinić

## Apstrakt

kao nešto pametno

## 1. Uvod

### 1.1 Motivacija

Unakrsna alergijska reaktivnost je fenomen u kojem IgE antitela senzibilisanog organizma prepoznaju proteine različitih, taksonomski često nepovezanih alergenih izvora. Proteinska sličnost je jedan od faktora koji doprinose ovom fenomenu, ali sama sekvencijalna sličnost nije dovoljna da ga u potpunosti objasni: reaktivnost zavisi od kombinacije osobina proteinske sekvence, konzerviranih strukturnih elemenata, lokalne dostupnosti epitopa i drugih molekularnih karakteristika. Zbog toga alergeni s relativno niskom sekvencijalnom sličnošću mogu pokazivati unakrsnu reaktivnost (npr. panalergeni poput profilina), dok visoka sličnost sama po sebi nije dovoljan dokaz zajedničke reaktivnosti.

Razvoj proteinskih jezičkih modela (protein language models) otvorio je mogućnost da se proteini predstave kao vektori u naučenom višedimenzionalnom prostoru, umesto isključivo kroz mere poravnanja sekvenci. Takve reprezentacije potencijalno opisuju obrasce koji nisu direktno vidljivi kroz klasične mere sličnosti, ali pitanje da li one zaista nose korisnu, generalizabilnu informaciju za predikciju unakrsne reaktivnosti — a ne artefakt konkretnog skupa podataka ili načina njegove podele — nije trivijalno i zahteva sistematsku, statistički rigoroznu proveru.

### 1.2 Formulacija problema — dva odvojena zadatka

Rad razmatra dva srodna, ali metodološki različita zadatka rangiranja, koja se u literaturi i u praksi često ne razdvajaju eksplicitno:

1. **Rangiranje unakrsno reaktivnih parova alergena (pair-level ranking).** Za dati alergen kao upit, zadatak je rangirati sve ostale proteine u referentnom skupu prema verovatnoći da čine unakrsno reaktivan par s upitom. Ovaj zadatak se evaluira direktno nad kurirano-verifikovanim skupom poznatih parova (poglavlje 2.1) i koristi se za treniranje i selekciju modela.
2. **Rangiranje kandidata za konkretnog pacijenta (patient-level ranking).** Za pacijenta sa jednim ili više već poznatih (pozitivnih i/ili negativnih) alergijskih nalaza, zadatak je rangirati preostale alergene iz istog referentnog skupa prema prioritetu za dalje testiranje. Signal za ovaj zadatak dobija se agregacijom pair-level rangova preko svih poznatih pozitivnih nalaza istog pacijenta (poglavlje 2.3, protokol evaluacije u poglavlju 3).

Razlikovanje ova dva zadatka je centralno za rad iz dva razloga. Prvo, prvi zadatak se evaluira nad skupom čija je konstrukcija (izvor dokaza, familijska struktura, gustina grafa) poznata i kontrolisana, dok se drugi zadatak evaluira nad nezavisnim, spolja pristiglim kliničkim slučajevima čija distribucija ne mora odgovarati distribuciji trening skupa. Drugo, kao što će pokazati poglavlje 4, model koji na prvom zadatku ne pokazuje prednost nad jednostavnijim baznim metodama može na drugom zadatku pokazati statistički značajnu prednost — nalaz koji bi ostao neprimećen da se ova dva zadatka ne razdvoje eksplicitno.

### 1.3 Istraživačka pitanja i hipoteze

Rad testira četiri konkretne, unapred formulisane hipoteze, umesto da izveštava rezultate niza pojedinačno motivisanih eksperimenata:

- **H1 (embedding signal):** Reprezentacije proteina dobijene proteinskim jezičkim modelom (ESM-2) nose diskriminativan signal za unakrsnu reaktivnost koji nije svodljiv na jednostavnu sekvencijalnu sličnost (BLAST).
- **H2 (naučena kombinacija embeddinga):** Nadgledani model koji uči nelinearnu kombinaciju para embeddinga izvlači više diskriminativnog signala iz iste reprezentacije nego fiksna mera sličnosti (cosine) nad istom reprezentacijom.
- **H3 (fuzija nezavisnih signala):** Kombinovanje više nezavisnih izvora sličnosti (sekvencijalne, strukturne, embedding-bazirane, topologije poznatog grafa reaktivnosti) poboljšava rangiranje u odnosu na bilo koji pojedinačni signal, pri čemu je veličina i pouzdanost ovog poboljšanja osetljiva na (ne)zavisnost primera u trening skupu.
- **H4 (generalizacija na klinički kontekst):** Relativan poredak metoda dobijen validacijom nad kuriranim skupom podataka (zadatak 1) predstavlja pouzdan prediktor relativnog poretka istih metoda na nezavisnim, stvarnim pacijentskim slučajevima (zadatak 2).

Rezultati u poglavlju 4 pokazuju da H1 i H2 važe u ograničenom, precizno definisanom obliku, da H3 važi delimično (deo dobitka fuzije ne opstaje pod strožom, study-level proverom nezavisnosti), i da **H4 ne važi** — ovo poslednje predstavlja centralni empirijski nalaz rada.

### 1.4 Naučni doprinos

Doprinos rada nije nov algoritam ili arhitektura, već **sistematska, statistički kontrolisana evaluacija postojećih izvora signala** (sekvencijalnog, strukturnog, embedding-baziranog i topološkog) za zadatak predikcije unakrsne reaktivnosti, sprovedena tako da se prividna poboljšanja koja ne opstaju pod strožom validacijom eksplicitno identifikuju i izveštavaju kao takva, umesto da se prećute. Poseban doprinos predstavlja direktno poređenje performansi na kuriranom skupu podataka i na nezavisnim pacijentskim slučajevima, koje pokazuje da ova dva konteksta evaluacije mogu dati suprotstavljene zaključke o istom modelu (poglavlje 4.5, 4.6).

## 2. Metodologija

### 2.1 Konstrukcija skupa podataka

Skup podataka konstruisan je objedinjavanjem informacija iz više javno dostupnih baza proteinskih alergena (AllergenOnline, WHO/IUIS Allergen Nomenclature, SDAP) i objavljenih naučnih radova koji opisuju eksperimentalno potvrđene ili pretpostavljene slučajeve unakrsne alergijske reaktivnosti. Za svaki par zabeležen je izvor dokaza, tip potvrde i proteinska familija kada je bila dostupna.

Referentni skup kandidata za rangiranje (poglavlje 1.2) obuhvata **1.536 proteinskih alergena**. Nad njima je definisano **1.922 poznata para** unakrsne reaktivnosti, koji pokrivaju 477 pojedinačnih alergena iz **74 proteinske familije**, izvedenih iz **317 nezavisnih literaturnih izvora**. Neravnomerna zastupljenost izvora — mali broj radova opisuje veliki broj parova (npr. populacione kohortne studije) — direktno motiviše potrebu za study-level bootstrap analizom (poglavlje 3).

Svaki par klasifikovan je prema pouzdanosti dokaza u jednu od četiri kategorije, sa brojem parova po kategoriji:

| Nivo dokaza | Opis | Broj parova (%) |
|---|---|---:|
| Confirmed | Direktan eksperimentalni dokaz (npr. IgE inhibicioni test) | 138 (7,2%) |
| Strong | Više nezavisnih studija ili velike kliničke kohorte | 377 (19,6%) |
| Suspected | Literatura ukazuje na moguću reaktivnost, dokaz nedovoljan za potvrdu | 277 (14,4%) |
| Inferred | Izvedeno iz homologije/pripadnosti istoj familiji, bez direktnog testa konkretnog para | 1.093 (56,9%) |

Parovi iz kategorije **Inferred** — više od polovine skupa — nisu korišćeni za treniranje nadgledanih modela, ali su zadržani kao evaluacioni ciljevi (poglavlje 3). Ova odluka odvaja pitanje "da li model dobro rangira i manje pouzdane parove" (evaluacija) od pitanja "da li model treba da uči iz njih kao iz pouzdanog signala" (trening).

### 2.2 Problem negativnih i nepoznatih parova

Gold skup sadrži isključivo **pozitivne** primere — parove za koje postoji literaturni dokaz reaktivnosti. Odsustvo para iz skupa **ne znači potvrđenu odsutnost** unakrsne reaktivnosti, već samo odsustvo objavljenog dokaza; zadatak je time strukturno Positive-Unlabeled (PU), a ne standardna binarna klasifikacija. Negativni primeri korišćeni za treniranje generisani su nasumičnim uzorkovanjem parova van gold skupa, pod pretpostavkom da je verovatnoća da nasumično izabran par bude neotkriven pravi pozitiv niska.

Ova pretpostavka je proverena i za jedan konkretan slučaj eksplicitno odbačena: ciljano uzorkovanje "teških" negativa unutar iste proteinske familije nije korišćeno, jer su upravo neoznačeni parovi unutar familije mesto gde je najverovatnije da se kriju neotkriveni pravi pozitivi — isti razlog zbog kog je i sam gold skup rastao dodavanjem novih, ranije neobuhvaćenih parova unutar poznatih familija. Nasumično uzorkovanje iz celog pool-a je zbog toga zadržano kao podrazumevana, konzervativnija strategija u svim eksperimentima treniranja opisanim u ovom radu.

### 2.3 Formalizacija zadataka i metrike

Za oba zadatka iz poglavlja 1.2 koristi se ista osnovna operacija: rangiranje kandidata prema skalarnom skoru. Kod **zadatka 1** (pair-level), skor kandidata $c$ za upit $q$ direktno je izlaz posmatranog signala — cosine sličnost, BLAST identitet, ili verovatnoća nadgledanog klasifikatora. Kod **zadatka 2** (patient-level), skor kandidata $c$ za pacijenta sa poznatim pozitivnim nalazima $P=\{p_1,\dots,p_n\}$ dobija se sabiranjem recipročnih rangova kandidata prema svakom poznatom pozitivu, istim funkcionalnim oblikom kao RRF fuzija signala (poglavlje 2.4):

$$
\mathrm{score}(c \mid P) = \sum_{p \in P} \frac{1}{K + r_p(c)},
$$

gde je $r_p(c)$ rang kandidata $c$ u listi dobijenoj upitom $p$. Ovim se poznati pozitivi jednog pacijenta tretiraju kao više nezavisnih upita čiji se doprinosi sabiraju — isti mehanizam koji poglavlje 2.4 koristi za fuziju više *signala* ovde se koristi za fuziju više *upita*.

Performanse oba zadatka mere se pomoću **Mean Reciprocal Rank (MRR)** i **Hits@K** ($K\in\{1,5,10\}$) — udela upita kod kojih se tačan kandidat nalazi među $K$ najbolje rangiranih.

### 2.4 Reciprocal Rank Fusion (RRF)

Pojedinačni izvori sličnosti između proteinskih alergena opisuju različite aspekte njihove potencijalne unakrsne reaktivnosti. Sličnost embedding reprezentacija dobijenih proteinskim jezičkim modelom opisuje globalnu sličnost proteinskih sekvenci u naučenom reprezentacionom prostoru, BLAST meri lokalnu sekvencijalnu homologiju zasnovanu na poravnanju sekvenci, dok Foldseek TM-score procenjuje sličnost trodimenzionalne strukture proteina. Nijedan od ovih signala pojedinačno ne predstavlja potpun opis unakrsne reaktivnosti, zbog čega je korišćena metoda fuzije rangova.

Za kombinovanje nezavisnih rang-lista korišćen je algoritam **Reciprocal Rank Fusion (RRF)**. Za svaki izvor sličnosti formira se rang svih kandidata, a konačan skor kandidata dobija se sabiranjem recipročnih vrednosti njihovih rangova:

$$
RRF(d)=\sum_{i=1}^{m}\frac{1}{K+r_i(d)}
$$

gde je $r_i(d)$ rang kandidata $d$ prema $i$-tom signalu, $m$ broj korišćenih signala, a $K$ konstanta koja umanjuje uticaj veoma visokih rangova.

U ovom radu korišćena su tri osnovna signala:

- cosine sličnost ESM-2 embedding reprezentacija,
- BLAST skor sekvencijalnog poravnanja,
- Foldseek TM-score strukturne sličnosti.

Konstanta $K$ nije odabrana proizvoljno. Testirano je više vrednosti parametra na trening komponentama LOCO validacije, pri čemu je izabrana vrednost koja je pokazala najstabilnije performanse na nezavisnim komponentama. U svim narednim eksperimentima korišćena je ista vrednost parametra kako bi evaluacija ostala konzistentna.

Prednost RRF algoritma je što kombinuje rangove umesto sirovih skorova, pa nije potrebno normalizovati vrednosti različitih metoda niti pretpostaviti da su njihovi skorovi međusobno uporedivi. Time se omogućava spajanje heterogenih izvora informacija bez dodatnog treniranja modela.

#### 2.4.1 Graph propagation kao dodatni signal

Pored tri osnovna signala, testirana je i proširena varijanta RRF-a u kojoj je dodat četvrti signal zasnovan na poznatim vezama u grafu unakrsne reaktivnosti.

Za dati upit, ako je poznato da je alergen već unakrsno reaktivan sa jednim ili više drugih alergena, kandidat dobija dodatni skor ukoliko je povezan sa tim susedima u grafu poznatih interakcija. Ovaj signal predstavlja propagaciju informacija kroz mrežu poznatih unakrsnih reaktivnosti i koristi isključivo veze dostupne u trening delu validacije.

Da bi se izbeglo curenje informacija, ovaj signal nije evaluiran u LOCO protokolu, jer u tom slučaju test komponenta nema nijednog poznatog suseda u trening grafu. Umesto toga korišćena je **leave-one-edge-out** evaluacija, pri kojoj se za svaki poznati par privremeno uklanja samo testirana veza, dok ostale veze istog alergena ostaju dostupne modelu. Ovakav protokol odgovara realnoj kliničkoj situaciji u kojoj je za pacijenta poznata jedna ili više potvrđenih alergija, a cilj je predvideti dodatne potencijalno unakrsno reaktivne alergene.

### 2.5 Modeli mašinskog učenja nad proteinskim embedding reprezentacijama

Pored metoda zasnovanih na direktnom rangiranju proteinske sličnosti, u radu su ispitani i nadgledani modeli mašinskog učenja čiji je cilj procena verovatnoće da dva alergena čine unakrsno reaktivan par. Svi modeli koriste embedding reprezentacije proteinskih sekvenci dobijene proteinskim jezičkim modelom ESM-2, dok se razlikuju u načinu predstavljanja para proteina i arhitekturi klasifikatora.

### 2.5.1 ESM-2 embedding reprezentacije

Za generisanje embedding reprezentacija korišćen je proteinski jezički model **ESM-2**. Model je prethodno treniran na velikom broju proteinskih sekvenci metodom samonadziranog učenja, pri čemu svakoj aminokiselini dodeljuje vektor koji sadrži informacije o lokalnom i globalnom kontekstu sekvence.

U radu su korišćene dve veličine modela:

- **ESM-2 650M**, sa približno 650 miliona parametara, korišćen za većinu eksperimenata i kao osnovna embedding reprezentacija.
- **ESM-2 3B**, sa približno 3 milijarde parametara, korišćen za proveru da li veći model pruža dodatni diskriminativni signal u zadatku predikcije unakrsne reaktivnosti.

Za svaki protein izdvojeni su **per-residue embeddingi**, nakon čega je primenjen **mean pooling** preko cele sekvence kako bi se dobio jedan vektor fiksne dimenzionalnosti po proteinu. Ovakva reprezentacija korišćena je u svim globalnim modelima mašinskog učenja.

### 2.5.2 Konstrukcija ulaznih karakteristika

Za svaki par proteina konstruisan je vektor karakteristika koji opisuje njihov međusobni odnos. Tokom rada ispitano je više načina kombinovanja embeddinga.

**Apsolutna razlika embeddinga (absolute difference)** definisana je kao

$$
x = |u-v|,
$$

gde su $u$ i $v$ embedding reprezentacije dva proteina. Ovakva reprezentacija korišćena je u početnim MLP i Random Forest modelima, uz dodatnu cosine sličnost kao posebnu numeričku karakteristiku.

Drugi pristup predstavlja **Hadamard produkt**

$$
x = u \odot v,
$$

odnosno element-wise množenje odgovarajućih dimenzija embedding vektora. Za razliku od apsolutne razlike, Hadamard produkt zadržava informacije o zajedničkoj aktivaciji pojedinačnih dimenzija embedding prostora i omogućava modelu da uči interakcije između istih latentnih osobina dva proteina.

Pored ove dve reprezentacije, eksperimentalno je ispitana i bilinearna reprezentacija zasnovana na spoljašnjem proizvodu (outer product), ali ona nije uključena u završni model zbog znatno većeg broja parametara i slabije stabilnosti pri validaciji.

### 2.5.3 MLP klasifikator

Osnovni neuronski model predstavlja višeslojni perceptron (Multi-Layer Perceptron, MLP) za binarnu klasifikaciju proteinskih parova.

Ulaz modela čini vektor karakteristika dobijen iz embedding reprezentacija dva proteina, dok izlaz predstavlja logit koji odgovara procenjenoj verovatnoći unakrsne reaktivnosti.

Početna arhitektura sastoji se od dva potpuno povezana skrivena sloja sa ReLU aktivacionom funkcijom i dropout regularizacijom:

$$
1281 \rightarrow 256 \rightarrow 64 \rightarrow 1.
$$

Model je treniran korišćenjem funkcije gubitka **Binary Cross-Entropy with Logits**, optimizatora **AdamW** i ranog zaustavljanja (early stopping) na osnovu performansi na validacionom skupu.

Za modele koji koriste reprezentaciju apsolutne razlike ulazne karakteristike standardizovane su korišćenjem z-score normalizacije izračunate isključivo na trening podacima svakog LOCO folda.

### 2.5.4 Hadamard MLP

Najuspešnija varijanta neuronskog modela koristi Hadamard produkt embedding vektora kao ulaz u isti MLP klasifikator.

Za ovu reprezentaciju standardizacija ulaznih karakteristika nije primenjivana, jer je tokom preliminarnih eksperimenata utvrđeno da normalizacija značajno narušava distribuciju Hadamard produkta i dovodi do gubitka diskriminativnog signala. Zbog toga je model treniran direktno nad sirovim Hadamard karakteristikama.

Ova arhitektura zadržava mali broj parametara u odnosu na dimenzionalnost skupa podataka i omogućava učenje nelinearne kombinacije latentnih osobina dva proteinska embeddinga.

### 2.5.5 Bilinearni modeli

Radi ispitivanja složenijih interakcija između embedding dimenzija implementiran je i bilinearni model zasnovan na spoljašnjem proizvodu embedding vektora.

Za embeddinge $u$ i $v$ bilinearni skor definiše se kao

$$
s = u^T W v,
$$

gde je $W$ naučena matrica parametara. Zbog veoma velikog broja mogućih parametara puna matrica nije korišćena, već niskorangna (low-rank) faktorizacija koja projektuje embeddinge u prostor manje dimenzionalnosti pre bilinearne interakcije.

Ovaj pristup predstavlja znatno izražajniji model od Hadamard produkta, ali istovremeno povećava broj parametara i rizik od preprilagođavanja na relativno malom broju nezavisnih trening primera.

### 2.5.6 Trening modela

Svi nadgledani modeli trenirani su i validirani protokolom opisanim u poglavlju 3 (LOCO). Negativni primeri uzorkovani su strategijom opisanom u poglavlju 2.2; tokom razvoja ispitane su i alternativne strategije (hard-negative uzorkovanje, Positive-Unlabeled bagging), evaluirane odvojeno od osnovnog MLP modela i diskutovane u poglavlju 5.

## 3. Eksperimentalni protokol

### 3.1 Kontrola curenja informacija — LOCO i leave-one-edge-out

Naivna slučajna podela parova na trening i test skup ne kontroliše zavisnost između povezanih primera: ako su alergeni $A$–$B$ i $A$–$C$ oba u gold skupu, a jedan završi u trening a drugi u test delu, model može posredno "videti" test par preko zajedničkog suseda $A$ već tokom treninga. Da bi se ovo sprečilo, gold parovi se posmatraju kao ivice grafa unakrsne reaktivnosti, a validacija se sprovodi na nivou **povezanih komponenti** tog grafa — maksimalnih podskupova alergena međusobno dostižnih preko lanca poznatih parova.

Kod **Leave-One-Connected-Component-Out (LOCO)** validacije, u svakom foldu se jedna cela povezana komponenta u potpunosti izdvaja iz treninga (svi njeni čvorovi i sve njene ivice) i koristi isključivo za testiranje. Time je zagarantovano da nijedan test par, niti bilo koji njemu susedan trening primer iz iste komponente, nije video model tokom učenja. Ovo je stroži zahtev od uobičajene k-fold podele na nivou pojedinačnih parova, koja bi mogla ostaviti direktno povezan par na obe strane podele.

Signal **graph propagation** (poglavlje 2.4.1) po definiciji koristi poznate susede upita — pod LOCO protokolom test komponenta nema nijednog vidljivog suseda u trening grafu, pa bi ovaj signal bio identički nula za svaki test upit. Za njega je zato korišćena **leave-one-edge-out** evaluacija: za svaki poznati par privremeno se uklanja samo ta jedna ivica, dok ostale ivice istog alergena (veze ka drugim poznatim partnerima) ostaju dostupne. Ovaj protokol modeluje realniju situaciju zadatka 2 (poglavlje 1.2): pacijentu je već poznata bar jedna reaktivnost, a cilj je predvideti sledeću.

### 3.2 Statistička procena značajnosti

Značajnost razlika između metoda procenjivana je bootstrap resamplovanjem, u dve varijante:

- **pair-level bootstrap** — resamplovanje pojedinačnih parova; tretira svaki gold par kao nezavisan primer,
- **study-level bootstrap** — resamplovanje po *literaturnom izvoru* (poglavlje 2.1) umesto po paru; kontroliše za slučaj da više parova potiče iz iste studije i time nije statistički nezavisno.

Kad god se ove dve metode razilaze, rad izveštava study-level rezultat kao merodavniji, a razliku eksplicitno komentariše (poglavlje 4.3) — pair-level bootstrap sam po sebi može precenjivati značajnost u prisustvu neravnomerno zastupljenih izvora.

### 3.3 Validacija na stvarnim pacijentima (zadatak 2)

Nezavisno od LOCO validacije nad gold skupom, model se dodatno evaluira na literaturno dokumentovanim slučajevima stvarnih pacijenata, korišćenjem **leave-one-patient-out** protokola: za svakog pacijenta sa $n\geq 2$ poznata nalaza, svaki nalaz se redom privremeno sakriva, preostali poznati pozitivi koriste se kao upiti (formula u poglavlju 2.3), i beleži se rang sakrivenog alergena među svim kandidatima. Ovaj skup je potpuno odvojen od gold skupa korišćenog za trening — pacijentski slučajevi potiču iz drugih, nezavisno pronađenih literaturnih izvora i ne učestvuju ni u jednom treningu.

Poređenja modela na ovom skupu izvode se **uparenim** testovima na identičnim (pacijent, sakriveni protein) upitima za oba modela — Wilcoxon signed-rank na razlici MRR po pacijentu, permutacioni test koji permutuje oznaku modela unutar pacijenta (ne ishod), i bootstrap sa resamplovanjem po pacijentu. Sva tri testa se izveštavaju zajedno; zaključak "model A bolji od B" se ne izvodi iz dva odvojena testa "iznad slučajnosti" za svaki model posebno, jer to ne dokazuje razliku između modela (poglavlje 4.5).

## 4. Rezultati

### 4.1 Cosine baseline i klasične metode klasifikacije

Cosine sličnost ESM-2 embedding reprezentacija korišćena je kao referentni signal. Pod LOCO validacijom ostvarila je mikro-prosečan **MRR = 0.1209**.

| Model                                     | MRR (LOCO) | Napomena                                          |
| ----------------------------------------- | ---------: | ------------------------------------------------- |
| Cosine                                    |     0.1209 | Referenca                                         |
| Random Forest + BLAST + Foldseek TM-score |     0.1249 | Nije statistički značajno bolje od cosine         |
| RF, sweep hiperparametara                 |     0.1245 | Najbolje podešavanje nije se replikovalo pod LOCO |
| PU Bagging (RF + BLAST)                   |     0.2038 | Nije bolje od RF + BLAST pod LOCO                 |
| XGBoost + BLAST                           |     0.1139 | Lošije od RF + BLAST                              |

RF modeli sa različitim podešavanjima i kombinacijama BLAST i Foldseek signala nisu pokazali statistički značajno poboljšanje u odnosu na cosine baseline pod LOCO validacijom. PU Bagging i XGBoost takođe nisu ostvarili prednost u odnosu na osnovne pristupe.

### 4.2 Neuronski modeli nad embedding reprezentacijama

| Model                     |       MRR (LOCO) | Rezultat                              |
| ------------------------- | ---------------: | ------------------------------------- |
| MLP, apsolutna razlika    | 0.1060 do 0.1737 | Lošije od cosine                      |
| Hadamard bilinearni model |           0.1004 | Statistički značajno lošije           |
| **MLP, Hadamard produkt** |       **0.1209** | **Statistički izjednačeno sa cosine** |

MLP nad apsolutnom razlikom embeddinga bio je lošiji od cosine baseline-a u svim testiranim konfiguracijama. MLP nad Hadamard produktom ostvario je **MRR = 0.1209**, što odgovara rezultatu cosine baseline-a. Standardizacija Hadamard obeležja nije korišćena u konačnoj konfiguraciji.

Provera veličine ESM-2 backbone-a pokazala je različite rezultate za 650M i 3B varijantu. MLP nad Hadamard produktom sa ESM-2 3B embeddingima ostvario je **MRR = 0.1131 do 0.1136** i bio je statistički značajno lošiji od BLAST-a i 650M varijante. Cosine sličnost u 3B prostoru nije pokazala statistički značajnu razliku u odnosu na 650M prostor.

### 4.3 Fuzija nezavisnih signala

| Model                                     |       MRR (LOCO) | Poređenje                                 |
| ----------------------------------------- | ---------------: | ----------------------------------------- |
| RRF-3, cosine + BLAST + Foldseek TM-score |           0.1294 | Bolje od cosine na pair-level bootstrap-u |
| **RRF-4, + graph propagation**            |       **0.1304** | Najviši MRR                               |
| RRF + MLP(Hadamard)                       |           0.1322 | Bez statistički značajne dodatne koristi  |
| Weighted RRF                              | 0.1309 do 0.1332 | Nije bolje od uniformne RRF-4             |

Za RRF-3 u odnosu na cosine, study-level bootstrap dao je 95% CI razlike **[+0.0032, +0.0301]**. Za RRF-3 u odnosu na BLAST CI je bio **[-0.0139, +0.0113]**, dok je za RRF-4 u odnosu na RRF-3 bio **[-0.0015, +0.0172]**.

| Poređenje        | Delta MRR | 95% CI, pair-level | 95% CI, study-level |
| ---------------- | --------: | ------------------ | ------------------- |
| RRF-3 vs. cosine |   +0.0113 | [+0.0028, +0.0142] | [+0.0032, +0.0301]  |
| RRF-3 vs. BLAST  |   +0.0057 | [+0.0001, +0.0116] | [-0.0139, +0.0113]  |
| RRF-4 vs. RRF-3  |   +0.0060 | [+0.0016, +0.0101] | [-0.0015, +0.0172]  |

### 4.4 Lokalna i strukturna reprezentacija

Sliding-window pristup sa max i top-3 agregacijom nije pokazao poboljšanje u odnosu na cosine sličnost celog proteina. Na celom skupu delta MRR iznosila je približno **−0.0015**.

LSE pooling preko matrice lokalnih sličnosti dao je različite rezultate između proteinskih familija:

| Familija | Delta MRR (LOCO) | 95% CI             | Značajno |
| -------- | ---------------: | ------------------ | -------- |
| nsLTP    |          +0.0218 | [+0.0116, +0.0329] | Da       |
| Profilin |          +0.0334 | [+0.0183, +0.0487] | Da       |
| PR-10    |          −0.0012 | [-0.0181, +0.0131] | Ne       |

Attention-MIL nije pokazao prednost u odnosu na LSE pooling. Za nsLTP delta u odnosu na cosine iznosila je **+0.0075**, sa 95% CI **[-0.0048, +0.0200]**. Za Profilin delta je iznosila **+0.0341**, sa 95% CI **[+0.0189, +0.0502]**.

### 4.5 Validacija na stvarnim pacijentima

Nezavisna validacija izvršena je na literaturno dokumentovanim slučajevima stvarnih pacijenata korišćenjem leave-one-patient-out protokola.

Za RRF-4, primarna analiza svih pacijenata nije pokazala statistički značajnu razliku (**cluster-permutacija p = 0.517**). U analizi bez dominantne kohorte rezultat je bio značajan (**cluster-permutacija p = 0.044**, **Wilcoxon p = 0.047**).

Dodavanje LSE signala u RRF-5 nije poboljšalo rezultat. U istoj podgrupi cluster-permutacija je dala **p = 0.131**, a Wilcoxon **p = 0.156**.

Dodavanje MLP(Hadamard) signala u RRF-6 pomerilo je rezultat u pozitivnom smeru. Na svim pacijentima cluster-permutacija je dala **p = 0.168**, dok je u analizi bez dominantne kohorte rezultat bio **p = 0.009**.

Direktno poređenje MLP(Hadamard) i BLAST signala izvršeno je na identičnim pacijentskim upitima korišćenjem Wilcoxon testa, cluster-permutacije i patient-level bootstrap-a.

| Podskup                      | n (upiti / pacijenti) | Wilcoxon p | Cluster-permutacija p | Bootstrap 95% CI       |
| ---------------------------- | --------------------: | ---------: | --------------------: | ---------------------- |
| Svi upiti                    |              176 / 54 | **0.0116** |            **0.0304** | **[+0.0019, +0.0237]** |
| Hard, full-text verifikovano |              148 / 44 | **0.0042** |            **0.0080** | **[+0.0043, +0.0295]** |

MLP(Hadamard) je ostvario statistički značajno bolji rezultat od BLAST-a na oba podskupa. Isti obrazac dobijen je i sa ESM-2 3B backbone-om.

**Cosine sličnost kao samostalan pacijentski signal.** Da bi se utvrdilo da li prednost MLP(Hadamard)-a nad BLAST-om potiče od same embedding reprezentacije ili od naučene transformacije, testirana je i sirova cosine sličnost (bez treniranja) kao treći samostalan ranker, istim mehanizmom i na identičnom skupu upita:

| Poređenje | Podskup | Wilcoxon p | Cluster-permutacija p | Bootstrap 95% CI |
|---|---|---:|---:|---|
| Cosine vs. BLAST | Svi upiti | 0.7994 | **0.0205** | **[−0.0921, −0.0121]** |
| Cosine vs. BLAST | Hard | 0.4692 | **0.0057** | **[−0.1105, −0.0166]** |
| Cosine vs. MLP(Hadamard) | Svi upiti | **0.0172** | **0.0026** | **[−0.1042, −0.0214]** |
| Cosine vs. MLP(Hadamard) | Hard | **0.0009** | **0.0002** | **[−0.1248, −0.0301]** |

Cosine je najslabiji od sva tri signala na pacijentskom skupu — statistički značajno lošiji od MLP(Hadamard)-a (sva tri testa, oba podskupa) i značajno lošiji od BLAST-a (dva od tri testa). Ovim se precizira centralni nalaz rada: prednost nad BLAST-om ne potiče iz same ESM-2 reprezentacije, već specifično iz naučene, nelinearne Hadamard-MLP transformacije te reprezentacije. Kompletno rangiranje samostalnih signala na pacijentskom skupu je **MLP(Hadamard) > BLAST > cosine**.

### 4.6 Sažetak glavnih rezultata

| Model                   | Gold skup podataka, LOCO                           | Real-world pacijenti          |
| ----------------------- | -------------------------------------------------- | ----------------------------- |
| Cosine                  | MRR = 0.1209                                       | Najslabiji signal — značajno lošiji od BLAST-a i MLP-a |
| BLAST                   | MRR = 0.1243                                       | Referentni signal             |
| **MLP(Hadamard), 650M** | **MRR = 0.1259, bez značajne razlike od BLAST-a**  | **Značajno bolji od BLAST-a** |
| MLP(Hadamard), 3B       | MRR = 0.1131 do 0.1136, značajno lošiji od BLAST-a | Značajno bolji od BLAST-a     |
| RRF-4                   | MRR = 0.1304                                       | Primarni test nije značajan   |

Najizraženija razlika između dve evaluacije javlja se kod MLP(Hadamard) modela. Na gold skupu podataka rezultat je statistički izjednačen sa BLAST-om, dok na nezavisnim slučajevima stvarnih pacijenata pokazuje statistički značajnu prednost.

## 5. Diskusija

### 5.1 Ograničenje dostupnih podataka
Rezultati ukazuju da je jedan od glavnih ograničavajućih faktora veličina i struktura dostupnog skupa podataka. Ukupan broj poznatih parova je relativno veliki, ali je broj nezavisnih primera manji zbog povezanosti proteina unutar familija i zajedničkih literaturnih izvora. Zbog toga bi dalje proširenje gold skupa verovatno bilo korisno, posebno dodavanjem eksperimentalno potvrđenih parova iz trenutno slabije zastupljenih familija i parova sa manjom sekvencijskom sličnošću.

### 5.2 Očekivanja od strukturnih embeddinga
Na osnovu dobijenih rezultata ne očekuje se da bi sama zamena ESM-2 embeddinga AlphaFold ili OpenFold embeddingima donela veliki napredak. Strukturna sličnost već je testirana kao dodatni signal, ali nije pokazala dovoljno snažan samostalan signal. Problem se posebno vidi kod proteinskih familija čiji su članovi veoma slični na globalnom nivou, dok se unakrsna reaktivnost može razlikovati zbog lokalnih i imunološki relevantnih osobina. Zbog toga bi veći potencijal imalo kombinovanje strukturnih reprezentacija sa informacijama o epitopima i površinskoj dostupnosti, a ne samo zamena jednog embedding modela drugim.

### 5.3 Razlika između gold skupa i pacijenata
Razlika između rezultata na gold skupu i stvarnim pacijentima može biti posledica različite distribucije podataka. Gold skup je sastavljen od već poznatih i literaturno dokumentovanih parova, dok pacijentski slučajevi predstavljaju širi i manje selektovan problem rangiranja.
BLAST je posebno zavisan od toga da sekvencijska sličnost prati unakrsnu reaktivnost. To ne mora važiti za udaljene proteine koji dele relevantne lokalne ili strukturne osobine. MLP nad Hadamard produktom može koristiti složenije odnose između embeddinga, što može objasniti njegovu prednost na pacijentskim slučajevima iako na gold skupu nije značajno bolji od BLAST-a.

### 5.4 Da li proteinski embeddingi imaju budućnost?
Rezultati podržavaju korišćenje proteinskih embeddinga, ali ne pokazuju da veći model automatski daje bolju predikciju. ESM-2 3B nije doneo poboljšanje u odnosu na 650M, dok je jednostavnija Hadamard reprezentacija para bila korisnija od složenijih bilinearnih modela.
To sugeriše da je za ovaj problem važniji način poređenja dve proteinske reprezentacije nego samo povećavanje kapaciteta osnovnog proteinskog modela. Embeddinge zato treba posmatrati kao jednu komponentu sistema, koju je korisno kombinovati sa sekvencijskim, strukturnim i eksperimentalnim informacijama.

## 6. Budući rad

Najvažniji pravci daljeg rada su proširenje gold skupa novim nezavisnim eksperimentalnim dokazima, povećanje broja nezavisnih pacijentskih slučajeva i uključivanje biološki relevantnih informacija o epitopima i strukturi. Posebno bi bilo važno proveriti da li se prednost MLP(Hadamard) modela održava na većem i raznovrsnijem skupu pacijenata.
Ukupno, rezultati pokazuju da proteinski embeddingi mogu sadržati signal relevantan za predikciju unakrsne reaktivnosti, ali da napredak verovatnije zavisi od kvaliteta podataka i načina reprezentacije odnosa između proteina nego od samog povećavanja modela.

Krajnji cilj ovakvog sistema mogao bi biti razvoj asistivnog alata koji, na osnovu poznatih alergija pacijenta, rangira potencijalno unakrsno reaktivne alergene i predlaže prioritete za dalje alergološko testiranje. U proširenoj verziji sistem bi mogao da koristi poznate pozitivne i negativne nalaze za personalizovano rangiranje novih kandidata. Takav sistem ne bi zamenio kliničku procenu, već bi služio kao pomoć pri izboru prioriteta za dalje testiranje.

