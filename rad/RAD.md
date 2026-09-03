---
bibliography: references.bib
---

# Predikcija unakrsne reaktivnosti proteinskih alergena korišćenjem proteinskih jezičkih modela i mašinskog učenja

Lana Lejić            
mentor: Stefan Nožinić
seminar: Računarstvo

## Apstrakt

kao nešto pametno

## 1. Uvod

### 1.1 Motivacija

Unakrsna alergijska reaktivnost je fenomen u kojem IgE antitela senzibilisanog organizma prepoznaju proteine različitih, taksonomski često nepovezanih alergenih izvora. Proteinska sličnost je jedan od faktora koji doprinose ovom fenomenu, ali sama sekvencijalna sličnost nije dovoljna da ga u potpunosti objasni: reaktivnost zavisi od kombinacije osobina proteinske sekvence, konzerviranih strukturnih elemenata, lokalne dostupnosti epitopa i drugih molekularnih karakteristika. Zbog toga alergeni s relativno niskom sekvencijalnom sličnošću mogu pokazivati unakrsnu reaktivnost (npr. panalergeni poput profilina), dok visoka sličnost sama po sebi nije dovoljan dokaz zajedničke reaktivnosti.

Razvoj proteinskih jezičkih modela (protein language models) otvorio je mogućnost da se proteini predstave kao vektori u naučenom višedimenzionalnom prostoru, umesto isključivo kroz mere poravnanja sekvenci. Takve reprezentacije potencijalno opisuju obrasce koji nisu direktno vidljivi kroz klasične mere sličnosti, ali pitanje da li one zaista nose korisnu, generalizabilnu informaciju za predikciju unakrsne reaktivnosti (a ne artefakt konkretnog skupa podataka ili načina njegove podele) nije trivijalno i zahteva sistematsku, statistički rigoroznu proveru.

### 1.2 Formulacija problema: dva odvojena zadatka

Rad razmatra dva srodna, ali metodološki različita zadatka rangiranja, koja se u literaturi i u praksi često ne razdvajaju eksplicitno:

1. **Rangiranje unakrsno reaktivnih parova alergena (pair-level ranking).** Za dati alergen kao upit, zadatak je rangirati sve ostale proteine u skupu kandidata prema verovatnoći da čine unakrsno reaktivan par s upitom. Ovaj zadatak se evaluira direktno nad kurirano-verifikovanim skupom poznatih parova i koristi se za treniranje i selekciju modela.
2. **Rangiranje kandidata za konkretnog pacijenta (patient-level ranking).** Za pacijenta sa jednim ili više već poznatih (pozitivnih i/ili negativnih) alergijskih nalaza, zadatak je rangirati preostale alergene iz istog skupa kandidata prema prioritetu za dalje testiranje. Signal za ovaj zadatak dobija se agregacijom rangova na nivou para (pair-level) preko svih poznatih pozitivnih nalaza istog pacijenta (formalizacija u poglavlju 2.3, protokol evaluacije u poglavlju 3.3).

Razlikovanje ova dva zadatka je centralno za rad iz dva razloga. Prvo, prvi zadatak se evaluira nad skupom čija je konstrukcija (izvor dokaza, familijska struktura, gustina grafa) poznata i kontrolisana, dok se drugi zadatak evaluira nad nezavisnim, spolja pristiglim kliničkim slučajevima čija distribucija ne mora odgovarati distribuciji trening skupa. Model koji na prvom zadatku ne pokazuje prednost nad jednostavnijim baznim metodama može na drugom zadatku pokazati statistički značajnu prednost, nalaz koji bi ostao neprimećen da se ova dva zadatka ne razdvoje eksplicitno.


## 2. Metodologija

### 2.1 Konstrukcija skupa podataka

Skup kandidata za rangiranje (pool) izveden je iz zvanične WHO/IUIS Allergen Nomenclature baze, dopunjene sa 1.922 para unakrsne reaktivnosti prikupljena iz objavljenih naučnih radova. U nastavku je konstrukcija oba dela opisana korak po korak, radi reprodukovanja.

**Pool kandidata: koraci obrade.** Polazni izvoz iz WHO/IUIS baze sadrži 1.638 zapisa, sa poljima za naziv alergena, izoformu, organizam, UniProt identifikator i FASTA sekvencu. Obrada se sastoji od sledećih koraka, primenjenih tim redosledom:

1. **Čišćenje sekvence.** Iz svake FASTA sekvence uklanjaju se svi karakteri koji nisu jedno od 20 standardnih aminokiselinskih slova; rezultat se svodi na velika slova.
2. **Filtriranje po dužini.** Zapisi sa očišćenom sekvencom kraćom od 30 aminokiselina se uklanjaju.
3. **Uklanjanje potpuno dupliranih redova** (identični svi podaci).
4. **Deduplikacija po sekvenci.** Zapisi sa identičnom FASTA sekvencom (posle koraka 1) se svode na jedan; kada više zapisa deli istu sekvencu, zadržava se onaj čiji je zvanični naziv alfabetski prvi. Ovo je proizvoljno, ne biološko pravilo, i ima konkretnu, dokumentovanu posledicu: tropomiozin škampa Pen a 1 i Pen m 1 imaju sekvencu identičnu već zadržanom Lit v 1 (drugi škamp iz iste porodice), pa su tokom ovog koraka uklonjeni iz pool-a kao "duplikati" iako nose sopstvenu WHO/IUIS registraciju. Za upite koji se oslanjaju na takva imena, rešenje usvojeno u ovom radu je eksplicitno mapiranje imena na zadržani pool-zapis iste sekvence (poglavlje 3.3), a ne ponovno uvođenje veštački dupliranog embeddinga u pool.

Rezultat je **1.536 proteinskih alergena** u konačnom pool-u.

**Izoforme.** Izoforme istog alergena (npr. Aca s 2.0101 i Aca s 2.0102) nisu kolabirane niti spojene u jedan zapis: ako se dve izoforme razlikuju u sekvenci, obe ostaju kao odvojeni, nezavisni kandidati u pool-u; kolabiraju samo ako im je sekvenca posle čišćenja identična (korak 4 iznad), po istom, ne-biološkom pravilu.

**Fragmenti.** Van filtriranja po minimalnoj dužini, pool ne sadrži sistematsku proveru da li je zapisana sekvenca puna ili predstavlja fragment proteina. Ovo je poznato ograničenje: naknadna provera protiv UniProt zapisa (van glavnog pipeline-a, nije primenjena retroaktivno na pool) potvrdila je barem jedan slučaj (Gly m 1) gde je zapisana sekvenca UniProt-om označena kao fragment. Dalja diskusija u poglavlju 5.5.

**Parovi unakrsne reaktivnosti.** Nad ovim pool-om definisano je **1.922 poznata para** unakrsne reaktivnosti, prikupljena iz objavljenih naučnih radova koji opisuju eksperimentalno potvrđene ili pretpostavljene slučajeve. Za svaki par zabeležen je izvor dokaza (literaturna referenca), tip potvrde i proteinska familija oba člana para kada je bila dostupna u izvoru. Ovih 1.922 para pokrivaju 477 pojedinačnih alergena iz pool-a, grupisanih (po sopstvenoj familijskoj oznaci iz izvora, ne iz pool-a) u 74 proteinske familije, izvedenih iz 317 nezavisnih literaturnih izvora. Neravnomerna zastupljenost izvora (mali broj radova opisuje veliki broj parova, npr. populacione kohortne studije) direktno motiviše potrebu za bootstrap analizom na nivou studije.

#### 2.1.1 Klasifikacija nivoa dokaza

Svaki par klasifikovan je prema pouzdanosti dokaza u jednu od četiri kategorije, sa brojem parova po kategoriji:

| Nivo dokaza | Opis | Broj parova (%) |
|---|---|---:|
| Confirmed | Direktan eksperimentalni dokaz (npr. IgE inhibicioni test) | 138 (7,2%) |
| Strong | Više nezavisnih studija ili velike kliničke kohorte | 377 (19,6%) |
| Suspected | Literatura ukazuje na moguću reaktivnost, dokaz nedovoljan za potvrdu | 277 (14,4%) |
| Inferred | Izvedeno iz homologije/pripadnosti istoj familiji, bez direktnog testa konkretnog para | 1.093 (56,9%) |

Parovi iz kategorije **Inferred** nisu korišćeni za treniranje nadgledanih modela, ali su zadržani kao evaluacioni ciljevi. Ova odluka odvaja pitanje "da li model dobro rangira i manje pouzdane parove" (evaluacija) od pitanja "da li model treba da uči iz njih kao iz pouzdanog signala" (trening).

### 2.2 Problem negativnih i nepoznatih parova

Referentni skup podataka sadrži isključivo **pozitivne** primere, parove za koje postoji literaturni dokaz reaktivnosti. Odsustvo para iz skupa **ne znači potvrđenu odsutnost** unakrsne reaktivnosti, već samo odsustvo objavljenog dokaza; zadatak je time strukturno pozitivno-neoznačen (Positive-Unlabeled, PU), a ne standardna binarna klasifikacija. Negativni primeri korišćeni za treniranje generisani su nasumičnim uzorkovanjem parova van referentnog skupa podataka, pod pretpostavkom da je verovatnoća da nasumično izabran par bude neotkriven pravi pozitiv niska.

Ova pretpostavka je proverena i za jedan konkretan slučaj eksplicitno odbačena: ciljano uzorkovanje "teških" negativa unutar iste proteinske familije nije korišćeno, jer su upravo neoznačeni parovi unutar familije mesto gde je najverovatnije da se kriju neotkriveni pravi pozitivi, isti razlog zbog kog je i sam referentni skup podataka rastao dodavanjem novih, ranije neobuhvaćenih parova unutar poznatih familija. Nasumično uzorkovanje iz celog skupa kandidata je zbog toga zadržano kao podrazumevana, konzervativnija strategija u svim eksperimentima treniranja opisanim u ovom radu.

U svim eksperimentima treniranja odnos negativnih naspram pozitivnih primera bio je fiksiran na 10:1, to jest za svaki pozitivan par iz trening dela referentnog skupa podataka nasumično se uzorkuje deset negativnih parova. Ovaj odnos je izabran empirijski, kao kompromis između dovoljno negativnih primera za stabilnu procenu granice odlučivanja i preteranog razblaživanja retkih pozitivnih primera unutar trening skupa; isti odnos korišćen je dosledno u svim modelima i svim protokolima validacije opisanim u ovom radu, radi uporedivosti rezultata.

### 2.3 Formalizacija zadataka i metrike

Za oba zadatka koristi se ista osnovna operacija: rangiranje kandidata prema skalarnom skoru. Kod **zadatka 1** (pair-level), skor kandidata $c$ za upit $q$ direktno je izlaz posmatranog signala cosine sličnost, BLAST identitet, ili verovatnoća nadgledanog klasifikatora. Kod **zadatka 2** (patient-level), skor kandidata $c$ za pacijenta sa poznatim pozitivnim nalazima $P=\{p_1,\dots,p_n\}$ dobija se sabiranjem recipročnih rangova kandidata prema svakom poznatom pozitivu, istim funkcionalnim oblikom kao RRF fuzija signala:

$$
\mathrm{score}(c \mid P) = \sum_{p \in P} \frac{1}{K + r_p(c)},
$$

gde je $r_p(c)$ rang kandidata $c$ u listi dobijenoj upitom $p$. Ovim se poznati pozitivi jednog pacijenta tretiraju kao više nezavisnih upita čiji se doprinosi sabiraju, isti mehanizam koji se koristi za fuziju više *signala* ovde se koristi za fuziju više *upita*.

Performanse oba zadatka mere se pomoću dve dopunske metrike rangiranja, obe računate nad skupom upita $Q$.

**Srednji recipročni rang (Mean Reciprocal Rank, MRR)** definiše se kao

$$
\mathrm{MRR} = \frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\mathrm{rang}(q)},
$$

gde je $\mathrm{rang}(q)$ pozicija tačnog kandidata u rangiranoj listi za upit $q$. Zbog recipročnog oblika, MRR je osetljiv na finu razliku između bliskih rangova (rang 3 naspram ranga 30 menja vrednost primetno), ali je manje osetljiv na razliku između već udaljenih rangova (rang 300 naspram ranga 600 doprinosi skoro identično malo).

**Hits@K** ($K\in\{1,5,10\}$) meri udeo upita kod kojih se tačan kandidat nalazi među $K$ najbolje rangiranih kandidata:

$$
\mathrm{Hits@K} = \frac{1}{|Q|}\sum_{q\in Q}\mathbb{1}\left[\mathrm{rang}(q)\leq K\right],
$$

gde je $\mathbb{1}[\cdot]$ indikatorska funkcija (1 ako je uslov tačan, inače 0). Hits@K je lakše klinički protumačiti (na primer, Hits@5 direktno odgovara na pitanje "da li se tačan kandidat nalazi među prvih pet predloženih"), ali ne razlikuje da li je promašeni kandidat na rangu 6 ili rangu 600. Zbog toga se dve metrike izveštavaju zajedno: MRR kao osetljivija, sveobuhvatnija mera kvaliteta rangiranja, Hits@K kao direktnije, praktično protumačiva mera.

### 2.4 Reciprocal Rank Fusion (RRF)

Pojedinačni izvori sličnosti između proteinskih alergena opisuju različite aspekte njihove potencijalne unakrsne reaktivnosti. Sličnost reprezentacija dobijenih proteinskim jezičkim modelom opisuje globalnu sličnost proteinskih sekvenci u naučenom reprezentacionom prostoru, BLAST meri lokalnu sekvencijalnu homologiju zasnovanu na poravnanju sekvenci, dok Foldseek TM-score procenjuje sličnost trodimenzionalne strukture proteina. Nijedan od ovih signala pojedinačno ne predstavlja potpun opis unakrsne reaktivnosti, zbog čega je korišćena metoda fuzije rangova.

Za kombinovanje nezavisnih rang-lista korišćen je algoritam **Reciprocal Rank Fusion (RRF)**. Za svaki izvor sličnosti formira se rang svih kandidata, a konačan skor kandidata dobija se sabiranjem recipročnih vrednosti njihovih rangova:

$$
RRF(d)=\sum_{i=1}^{m}\frac{1}{K+r_i(d)}
$$

gde je $r_i(d)$ rang kandidata $d$ prema $i$-tom signalu, $m$ broj korišćenih signala, a $K$ konstanta koja umanjuje uticaj veoma visokih rangova.

U ovom radu korišćena su tri osnovna signala:

- cosine sličnost ESM-2 reprezentacija,
- BLAST skor sekvencijalnog poravnanja,
- Foldseek TM-score strukturne sličnosti.

Konstanta $K$ nije odabrana proizvoljno. Testirano je više vrednosti parametra na trening komponentama LOCO validacije, pri čemu je izabrana vrednost koja je pokazala najstabilnije performanse na nezavisnim komponentama. U svim narednim eksperimentima korišćena je ista vrednost parametra kako bi evaluacija ostala konzistentna.

Prednost RRF algoritma je što kombinuje rangove umesto sirovih skorova, pa nije potrebno normalizovati vrednosti različitih metoda niti pretpostaviti da su njihovi skorovi međusobno uporedivi. Time se omogućava spajanje heterogenih izvora informacija bez dodatnog treniranja modela.

#### 2.4.1 Graph propagation kao dodatni signal

Pored tri osnovna signala, testirana je i proširena varijanta RRF-a u kojoj je dodat četvrti signal zasnovan na poznatim vezama u grafu unakrsne reaktivnosti.

Za dati upit, ako je poznato da je alergen već unakrsno reaktivan sa jednim ili više drugih alergena, kandidat dobija dodatni skor ukoliko je povezan sa tim susedima u grafu poznatih interakcija. Ovaj signal predstavlja propagaciju informacija kroz mrežu poznatih unakrsnih reaktivnosti i koristi isključivo veze dostupne u trening delu validacije.

Da bi se izbeglo curenje informacija, ovaj signal nije evaluiran u LOCO protokolu, jer u tom slučaju test komponenta nema nijednog poznatog suseda u trening grafu. Umesto toga korišćena je **leave-one-edge-out** evaluacija, pri kojoj se za svaki poznati par privremeno uklanja samo testirana veza, dok ostale veze istog alergena ostaju dostupne modelu. Ovakav protokol odgovara realnoj kliničkoj situaciji u kojoj je za pacijenta poznata jedna ili više potvrđenih alergija, a cilj je predvideti dodatne potencijalno unakrsno reaktivne alergene.

### 2.5 Modeli mašinskog učenja nad proteinskim reprezentacijama

Pored metoda zasnovanih na direktnom rangiranju proteinske sličnosti, u radu su ispitani i nadgledani modeli mašinskog učenja čiji je cilj procena verovatnoće da dva alergena čine unakrsno reaktivan par. Svi modeli koriste reprezentacije proteinskih sekvenci dobijene proteinskim jezičkim modelom ESM-2, dok se razlikuju u načinu predstavljanja para proteina i arhitekturi klasifikatora.

### 2.5.1 ESM-2 reprezentacije

Za generisanje vektorskih reprezentacija (embedding) korišćen je proteinski jezički model **ESM-2**. Model je prethodno treniran na velikom broju proteinskih sekvenci metodom samonadziranog učenja, pri čemu svakoj aminokiselini dodeljuje vektor koji sadrži informacije o lokalnom i globalnom kontekstu sekvence.

U radu su korišćene dve veličine modela:

- **ESM-2 650M**, sa približno 650 miliona parametara, korišćen za većinu eksperimenata i kao osnovna reprezentacija.
- **ESM-2 3B**, sa približno 3 milijarde parametara, korišćen za proveru da li veći model pruža dodatni diskriminativni signal u zadatku predikcije unakrsne reaktivnosti.

Za svaki protein izdvojeni su vektori po pojedinačnoj aminokiselini (**per-residue**), nakon čega je primenjen **mean pooling** preko cele sekvence kako bi se dobio jedan vektor fiksne dimenzionalnosti po proteinu. Ovakva reprezentacija korišćena je u svim globalnim modelima mašinskog učenja.

### 2.5.2 Konstrukcija ulaznih karakteristika

Za svaki par proteina konstruisan je vektor karakteristika koji opisuje njihov međusobni odnos. Tokom rada ispitano je više načina kombinovanja reprezentacija.

**Apsolutna razlika reprezentacija (absolute difference)** definisana je kao

$$
x = |u-v|,
$$

gde su $u$ i $v$ reprezentacije dva proteina. Ovakva reprezentacija korišćena je u početnim MLP i Random Forest modelima, uz dodatnu cosine sličnost kao posebnu numeričku karakteristiku.

Drugi pristup predstavlja **Hadamard produkt**

$$
x = u \odot v,
$$

odnosno poelementno (element-wise) množenje odgovarajućih dimenzija vektora reprezentacije. Za razliku od apsolutne razlike, Hadamard produkt zadržava informacije o zajedničkoj aktivaciji pojedinačnih dimenzija prostora reprezentacije i omogućava modelu da uči interakcije između istih latentnih osobina dva proteina. Za razliku od apsolutne razlike, ulazne karakteristike zasnovane na Hadamard produktu se pri treniranju **ne standardizuju**: preliminarni eksperimenti pokazali su da z-score normalizacija narušava prirodnu skalu Hadamard produkta i značajno narušava diskriminativni signal, pa se u svim eksperimentima sa ovom reprezentacijom trenira direktno nad sirovim vrednostima.

Pored ove dve reprezentacije, eksperimentalno je ispitan i bilinearni model zasnovan na spoljašnjem proizvodu (outer product) para reprezentacija; opisan je ukratko u Prilogu A, jer nije uključen u završni model zbog znatno većeg broja parametara i slabije stabilnosti pri validaciji na relativno malom broju nezavisnih trening primera.

### 2.5.3 MLP klasifikator

Osnovni neuronski model predstavlja višeslojni perceptron (Multi-Layer Perceptron, MLP) za binarnu klasifikaciju proteinskih parova.

Ulaz modela čini vektor karakteristika dobijen iz reprezentacija dva proteina, dok izlaz predstavlja logit koji odgovara procenjenoj verovatnoći unakrsne reaktivnosti.

Početna arhitektura sastoji se od dva potpuno povezana skrivena sloja sa ReLU aktivacionom funkcijom i dropout regularizacijom:

$$
1281 \rightarrow 256 \rightarrow 64 \rightarrow 1.
$$

Model je treniran korišćenjem funkcije gubitka **Binary Cross-Entropy with Logits**, optimizatora **AdamW** i ranog zaustavljanja (early stopping) na osnovu performansi na validacionom skupu.

Za modele koji koriste reprezentaciju apsolutne razlike ulazne karakteristike standardizovane su korišćenjem z-score normalizacije izračunate isključivo na trening podacima svakog LOCO folda; za Hadamard produkt standardizacija se ne primenjuje,zadržava mali broj parametara u odnosu na dimenzionalnost referentnog skupa podataka i pokazala se kao najuspešnija neuronska varijanta.

### 2.5.4 Bilinearni model (kratko, detalji u Prilogu A)

Kao izražajnija alternativa Hadamard produktu, ispitan je i bilinearni model $s=u^TWv$ nad niskorangnom (low-rank) faktorizacijom parova reprezentacija. Model nije uključen u završnu konfiguraciju: veći broj parametara doneo je veći rizik od preprilagođavanja na relativno malom broju nezavisnih trening primera, bez poboljšanja u odnosu na Hadamard produkt (detaljni rezultati, formulacija i diskusija u Prilogu A).

### 2.5.5 Trening modela

Svi nadgledani modeli trenirani su i validirani protokolom opisanim u poglavlju 3 (LOCO). Negativni primeri uzorkovani su strategijom opisanom u poglavlju 2.2; tokom razvoja ispitane su i alternativne strategije (teški negativni primeri, hard-negative uzorkovanje, i pozitivno-neoznačeni, Positive-Unlabeled, bagging), evaluirane odvojeno od osnovnog MLP modela i diskutovane u poglavlju 5.

## 3. Eksperimentalni protokol

### 3.1 Kontrola curenja informacija: LOCO i leave-one-edge-out

Naivna slučajna podela parova na trening i test skup ne kontroliše zavisnost između povezanih primera: ako su alergeni $A$–$B$ i $A$–$C$ oba u referentnom skupu podataka, a jedan završi u trening a drugi u test delu, model može posredno "videti" test par preko zajedničkog suseda $A$ već tokom treninga. Da bi se ovo sprečilo, referentni parovi se posmatraju kao ivice grafa unakrsne reaktivnosti, a validacija se sprovodi na nivou **povezanih komponenti** tog grafa, maksimalnih podskupova alergena međusobno dostižnih preko lanca poznatih parova.

Kod **Leave-One-Connected-Component-Out (LOCO)** validacije, u svakom foldu se jedna cela povezana komponenta u potpunosti izdvaja iz treninga (svi njeni čvorovi i sve njene ivice) i koristi isključivo za testiranje. Time je zagarantovano da nijedan test par, niti bilo koji njemu susedan trening primer iz iste komponente, nije video model tokom učenja. Ovo je stroži zahtev od uobičajene k-fold podele na nivou pojedinačnih parova, koja bi mogla ostaviti direktno povezan par na obe strane podele.

Signal **graph propagation** po definiciji koristi poznate susede upita. Pod LOCO protokolom test komponenta nema nijednog vidljivog suseda u trening grafu, pa bi ovaj signal bio identički nula za svaki test upit. Za njega je zato korišćena **leave-one-edge-out** evaluacija: za svaki poznati par privremeno se uklanja samo ta jedna ivica, dok ostale ivice istog alergena (veze ka drugim poznatim partnerima) ostaju dostupne. Ovaj protokol modeluje realniju situaciju zadatka 2: pacijentu je već poznata bar jedna reaktivnost, a cilj je predvideti sledeću.

### 3.2 Statistička procena značajnosti

Značajnost razlika između metoda procenjivana je bootstrap resamplovanjem, u dve varijante:

- bootstrap na nivou para (**pair-level**), resamplovanje pojedinačnih parova; tretira svaki referentni par kao nezavisan primer,
- bootstrap na nivou studije (**study-level**), resamplovanje po *literaturnom izvoru*
umesto po paru; kontroliše za slučaj da više parova potiče iz iste studije i time nije statistički nezavisno.

### 3.3 Validacija na stvarnim pacijentima (zadatak 2)

Nezavisno od LOCO validacije nad referentnim skupom podataka, model se dodatno evaluira na literaturno dokumentovanim slučajevima stvarnih pacijenata, korišćenjem **leave-one-patient-out** protokola: za svakog pacijenta sa $n\geq 2$ poznata nalaza, svaki nalaz se redom privremeno sakriva, preostali poznati pozitivi koriste se kao upiti, i beleži se rang sakrivenog alergena među svim kandidatima. Ovaj skup je potpuno odvojen od referentnog skupa podataka korišćenog za trening. Pacijentski slučajevi potiču iz drugih, nezavisno pronađenih literaturnih izvora i ne učestvuju ni u jednom treningu.

Poređenja modela na ovom skupu izvode se **uparenim** testovima na identičnim (pacijent, sakriveni protein) upitima za oba modela. Wilcoxon signed-rank na razlici MRR po pacijentu, permutacioni test koji permutuje oznaku modela unutar pacijenta (ne ishod), i bootstrap sa resamplovanjem po pacijentu. Sva tri testa se izveštavaju zajedno; zaključak "model A bolji od B" se ne izvodi iz dva odvojena testa "iznad slučajnosti" za svaki model posebno, jer to ne dokazuje razliku između modela.

## 4. Rezultati

### 4.1 Cosine kao polazni model i klasične metode klasifikacije

Cosine sličnost ESM-2 reprezentacija korišćena je kao polazni model (baseline). Pod LOCO validacijom ostvarila je mikro-prosečan **MRR = 0,1209**.

| Model | MRR | Protokol | Δ vs. polazni model | Statistička provera |
|---|---:|---|---:|---|
| Cosine (polazni model) | 0,1209 | LOCO | - | referenca |
| Random Forest + BLAST + Foldseek TM-score | 0,1249 | LOCO | +0,0040 | nije statistički značajno bolje |
| RF, sweep hiperparametara | 0,1245 | LOCO | +0,0036 | najbolje pojedinačno podešavanje se nije replikovalo pod LOCO |
| PU Bagging (RF + BLAST)$^{A.1}$ | 0,2038 | jednostruka podela, nije potvrđeno pod LOCO | nije direktno uporedivo | mešovit, nepotvrđen rezultat pod LOCO |
| XGBoost + BLAST$^{A.1}$ | 0,1139 | LOCO | −0,0070 | lošije od RF + BLAST |

Nijedan klasičan model zasnovan na ručno konstruisanim obeležjima nije pod LOCO protokolom statistički značajno prevazišao polazni model.

### 4.2 Neuronski modeli nad proteinskim reprezentacijama

| Model | MRR | Protokol | Δ vs. polazni model | Statistička provera |
|---|---:|---|---:|---|
| MLP, apsolutna razlika$^{A.2}$ | 0,1060 do 0,1737 | LOCO, sensitivity sweep (svaka konfiguracija poređena sa sopstvenim polaznim modelom na istoj podeli) | dosledno negativno | lošije od polaznog modela u svih osam testiranih konfiguracija |
| Bilinearni model$^{A.3}$ | 0,1004 | LOCO | −0,0205 | statistički značajno lošije |

Nijedna od ove dve alternativne reprezentacije para ne dostiže polazni model. MLP(Hadamard), konačna konfiguracija korišćena u ostatku rada, testiran je direktno naspram BLAST-a (umesto naspram cosine polaznog modela) pod metodološki finalnim LOCO protokolom (čist trening, referentni skup podataka bez Inferred parova, poglavlje 2.1), zajedno sa proverom veličine ESM-2 baznog modela (backbone):

| Model | MRR | Protokol | Δ vs. BLAST | Δ vs. MLP(Hadamard) 650M | Statistička provera |
|---|---:|---|---:|---:|---|
| BLAST | 0,1243 | LOCO, čist trening | - | - | referenca |
| MLP(Hadamard), ESM-2 650M | 0,1259 | LOCO, čist trening | +0,0016 | - | CI uključuje nulu, statistički izjednačeno sa BLAST-om |
| MLP(Hadamard), ESM-2 3B | 0,1131 do 0,1136 | LOCO, čist trening | −0,0107 do −0,0112 | −0,0123 do −0,0128 | statistički značajno lošije i od BLAST-a i od 650M varijante |
| Cosine, ESM-2 3B prostor (bez treninga) | - | LOCO | nije primenjivo | −0,0007 (naspram cosine u 650M prostoru) | CI uključuje nulu, nije statistički značajno drugačije od 650M prostora |

MLP(Hadamard) na ESM-2 650M je jedini neuronski model u radu koji dostiže performanse BLAST-a; ova vrednost (MRR = 0,1259) koristi se u ostatku rada kao referentna za MLP(Hadamard) (poglavlja 4.3, 4.5, 4.6, 4.7).

### 4.3 Fuzija nezavisnih signala

| Model | MRR | Δ vs. prethodni korak | 95% CI (nivo para) | 95% CI (nivo studije) | Značajno na nivou studije? |
|---|---:|---:|---|---|---|
| Cosine (polazni model) | 0,1209 | - | - | - | - |
| RRF-3 (cosine + BLAST + Foldseek TM-score) | 0,1294 | +0,0113 | [+0,0028, +0,0142] | [+0,0032, +0,0301] | da |
| RRF-3 vs. BLAST samostalno (isti model, poređenje sa BLAST-om a ne sa prethodnim korakom) | - | +0,0057 | [+0,0001, +0,0116] | [−0,0139, +0,0113] | ne |
| RRF-4 (+ graph propagation) | 0,1304 | +0,0060 | [+0,0016, +0,0101] | [−0,0015, +0,0172] | ne |
| RRF + MLP(Hadamard) | 0,1322 | opisno: bez statistički značajne dodatne koristi u odnosu na RRF-3 | nije izračunato | nije izračunato | ne |
| Weighted RRF (naučene težine signala) | 0,1309 do 0,1332 | ne prevazilazi RRF-4 sa uniformnim težinama | nije izračunato | nije izračunato | ne |

RRF-4 je model sa najvišim MRR na referentnom skupu podataka pod LOCO protokolom. Pouzdanost fuzionih dobitaka je, međutim, neravnomerna: dobitak RRF-3 naspram polaznog modela ostaje značajan i na nivou studije, dok dobici RRF-3 naspram BLAST-a i RRF-4 naspram RRF-3 (graph propagation) gube značajnost kada se nezavisnost proveri na nivou literaturnog izvora umesto na nivou pojedinačnog para. Tačke procene ostaju pozitivne u sva tri slučaja, ali gubitak značajnosti na nivou studije pokazuje da je pouzdanost fuzionog dobitka manja nego što bi ukazivao bootstrap na nivou para.

### 4.4 Lokalna i strukturna reprezentacija

Sliding-window pristup sa max i top-3 agregacijom nije pokazao poboljšanje u odnosu na polazni model. Na celom skupu Δ MRR iznosila je približno −0,0015, statistički neznačajno.

LSE pooling preko matrice lokalnih sličnosti dao je različite rezultate između proteinskih familija:

| Model | MRR | Δ vs. polazni model | 95% CI (nivo para) | Značajno? |
|---|---|---:|---|---|
| LSE pooling, nsLTP | - | +0,0218 | [+0,0116, +0,0329] | da |
| LSE pooling, Profilin | - | +0,0334 | [+0,0183, +0,0487] | da |
| LSE pooling, PR-10 | - | −0,0012 | [−0,0181, +0,0131] | ne |
| Attention-MIL, nsLTP | - | +0,0075 | [−0,0048, +0,0200] | ne |
| Attention-MIL, Profilin | - | +0,0341 | [+0,0189, +0,0502] | da |

Attention-MIL, izražajniji model od LSE poolinga, nije pokazao doslednu prednost: rezultat za Profilin je uporediv sa LSE poolingom, ali za nsLTP gubi značajnost koju je LSE pooling imao. Dodatna izražajnost modela ovde nije donela robusniji rezultat.

### 4.5 Validacija na stvarnim pacijentima

Nezavisna validacija izvršena je na literaturno dokumentovanim slučajevima stvarnih pacijenata korišćenjem leave-one-patient-out protokola (poglavlje 3.3).

| Model | Svi pacijenti, cluster-permutacija p | Podgrupa bez dominantne kohorte, cluster-permutacija p | Podgrupa, Wilcoxon p |
|---|---:|---:|---:|
| RRF-4 | 0,517 (n.z.) | 0,044 (značajno) | 0,047 (značajno) |
| RRF-5 (+ LSE signal, poglavlje 4.4) | nije poboljšano naspram RRF-4 | 0,131 (n.z.) | 0,156 (n.z.) |
| RRF-6 (+ MLP(Hadamard) signal) | 0,168 (n.z., ali bliže značajnosti) | 0,009 (značajno) | nije izračunato |

Dodavanje LSE signala (interno LOCO-potvrđenog dobitka za nsLTP/Profilin, poglavlje 4.4) u RRF-5 pogoršalo je rezultat na pacijentima; dodavanje MLP(Hadamard) signala u RRF-6 ga je poboljšalo. Ovaj kontrast je sam po sebi nalaz: interni (LOCO) dobitak signala ne predviđa pouzdano njegov doprinos na pacijentskom skupu, dalje razmotreno u poglavlju 5.3.

Direktno poređenje MLP(Hadamard) i BLAST signala, svaki samostalno bez RRF fuzije, izvršeno je na identičnim pacijentskim upitima korišćenjem Wilcoxon testa, cluster-permutacije i patient-level bootstrap-a.

| Podskup                      | n (upiti / pacijenti) | Wilcoxon p | Cluster-permutacija p | Bootstrap 95% CI       |
| ---------------------------- | --------------------: | ---------: | --------------------: | ---------------------- |
| Svi upiti                    |              176 / 54 | **0,0116** |            **0,0304** | **[+0,0019, +0,0237]** |
| Hard, full-text verifikovano |              148 / 44 | **0,0042** |            **0,0080** | **[+0,0043, +0,0295]** |

MLP(Hadamard) je ostvario statistički značajno bolji rezultat od BLAST-a na oba podskupa. Isti obrazac dobijen je i sa ESM-2 3B baznim modelom.

**Cosine sličnost kao samostalan pacijentski signal.** Da bi se utvrdilo da li prednost MLP(Hadamard)-a nad BLAST-om potiče od same reprezentacije ili od naučene transformacije, testirana je i sirova cosine sličnost (bez treniranja) kao treći samostalan ranker, istim mehanizmom i na identičnom skupu upita:

| Poređenje | Podskup | Wilcoxon p | Cluster-permutacija p | Bootstrap 95% CI |
|---|---|---:|---:|---|
| Cosine vs. BLAST | Svi upiti | 0,7994 | **0,0205** | **[−0,0921, −0,0121]** |
| Cosine vs. BLAST | Hard | 0,4692 | **0,0057** | **[−0,1105, −0,0166]** |
| Cosine vs. MLP(Hadamard) | Svi upiti | **0,0172** | **0,0026** | **[−0,1042, −0,0214]** |
| Cosine vs. MLP(Hadamard) | Hard | **0,0009** | **0,0002** | **[−0,1248, −0,0301]** |

Cosine je najslabiji od sva tri signala na pacijentskom skupu, statistički značajno lošiji od MLP(Hadamard)-a (sva tri testa, oba podskupa) i značajno lošiji od BLAST-a (dva od tri testa). Ovim se precizira centralni nalaz rada: prednost nad BLAST-om ne potiče iz same ESM-2 reprezentacije, već specifično iz naučene, nelinearne Hadamard-MLP transformacije te reprezentacije. Kompletno rangiranje samostalnih signala na pacijentskom skupu je **MLP(Hadamard) > BLAST > cosine**.

### 4.6 Sažetak glavnih rezultata

| Model                   | Referentni skup podataka (LOCO)                    | Stvarni pacijenti              |
| ----------------------- | -------------------------------------------------- | ----------------------------- |
| Cosine                  | MRR = 0,1209                                       | Najslabiji signal, značajno lošiji od BLAST-a i MLP-a |
| BLAST                   | MRR = 0,1243                                       | Referentni signal             |
| **MLP(Hadamard), 650M** | **MRR = 0,1259, bez značajne razlike od BLAST-a**  | **Značajno bolji od BLAST-a** |
| MLP(Hadamard), 3B       | MRR = 0,1131 do 0,1136, značajno lošiji od BLAST-a | Značajno bolji od BLAST-a     |
| RRF-4                   | MRR = 0,1304                                       | Primarni test nije značajan   |

Najizraženija razlika između dve evaluacije javlja se kod MLP(Hadamard) modela. Na referentnom skupu podataka rezultat je statistički izjednačen sa BLAST-om, dok na nezavisnim slučajevima stvarnih pacijenata pokazuje statistički značajnu prednost.

### 4.7 Ablaciona studija: koji deo modela zaista doprinosi

Da bi se utvrdilo koja komponenta MLP(Hadamard) modela nosi najviše diskriminativnog signala, sprovedena je ablaciona studija na istom 57-pacijentskom skupu (176 upita, 54 pacijenta). Svaka komponenta zamenjena je pojednostavljenom alternativom, dok su ostale komponente ostale nepromenjene, i rezultat je poređen sa produkcionim polaznim modelom istom uparenom metodologijom kao u poglavlju 4.5.

**Arhitektonske komponente.** Testirane su tri zamene: (1) ESM-2 reprezentacija zamenjena aminokiselinskim sastavom proteina (20-dimenzioni vektor frekvencija, bez informacije o rasporedu ili motivima), (2) Hadamard kombinovanje para zamenjeno apsolutnom razlikom, (3) MLP klasifikator zamenjen linearnim modelom (logistička regresija nad istim Hadamard ulazom, bez skrivenih slojeva).

| Zamenjena komponenta | Δ MRR (bootstrap 95% CI) | Značajno? |
|---|---|---|
| ESM-2 reprezentacija → aminokiselinski sastav | −0,159 [−0,275, −0,059] | da, sva tri testa |
| Hadamard kombinovanje → apsolutna razlika | −0,055 [−0,099, −0,016] | da, dva od tri testa |
| MLP → linearni model | −0,009 [−0,020, +0,001] | ne, nijedan test |

Rezultati pokazuju izrazito neravnomeran doprinos komponenti. Zamena proteinske reprezentacije trivijalnim aminokiselinskim sastavom uništava najveći deo performansi modela, i taj pad se vidi već i na sopstvenom trening skupu (validaciona AUC pada sa 0,983 na 0,733), što znači da problem nije samo slabija generalizacija nego suštinski siromašnija reprezentacija. Izbor kombinovanja para (Hadamard naspram apsolutne razlike) doprinosi umereno, u skladu sa ranije opisanim LOCO nalazom (poglavlje 4.2), sada potvrđenim i na pacijentima. Nelinearnost MLP klasifikatora doprinosi zanemarljivo: linearni model nad istim Hadamard ulazom postiže statistički identičan rezultat. Redosled važnosti komponenti je jasan: reprezentacija je daleko najvažnija, zatim način kombinovanja para, dok arhitektonska složenost klasifikatora gotovo da nije bitna.

**Trening podaci: strože filtriranje nivoa dokaza.** Odvojeno je testirano da li ograničavanje trening skupa na najpouzdanije nivoe dokaza (Confirmed i Strong, ukupno 511 parova naspram 825 u produkcionom skupu, poglavlje 2.1) poboljšava rangiranje. Ovaj model (u daljem tekstu strict) nije se statistički razlikovao od polaznog modela na celom skupu upita, ali je zadržao značajnu prednost nad BLAST-om (Wilcoxon p = 0,0012, cluster-permutacija p = 0,0310, bootstrap CI [+0,0022, +0,0295]). U familijama sa dijagnostikovanim zagušenjem (nsLTP, Profilin, PR-10, poglavlje 4.5) strict model je jedini kandidat sa robustnim, statistički značajnim poboljšanjem u odnosu na polazni model (cluster-permutacija p = 0,0184, bootstrap CI [+0,0010, +0,0047]), pretežno zahvaljujući boljem potiskivanju negativa kod profilina (medijan percentil 34,3% naspram 66,2%, gotovo izjednačeno sa BLAST-ovih 66,9%) i nsLTP-a, dok se kod PR-10 rezultat pogoršao. Alternativni pristup, u kom su parovi iz kategorije Suspected zadržani u treningu ali sa smanjenom težinom u funkciji gubitka, dao je suprotan rezultat: model je bio statistički značajno lošiji od polaznog modela i izgubio je prednost nad BLAST-om. Detaljni rezultati oba pristupa, uključujući raspad po proteinskoj familiji, dati su u zasebnom dokumentu (`rad/ablacioni_test.md`).

## 5. Diskusija

### 5.1 Ograničenje dostupnih podataka

Rezultati u poglavlju 4 ukazuju da je veličina i struktura dostupnog referentnog skupa podataka jedan od glavnih ograničavajućih faktora za ceo zadatak, ne samo za pojedinačne modele. Ukupan broj poznatih parova (1.922) deluje relativno velikim, ali broj *nezavisnih* primera je znatno manji zbog dva efekta koja se preklapaju: povezanosti proteina unutar familija (mali broj proteina učestvuje u velikom broju parova, poglavlje 3.1) i koncentracije dokaza u malom broju literaturnih izvora (317 izvora za 1.922 para, poglavlje 2.1). Study-level bootstrap analiza direktno kvantifikuje posledicu ove strukture: dva od tri fuziona dobitka koja izgledaju značajno kada se parovi tretiraju kao nezavisni gube značajnost čim se nezavisnost proveri na nivou izvora (poglavlje 4.3). Ovo nije dokaz da su ti dobici lažni, tačke procene ostaju pozitivne, ali pokazuje da trenutni skup podataka nosi manje statističke snage nego što njegova nominalna veličina sugeriše.

Ovo ograničenje ima direktnu posledicu za dizajn budućih proširenja referentnog skupa podataka: dodavanje još parova iz već dobro zastupljenih izvora (npr. još jedna kohorta iz iste populacione studije) donosi manje nove statističke snage nego dodavanje jednog para iz potpuno novog, nezavisnog izvora. Prioritet bi trebalo da bude dodavanje eksperimentalno potvrđenih parova iz trenutno slabije zastupljenih proteinskih familija i parova sa manjom sekvencijskom sličnošću, ne prosto uvećanje broja parova.

### 5.2 Očekivanja od strukturnih reprezentacija

Rezultati u poglavlju 4.7 (komponentna ablacija) direktno informišu ovo pitanje: kada je ESM-2 reprezentacija zamenjena znatno siromašnijom (aminokiselinski sastav), pad performansi je bio drastičan; obrnuto pitanje, da li bogatija reprezentacija (npr. strukturna, dobijena iz AlphaFold ili OpenFold predikcije) donosi dodatni napredak, rad ne testira direktno, ali daje indirektan razlog za skepsu. Foldseek TM-score, strukturni signal uključen u RRF fuziju, nije samostalno testiran kao jedini signal u ovom radu, ali njegovo prisustvo u RRF-3 nije bilo dovoljno da RRF-3 naspram BLAST-a ostane značajno na nivou studije (poglavlje 4.3), što ukazuje da strukturna sličnost, bar u obliku globalnog TM-score poravnanja, ne nosi snažan nezavisan signal za ovaj zadatak.

Ovo je značajno zbog toga što je unakrsna reaktivnost, po definiciji iz poglavlja 1.1, fenomen koji zavisi od lokalnih, ne globalnih, strukturnih osobina: dostupnosti i konformacije konkretnih epitopa, ne ukupnog oblika proteina. Globalni TM-score, kao i globalni cosine nad mean-pooled reprezentacijom (poglavlje 2.5.1), agregira informaciju preko cele sekvence i time potencijalno razblažuje baš onaj lokalni signal koji bi bio najrelevantniji. LSE pooling nad lokalnim prozorima (poglavlje 2.4) (poglavlje 4.4) delimično podržava ovo tumačenje, pokazuje realan dobitak za dve od tri testirane familije, ali ne za sve, što znači da "lokalnija" reprezentacija nije univerzalno rešenje. Na osnovu ovoga, veći potencijal bi verovatno imalo kombinovanje strukturnih reprezentacija sa eksplicitnom informacijom o epitopima i površinskoj dostupnosti aminokiselinskih ostataka, a ne prosta zamena jednog globalnog modela reprezentacije drugim, globalnim ali strukturnim modelom.

### 5.3 Razlika između referentnog skupa podataka i pacijenata

Ovo je centralni empirijski nalaz rada (poglavlje 1.3, hipoteza H4) i zaslužuje detaljnije razmatranje mogućih objašnjenja, ne samo konstataciju da razlika postoji.

**Prvo moguće objašnjenje, razlika u distribuciji.** Referentni skup podataka sastavljen je od već poznatih, literaturno dokumentovanih parova, čiji je proces otkrića i publikovanja sam po sebi selektivan (poglavlje 2.1: parovi iz porodica sa dužom istorijom istraživanja su nadzastupljeni). Pacijentski slučajevi (poglavlje 3.3) predstavljaju širi, manje selektovan uzorak stvarne kliničke reaktivnosti, uključujući kombinacije alergena koje možda nikada nisu bile predmet posvećene naučne studije. Ako model uči obrasce specifične za to KAKO je referentni skup podataka konstruisan (npr. koje familije su dobro istražene), a ne obrasce same biologije unakrsne reaktivnosti, očekivano je da će se njegova relativna prednost promeniti kada se pređe na drugačije distribuiran skup upita.

**Drugo moguće objašnjenje, mehanizam signala.** BLAST po definiciji meri sekvencijalnu homologiju; njegova korisnost je direktno uslovljena time da li sekvencijalna sličnost prati unakrsnu reaktivnost za dati par. Ovo ne mora važiti za proteine koji dele relevantne lokalne ili strukturne osobine bez visoke globalne sekvencijalne sličnosti, upravo onaj slučaj panalergena pomenut u poglavlju 1.1. MLP(Hadamard) uči nelinearnu kombinaciju latentnih osobina ESM-2 reprezentacije (poglavlje 2.5.2), koja teorijski može kodirati takve odnose bez eksplicitnog poravnanja sekvenci. Ovo bi objasnilo zašto MLP(Hadamard) prednjači baš na pacijentskim slučajevima, koji verovatno sadrže veći udeo "netipičnih" (niska sekvencijalna sličnost, visoka klinička reaktivnost) parova nego kurirani referentni skup podataka.

**Treće, dokazano objašnjenje koje isključuje jednu alternativu.** Poglavlje 4.5 direktno testira da li je prednost MLP(Hadamard)-a nad BLAST-om posledica same ESM-2 reprezentacije (u kom slučaju bi i cosine, koji koristi istu reprezentaciju bez treniranja, morao pokazati sličnu prednost) ili posledica naučene transformacije. Cosine je na istom pacijentskom skupu bio najslabiji od sva tri signala, ne najbolji. Ovim se isključuje objašnjenje "ESM-2 reprezentacija sama po sebi bolje odgovara pacijentskim slučajevima"; prednost je specifično vezana za MLP-ovu naučenu, nelinearnu kombinaciju te reprezentacije, ne za reprezentaciju samu.

**Napomena o smeru efekta kod alternativnih enkodiranja.** Vredi eksplicitno primetiti da se ovaj obrazac (razlika između referentnog skupa podataka i pacijenata) ne kreće uvek u istom, "pacijenti su blagonakloniji" smeru. U ablacionoj studiji opisanoj u `rad/ablacioni_test.md`, kombinovanje para bogatijom reprezentacijom (spajanje sirovih vektora i njihove razlike, umesto Hadamard produkta) na ESM-2 3B reprezentaciji dalo je rezultat *bliži* BLAST-u na referentnom skupu podataka nego standardni Hadamard pristup, ali je na pacijentima bilo dosledno *lošije* i od BLAST-a i od produkcionog MLP(Hadamard) modela, suprotan smer od glavnog nalaza. Ovo pokazuje da razlika između dva konteksta evaluacije nije jednosmerna "pacijenti uvek nagrađuju composniju reprezentaciju" pravilnost, već da je svaka kombinacija (reprezentacija, enkodiranje, model) potrebno nezavisno proveriti na oba nivoa pre nego što se izvede zaključak o generalizaciji.

### 5.4 Metodološke implikacije fuzije signala i nezavisnosti uzoraka

Nalaz iz poglavlja 4.3, da RRF-3 naspram BLAST-a i RRF-4 naspram RRF-3 gube statističku značajnost pod study-level bootstrap-om, ima implikacije šire od ovog konkretnog skupa podataka. U mnogim biomedicinskim zadacima rangiranja i klasifikacije, "nezavisni" primeri u kuriranom skupu podataka zapravo dele zajednički izvor (istu kliničku studiju, istu kohortu, isti laboratorijski protokol), što krši pretpostavku nezavisnosti na kojoj se zasniva standardni bootstrap ili unakrsna validacija na nivou pojedinačnog primera. Kada se ta zavisnost ne kontroliše, prividna statistička značajnost može biti artefakt strukture skupa podataka, a ne stvarnog efekta modela. Praktična preporuka koja sledi iz ovog nalaza jeste da se svaka tvrdnja o poboljšanju u ovakvom kontekstu prijavljuje sa obe procene (na nivou primera i na nivou izvora), a ne samo sa onom koja daje povoljniji rezultat.

Sličan metodološki nalaz odnosi se i na poređenje modela na pacijentskom skupu (poglavlje 3.3, 4.5): dva odvojena testa "model iznad slučajnosti" nisu zamena za jedan pravi upareni test na istim primerima. Ova greška je lako napraviti, jer oba pojedinačna testa mogu izgledati ubedljivo kada se prijave jedan pored drugog, a ne dokazuju razliku između modela dok se ne sprovede pravo upareno poređenje.

### 5.5 Ograničenja

Rad ima nekoliko ograničenja koja utiču na to koliko se dobijeni nalazi mogu generalizovati.

**Proizvoljnost u konstrukciji pool-a kandidata.** Kao što je opisano u poglavlju 2.1, deduplikacija pool-a kandidata po identičnoj FASTA sekvenci koristi proizvoljno (alfabetsko) pravilo za biranje koji zapis preživljava kada više njih deli istu sekvencu; ovo je uklonilo, na primer, dva zvanično registrovana alergena (Pen a 1, Pen m 1) čija je sekvenca identična već zadržanom Lit v 1. Iako je posledica ovog konkretnog slučaja ublažena eksplicitnim mapiranjem imena, ne može se isključiti da postoje slični, još neotkriveni slučajevi koji utiču na broj dostupnih kandidata za retke familije. Pool takođe ne sadrži sistematsku proveru fragmenata (poglavlje 2.1); bar jedan potvrđen slučaj (Gly m 1) pokazuje da se u pool-u mogu naći nepotpune sekvence.

**Problem nepoznatih negativa.** Kao što je detaljno opisano u poglavlju 2.2, odsustvo para iz referentnog skupa podataka ne znači potvrđenu odsutnost unakrsne reaktivnosti. Nasumično uzorkovani "negativni" primeri korišćeni za treniranje mogu sadržati neotkrivene prave pozitive, posebno unutar gusto povezanih proteinskih familija. Ovo strukturno ograničenje ne može se u potpunosti otkloniti bez dodatnih eksperimentalnih podataka, i primenjuje se podjednako na sve nadgledane modele opisane u ovom radu, ne samo na MLP(Hadamard).

**Mali broj nezavisnih pacijentskih slučajeva.** Iako pacijentski skup (poglavlje 3.3) igra centralnu ulogu u glavnom nalazu rada, on ostaje mali u apsolutnom broju nezavisnih, uparenih pacijenata (54 do 57 u različitim analizama) u poređenju sa veličinom koja bi bila poželjna za pouzdanu procenu efekta u kliničkom kontekstu. Manji podskupovi korišćeni za analizu po proteinskoj familiji (poglavlje 4.5, 4.7) su još manji, i tačke procene za te podgrupe treba tumačiti sa dodatnim oprezom.

**Neravnomerna zastupljenost proteinskih familija.** I referentni skup podataka i pacijentski skup pokazuju veliku neravnomernost u zastupljenosti proteinskih familija (poglavlje 4.5: familije poput nsLTP i Profilina su brojno dobro zastupljene ali sistemski teže za rangiranje, dok su druge familije zastupljene sa premalo primera da bi se o njima moglo pouzdano zaključivati). Nalazi specifični za pojedinačne familije treba čitati kao ilustrativne, ne kao definitivne za tu familiju u opštem slučaju.

**Domet zaključaka o kliničkoj primeni.** Rezultati ovog rada pokazuju da predloženi pristup može korisno *rangirati* i *prioritizovati* kandidate za dalje alergološko testiranje, ne da može *dijagnostikovati* unakrsnu reaktivnost. Nijedan model opisan u ovom radu nije validiran kao dijagnostičko sredstvo, niti je za to dizajniran; sva izveštavanja o "prednosti" ili "tačnosti" odnose se isključivo na relativni kvalitet rangiranja kandidata, ne na apsolutnu kliničku pouzdanost pojedinačne predikcije. Svaka buduća primena ovakvog sistema u kliničkom kontekstu zahtevala bi odvojenu, prospektivnu validaciju koja izlazi iz okvira ovog rada.

### 5.6 Da li proteinske reprezentacije imaju budućnost u ovom zadatku?

Rezultati podržavaju korišćenje proteinskih reprezentacija (poglavlje 4.7: uklanjanje ESM-2 reprezentacije ima daleko najveći negativan efekat od svih testiranih izmena), ali ne podržavaju pretpostavku da veći ili izražajniji model automatski daje bolju predikciju. ESM-2 3B nije doneo poboljšanje u odnosu na 650M (poglavlje 4.2), jednostavnija Hadamard kombinacija para bila je korisnija od izražajnijeg bilinearnog modela (poglavlje 4.2, Prilog A), a čak i unutar same MLP(Hadamard) arhitekture, nelinearnost skrivenih slojeva doprinosi zanemarljivo u odnosu na linearni model (poglavlje 4.7).

Ovaj obrazac, doslednog izostanka koristi od povećanja kapaciteta modela ili reprezentacije, ponavlja se na tri nezavisna nivoa (veličina proteinskog jezičkog modela, izražajnost kombinovanja para, dubina klasifikatora) i sugeriše da za ovaj konkretan zadatak, sa ovom veličinom referentnog skupa podataka, način poređenja dve proteinske reprezentacije nosi više informacije nego sirovi kapacitet bilo kog pojedinačnog dela sistema. Ovo je u skladu sa opštim nalazom rada (poglavlje 1.4) da je doprinos sistematska evaluacija izvora signala, ne nov algoritam: rezultati sugerišu da bi dalji napredak pre trebalo tražiti u kvalitetu i strukturi podataka (poglavlje 5.1, 5.5) i u načinu na koji se predstavlja odnos između dva proteina (poglavlje 5.2), nego u zameni ESM-2 nekim većim ili složenijim proteinskim jezičkim modelom.

## 6. Budući rad

Najvažniji pravci daljeg rada su proširenje referentnog skupa podataka novim nezavisnim eksperimentalnim dokazima, povećanje broja nezavisnih pacijentskih slučajeva i uključivanje biološki relevantnih informacija o epitopima i strukturi. Posebno bi bilo važno proveriti da li se prednost MLP(Hadamard) modela održava na većem i raznovrsnijem skupu pacijenata.
Ukupno, rezultati pokazuju da proteinske reprezentacije mogu sadržati signal relevantan za predikciju unakrsne reaktivnosti, ali da napredak verovatnije zavisi od kvaliteta podataka i načina reprezentacije odnosa između proteina nego od samog povećavanja modela.

Krajnji cilj ovakvog sistema mogao bi biti razvoj asistivnog alata koji, na osnovu poznatih alergija pacijenta, rangira potencijalno unakrsno reaktivne alergene i predlaže prioritete za dalje alergološko testiranje (poglavlje 5.5, domet ovakve primene ograničen je na rangiranje i prioritizaciju, ne dijagnozu). U proširenoj verziji sistem bi mogao da koristi poznate pozitivne i negativne nalaze za personalizovano rangiranje novih kandidata. Takav sistem ne bi zamenio kliničku procenu, već bi služio kao pomoć pri izboru prioriteta za dalje testiranje.

## Prilog A: Dodatni eksperimenti i detalji

Ovaj prilog sadrži modele i konfiguracije koje nisu ušle u završnu konfiguraciju opisanu u glavnom tekstu, uključene ovde radi potpunosti i reproducibilnosti, ne zato što su centralne za glavni nalaz rada.

### A.1 PU Bagging i XGBoost

**PU Bagging (Random Forest + BLAST).** Ansambl Random Forest klasifikatora treniranih u Positive-Unlabeled bagging režimu (poglavlje 2.2) nad istim BLAST + reprezentacija obeležjima kao osnovni Random Forest model. Na ranijoj, jednostrukoj podeli podataka (pre uvođenja LOCO protokola u ovaj rad) ostvario je MRR = 0,2038, viši od svih ostalih klasičnih metoda, ali ovaj rezultat nije potvrđen pod LOCO protokolom (poglavlje 3.1); ponovljena provera pod LOCO dala je mešovite, nekonzistentne rezultate po komponenti, zbog čega broj nije uključen u glavnu tabelu poređenja (tabela u poglavlju 4.1) kao direktno uporediv.

**XGBoost + BLAST.** Model gradijentno pojačanih stabala odluke (gradient boosting) nad istim skupom obeležja kao Random Forest + BLAST. Pod LOCO protokolom ostvario je MRR = 0,1139, dosledno lošije od Random Forest varijante sa istim obeležjima, bez dodatnih podešavanja hiperparametara van podrazumevanih vrednosti biblioteke.

### A.2 Sensitivity sweep: MLP nad apsolutnom razlikom

Osam konfiguracija MLP klasifikatora nad apsolutnom razlikom reprezentacija (poglavlje 2.5.2), sa različitim kombinacijama veličine skrivenih slojeva, dropout regularizacije i prisustva/odsustva dodatne cosine karakteristike, testirano je pod LOCO protokolom. MRR se kretao u opsegu 0,1060 do 0,1737 u zavisnosti od konfiguracije; u svih osam slučajeva, rezultat je bio lošiji od polaznog modela na istoj podeli podataka. Nijedna kombinacija hiperparametara nije promenila ovaj kvalitativni zaključak, što je motivisalo prelazak na Hadamard produkt kao alternativnu reprezentaciju para (poglavlje 2.5.2).

### A.3 Bilinearni model, pun izvod

Bilinearni skor za par reprezentacija $u$, $v$ definiše se kao

$$
s = u^T W v,
$$

gde je $W\in\mathbb{R}^{d\times d}$ naučena matrica parametara, $d$ dimenzionalnost reprezentacije. Zbog veličine ESM-2 reprezentacije ($d=1280$ za 650M model), puna matrica $W$ bi imala preko 1,6 miliona parametara, znatno više od broja nezavisnih trening primera (poglavlje 2.1); zbog toga je korišćena niskorangna (low-rank) faktorizacija koja prvo projektuje $u$ i $v$ u prostor manje dimenzionalnosti, pa tek onda računa bilinearnu interakciju u tom manjem prostoru.

Pod LOCO protokolom, ovaj model je ostvario MRR = 0,1004, statistički značajno lošije od polaznog modela (poglavlje 4.2), i lošije od MLP(Hadamard) uz veći broj parametara i veći rizik od preprilagođavanja. Zaključak je da veća izražajnost modela interakcije para, sama po sebi, ne nadoknađuje ograničenu veličinu referentnog skupa podataka, nalaz konzistentan sa opštim obrascem opisanim u poglavlju 5.6.

### Zahvalnica

Želim da se zahvalim svom mentoru Stefanu Nožiniću na stručnom vođstvu, savetima i kontinuiranoj podršci tokom razvoja ovog istraživanja.

Posebnu zahvalnost dugujem Mariji Stefanović na pomoći u razumevanju biološke pozadine problema, savetima u vezi sa alergenima i korisnim komentarima tokom rada.

Zahvaljujem se i svim osobama koje su ustupile svoje rezultate alergoloških testiranja, čime su omogućile nezavisnu validaciju modela na stvarnim slučajevima i značajno doprinele ovom istraživanju.


