---
bibliography: references.bib
---

# Šta model uparenih vrednosti baziran na ESM-2 uči o unakrsnoj reaktivnosti proteina?

Lana Lejić

Mentor: Stefan Nožinić

---

### Apstrakt

Unakrsna alergijska reaktivnost između proteina ne može se pouzdano opisati jednom merom sličnosti sekvenci. Ovaj rad ispituje da li reprezentacije proteinskog jezičkog modela ESM-2 sadrže signal povezan sa unakrsnom reaktivnošću koji nije obuhvaćen klasičnim metodama poređenja sekvenci. Na skupu proteinskih alergena, ESM-2 reprezentacije korišćene su za učenje modela koji rangira potencijalno unakrsno reaktivne partnere.

Sama kosinusna sličnost ESM-2 embeddinga nije nadmašila BLAST. Međutim, nadgledani model zasnovan na kombinovanju reprezentacija dva proteina pokazao je komplementaran signal. Najveće poboljšanje u odnosu na BLAST javlja se kod kandidata za koje BLAST daje slabiji signal. Na nezavisnim pacijentskim slučajevima ESM-2 model bolje rangira pozitivne kandidate, dok BLAST bolje razlikuje negativne kandidate.

Rezultati pokazuju da ESM-2 reprezentacije mogu pružiti informaciju koja dopunjuje klasičnu sličnost sekvenci. Kombinovanje ova dva izvora informacije može biti korisno za prioritizaciju kandidata za dalje eksperimentalno ispitivanje.

### Ključne reči

unakrsna reaktivnost alergena; proteinski jezički modeli; ESM-2; Hadamardov proizvod; LOCO validacija; BLAST


# 1. Uvod

### 1.1. Unakrsna reaktivnost kao problem reprezentacije proteina

Unakrsna alergijska reaktivnost predstavlja sposobnost IgE antitela da prepoznaju homologne proteine iz različitih alergenih izvora [@aalberse2001cross; @eaaci2022guide]. Ovaj fenomen nastaje zbog očuvanih molekularnih karakteristika između proteina, ali nije određen isključivo njihovom sekvencijalnom sličnošću. Proteini sa relativno niskim identitetom sekvence mogu izazivati unakrsnu reaktivnost, dok visoka sekvencijalna sličnost sama po sebi ne predstavlja dovoljan uslov za zajedničko imunološko prepoznavanje.

Zbog toga se predikcija unakrsne reaktivnosti može posmatrati kao problem reprezentacije proteina: potrebno je pronaći prikaz proteinske sekvence koji zadržava informacije relevantne za funkcionalnu i imunološku srodnost, a ne samo za evolutivnu sličnost.

Postojeći bioinformatički resursi za alergene baze podataka kao što su WHO/IUIS Allergen Nomenclature [@allergen2026who], Allergome [@allergome2026], AllFam [@meduniwien2026allfam], SDAP [@sdap2026database], COMPARE [@hesi2026compare] i AllergenOnline [@allergenonline2026], kao i alati poput Allermatch-a [@allermatch2026] prvenstveno se oslanjaju na kurirane zapise i jednostavna pravila sekvencijalne sličnosti (pragove procenta identiteta i dužine preklapanja) za procenu potencijalne unakrsne reaktivnosti.

**Istraživački jaz.** Iz navedenog proizlaze tri konkretna nedostatka koje ovaj rad adresira: 
1) nijedan od postojećih resursa sistematski ne ispituje da li naučene reprezentacije proteinskog jezičkog modela (ESM-2) nose informaciju o unakrsnoj reaktivnosti komplementarnu sekvencijalnom poravnanju (BLAST); 
2) nijedan to ne čini uz protokol koji eksplicitno kontroliše curenje informacija kroz povezane proteine, umesto nasumične podele parova (LOCO, 2.5.1); 
3) nijedan ne testira da li se takav signal, ako postoji, prenosi na nezavisne, dokumentovane slučajeve pacijenata, a ne samo na kurirani literaturni skup. Ovaj rad se bavi upravo tom prazninom: da li i kada embedding-based signal nadmašuje ili dopunjuje klasično sekvencijalno poravnanje.

### 1.2. Jezički modeli za proteine i naučene reprezentacije proteina

Proteinski jezički modeli uče reprezentacije proteina treniranjem nad velikim skupovima aminokiselinskih sekvenci bez eksplicitnih anotacija o njihovoj funkciji ili strukturi. Umesto ručno definisanih osobina, model svakoj sekvenci dodeljuje kontekstualnu vektorsku reprezentaciju (embedding) koja sažima obrasce naučene iz prirodne distribucije proteinskih sekvenci.

Takve reprezentacije organizuju proteine u višedimenzionalnom prostoru u kojem geometrijski odnosi između embeddinga mogu odražavati različite biološke osobine. Centralno pitanje ovog rada jeste da li embedding prostor ESM-2 modela sadrži informacije relevantne za unakrsnu reaktivnost koje nisu trivijalno dostupne iz klasičnih mera sekvencijalne sličnosti.

### 1.3. Od pojedinačnih reprezentacija do predikcije parova proteina

Ideja rada nije klasifikacija pojedinačnih proteina, već predikcija odnosa između dva proteina. Zbog toga kvalitet modela ne zavisi samo od reprezentacije svakog proteina pojedinačno već i od načina na koji se dve reprezentacije kombinuju u zajedničku reprezentaciju para (pair representation).

Različite strategije enkodiranja parova mogu naglasiti različite tipove odnosa između embeddinga, pa izbor pairwise encoding-a postaje sastavni deo modelovanja, a ne samo tehnički detalj ulaza u klasifikator.

### 1.4. Istraživačka pitanja

Rad ispituje četiri istraživačka pitanja:

* **RQ1.** Da li embedding reprezentacije dobijene modelom ESM-2 sadrže informacije korisne za predikciju unakrsne reaktivnosti izvan jednostavne sekvencijalne sličnosti?
* **RQ2.** Kako način kombinovanja dve ESM-2 reprezentacije utiče na sposobnost modela da prepozna unakrsno reaktivne parove proteina?
* **RQ3.** Da li poboljšanje performansi potiče prvenstveno od nelinearnog klasifikatora ili od izbora reprezentacije i pairwise encoding-a?
* **RQ4.** Koje osobine naučenog embedding prostora objašnjavaju ponašanje modela i kada embedding pruža informacije komplementarne sekvencijalnim metodama poput BLAST-a?

---
# 2. Metodologija

## 2.1. Skup podataka o unakrsnoj reaktivnosti
### 2.1.1. Skup kandidata proteina

Skup kandidata formiran je iz WHO/IUIS Allergen Nomenclature baze [@allergen2026who] i obuhvata proteinske alergene za koje su bile dostupne odgovarajuće aminokiselinske sekvence. Nakon uklanjanja nevalidnih sekvenci, sekvenci kraćih od 30 aminokiselina, potpunih duplikata i proteina bez validnog ESM-2 embeddinga, konačni skup sadrži **1.535 proteinskih alergena**. Izoforme sa različitim aminokiselinskim sekvencama tretirane su kao zasebni kandidati.

### 2.1.2. Kurirani unakrsno reaktivni parovi

Poznati odnosi unakrsne reaktivnosti prikupljeni su iz objavljene naučne literature i povezani sa proteinima iz konačnog skupa kandidata. Ukupno je identifikovano 1.916 jedinstvenih parova koji obuhvataju 477 alergena i 317 literaturnih izvora. Za svaki par zabeležen je izvor dokaza i nivo pouzdanosti, a kada je bio dostupan i pripadnost proteinskoj familiji.

Parovi su klasifikovani u četiri nivoa dokaza: Confirmed (138), Strong (377), Suspected (275) i Inferred (1.089), što čini 1.879 parova sa pozitivnim dokazom. Preostalih 37 parova nisu pozitivni dokazi: 36 su "Reported negative" (dokumentovan nalaz odsustva unakrsne reaktivnosti, korišćen kao kontrola) i 1 "Risky/Contested" (sporna, kontradiktorna evidencija). Inferred parovi (unakrsna reaktivnost izvedena iz homologije/familije, bez direktnog eksperimentalnog dokaza) nisu korišćeni kao pozitivni trening primeri, ali su zadržani u evaluaciji.

**Od kandidata do parova — tok podataka.** Dva broja koja se pojavljuju u radu (1.535, 477) opisuju dva različita, ugnježdena skupa, ne dve verzije istog skupa podataka:

```
1.535 kandidata sa embeddingom (WHO/IUIS, posle čišćenja)  →  CEO candidate pool za rangiranje (BLAST/cosine/MLP)
        │  samo proteini koji se pojavljuju u BAR JEDNOM kurirano potvrđenom paru
        ▼
   477 alergena  →  jedini proteini koji ikad figurišu kao POZITIVAN primer (trening ili evaluacija)
```

Preostalih ~1.058 kandidata (1.535 − 477) nikada se ne pojavljuju kao pozitivan par. Služe samo kao distraktori u kandidatskom pool-u i kao izvor za uzorkovanje negativa (2.1.3), čineći zadatak realističnim (model mora da izdvoji tačnog partnera iz mnogo više kandidata nego što ima poznatih pozitivnih odnosa).

### 2.1.3. Pozitivno-neobeleženo okruženje i uzorkovanje negativa

Skup podataka ima karakteristike positive-unlabeled (PU) problema: činjenica da par nije zabeležen u literaturi ne znači da je eksperimentalno potvrđeno odsustvo unakrsne reaktivnosti, pa se odsustvo iz kuriranog skupa ne može direktno tumačiti kao negativna klasa. Negativni primeri za trening zato su uzorkovani iz parova van kurirane pozitivne relacije, nezavisno od proteinskih familija (da se ne uvede implicitna familijska pretpostavka), u fiksnom odnosu **1 pozitivan : 10 negativnih**.

Pod LOCO protokolom (2.5.1), negativni parovi za svaki fold uzorkuju se isključivo iz proteina TRENING skupa tog folda. Nijedan protein iz test komponente ne učestvuje u konstrukciji trening negativa i generišu se nezavisno (novo seme) za svaki fold, bez deljenog fiksnog skupa negativa.

## 2.2. ESM-2 vektorske reprezentacije (embeddings) proteina

### 2.2.1. ESM-2 model

Za generisanje vektorskih reprezentacija proteinskih sekvenci korišćen je **ESM-2 (Evolutionary Scale Modeling 2)** proteinski jezički model, koji uči kontekstualne reprezentacije amino kiselina na osnovu njihovog položaja i konteksta u sekvenci. Kao primarni model korišćen je **ESM-2 650M** (~650M parametara); **ESM-2 3B** (~3B parametara) korišćen je u kontrolnom eksperimentu za ispitivanje uticaja veličine modela na dobijene reprezentacije.

### 2.2.2. Reprezentacije po aminokiselini i agregacija srednjom vrednošću (mean pooling)

Za svaku aminokiselinu u ulaznoj sekvenci dužine $L$ ESM-2 generiše kontekstualni vektor $h_i$. Pošto modeli mašinskog učenja zahtevaju reprezentaciju fiksne dimenzionalnosti, per-residue reprezentacije agregirane su primenom **mean pooling-a**: 

$u = \frac{1}{L}\sum_{i=1}^{L} h_i$. 

Na ovaj način svaka proteinska sekvenca predstavljena je jednim vektorom $u$. za ESM-2 650M dimenzionalnosti **1280**, za ESM-2 3B odgovarajućom (većom) izlaznom dimenzionalnošću.

## 2.3. Reprezentacija parova dva proteina

Pojedinačni embeddingi $u$ i $v$ opisuju proteine zasebno. Za predikciju unakrsne reaktivnosti potrebno je iz njih formirati reprezentaciju koja opisuje njihov međusobni odnos. U radu su ispitana dva načina enkodiranja para zasnovana na direktnim operacijama nad embedding prostorom.

### 2.3.1. Enkodiranje apsolutnom razlikom (Absolute-difference encoding)

Reprezentacija para apsolutnom razlikom definisana je kao 

$x = |u-v|$

, svaka komponenta opisuje udaljenost proteina duž odgovarajuće dimenzije reprezentacije. Enkodiranje je simetrično u odnosu na redosled proteina — zamenom $u$ i $v$ dobija se ista reprezentacija.

### 2.3.2. Hadamardov proizvod (Hadamard encoding)

Hadamardov proizvod definisan je kao 

$x = u \odot v$

,odnosno $x_i = u_i v_i$ po komponenti. Poelementno množenje odgovarajućih komponenti dve proteinske reprezentacije. Svaka dimenzija tako predstavlja zajedničku aktivaciju odgovarajuće latentne osobine kod oba proteina, čime se eksplicitno uvodi interakcija između njihovih naučenih osobina. Kao i apsolutna razlika, Hadamardov proizvod je simetričan prema zameni proteina $u$ i $v$.

### 2.3.3. Kontrola složenijih interakcija

Kao kontrolni eksperiment (da li eksplicitno modelovanje složenijih interakcija pruža dodatnu informaciju u odnosu na element-wise enkodiranje) ispitan je i bilinearni pristup zasnovan na spoljašnjem proizvodu:

$X = uv^\mathsf{T}$

,koji modeluje interakciju svake dimenzije jednog proteina sa svakom dimenzijom drugog. Zbog velike dimenzionalnosti punog outer product-a ($1280\times1280$) korišćena je low-rank parametrizacija (projekcija na 64 dimenzije). Zbog velike dimenzionalnosti i slabije stabilnosti nije uključen u završni model.

## 2.4. Prediktivni modeli

### 2.4.1. MLP(Hadamard)

Za predikciju unakrsne reaktivnosti korišćen je višeslojni perceptron (MLP) nad Hadamardovom reprezentacijom para: nelinearna funkcija 

$\hat{y} = f_{\theta}(x)$ 

koja mapira reprezentaciju para u verovatnoću unakrsne reaktivnosti. Model je treniran minimizacijom binarne unakrsne entropije (BCE) koristeći AdamW, uz dropout, weight decay i early stopping na validacionoj AUC.

Finalna arhitektura: 

$\mathrm{Hadamard}(1280){\to}\mathrm{Linear}(1280{\to}32){\to}\mathrm{ReLU}{\to}\mathrm{Dropout}(0{,}3){\to}\mathrm{Linear}(32{\to}1)$. 

Ulazne dimenzije Hadamard proizvoda nisu standardizovane (z-score standardizacija je dijagnostikovano štetna za ovaj ulaz. Ona narušava prirodnu skalu proizvoda koja sama nosi deo signala). Svi hiperparametri su fiksirani pre završne evaluacije i identični za svih 40 foldova.

### 2.4.2. Kontrola linearnim klasifikatorom

Da bi se odvojio doprinos nelinearnosti klasifikatora od informacije sadržane u samoj reprezentaciji para, kao kontrolni model korišćena je logistička regresija nad **istim Hadamardovim ulazom** kao MLP, bez skrivenih nelinearnih slojeva: 
$\hat{y} = \sigma(w^\mathsf{T}x+b)$

, gde je $\sigma$ sigmoidna funkcija. Poređenjem ova dva modela uz isti ulaz ispituje se da li dodatna prediktivna sposobnost potiče od same Hadamard reprezentacije ili od sposobnosti MLP-a da nad njom modeluje nelinearne odnose.

### 2.4.3. Baseline modeli

**BLAST** [@ncbi2026blast]: sekvencijalni signal (2.7.1), bez treniranja; kandidati se rangiraju direktno po BLAST skoru poravnanja upit-kandidat sekvenci.

**Cosine**: kosinusna sličnost ESM-2 embeddinga ($u$, $v$) bez ikakve naučene transformacije. Netreniran signal nad istom reprezentacijom koju koristi MLP(Hadamard), koristi se da se odvoji doprinos same reprezentacije od doprinosa treninga.

**ESM-2 3B backbone**: identičan MLP(Hadamard) klasifikator i protokol kao primarni model (2.4.1), jedina razlika je backbone (ESM-2 3B umesto 650M, veća izlazna dimenzionalnost embeddinga)

**Bilinearni (low-rank outer product)**: opisan u 2.3.3; treniran istim postupkom (BCE, AdamW, early stopping) kao MLP(Hadamard), ulaz je projektovan outer product umesto Hadamard proizvoda.

## 2.5. Protokol validacije

Glavna izveštavana metrika je **Mean Reciprocal Rank (MRR)**: za svaki upit $q$ model rangira kandidate

$\mathrm{rr}(q) = 1/r(q)$ gde je $r(q)$ rang tačnog kandidata; 
$\mathrm{MRR} = \mathrm{mean}_q(\mathrm{rr}(q))$, mikro-prosek preko svih pojedinačnih upita (ne makro-prosek po grupi/proteinu), osim gde je eksplicitno drugačije navedeno. 

AUC (korišćena u 3.1 i za early stopping u 2.4.1) je standardna površina ispod ROC krive validacionog skupa binarnog klasifikatora, nezavisna od MRR-a.

### 2.5.1. Validacija izostavljanjem povezane komponente (Leave-One-Connected-Component-Out — LOCO)

Random podela parova na trening i test skup nije odgovarajuća zbog povezanosti proteina kroz mrežu unakrsne reaktivnosti: ako su dva proteina u istoj komponenti, njihovo razdvajanje između treninga i testa može omogućiti modelu da indirektno iskoristi informacije iz test komponente. Zbog toga je evaluacija sprovedena metodom **Leave-One-Connected-Component-Out (LOCO)**. Ovde se jednom i eksplicitno definiše koji nivo dokaza (2.1.2) ulazi u graf, u trening i u evaluaciju.

- **Graf i povezane komponente.** Čvorovi grafa su svi proteini iz kandidatskog skupa (2.1.1). Grane grafa su svi parovi sa **pozitivnim dokazom bilo kog nivoa** Confirmed, Strong, Suspected i Inferred "Reported negative" i "Risky/Contested" parovi (37 parova) **nikada nisu grane** i ne utiču na povezanost komponenti. Povezane komponente ovog grafa čine 40 foldova. U svakoj iteraciji jedna komponenta se izostavlja iz treninga i koristi kao test skup, a model se trenira isključivo nad preostalim komponentama.
- **Trening (pozitivni primeri).** Unutar trening dela svakog folda, kao pozitivni primeri koriste se samo parovi nivoa Confirmed, Strong i Suspected. Inferred parovi **nisu** pozitivni trening primeri, iako jesu grane grafa koje su odredile sam oblik komponenti/foldova.
- **Evaluacija (ispravni ciljevi rangiranja).** Test skup svakog folda koristi **isti, širi skup grana kao i graf** Confirmed, Strong, Suspected i Inferred kao ispravne (pozitivne) ciljeve pri računanju ranga $r(q)$. 

Posledica ove definicije: 1.089 Inferred parova oblikuje sastav 40 povezanih komponenti (i time koji proteini padaju u koji fold) i broje se kao ispravni evaluacioni ciljevi, ali nikada nisu pozitivan trening signal.

Rang $r(q)$ pod ovim protokolom računa se nad **celim** kandidatskim skupom (samo je sâm upit isključen). Ostali poznati pozitivni partneri upita nisu uklonjeni iz kandidatske liste pre rangiranja (rangiranje nije "filtrirano" u smislu standardnog knowledge-graph-completion protokola). Ovo je namerno povezano sa analizom "zagušenja" (crowding) u 3.7.2–3.7.3: proteini iz familija sa mnogo međusobno pozitivnih parova mogu imati niži $r(q)$ zbog konkurencije drugih tačnih odgovora, ne zbog lošijeg modela; pacijentski protokol (2.5.2). Nasuprot tome poznati nalazi istog pacijenta eksplicitno se isključuju iz kandidatske liste pre rangiranja. Ovakva podela omogućava procenu generalizacije na proteinske odnose koji nisu povezani sa primerima dostupnim tokom treniranja.

### 2.5.2. Nezavisna evaluacija na nivou pacijenata

Generalizacija modela dodatno je ispitana na nezavisnim podacima iz dokumentovanih slučajeva pacijenata koji nisu korišćeni za treniranje. Za svakog pacijenta model rangira kandidate na osnovu dostupnih pozitivnih nalaza, a rezultat se procenjuje na kandidatima čiji status nije korišćen za formiranje upita. Time se proverava da li se naučeni odnos prenosi na podatke nezavisne od kuriranog trening skupa.

Svi pacijentski slučajevi transkribovani su iz prethodno objavljenih, javno dostupnih kliničkih izveštaja (citirani izvor po slučaju); nijedan slučaj nije simuliran niti direktno prikupljen od autora. Pošto je reč isključivo o sekundarnoj analizi već objavljenih, neidentifikovanih podataka, nije bilo potrebno posebno etičko odobrenje.Y
<!-- ???????? jel moze ovako -->

### 2.5.3. Statističko testiranje i bootstrap intervali poverenja

Poređenja modela sprovedena su na istim upitima (upareni uslovi), korišćenjem **Wilcoxon signed-rank test-a** (ne zahteva normalnost razlika). Na pacijentskom nivou (2.5.2) korišćen je i **cluster-permutacioni test** (10.000 permutacija, "klaster" = sve probe jednog pacijenta permutovane zajedno) kao nezavisna, robusnija provera koja poštuje zavisnost proba unutar pacijenta. Bootstrap intervali poverenja računati su odvojeno. Uzorkovanje na nivou literaturnog izvora ili pacijenta, umesto pojedinačnih parova, kad god je struktura zavisnosti to zahtevala i interpretirani zajedno sa veličinom razlike, ne samo p-vrednošću.

## 2.6. Mehanističke analize

Nakon evaluacije prediktivnih performansi, sprovedene su dodatne analize sa ciljem da se ispita koje karakteristike embedding prostora doprinose ponašanju modela.

### 2.6.1. Stabilnost odabranih dimenzija embeddinga kroz različita slučajna semena

Stabilnost važnih dimenzija ispitana je ponavljanjem treniranja sa 5 različitih slučajnih semena i merenjem preklapanja top-k skupova dimenzija (po $|$težini$|$) **Jaccardovim indeksom** 
$J(A,B)=|A\cap B|/|A\cup B|$. 

Visoko preklapanje ukazuje na stabilan izbor dimenzija.

### 2.6.2. Povezanost dimenzija embeddinga sa biohemijskim i strukturnim deskriptorima

Da bi se ispitalo da li pojedinačne dimenzije imaju prepoznatljivu biohemijsku ili strukturnu interpretaciju, njihove vrednosti korelisane su sa poznatim osobinama proteina (dužina, naboj, hidrofobnost, sekundarna struktura, aminokiselinski sastav i drugo. Ove analize su **post-hoc asocijacije**: ne predstavljaju dokaz da određena osobina uzrokuje prediktivni signal modela.

### 2.6.3. Orezivanje dimenzija (Dimension pruning)

Ispitano je da li dimenzije koje pojedinačno pokazuju diskriminativnu sposobnost (Cohenov $d$, računat po train-fold-u) mogu zajedno da reprodukuju ponašanje punog modela, zadržavanjem top 50% najbolje rangiranih dimenzija po foldu (ostatak nuliran; pun protokol u Supplementary S2.6.3). Cilj je utvrditi da li je prediktivni signal koncentrisan u malom broju dimenzija ili zahteva širu reprezentaciju.

### 2.6.4. Test međudimenzionalnih interakcija

Da bi se direktno ispitalo da li model koristi interakcije između različitih dimenzija, poređen je aditivni logistički model sa modelom koji dodaje eksplicitan proizvod između para dimenzija, preko svih 861 para relevantnih dimenzija. Ovim poređenjem testira se da li dodatno modelovanje međudimenzionalnih interakcija pruža informaciju koja nije već sadržana u pojedinačnim dimenzijama.

## 2.7. Komplementarnost sa BLAST-om

### 2.7.1. Osnovna linija rangiranja pomoću BLAST-a (BLAST ranking baseline)

BLAST je korišćen kao sekvencijalna baseline metoda: za svaki proteinski upit $q$ kandidati $c$ rangirani su direktno prema BLAST skoru poravnanja upit-kandidat sekvenci. Isti skup upita i kandidata koristi se za poređenje sa MLP modelom.

### 2.7.2. Dobitak MLP-a na nivou upita (Query-level MLP gain)

Da bi se ispitalo gde MLP pruža korist u odnosu na BLAST, za svaki upit izračunata je razlika recipročnih rangova:

$$
\Delta_q =
\frac{1}{r_{\mathrm{MLP}}(q)}
-
\frac{1}{r_{\mathrm{BLAST}}(q)},
$$

gde su $r_{\mathrm{MLP}}(q)$ i $r_{\mathrm{BLAST}}(q)$ rang tačnog kandidata prema MLP-u i BLAST-u. Pozitivna vrednost označava da je MLP bolje rangirao tačan kandidat; na ovaj način ukupna razlika u MRR-u može se analizirati na nivou pojedinačnih upita.

### 2.7.3. Stratifikacija prema jačini BLAST-a i analiza BLAST-slabih upita

Da bi se ispitalo da li doprinos MLP-a zavisi od dostupnosti sekvencijalnog signala, upiti su podeljeni u grupe prema uspešnosti BLAST-a (npr. kvartili BLAST rr), sa $\Delta_q$ analiziranim po grupi — posebna pažnja posvećena je upitima kod kojih BLAST ne obezbeđuje snažan signal (**BLAST-weak**). Unutar te grupe, upiti su dalje podeljeni na slučajeve gde je MLP nadmašio BLAST (**MLP-wins**) naspram slučajeva gde je BLAST ostao bolji (**MLP-loses**), i grupe su poređene prema karakteristikama proteinskih reprezentacija i rangiranja. Cilj je utvrditi da li MLP prvenstveno dopunjuje BLAST kada je sekvencijalni signal slab, i da li postoje sistematske karakteristike upita gde embedding-based model uspeva da nadomesti taj nedostatak.


# 3. Rezultati

## 3.0. Glavna tabela: svi modeli i baseline-ovi

**LOCO (zlatni skup, 40 foldova, mikro-MRR preko svih upita):**

| Model | MRR | Δ vs. BLAST | Značajno? |
|---|---:|---:|---|
| BLAST (2.7.1) | 0,1243 | referenca | — |
| Cosine, ESM-2 650M (bez treninga) | 0,1209 | −0,0034 | ne |
| **MLP(Hadamard), ESM-2 650M** (primarni model) | 0,1259 | +0,0016 | ne, CI uključuje nulu |
| MLP(Hadamard), ESM-2 3B | 0,1131–0,1136 | −0,0107 do −0,0112 | da, lošije |
| Bilinearni (low-rank outer product, 2.3.3) | 0,1004 | −0,0239 | da, lošije |

**Nezavisna validacija na pacijentima** (dve odvojene metrike — 3.8.1; senzitivnost/specifičnost, ne jedan MRR):

| Poređenje | MLP | BLAST | Δ | Wilcoxon p | Cluster-perm. p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| MRR₊ (senzitivnost, n=100/34 pac.) | 0,203 | 0,174 | +0,029 | 0,0156 | 0,0227 | [+0,0045, +0,0423] |
| NR₋ (specifičnost, n=76/37 pac.) | 0,441 | 0,599 | −0,158 | 0,0006 | <0,0001 | [−0,2135, −0,0787] |
| Cosine vs. BLAST (svi upiti) | — | — | — | — | 0,0205 | [−0,0921, −0,0121] |
| Cosine vs. MLP(Hadamard) (svi upiti) | — | — | — | — | 0,0026 | [−0,1042, −0,0214] |

Δ je definisana kao srednja **uparena razlika po pacijentu** (MLP−BLAST, makro-prosek preko 34, odn. 37 uparenih pacijenata), ne razlika agregatnih proseka u koloni MLP/BLAST — ista jedinica na kojoj su računati i bootstrap CI (resampling na nivou pacijenta) i cluster-permutacioni test (2.5.3).

### 3.0.1. Stabilnost glavnog LOCO rezultata preko 5 nezavisnih semena

Gornji LOCO rezultat za primarni model (0,1259) potiče od jednog semena (42). Da bi se proverilo da li je ta vrednost reprezentativna ili artefakt jednog seed-a, isti protokol (2.4.1) ponovljen je za 5 nezavisnih semena; BLAST referenca je determinstička (0,1243 za sve semenove):

| Seme | 42 | 137 | 271 | 314 | 500 | Mean (std) |
|---|---:|---:|---:|---:|---:|---:|
| LOCO MRR, MLP(Hadamard) 650M | 0,1254 | 0,1229 | 0,1258 | 0,1245 | 0,1162 | **0,1230 (0,0039)** |

Prosek preko 5 semena (0,1230) leži blago ispod BLAST reference (0,1243), unutar jednog standardnog odstupanja — konzistentno sa nalazom da razlika nije statistički značajna (CI uključuje nulu, gornja tabela). Ovo pojačava, a ne slabi, glavni zaključak: LOCO MRR MLP(Hadamard)-a naspram BLAST-a nije robusno pozitivan preko semena; komplementarnost sa BLAST-om po podgrupama upita (3.7) i na pacijentima (3.8) ostaje pouzdaniji, seed-nezavisan nalaz od same tačkaste LOCO MRR razlike.

*Napomena o preciznosti:* seme 42 u ovoj proveri (0,1254) blago odstupa od vrednosti 0,1259 citirane kao rezultat primarnog modela — razlika (0,0005) potiče od manje razlike u verziji skripte (`analysis/hadamard_standardize_multiseed_1548.py`, pisana radi ove provere, naspram originalnog `ml/loco_blast_vs_mlp_hadamard_only_1548.py`), a ne od promene protokola. Razlika je osam puta manja od standardnog odstupanja preko semena (0,0039) i ne menja nijedan zaključak.

Detaljni protokoli, dodatne konfiguracije (sweep apsolutne razlike, linearni klasifikator) i mehanističke analize dimenzija embeddinga dati su u nastavku (3.1–3.9).

## 3.1. ESM-2 pruža znatno bogatije prediktivne informacije od jednostavnog sastava

| Zamenjena komponenta | Protokol | Δ MRR (bootstrap 95% CI) | Značajno? |
|---|---|---|---|
| ESM-2 reprezentacija → aminokiselinski sastav (20-dim) | Pacijenti (176 upita/54 pac.) | −0,159 [−0,275, −0,059] | da, sva tri testa |

Zamena ESM-2 reprezentacije aminokiselinskim sastavom uništava najveći deo performansi modela; pad je vidljiv i na sopstvenom trening skupu (validaciona AUC 0,983→0,733), što isključuje objašnjenje da je reč samo o slabijoj generalizaciji — reprezentacija je suštinski siromašnija. Ovo je najveći pojedinačni efekat izmeren u celom radu.

## 3.2. Enkodiranje parova: Hadamard nadmašuje apsolutnu razliku

| Enkodiranje | Protokol | MRR / Δ MRR | Značajno? |
|---|---|---|---|
| Apsolutna razlika, 8 konfiguracija (sweep arhitekture: veličina skrivenih slojeva, dropout, weight decay) | LOCO, svaka konfig. naspram sopstvenog polaznog modela na istoj podeli | MRR 0,1060–0,1429, dosledno negativno | lošije u svih 8 konfiguracija |
| Hadamard → apsolutna razlika (ista arhitektura) | Pacijenti (176/54) | Δ = −0,055 [−0,099, −0,016] | da, 2 od 3 testa |

Nijedna testirana konfiguracija apsolutne razlike nije dostigla polazni model pod LOCO-om; ovaj nalaz je motivisao prelazak na Hadamard produkt. Prednost Hadamard enkodiranja potvrđena je nezavisno i na pacijentskom skupu.

## 3.3. Povećanje kapaciteta modela ne poboljšava performanse

### 3.3.1. Veća ESM-2 osnova (backbone) ne poboljšava nadgledani model

| Model | MRR | Protokol | Δ vs. BLAST | Značajno? |
|---|---:|---|---:|---|
| BLAST | 0,1243 | LOCO, čist trening | – | referenca |
| MLP(Hadamard), ESM-2 650M | 0,1259 | LOCO, čist trening | +0,0016 | ne, CI uključuje nulu |
| MLP(Hadamard), ESM-2 3B | 0,1131–0,1136 | LOCO, čist trening | −0,0107 do −0,0112 | da, značajno lošije |
| Cosine, ESM-2 3B prostor (bez treninga) | – | LOCO | −0,0007 vs. cosine 650M | ne |

### 3.3.2. Veći kapacitet interakcije parova ne poboljšava model

| Model | MRR | Protokol | Δ vs. BLAST (0,1243) | Δ vs. MLP(Hadamard) (0,1259) | Δ vs. cosine (0,1209, embedding baseline) | Značajno? |
|---|---:|---|---:|---:|---:|---|
| Bilinearni model (low-rank outer product, rang 64 — 2.3.3) | 0,1004 | LOCO | −0,0239 | −0,0255 | −0,0205 | da, značajno lošije |

### 3.3.3. Nelinearna MLP klasifikacija donosi malo u odnosu na linearni klasifikator

| Zamenjena komponenta | Protokol | Δ MRR (bootstrap 95% CI) | Značajno? |
|---|---|---|---|
| MLP → linearni model (logistička regresija, isti Hadamard ulaz) | Pacijenti (176/54) | −0,009 [−0,020, +0,001] | ne, nijedan test |

Nijedan od tri nezavisna oblika povećanja kapaciteta — veći jezički model, izražajnija reprezentacija para, dublji klasifikator — nije doneo merljivo poboljšanje; kod backbone-a i interakcije para efekat je značajno negativan. Redosled važnosti komponenti: kvalitet reprezentacije ≫ način kombinovanja para > dubina klasifikatora.

## 3.4. Model se oslanja na stabilan podskup dimenzija embeddinga

Sve analize u ovom odeljku sprovedene su na linearnom Hadamard modelu treniranom nad celim trening skupom (bez LOCO/pacijentskog holdout-a), preko 5 nezavisnih semena (42, 137, 271, 314, 500).

### 3.4.1. Stabilnost najbolje rangiranih dimenzija kroz slučajna semena (cross-seed stability)

| Skup | Prosečan Jaccard indeks preko 5 semena |
|---|---:|
| Top-20 dimenzija po \|težini\| | 0,80 |
| Top-50 dimenzija po \|težini\| | 0,77 |

17/20, odnosno 42/50 dimenzija pojavljuje se u top-skupu kod ≥4 od 5 semena — model dosledno koristi skoro isti mali podskup dimenzija, ne nasumičan izbor pri svakom treningu. Za poređenje, očekivani Jaccard indeks pri potpuno nasumičnom izboru dimenzija iz 1280 (dva nezavisna nasumična podskupa iste veličine) iznosi ≈0,008 za top-20 i ≈0,020 za top-50 (hipergeometrijska očekivana vrednost, potvrđeno Monte Carlo simulacijom) — opaženi Jaccard indeks je znatno veći od ove referentne, nasumične vrednosti, što pojačava tvrdnju o stabilnosti.

### 3.4.2. Pojedinačna diskriminativnost ne objašnjava u potpunosti upotrebu dimenzija u modelu

Preklapanje top-20/top-50 dimenzija po \|težini\| sa top-20/top-50 dimenzija po Cohenovom $d$ (unutar istog semena) iznosi svega 6–12%, dosledno preko svih 5 semena. Ovo ne protivreči umerenoj globalnoj korelaciji \|težina\|↔Cohen's $d$ (Pearson ≈0,41, Spearman ≈0,51, p<10⁻⁵⁸) — globalna korelacija preko 1280 dimenzija ne garantuje poklapanje u samom vrhu raspodele.

### 3.4.3. Orezivanje pojedinačno diskriminativnih dimenzija ne reprodukuje performanse

| Odsečena varijanta | Protokol | Δ (5 semena, upareno) | Pobeda odsečene varijante |
|---|---|---|---|
| Zadrži top 50% dimenzija po train-fold Cohen's $d$, ostatak nuliran | LOCO (40 folda) | mean Δ = +0,0003 (std 0,0037) | 1/5 semena |

Sa samo 5 semena i ovako malim efektom (mean +0,0003, std 0,0037), najpošteniji zaključak je: **nije detektovan merljiv efekat odsecanja preko ovih 5 semena**. Odsecanje po Cohen's $d$ uklanja i deo dimenzija koje model stvarno koristi (3.4.2), pa ne uspeva pouzdano da odvoji šum od signala; jača tvrdnja (u bilo kom pravcu) zahtevala bi više semena ili test ekvivalencije (npr. TOST), što nije sprovedeno.

## 3.5. Većina ključnih dimenzija prati merljiva svojstva proteina

Za svih ~1535 proteina u pool-u izračunata su realna biofizička i strukturna svojstva direktno iz FASTA sekvenci (dužina, GRAVY hidrofobnost, naboj na pH 7, aromatičnost, izoelektrična tačka, indeks nestabilnosti, udeo alfa-heliksa, pun aminokiselinski sastav — 27 pojedinačnih deskriptora ukupno), i korelisana (Spearman) sa vrednošću svake od 42 dimenzije (3.4.1) preko celog pool-a, u dve faze (3.5.1, 3.5.2; ukupno 630 testova). Na kompletan skup p-vrednosti primenjena je Benjamini–Hochberg FDR korekcija; 484/630 testova ostaje značajno na $q<0{,}05$.

### 3.5.1. Povezanost sa biohemijskim svojstvima

| Svojstvo | Broj  dimenzija (\|r\|>0,3) | Raspon \|r\| |
|---|---:|---|
| Naboj / izoelektrična tačka | 10 | 0,33–0,50 |
| Dužina proteina | 5 | 0,30–0,47 |
| Hidrofobnost (GRAVY) | 3 | 0,31–0,39 |
| Aromatičnost | 3 | 0,36–0,40 |
| Indeks nestabilnosti | 3 | 0,32–0,40 |

### 3.5.2. Povezanost sa sekundarnom strukturom i aminokiselinskim sastavom

Preostalih 18 dimenzija (bez korelata u 3.5.1) testirano je protiv udela sekundarne strukture i 20 aminokiselinskih frekvencija: 16/18 pokazuje \|r\|>0,3 (najjače: udeo alfa-heliksa, r=0,32–0,45; specifične aminokiseline — serin, triptofan, glicin, glutamin(ska kiselina), izoleucin, r=0,30–0,41).

### 3.5.3. Interpretacija i ograničenja analize latentnih dimenzija

Ukupno **40/42 (95%) relevantnih dimenzija** korelira (na FDR-korigovanom $q<0{,}05$) sa bar jednim od 27 testiranih realnih deskriptora. Preostale dve (nazvane po formalnom indeksu dimenzije) ne prate nijedno kontinuirano svojstvo, ali formalna provera (Mann–Whitney) pokazuje da svaka kodira kategorijsku pripadnost proteinskoj familiji (PR-10, p=1,4×10⁻¹⁷; Tropomyosin, p=4,1×10⁻⁹) — objašnjenje zašto ih korelacija sa kontinuiranim svojstvima nije uhvatila. Nijedna dimenzija ne prelazi \|r\|=0,5; ni jedna nije "čist" enkoder jedne osobine. Ovo su korelacione, ne uzročne asocijacije — ne dokazuju da navedena svojstva pokreću prediktivni signal modela.

## 3.6. Eksplicitne međudimenzionalne interakcije ne pružaju merljiv dodatni signal

Za svih $\binom{42}{2}=861$ parova relevantnih dimenzija, poređen je aditivni logistički model (dve dimenzije) sa modelom koji dodaje eksplicitan proizvod (interakcioni član), 5-strukom unakrsnom validacijom nad celim trening skupom (jedno seme, 42).

| Mera | Vrednost preko 861 parova |
|---|---:|
| Maksimalni dobitak od interakcionog člana (ΔAUC) | 0,0006 |
| Medijalni dobitak | ≈0,000004 |

Nijedan par ne pokazuje merljiv dobitak od eksplicitne interakcije — aditivna kombinacija dve dimenzije već sadrži skoro svu njihovu zajedničku prediktivnu informaciju. Ovo je nezavisna potvrda nalaza iz 3.3.3: interakcija koju Hadamard produkt nosi je ona ugrađena po konstrukciji (ista dimenzija, dva proteina), ne interakcija između različitih dimenzija unutar predstave. Za razliku od 3.4 (5 semena), ova analiza je sprovedena na jednom semenu — navedeno kao ograničenje obima, ne kao dokazano stabilan nalaz preko inicijalizacija.

## 3.7. BLAST-komplementarnost po jačini signala

MLP(Hadamard) 650M i BLAST su statistički izjednačeni pod LOCO-om u celini (0,1259 vs. 0,1243, tabela 3.3.1, CI uključuje nulu); sledeći odeljci pokazuju da ta ukupna izjednačenost krije jak, sistematski obrazac po podgrupama upita.

### 3.7.2. Dobitak MLP-a raste kako se performanse BLAST-a smanjuju

| Kvartil BLAST rr (LOCO, ne-crowded upiti, n=1928) | mean BLAST rr | mean MLP rr | Δ (MLP−BLAST) |
|---|---:|---:|---:|
| Q1 (najslabiji) | 0,026 | 0,055 | +0,029 |
| Q2 | 0,075 | 0,136 | +0,061 |
| Q3 | 0,175 | 0,200 | +0,025 |
| Q4 (najjači) | 0,607 | 0,411 | −0,196 |

Spearman(BLAST rr, Δ) = −0,404 (p=1,5×10⁻⁷⁶). MLP nadmašuje BLAST u donja tri kvartila (75% upita); u gornjem kvartilu, gde je BLAST već blizu maksimuma, MLP zaostaje. Familije sa dijagnostikovanim "zagušenjem" kandidata (nsLTP/Profilin/PR-10) imaju sistemski nizak BLAST rr (mean 0,052 naspram 0,193 za ostale familije) — crowding je najekstremniji, ne poseban, slučaj ovog istog obrasca.

**Metodološki oprez.** Pošto je $\Delta_q$ definisan kao razlika dva recipročna ranga koji su oba ograničena na $[0,1]$, deo negativne korelacije u Q4 je delimično i matematička posledica te definicije: kada je $r_{\mathrm{BLAST}}(q)$ već blizu 1 (BLAST rr blizu maksimuma), $\Delta_q$ ne može biti pozitivan bez obzira na MLP, jer nema više prostora na skali prema gore ("plafon"-efekat). Ovo ne poništava nalaz — obrazac je dosledan i u Q1–Q3, gde plafon-efekat nije aktivan a MLP i dalje sistematski dobija — ali znači da se sam koeficijent korelacije preko svih kvartila zajedno ne sme tumačiti kao čista mera komplementarnosti; nosilac dokaza je obrazac po kvartilima, ne jedan globalni Spearman broj.

### 3.7.3. Upitima sa slabim BLAST-om dominira konkurencija kandidata, a ne nužno nizak identitet sekvence

| Grupa (LOCO) | mean sequence identity % | % upita koji dodiruju crowded familiju |
|---|---:|---:|
| BLAST_jak (rang iznad medijane) | 60–62% | 15,6–27,1% |
| BLAST_slab (rang ispod medijane) | 48–53% | 73,8–81,9% |

BLAST_slab populacija ima umeren, ne nizak, sirov identitet — pada u srednji tercil sekvencijalne sličnosti definisan u 3.7.4/pacijentskoj stratifikaciji, ne u niski. Nizak *rang* BLAST-a je posledica konkurencije mnogo sličnih kandidata unutar iste familije, ne odsustva homologije.

"Crowded familija" ovde je operativno definisana kao fiksan, unapred određen skup od tri proteinske familije (nsLTP, Profilin, PR-10), identifikovan u odvojenoj analizi (raspon MRR-a 13× između familija) kao familije sa neuobičajeno velikim brojem međusobno pozitivnih parova. Ovo je dakle svojstvo familije kojoj upit pripada ("crowded familija"), a ne dinamička, po-upitna mera gustine kandidata ("crowded upit" u smislu "N kandidata unutar praga sličnosti X") — takva gušća, kontinuirana definicija nije korišćena.

### 3.7.4. Nijedan jednostavan biohemijski potpis ne razlikuje pobede MLP-a od njegovih poraza

| Svojstvo (unutar BLAST_slab, n=1086 pobeda MLP-a / 714 poraza) | p (Mann–Whitney) | Rank-biserial efekat |
|---|---:|---:|
| BLAST skor (sirov) | 2,8×10⁻⁵ | 0,117 (mali) |
| Naboj, hidrofobnost, aromatičnost, instabilnost, heliks (razlika para) | 0,37–0,80 | 0,01–0,03 (zanemarljivo) |

Ni sa velikom statističkom snagom (n=1800) nijedno testirano biofizičko svojstvo para ne razdvaja pobede od poraza MLP-a unutar BLAST-slabe zone; jedini (mali) signal je da MLP dodatno dobija kada je BLAST skor i unutar te zone niži — isti mehanizam iz 3.7.2, na finijoj rezoluciji, ne nov nezavisan signal.

## 3.8. Nezavisna validacija na nivou pacijenata

### 3.8.1. MLP(Hadamard) naspram BLAST-a

Skriveni nalaz (pozitivan ili negativan) evaluira se povratkom njegovog ranga; pošto je "uspeh" suprotno definisan za dva tipa proba (nizak rang za pozitiv, visok rang za negativ), izveštavaju se **dve odvojene, uparene metrike**, ne jedan kombinovan MRR:

$$\mathrm{MRR}_{+} = \mathrm{mean}(1/\mathrm{rang}), \quad \mathrm{NR}_{-} = \mathrm{mean}\left(\frac{\mathrm{rang}-1}{N-1}\right)\ (\text{veće} = \text{bolje potisnuto})$$

| Metrika | MLP | BLAST | Δ (uparena, po pacijentu) | Wilcoxon p | Cluster-perm. p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| MRR₊ (n=100 proba, 34 uparena pac.) | 0,203 | 0,174 | +0,029 | 0,0156 | 0,0227 | [+0,0045, +0,0423] |
| NR₋ (n=76 proba, 37 uparenih pac.) | 0,441 | 0,599 | −0,158 | 0,0006 | <0,0001 | [−0,2135, −0,0787] |

Δ je srednja razlika (MLP−BLAST) po pacijentu, makro-usrednjena preko uparenih pacijenata — ne prosta razlika prikazanih MLP/BLAST proseka u koloni levo (koje su mikro-proseci preko svih proba). MLP(Hadamard) značajno bolje prioritizuje prave unakrsno reaktivne partnere; BLAST bolje potiskuje prave negativne kandidate, efektom veće apsolutne i relativne veličine. Modeli su komplementarni specijalisti, ne jedan univerzalno superioran.

### 3.8.2. ESM kosinusna sličnost naspram MLP(Hadamard)

| Poređenje | Cluster-permutacija p | Bootstrap 95% CI |
|---|---:|---|
| Cosine vs. BLAST (svi upiti) | 0,0205 | [−0,0921, −0,0121] |
| Cosine vs. MLP(Hadamard) (svi upiti) | 0,0026 | [−0,1042, −0,0214] |

Cosine (netreniran signal nad istom reprezentacijom) je najslabiji od sva tri signala — prednost MLP(Hadamard)-a nad BLAST-om ne potiče iz same ESM-2 reprezentacije, već iz naučene Hadamard transformacije nad njom.

## 3.9. Sažetak nalaza

| Istraživačko pitanje | Eksperiment | Glavni nalaz |
|---|---|---|
| RQ1 | ESM naspram sastava (3.1) / cosine naspram MLP (3.8.2) | ESM reprezentacija nosi dominantan signal; sam trening (ne sama reprezentacija) daje prednost nad BLAST-om |
| RQ2 | Apsolutna razlika naspram Hadamard produkta (3.2) | Hadamard značajno bolji, potvrđeno na LOCO i pacijentima |
| RQ3 | Backbone/bilinear/MLP naspram linearnog (3.3, 3.6) | Veći kapacitet ne pomaže ni na jednom od tri testirana nivoa; interakcije između dimenzija ne postoje merljivo |
| RQ4 | Stabilnost i interpretacija dimenzija (3.4–3.5) + BLAST-komplementarnost (3.7–3.8) | Model koristi stabilan, delom biohemijski interpretabilan podskup dimenzija; prednost nad BLAST-om raste kontinuirano kako BLAST slabi, i statistički je značajna u oba smera na pacijentima (senzitivnost naspram specifičnosti) |

---
# 4. Diskusija

## 4.1. Kvalitet reprezentacije i način formiranja para važniji su od povećavanja složenosti modela

Rezultati pokazuju da povećavanje složenosti modela nije samo po sebi dovelo do boljeg predviđanja unakrsne reaktivnosti. ESM-2 reprezentacije sadržale su informaciju korisnu za razlikovanje cross-reactive parova koja se ne može objasniti samo sastavom aminokiselina. Istovremeno, veći ESM-2 model nije doneo poboljšanje u odnosu na model sa 650 miliona parametara. Slično tome, eksplicitno povećavanje prostora interakcija pomoću bilinearne reprezentacije nije poboljšalo rezultat, dok poređenje MLP-a sa linearnim klasifikatorom nije pokazalo jasnu prednost dodatne nelinearne složenosti).

Nasuprot tome, način na koji su reprezentacije dva proteina pretvorene u zajedničku reprezentaciju imao je izraženiji uticaj. Hadamardov proizvod pokazao se pogodnijim od apsolutne razlike za formiranje proteinskih parova, što ukazuje da informacija sadržana u pojedinačnim ESM-2 embeddingima nije dovoljna sama po sebi. Važno je i kako se ta informacija međusobno povezuje između dva proteina.

Ovaj rezultat ukazuje na drugačiji način razumevanja uloge neuronskog modela u ovom zadatku. MLP ne mora da bude posebno dubok ili velik da bi iskoristio informaciju iz ESM-2 reprezentacije. Veći značaj ima izbor ulazne reprezentacije koja omogućava modelu da izrazi relevantan odnos između dva proteina. Drugim rečima, rezultati više podržavaju hipotezu da je **kvalitet naučene reprezentacije i njena transformacija u reprezentaciju para važnija od prostog povećavanja kapaciteta klasifikatora**.

Ovaj zaključak ne znači da složenost modela nema nikakav značaj. Pokazano je samo da dodatni kapacitet u ispitivanim varijantama nije predstavljao ograničavajući faktor. Kada osnovna reprezentacija već sadrži relevantan signal, njegovo efikasno iskorišćavanje može biti važnije od dodavanja novih parametara ili složenijih funkcija interakcije.

## 4.2. Šta doprinosi enkodiranje Hadamardovim proizvodom

Bolje ponašanje Hadamardovog proizvoda u odnosu na apsolutnu razliku pokazuje da način formiranja reprezentacije proteinskog para utiče na to koji odnosi između dva embeddinga postaju dostupni klasifikatoru. Za dva embeddinga \(u\) i \(v\), Hadamardova reprezentacija definiše se kao

$$
x = u \odot v,
$$

gde se svaka komponenta prvog embeddinga množi sa odgovarajućom komponentom drugog embeddinga:

$$
x_i = u_i v_i.
$$

Na taj način svaka latentna dimenzija jednog proteina ulazi u model zajedno sa odgovarajućom dimenzijom drugog proteina. Ovo omogućava modelu da detektuje obrasce zajedničke aktivacije ili suprotne aktivacije određenih latentnih osobina. Kod apsolutne razlike, s druge strane, informacija je zasnovana na udaljenosti između odgovarajućih komponenti i ne zadržava njihov zajednički znak. Njegova uloga je uža pa omogućava **interakciju između odgovarajućih dimenzija reprezentacija dva proteina**, ne interakciju svih mogućih parova latentnih dimenzija.

Zbog toga se Hadamardov proizvod može posmatrati kao jednostavan način da se iz dva pojedinačna ESM-2 embeddinga formira reprezentacija njihovog odnosa. Rezultati ne pokazuju da ova reprezentacija eksplicitno modeluje sve moguće proteinske interakcije. Pokazuju da je upravo ova ograničena, dimenzijski usklađena forma interakcije bila pogodnija za zadatak predviđanja unakrsne reaktivnosti od testiranih alternativnih enkodiranja.

## 4.3. Zašto linearna predikcija može biti dovoljna nakon učenja reprezentacije

Poređenje MLP klasifikatora sa logističkom regresijom nad istom Hadamardovom reprezentacijom pokazalo je da dodatna nelinearnost klasifikatora nije donela jasno poboljšanje performansi. Ovaj rezultat sugeriše da značajan deo složenosti problema nije nužno potrebno učiti na nivou završnog klasifikatora. ESM-2 je prethodno transformisao proteinsku sekvencu u visokodimenzionalnu reprezentaciju, dok je Hadamardov proizvod omogućio da se informacije iz dve takve reprezentacije kombinuju u reprezentaciju proteinskog para.

U tom kontekstu, uloga klasifikatora može biti pre svega da kombinuje već postojeće signale iz pairwise reprezentacije. Ako su relevantni obrasci nakon ovog postupka dovoljno separabilni, linearni model može da ih iskoristi bez potrebe za dodatnim slojevima nelinearnih transformacija.

Ovaj nalaz ne znači da je odnos između proteina inherentno linearan. ESM-2 reprezentacija je rezultat nelinearnog procesa učenja, a Hadamardov proizvod uvodi multiplicativnu interakciju između odgovarajućih dimenzija dva proteina. Linearna priroda završnog klasifikatora zato ne treba da se tumači kao odsustvo nelinearnosti u celom modelu. Preciznije, rezultat pokazuje da **nakon formiranja odgovarajuće reprezentacije para nije pronađena potreba za dodatnom nelinearnom transformacijom na nivou klasifikatora**.

## 4.4. Signal je raspoređen širom embedding prostora

Analiza latentnih dimenzija pokazuje da informacija koju model koristi nije koncentrisana u malom broju izolovanih komponenti embeddinga. Pet različitih inicijalizacija pokazalo je visoku stabilnost skupa najvažnijih dimenzija: 17 od 20 najčešće odabranih dimenzija i 42 od 50 najčešće odabranih dimenzija pojavili su se u odgovarajućem skupu u najmanje četiri od pet pokretanja. To ukazuje da model kroz različite inicijalizacije dolazi do sličnog podskupa relevantnih komponenti.

Međutim, sama stabilnost najvažnijih dimenzija nije dovoljna da pokaže da je signal koncentrisan upravo u njima. Kada su dimenzije sa najvećom pojedinačnom diskriminativnošću uklonjene na osnovu Cohenovog \(d\), njihovo uklanjanje nije dovelo do sistematskog poboljšanja. Još važnije, zadržavanje samo najdiskriminativnijih dimenzija nije pouzdano reprodukovalo performanse pune reprezentacije. Ovo pokazuje da pojedinačna diskriminativnost dimenzije nije isto što i njen doprinos konačnoj predikciji.

Dodatnu podršku ovoj interpretaciji daje analiza bioloških svojstava. Od 42 izdvojene ključne dimenzije, 40 je pokazalo povezanost sa najmanje jednim od ispitivanih biohemijskih ili strukturnih deskriptora. Povezanosti su obuhvatale osobine kao što su naelektrisanje, izoelektrična tačka, dužina proteina, hidrofobnost, aromatičnost, nestabilnost, sekundarna struktura i učestalost pojedinačnih aminokiselina. Međutim, nijedna pojedinačna korelacija nije bila dovoljno jaka da objasni ponašanje dimenzije sama za sebe.

Ovi rezultati ukazuju da ESM-2 embedding ne treba posmatrati kao skup potpuno nezavisnih i lako interpretabilnih osobina. Latentne dimenzije mogu istovremeno nositi više povezanih informacija, dok njihov značaj zavisi od kombinacije sa odgovarajućom reprezentacijom drugog proteina. Zbog toga analiza pojedinačnih dimenzija daje samo delimičnu sliku načina na koji model donosi odluku.

Važan dodatni rezultat dolazi iz testa eksplicitnih interakcija između različitih latentnih dimenzija. Uvođenje proizvoda između svih parova od 42 ključne dimenzije nije donelo praktično poboljšanje u odnosu na aditivni model. To sugeriše da se korisna interakcija u Hadamardovoj reprezentaciji ne mora proširivati na sve moguće kombinacije dimenzija. Dovoljno je da se odgovarajuće dimenzije dva proteina kombinuju, dok dodatno modelovanje interakcija između različitih latentnih dimenzija nije pokazalo novu informaciju.

Zajedno, ovi nalazi podržavaju sliku **distribuiranog signala**. Model koristi stabilan skup latentnih komponenti, ali njihov doprinos nije moguće svesti na nekoliko izolovanih dimenzija. Biološke osobine povezane sa tim komponentama takođe nisu dovoljne da pojedinačno objasne predikciju. Relevantna informacija je raspoređena kroz embedding prostor i postaje korisna prvenstveno kroz kombinovanje reprezentacija dva proteina.

## 4.5. Šta model uči što jednostavni deskriptori ne obuhvataju

Poređenje ESM-2 reprezentacija sa jednostavnim deskriptorima pokazalo je da informacija korisna za predikciju nije obuhvaćena samo osnovnim biohemijskim osobinama proteina. Ipak, kao što je pokazano u 4.4, većina bitnih dimenzija je upravo sa takvim osobinama korelisana. Ova dva rezultata nisu kontradiktorna: ESM-2 embedding može sadržati informacije povezane sa poznatim biohemijskim osobinama, ali ih organizovati u višedimenzionalnu reprezentaciju koju jednostavni zbirni deskriptori ne mogu u potpunosti opisati; sama korelacija dimenzije sa svojstvom ne pokazuje da je to svojstvo mehanizam koji određuje predikciju.

Dodatno, analiza parova na kojima je MLP nadmašio BLAST naspram parova gde je BLAST bio bolji nije otkrila jednostavan biohemijski obrazac koji bi razdvojio ove dve grupe. Razlike u ispitivanim deskriptorima ne pružaju stabilan kriterijum za predviđanje kada će model biti uspešniji od klasičnog poravnanja.

Najopreznija interpretacija je da model koristi **finiju organizaciju informacija u latentnom prostoru ESM-2** koja nije direktno predstavljena pojedinačnim deskriptorima.

## 4.6. Komplementarnost sa poravnanjem sekvenci

Poređenje MLP-a sa BLAST-om pokazuje da ESM-2 reprezentacija ne predstavlja jednostavnu zamenu za klasično poravnanje sekvenci. Njihova relativna uspešnost zavisi od informacija koje su dostupne iz samih sekvenci i od strukture prostora kandidata.

Na LOCO evaluaciji ukupna razlika između MLP-a i BLAST-a bila je mala, što pokazuje da MLP nije univerzalno bolji prediktor. Međutim, analiza po jačini BLAST signala pokazala je izraženu promenu odnosa između dva pristupa. U najnižim kvartilima BLAST ranga MLP je ostvarivao veći recipročni rang, dok je u najjačem kvartilu BLAST imao jasnu prednost. Razlika između performansi MLP-a i BLAST-a pritom je negativno korelisala sa jačinom BLAST signala.

Ovaj obrazac pokazuje da vrednost ESM-2 reprezentacije nije ravnomerno raspoređena kroz sve slučajeve. Kada sekvencijalno poravnanje već daje snažan signal, dodatna informacija koju MLP izvlači iz embeddinga ima manji doprinos. Kada je BLAST signal slabiji, ESM-2 reprezentacija može pružiti informaciju koja nije dovoljno izražena kroz direktnu sekvencijalnu sličnost.

Analiza BLAST-slabih slučajeva dodatno pokazuje da slab BLAST *rang* ne znači nužno i nisku sekvencijalnu sličnost — termin „slab BLAST" u ovom radu treba razumeti prvenstveno kao slabiji rang kandidata, ne kao sinonim za nizak identitet sekvence. U značajnom broju takvih slučajeva problem nastaje zbog konkurencije između više kandidata sa sličnim, umerenim BLAST rezultatima (3.7.3), ne zbog odsustva homologije; niti je pronađen jednostavan biohemijski obrazac koji bi razlikovao parove na kojima MLP dobija od onih na kojima gubi (3.7.4). MLP u tim situacijama drugačije rangira kandidate koristeći informacije iz njihove latentne reprezentacije.

Zbog toga se najprirodnije tumačenje odnosa između ova dva pristupa ne zasniva na pitanju koji je model „bolji“. BLAST i MLP predstavljaju **dva različita izvora informacije o odnosu između proteina**. BLAST direktno koristi sekvencijalnu sličnost, dok MLP koristi obrasce prisutne u ESM-2 embedding prostoru. Njihova komplementarnost je naročito izražena kada sekvencijalni signal nije dovoljan da jednoznačno rangira kandidate — umesto da se ESM-2 model posmatra kao zamena za postojeće metode poravnanja, prirodnije je posmatrati ga kao dopunski izvor informacije.

## 4.7. Šta nam model ne govori?

Iako analiza embedding prostora pokazuje povezanost latentnih dimenzija sa različitim biohemijskim i strukturnim svojstvima, ovi rezultati ne omogućavaju zaključak da ta svojstva uzrokuju unakrsnu reaktivnost. Korelacija latentne dimenzije sa naelektrisanjem, izoelektričnom tačkom ili sekundarnom strukturom pokazuje samo da su informacije povezane u reprezentaciji.

Model takođe ne daje direktan dokaz konkretnog imunološkog mehanizma. Predviđanje proteinskog para kao cross-reactive ne pokazuje da je model naučio specifičan način vezivanja IgE antitela, određenu epitopsku interakciju ili drugi pojedinačni molekularni mehanizam.

Slično tome, pojedinačna dimenzija ESM-2 embeddinga ne treba da se tumači kao nosilac jedne jedine biološke osobine. Iako su neke dimenzije povezane sa merljivim svojstvima, rezultati o distribuiranom signalu pokazuju da se korisna informacija verovatnije formira kroz kombinaciju više komponenti.

Zbog toga interpretaciju modela treba ograničiti na ono što je direktno podržano eksperimentima: ESM-2 embedding sadrži informaciju korisnu za ovaj zadatak, Hadamardovo enkodiranje omogućava njeno korišćenje na nivou proteinskog para, a deo te informacije povezan je sa poznatim osobinama proteina. Precizan biološki mehanizam koji stoji iza tih obrazaca ostaje otvoreno pitanje.

## 4.8. Ograničenja

### 4.8.1. Pozitivno-neobeležena priroda skupa podataka

Skup podataka ne omogućava pouzdano razlikovanje svih negativnih parova od neobeleženih parova. Odsustvo dokumentovane unakrsne reaktivnosti ne znači nužno da ona ne postoji, zbog čega negativni primeri mogu sadržati neotkrivene pozitivne odnose. Ograničenje je posebno važno pri tumačenju performansi klasifikatora.

### 4.8.2. Ograničen broj nezavisnih bioloških primera

Iako skup sadrži veliki broj proteinskih parova, broj međusobno nezavisnih bioloških primera je ograničen njihovom povezanošću kroz zajedničke proteine i literaturne izvore. Zbog toga je korišćena LOCO validacija kako bi se smanjio uticaj ove zavisnosti. Ipak, rezultati treba da se tumače kao procena generalizacije na nove povezane komponente, a ne kao potpuno nezavisna procena na velikom broju nezavisnih eksperimenata.

### 4.8.3. Ograničena nezavisna kohorta pacijenata

Pacijentska evaluacija predstavlja nezavisnu proveru modela, ali je njen obim ograničen brojem dostupnih pacijenata i kompletnim nalazima. Rezultate stoga treba posmatrati kao potvrdu potencijala modela na nezavisnim podacima, a ne kao konačnu procenu kliničke primenljivosti.

### 4.8.4. Post-hoc interpretacija dimenzija embeddinga

Povezivanje latentnih dimenzija ESM-2 embeddinga sa biohemijskim i strukturnim osobinama izvršeno je nakon treniranja modela. Takve analize mogu ukazati na moguće značenje naučenih reprezentacija, ali ne dokazuju da određena osobina uzrokuje predikciju niti da latentna dimenzija ima jednu jasno definisanu biološku funkciju.

### 4.8.5. Namena modela: prioritizacija kandidata, ne dijagnoza

Model je razvijen i evaluiran kao alat za **prioritizaciju kandidata za dalje eksperimentalno testiranje**, rangiranje proteina prema verovatnoći unakrsne reaktivnosti radi usmeravanja skupljih laboratorijskih ili kliničkih provera (npr. skin-prick ili specifičnog IgE testiranja). On **nije** dijagnostički alat: nijedan nalaz modela ne predstavlja potvrdu niti isključenje unakrsne reaktivnosti kod konkretnog pacijenta, niti zamenu za klinički pregled i standardizovano alergološko testiranje.

# 5. Budući pravci istraživanja

Rezultati ovog rada otvaraju nekoliko pravaca za dalja istraživanja. Prvi je razvoj adaptivne fuzije BLAST-a i MLP-a, pri kojoj bi doprinos MLP-a mogao biti veći u slučajevima kada BLAST daje slab ili neodlučan signal. Ova ideja predstavlja hipotezu zasnovanu na uočenoj komplementarnosti dva pristupa i nije validirana u okviru ovog rada.

Dalji rad treba da uključi veće i potpuno nezavisne kohorte pacijenata kako bi se proverila reproduktivnost rezultata na različitim populacijama i eksperimentalnim protokolima. Takođe, analiza zasnovana na strukturi proteina i poznatim epitopima mogla bi pomoći u povezivanju obrazaca naučenih u embedding prostoru sa konkretnijim molekularnim svojstvima. Ove analize treba sprovoditi kao nezavisnu validaciju, a ne kao pretpostavljeni mehanizam modela.

Konačno, proširenje skupa podataka većim brojem nezavisno i eksperimentalno potvrđenih unakrsno reaktivnih parova predstavljalo bi jedan od najvažnijih narednih koraka. Takav skup bi smanjio neizvesnost izazvanu pozitivno-neobeleženom prirodom postojećih podataka i omogućio pouzdaniju procenu sposobnosti modela da generalizuje na nove biološke primere.

---

# 6. Zaključak

Ovaj rad pokazuje da potencijalna unakrsna reaktivnost proteinskih alergena ne može biti pouzdano opisana jednom merom sličnosti. Dok sama ESM-2 cosine sličnost nije nadmašila BLAST, nadgledano kombinovanje ESM-2 reprezentacija putem Hadamard proizvoda pokazalo je znatno korisniji signal i na nezavisnim pacijentskim slučajevima ostvarilo najbolje rangiranje. Ablacione analize ukazuju da ključ nije u samoj kompleksnosti modela, već u kvalitetu proteinske reprezentacije i načinu na koji se dve reprezentacije povezuju.

Rezultati zato ne ukazuju na jednostavnu zamenu BLAST-a novim modelom, već na **nov način korišćenja proteinskih reprezentacija kao dopune postojećim signalima**. Najvažniji nalaz ovog rada nije da je jedan algoritam univerzalno najbolji, već da se informacija relevantna za unakrsnu reaktivnost može nalaziti u odnosu između dva proteinska zapisa, a ne samo u njihovoj pojedinačnoj sličnosti. To predstavlja osnovu za razvoj budućih sistema koji bi sekvencijalne, reprezentacione, strukturne i kliničke informacije povezivali u jedinstven, strogo nezavisno validiran model.

---

# Dostupnost podataka i koda

Kod i svi korišćeni resursi dostupni na: https://github.com/tardigrafika/Allergorithm

# Zahvalnice

Želim da se zahvalim svom mentoru Stefanu Nožiniću na stručnom vođstvu, savetima i kontinuiranoj podršci tokom razvoja ovog istraživanja.

Posebnu zahvalnost dugujem Mariji Stefanović na pomoći u razumevanju biološke pozadine problema, savetima u vezi sa alergenima i korisnim komentarima tokom rada.


# Literatura
Bibliografija 




