# Šta model uparenih vrednosti baziran na ESM-2 uči o unakrsnoj reaktivnosti proteina?

Lana Lejić

Mentor: Stefan Nožinić

<!-- RECENZIJA (peer review, 2026-09): (1) Naslov je pitanje — neki časopisi to ne vole, a rad daje samo delimičan odgovor; razmotriti deklarativni naslov. (2) Autorstvo: mentor je za časopis gotovo sigurno koautor; definisati doprinose po CRediT taksonomiji. (3) VAŽNO: rad/RAD.md i rad/MLP_RAD.md se preklapaju ~70% — odlučiti koji je "taj" rad; slanje oba nosi rizik od salami-slicing / duple publikacije. Preporuka: konsolidovati u jedan rad. -->

---

## Apstrakt

nwsto smort

<!-- RECENZIJA: Apstrakt i ključne reči su placeholderi. Za časopis: strukturisan apstrakt (~200-250 reči: pozadina / metode / rezultati / zaključak). Glavna poruka mora biti "komplementarnost", ne "MLP bolji od BLAST-a" (v. §3.8.1). Ključne reči: cross-reactivity, protein language models, ESM-2, allergens, link prediction, pairwise representation, LOCO validation. -->

### Ključne reči ?

nez jel ovo treba d


# 1. Uvod

### 1.1. Unakrsna reaktivnost kao problem reprezentacije proteina

Unakrsna alergijska reaktivnost predstavlja sposobnost IgE antitela da prepoznaju homologne proteine iz različitih alergenih izvora. Ovaj fenomen nastaje zbog očuvanih molekularnih karakteristika između proteina, ali nije određen isključivo njihovom sekvencijalnom sličnošću. Proteini sa relativno niskim identitetom sekvence mogu izazivati unakrsnu reaktivnost, dok visoka sekvencijalna sličnost sama po sebi ne predstavlja dovoljan uslov za zajedničko imunološko prepoznavanje.

Zbog toga se predikcija unakrsne reaktivnosti može posmatrati kao problem reprezentacije proteina: potrebno je pronaći prikaz proteinske sekvence koji zadržava informacije relevantne za funkcionalnu i imunološku srodnost, a ne samo za evolutivnu sličnost.

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

<!-- RECENZIJA: RQ3 pretpostavlja da "poboljšanje performansi" postoji — a na LOCO-u je MLP ≈ BLAST (Δ +0,0016, n.z., §3.3.1), na pacijentima je ishod podeljen (§3.8.1). Preformulisati neutralno: "Da li eventualna razlika u performansama potiče od nelinearnog klasifikatora ili od izbora reprezentacije i pairwise enkodiranja?". RQ1/RQ4 formulacije koje impliciraju "prednost nad BLAST-om" takođe kvalifikovati. -->

---
# 2. Metodologija

## 2.1. Skup podataka o unakrsnoj reaktivnosti
### 2.1.1. Skup kandidata proteina

Skup kandidata formiran je iz WHO/IUIS Allergen Nomenclature baze i obuhvata proteinske alergene za koje su bile dostupne odgovarajuće aminokiselinske sekvence. Nakon uklanjanja nevalidnih sekvenci, sekvenci kraćih od 30 aminokiselina i potpunih duplikata, konačni skup sadrži 1.536 proteinskih alergena. Izoforme sa različitim aminokiselinskim sekvencama tretirane su kao zasebni kandidati.

<!-- RECENZIJA: "1.536" ovde vs "~1535 proteina" u §3.5 — uskladiti tačan broj. Deduplikacija po FASTA sekvenci koristi proizvoljno alfabetsko pravilo (uklonilo npr. Pen a 1, Pen m 1 kao "duplikate" Lit v 1 — v. RAD.md Prilog B); to je ograničenje, mora u Supplementary + Ograničenja. -->

Detaljni koraci čišćenja i dokumentovani granični slučajevi opisani su u Supplementary Material-u.

### 2.1.2. Kurirani unakrsno reaktivni parovi

Poznati odnosi unakrsne reaktivnosti prikupljeni su iz objavljene naučne literature i povezani sa proteinima iz konačnog skupa kandidata. Ukupno je identifikovano 1.922 jedinstvena para koji obuhvataju 477 alergena i 317 literaturnih izvora. Za svaki par zabeležen je izvor dokaza i nivo pouzdanosti, a kada je bio dostupan i pripadnost proteinskoj familiji.

Parovi su klasifikovani u četiri nivoa dokaza: Confirmed (138), Strong (377), Suspected (277) i Inferred (1.093). Kategorija Inferred obuhvata parove čija je unakrsna reaktivnost izvedena iz homologije ili pripadnosti istoj proteinskoj familiji bez direktnog eksperimentalnog dokaza. Ovi parovi nisu korišćeni kao pozitivni primeri tokom treniranja nadgledanih modela, ali su zadržani u evaluacionom skupu.

<!-- RECENZIJA: 138+377+277+1.093 = 1.885, a gore piše "1.922 jedinstvena para" — manjak 37 (ista greška u RAD.md §2.1.1). Ili nedostaje kategorija, ili 37 parova nema dodeljen nivo — objasniti. Propagira se u RAD.md §4.7 (511 vs 138+377=515; 825 vs 1922−1093=829 vs zbir-bez-Inferred 792). Predlog: nivoe dokaza iskoristiti kao gradiranu relevantnost za nDCG (v. komentar u §2.5). -->


### 2.1.3. Pozitivno-neobeleženo okruženje i uzorkovanje negativa

Skup podataka ima karakteristike positive-unlabeled (PU) problema: činjenica da određeni par nije zabeležen u literaturi ne znači da je eksperimentalno potvrđeno da između proteina ne postoji unakrsna reaktivnost. Zbog toga se odsustvo para iz kuriranog skupa ne može direktno interpretirati kao negativna klasa.

Za treniranje modela negativni primeri su stoga uzorkovani iz parova koji nisu prisutni u kuriranoj pozitivnoj relaciji. Negativno uzorkovanje vršeno je nezavisno od oznaka proteinskih familija kako bi se izbeglo uvođenje eksplicitne familijske pretpostavke u sam problem predikcije. Odnos pozitivnih i negativnih primera i konkretna procedura uzorkovanja navedeni su uz odgovarajuće eksperimente radi potpune reproduktivnosti.

<!-- RECENZIJA: Odnos pozitiva:negativa (10:1 u RAD.md) i procedura uzorkovanja se ovde odlažu, ali se nigde u MLP_RAD.md ne navode. Navesti eksplicitno u Metodologiji: odnos, resampling po foldu, iz kog skupa proteina (samo trening deo folda, nikad test komponenta). -->


## 2.2. ESM-2 vektorske reprezentacije (embeddings) proteina

### 2.2.1. ESM-2 model

Za generisanje vektorskih reprezentacija proteinskih sekvenci korišćen je **ESM-2 (Evolutionary Scale Modeling 2)** proteinski jezički model. Model je prethodno treniran nad velikim skupom proteinskih sekvenci i uči kontekstualne reprezentacije aminokiselina na osnovu njihovog položaja u sekvenci i konteksta ostalih aminokiselina.

Kao primarni model korišćen je **ESM-2 650M**, koji sadrži približno 650 miliona parametara. **ESM-2 3B**, sa približno 3 milijarde parametara, korišćen je u kontrolnom eksperimentu za ispitivanje uticaja veličine modela na dobijene reprezentacije.

### 2.2.2. Reprezentacije po aminokiselini i agregacija srednjom vrednošću (mean pooling)

Za svaku aminokiselinu u ulaznoj sekvenci ESM-2 generiše kontekstualni vektor:

$$
h_1, h_2, \ldots, h_L,
$$

gde je $L$ dužina proteinske sekvence. Svaki vektor predstavlja aminokiselinu u kontekstu cele sekvence.

Pošto modeli mašinskog učenja zahtevaju reprezentaciju fiksne dimenzionalnosti, per-residue reprezentacije agregirane su primenom **mean pooling-a**:

$$
u = \frac{1}{L}\sum_{i=1}^{L} h_i.
$$

Na ovaj način svaka proteinska sekvenca predstavljena je jednim vektorom $u$. Za ESM-2 650M korišćeni su vektori dimenzionalnosti **1280**, dok je ESM-2 3B korišćen sa svojom odgovarajućom izlaznom dimenzionalnošću.

<!-- RECENZIJA: Navesti tačnu dimenzionalnost ESM-2 3B (2560), sloj sa kog se uzima reprezentacija i da li je isti pooling. Mean pooling globalno agregira signal — relevantno za diskusiju o lokalnim epitopima (RAD.md §5.2); razmotriti napomenu ovde. -->


## 2.3. Reprezentacija parova dva proteina

Pojedinačni embeddingi $u$ i $v$ opisuju proteine zasebno. Za predikciju unakrsne reaktivnosti potrebno je iz njih formirati reprezentaciju koja opisuje njihov međusobni odnos. U radu su ispitana dva načina enkodiranja para zasnovana na direktnim operacijama nad embedding prostorom.

### 2.3.1. Enkodiranje apsolutnom razlikom (Absolute-difference encoding)

Reprezentacija para apsolutnom razlikom definisana je kao:

$$
x = |u-v|.
$$

Svaka komponenta vektora $x$ predstavlja apsolutnu razliku između odgovarajućih komponenti embeddinga dva proteina. Na ovaj način reprezentacija direktno opisuje udaljenost proteina duž svake dimenzije prostora reprezentacije.

Apsolutna razlika je simetrična u odnosu na redosled proteina. Zamenom $u$ i $v$ dobija se ista reprezentacija para.

### 2.3.2. Hadamardov proizvod (Hadamard encoding)

Hadamardov proizvod definisan je kao:

$$
x = u \odot v.
$$

Svaka komponenta rezultujućeg vektora dobija se poelementnim množenjem odgovarajućih komponenti dve proteinske reprezentacije:

$$
x_i = u_i v_i.
$$

Ovim enkodiranjem svaka dimenzija predstavlja zajedničku aktivaciju odgovarajuće latentne osobine kod oba proteina. Visoke vrednosti mogu nastati kada oba proteina imaju izraženu istu komponentu reprezentacije, čime se eksplicitno uvode interakcije između njihovih naučenih osobina.

Kao i apsolutna razlika, Hadamardov proizvod je simetričan prema zameni proteina $u$ i $v$.

### 2.3.3. Kontrola složenijih interakcija

Pored element-wise enkodiranja, ispitan je bilinearni pristup zasnovan na spoljašnjem proizvodu (outer product):

$$
X = uv^\mathsf{T}.
$$

Za razliku od prethodnih reprezentacija, outer product eksplicitno modeluje interakciju svake dimenzije embeddinga prvog proteina sa svakom dimenzijom embeddinga drugog proteina. Time se dobija znatno veća reprezentacija para.

Ovaj pristup korišćen je kao kontrolni eksperiment za proveru da li eksplicitno modelovanje složenijih interakcija između embeddinga pruža dodatnu informaciju. Zbog velike dimenzionalnosti i slabije stabilnosti nije uključen u završni model. Detalji eksperimenta dati su u Supplementary Material-u.
## 2.4. Prediktivni modeli

### 2.4.1. MLP(Hadamard)

Za predikciju unakrsne reaktivnosti korišćen je višeslojni perceptron (MLP) nad Hadamardovom reprezentacijom para. Ulaz modela je vektor Hadamardovog proizvoda dobijen iz embeddinga dva proteina.

MLP predstavlja nelinearnu funkciju koja mapira reprezentaciju para u verovatnoću unakrsne reaktivnosti:

$$
\hat{y} = f_{\theta}(x).
$$

Model je treniran minimizacijom binarne unakrsne entropije (binary cross-entropy) koristeći Adam optimizer. Tokom treniranja primenjena je regularizacija kroz dropout i weight decay. Treniranje je prekinuto primenom early stopping-a kada se performanse na validacionom skupu nisu dalje poboljšavale.

Finalna arhitektura MLP-a sastoji se od dva skrivena sloja sa nelinearnom aktivacionom funkcijom i dropout regularizacijom. Broj neurona po slojevima i ostali hiperparametri određeni su na razvojnom skupu i zatim fiksirani pre završne evaluacije.

<!-- RECENZIJA: Navesti finalne vrednosti (neurona po sloju, dropout, weight decay, lr, batch size, early-stopping kriterijum, veličina ulaza = 1280 za Hadamard). Opisati "razvojni skup" i proceduru izbora (grid? koliko konfiguracija?). Bez ovoga rad nije reproducibilan. -->


### 2.4.2. Kontrola linearnim klasifikatorom

Da bi se odvojio doprinos nelinearnosti klasifikatora od informacije sadržane u samoj reprezentaciji para, kao kontrolni model korišćena je logistička regresija.

Logistička regresija koristi **isti Hadamardov ulaz** kao MLP. Za razliku od MLP-a, model ne sadrži skrivene nelinearne slojeve. Njegov skor je određen linearnom kombinacijom komponenti ulaznog vektora:

$$
\hat{y} = \sigma(w^\mathsf{T}x+b)
$$

, gde je $\sigma$ sigmoidna funkcija.

Poređenjem ova dva modela uz isti ulaz može se ispitati da li dodatna prediktivna sposobnost potiče od same Hadamard reprezentacije ili od mogućnosti MLP-a da nad njom modeluje nelinearne odnose.

## 2.5. Protokol validacije

<!-- RECENZIJA — METRIKE (nedostaje ceo pododeljak): MLP_RAD.md nema definiciju evaluacionih metrika. Dodati §2.5.x "Metrike rangiranja":
  1. Ovo je formalno predikcija veza u grafu (link prediction). Koristiti FILTERED rang skrivene ivice (q,t) — pri skorovanju ukloniti iz liste ostale POZNATE prave partnere čvora q. Nefiltriran rang je pesimistički i neravnomerno pristrasan: kažnjava gusto povezane familije (nsLTP, profilin) — verovatno deo efekta "zagušenja" iz §3.7.
  2. Definisati jedinicu: MRR po skrivenoj ivici (leave-one-edge-out / leave-one-finding-out), MIKRO i MAKRO prosek (makro = po upitu-proteinu, pa po familiji). "Mikro-prosečan MRR" preteže gusto povezane familije.
  3. MRR zadržati kao osetljivu meru, ali dodati: Hits@{1,5,10} (klinička interpretacija; vraćeno iz RAD.md), nDCG@10 sa gain=f(nivo dokaza) (jedina metrika koja koristi Confirmed/Strong/Suspected/Inferred i više partnera po upitu), medijalni rang + IQR (robustan; MRR maskira rep).
  4. Za pacijentski zadatak: recall@k kriva / površina do klinički realnog k (~20), ili "work-saved over BLAST" — direktno meri čemu alat služi.
  5. PU: NIKAD ne računati precision@k / FPR / AUPRC protiv neoznačenih parova kao da su negativi — meriti samo gde padne poznati pozitiv. AUROC/AUC samo kao interni dijagnostički signal (kao u §3.1).
  6. Male MRR razlike (0,001-0,01) sa CI preko nule sugerišu slabu rezoluciju metrike na 40 komponenti / 34 pacijenta — komplementarne metrike čine efekat merljivijim.
-->

### 2.5.1. Validacija izostavljanjem povezane komponente (Leave-One-Connected-Component-Out — LOCO)

Random podela parova na trening i test skup nije odgovarajuća za ovaj problem zbog povezanosti proteina kroz mrežu unakrsne reaktivnosti. Ako su dva proteina povezana u istoj komponenti, njihovo razdvajanje između treninga i testa može omogućiti modelu da indirektno iskoristi informacije iz test komponente.

Zbog toga je evaluacija sprovedena metodom **Leave-One-Connected-Component-Out (LOCO)**. Graf unakrsne reaktivnosti formiran je tako da proteini predstavljaju čvorove, a poznati odnosi unakrsne reaktivnosti grane. U svakoj iteraciji jedna povezana komponenta izostavljena je iz treninga i korišćena kao test skup. Model je treniran isključivo nad preostalim komponentama.

Ovakva podela omogućava procenu sposobnosti modela da generalizuje na proteinske odnose koji nisu povezani sa primerima dostupnim tokom treniranja.

<!-- RECENZIJA: Navesti broj i raspodelu veličina povezanih komponenti (broj folda — "40 folda" se pominje tek u §3.4.3 i pripada ovde). Kako se tretiraju izolovani čvorovi / komponente veličine 2? Kako se uzorkuju negativi po foldu? -->


### 2.5.2. Nezavisna evaluacija na nivou pacijenata

Pored evaluacije na kuriranom skupu, generalizacija modela ispitana je na nezavisnim podacima prikupljenim iz dokumentovanih slučajeva pacijenata. Ovi podaci nisu korišćeni za treniranje modela.

Za svakog pacijenta model rangira kandidate na osnovu dostupnih pozitivnih nalaza. Rezultati se zatim procenjuju na kandidatima čiji status nije korišćen za formiranje datog upita. Time se proverava da li naučeni odnos između proteinskih reprezentacija može da se prenese na podatke nezavisne od kuriranog skupa za treniranje.

<!-- RECENZIJA: (1) PROVENIJENCIJA PODATAKA — nejasno i etički kritično: ovde "prikupljenim iz dokumentovanih slučajeva pacijenata", RAD.md §3.3 kaže "literaturno dokumentovani slučajevi", a Zahvalnice (obe verzije) kažu "osobe koje su ustupile svoje rezultate alergoloških testiranja". Ako su realni pacijenti: OBAVEZNO odobrenje etičke komisije / IRB + izjava o informisanom pristanku + anonimizacija — inače većina časopisa neće poslati rad u recenziju. Ako je literatura: navesti sve izvore (PMID) u Supplementary i izjaviti da odobrenje nije potrebno.
(2) PROTOKOL NEPOTPUN: konkretan protokol (leave-one-finding-out, skrivanje pozitivnog ILI negativnog nalaza, metrike svesne smera MRR+ i NR-) prvi put se pojavljuje tek u §3.8.1 — preneti ceo taj opis ovde. Navesti: broj pacijenata, raspodelu broja nalaza po pacijentu, udeo pozitivnih/negativnih, kriterijum n>=2. -->


### 2.5.3. Statističko testiranje i bootstrap intervali poverenja

Poređenja modela sprovedena su na istim upitima kako bi se razlike u rangiranju procenjivale u uparenim uslovima. Za poređenje performansi korišćen je **Wilcoxon signed-rank test**, koji ne zahteva pretpostavku normalne raspodele razlika između parova rezultata.

Intervali poverenja za razlike u performansama procenjivani su bootstrap postupkom. Kada je struktura podataka to zahtevala, resampling je vršen na nivou literaturnog izvora umesto na nivou pojedinačnih parova, čime se izbegava tretiranje više parova iz iste studije kao potpuno nezavisnih opažanja.

Statistička značajnost i intervali poverenja interpretirani su zajedno sa veličinom uočene razlike, a ne samo na osnovu p-vrednosti.

<!-- RECENZIJA: (1) Rezultati (§3.8.1) izveštavaju "cluster-permutacija p" — permutacioni/cluster-permutacioni test nije opisan ovde; dodati. (2) Study-level (po izvoru) i patient-level (po pacijentu) klasterisanje dosledno primeniti na SVE tvrdnje o razlici i izveštavati OBE procene (nivo primera i nivo klastera), ne samo povoljniju — to je jak metodološki doprinos (RAD.md §5.4), zadržati eksplicitno. (3) Višestruka poređenja: §3.5 (42 × ~21 korelacija) i §3.6 (861 par) — navesti FDR korekciju ili obrazložiti da je §3.6 nekorigovan jer je nalaz negativan. -->


## 2.6. Mehanističke analize

Nakon evaluacije prediktivnih performansi, sprovedene su dodatne analize sa ciljem da se ispita koje karakteristike embedding prostora doprinose ponašanju modela.

### 2.6.1. Stabilnost odabranih dimenzija embeddinga kroz različita slučajna semena

Stabilnost važnih dimenzija embeddinga ispitana je ponavljanjem treniranja modela sa različitim slučajnim semenom. Za svako seme izdvojene su dimenzije sa najvećim doprinosom modelu.

Stabilnost izbora procenjena je preklapanjem skupova najbolje rangiranih dimenzija. Korišćeni su **top-k overlap** i **Jaccardov indeks**:

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

Visoko preklapanje između različitih semena ukazuje na stabilan izbor dimenzija, dok nisko preklapanje ukazuje da se model može oslanjati na različite ekvivalentne dimenzije embedding prostora.

### 2.6.2. Povezanost dimenzija embeddinga sa biohemijskim i strukturnim deskriptorima

Da bi se ispitalo da li pojedinačne dimenzije embeddinga imaju prepoznatljivu biohemijsku ili strukturnu interpretaciju, njihove vrednosti povezane su sa poznatim osobinama proteina.

Analizirani su deskriptori kao što su dužina sekvence, izoelektrična tačka, ukupni naboj, GRAVY skor, aromatičnost, indeks nestabilnosti, sastav aminokiselina i karakteristike sekundarne strukture.

Ove analize predstavljaju **post-hoc asocijacije** između naučenih reprezentacija i poznatih proteinskih osobina. One ne predstavljaju dokaz da određena osobina uzrokuje prediktivni signal modela.

### 2.6.3. Orezivanje dimenzija (Dimension pruning)

Ispitano je da li dimenzije koje pojedinačno pokazuju diskriminativnu sposobnost mogu zajedno da reprodukuju ponašanje modela. Dimenzije su rangirane prema njihovoj diskriminativnosti korišćenjem **Cohenovog $d$**, nakon čega su formirani podskupovi sa različitim brojem najbolje rangiranih dimenzija.

Modeli trenirani nad ovim podskupovima poređeni su sa modelom koji koristi punu reprezentaciju. Cilj analize je da se utvrdi da li je prediktivni signal koncentrisan u malom broju dimenzija ili zahteva širu reprezentaciju embedding prostora.

### 2.6.4. Test međudimenzionalnih interakcija

Da bi se direktno ispitalo da li model koristi interakcije između različitih dimenzija embeddinga, analizirano je **861 par dimenzija** identifikovanih kao relevantne dimenzije reprezentacije.

<!-- RECENZIJA: Formulacija zbunjuje — nije 861 nezavisno identifikovanih parova, već svih C(42,2)=861 parova među 42 stabilne dimenzije iz §3.4.1. Preformulisati. Obrazložiti izbor praga "42" (iz top-50, a ne top-20) kao "relevantnog skupa". -->


Upoređena su dva modela. Prvi koristi samo aditivne efekte odabranih dimenzija. Drugi, pored aditivnih efekata, uključuje eksplicitne proizvode između odabranih parova dimenzija.

Ovim poređenjem testira se da li dodatno modelovanje međudimenzionalnih interakcija pruža informaciju koja nije sadržana u pojedinačnim dimenzijama posmatranim nezavisno.
## 2.7. Komplementarnost sa BLAST-om

### 2.7.1. Osnovna linija rangiranja pomoću BLAST-a (BLAST ranking baseline)

BLAST je korišćen kao sekvencijalna baseline metoda za rangiranje kandidata. Za svaki proteinski upit kandidati su rangirani prema BLAST signalu dobijenom poređenjem njegove sekvence sa sekvencama kandidata.

Za svaki upit $q$ i kandidata $c$ beleži se rang kandidata u BLAST rang-listi. Isti skup upita i kandidata koristi se za poređenje sa MLP modelom.

### 2.7.2. Dobitak MLP-a na nivou upita (Query-level MLP gain)

Da bi se ispitalo gde MLP pruža korist u odnosu na BLAST, za svaki upit izračunata je razlika između njihovih recipročnih rangova:

$$
\Delta_q =
\frac{1}{r_{\mathrm{MLP}}(q)}
-
\frac{1}{r_{\mathrm{BLAST}}(q)},
$$

gde su $r_{\mathrm{MLP}}(q)$ i $r_{\mathrm{BLAST}}(q)$ rang tačnog kandidata prema MLP-u i BLAST-u.

Pozitivna vrednost $\Delta_q$ označava da je MLP bolje rangirao tačan kandidat, dok negativna vrednost označava prednost BLAST-a. Na ovaj način ukupna razlika u MRR-u može se analizirati na nivou pojedinačnih upita.

### 2.7.3. Stratifikacija prema jačini BLAST-a

Da bi se ispitalo da li doprinos MLP-a zavisi od dostupnosti sekvencijalnog signala, upiti su podeljeni prema uspešnosti BLAST-a.

Za svaku grupu analiziran je MLP gain definisan razlikom recipročnih rangova. Posebna pažnja posvećena je upitima kod kojih BLAST ne obezbeđuje snažan signal.

<!-- RECENZIJA: Precizirati podelu — §3.7.2 koristi kvartile BLAST rr, §3.7.3 medijanu; opisati oba. Definisati OPERATIVNO "zagušenje/crowding" familije (koristi se u §3.7.2-3.7.3, §4.6-4.7 kao "dijagnostikovano", a kriterijum nigde nije dat — npr. broj kandidata iznad praga sličnosti u pool-u). Razjasniti "crowded upit" naspram "upit koji dodiruje crowded familiju" (§3.7.3). -->


Ova analiza omogućava da se utvrdi da li MLP predstavlja alternativni signal kada je sekvencijalna sličnost informativna ili prvenstveno dopunjuje BLAST u slučajevima u kojima je sekvencijalni signal slab.

### 2.7.4. Analiza upita sa slabim BLAST rezultatom (BLAST-weak queries)

Upiti sa slabim BLAST rezultatom analizirani su zasebno kako bi se ispitalo da li MLP upravo u ovom režimu pruža najveću dodatnu informaciju.

Unutar ove grupe upiti su podeljeni na slučajeve u kojima je MLP nadmašio BLAST (**MLP-wins**) i slučajeve u kojima je BLAST ostao bolji (**MLP-loses**). Grupe su zatim poređene prema karakteristikama njihovih proteinskih reprezentacija i rangiranja.

Motivacija iza analize je da se utvrdi da li postoje sistematske karakteristike upita kod kojih embedding-based model uspeva da nadomesti nedostatak sekvencijalnog signala.


# 3. Rezultati

## 3.1. ESM-2 pruža znatno bogatije prediktivne informacije od jednostavnog sastava

| Zamenjena komponenta | Protokol | Δ MRR (bootstrap 95% CI) | Značajno? |
|---|---|---|---|
| ESM-2 reprezentacija → aminokiselinski sastav (20-dim) | Pacijenti (176 upita/54 pac.) | −0,159 [−0,275, −0,059] | da, sva tri testa |

Zamena ESM-2 reprezentacije aminokiselinskim sastavom uništava najveći deo performansi modela; pad je vidljiv i na sopstvenom trening skupu (validaciona AUC 0,983→0,733), što isključuje objašnjenje da je reč samo o slabijoj generalizaciji — reprezentacija je suštinski siromašnija. Ovo je najveći pojedinačni efekat izmeren u celom radu.

<!-- RECENZIJA: (1) Δ MRR na pacijentima je po STAROJ, konflatirajućoj metrici (kombinovani MRR za pozitive+negative), prenetoj iz RAD.md — a §3.8.1 argumentuje da je ta metrika pogrešna za pacijentski skup. Preračunati po MRR+/NR- ili eksplicitno obrazložiti zašto je za ablacije kombinovana metrika prihvatljiva. Isto važi za §3.2 i §3.3.3.
(2) Tabela sa jednim redom — spojiti §3.1 + §3.2 + §3.3.3 u jednu ablacionu tabelu sa jasno označenim protokolom po redu.
(3) Δ = -0,159: MRR ne može ispod 0, pa je bazni MRR MLP(Hadamard) na pacijentima >= ~0,16-0,20 — a taj apsolutni broj se nigde ne navodi. Dati baznu vrednost. -->


## 3.2. Enkodiranje parova: Hadamard nadmašuje apsolutnu razliku

| Enkodiranje | Protokol | MRR / Δ MRR | Značajno? |
|---|---|---|---|
| Apsolutna razlika, 8 konfiguracija (sweep) | LOCO, svaka konfig. naspram sopstvenog polaznog modela na istoj podeli | MRR 0,1060–0,1737, dosledno negativno | lošije u svih 8 konfiguracija |
| Hadamard → apsolutna razlika (ista arhitektura) | Pacijenti (176/54) | Δ = −0,055 [−0,099, −0,016] | da, 2 od 3 testa |

Nijedna testirana konfiguracija apsolutne razlike nije dostigla polazni model pod LOCO-om; ovaj nalaz je motivisao prelazak na Hadamard produkt. Prednost Hadamard enkodiranja potvrđena je nezavisno i na pacijentskom skupu.

<!-- RECENZIJA: Raspon "MRR 0,1060–0,1737" zbunjuje: gornja granica je IZNAD BLAST-a (0,1243) i finalnog MLP(Hadamard) (0,1259), a tekst kaže "dosledno negativno". Razlog (svaka konfiguracija naspram sopstvenog baseline-a na svojoj podeli) je tačan, ali sirovi MRR ne sme u istu tabelu bez upozorenja. Izvestiti UPARENI Δ (raspon), ne sirovi MRR raspon; ili oba, uz jasnu napomenu da sirovi MRR nije uporediv sa ostalim tabelama. -->


## 3.3. Povećanje kapaciteta modela ne poboljšava performanse

### 3.3.1. Veća ESM-2 osnova (backbone) ne poboljšava nadgledani model

| Model | MRR | Protokol | Δ vs. BLAST | Značajno? |
|---|---:|---|---:|---|
| BLAST | 0,1243 | LOCO, čist trening | – | referenca |
| MLP(Hadamard), ESM-2 650M | 0,1259 | LOCO, čist trening | +0,0016 | ne, CI uključuje nulu |
| MLP(Hadamard), ESM-2 3B | 0,1131–0,1136 | LOCO, čist trening | −0,0107 do −0,0112 | da, značajno lošije |
| Cosine, ESM-2 3B prostor (bez treninga) | – | LOCO | −0,0007 vs. cosine 650M | ne |

<!-- RECENZIJA §3.3.1: (1) Prikazati stvarne CI (tabela kaže "CI uključuje nulu" ali CI se ne vidi) i Hits@k uz MRR. (2) "čist trening" — nedefinisan termin (nasleđen iz RAD.md); objasniti šta kontrastira. Ako znači "bez Inferred u treningu", protivreči §2.1.2 ("Inferred nisu korišćeni za trening") — razjasniti da li su raniji eksperimenti ipak koristili Inferred. (3) "cosine" kao ranker se koristi ovde, u §3.3.2 i §3.8.2, ali NIJE opisan u Metodologiji (§2 opisuje samo MLP i logističku regresiju, §2.7 samo BLAST) — dodati odeljak o baznim modelima (cosine ESM-2, BLAST, i bar jedan domenski prediktor: AllerCatPro / AlgPred / SDAP). (4) 3B na pacijentima: RAD.md je tvrdio da je 3B značajno BOLJI od BLAST-a na pacijentima — taj rezultat ovde nedostaje; ako je izostavljen, reći zašto. -->

### 3.3.2. Veći kapacitet interakcije parova ne poboljšava model

| Model | MRR | Protokol | Δ vs. cosine (0,1209) | Značajno? |
|---|---:|---|---:|---|
| Bilinearni model (low-rank outer product) | 0,1004 | LOCO | −0,0205 | da, značajno lošije |

<!-- RECENZIJA: Δ se ovde meri naspram cosine (0,1209), a u §3.3.1 naspram BLAST (0,1243) — uskladiti referentnu tačku kroz sve tabele (predlog: uvek naspram BLAST-a). Navesti dimenziju low-rank projekcije i broj parametara bilinearnog modela. -->


### 3.3.3. Nelinearna MLP klasifikacija donosi malo u odnosu na linearni klasifikator

| Zamenjena komponenta | Protokol | Δ MRR (bootstrap 95% CI) | Značajno? |
|---|---|---|---|
| MLP → linearni model (logistička regresija, isti Hadamard ulaz) | Pacijenti (176/54) | −0,009 [−0,020, +0,001] | ne, nijedan test |

Nijedan od tri nezavisna oblika povećanja kapaciteta — veći jezički model, izražajnija reprezentacija para, dublji klasifikator — nije doneo merljivo poboljšanje; kod backbone-a i interakcije para efekat je značajno negativan. Redosled važnosti komponenti: kvalitet reprezentacije ≫ način kombinovanja para > dubina klasifikatora.

<!-- RECENZIJA: (1) Δ na pacijentima ponovo po konflatirajućoj metrici — v. komentar u §3.1. (2) Dodati eksplicitan zaključak koji rad izbegava: MLP SAMOSTALNO ne prevazilazi BLAST u agregatu (LOCO 0,1259 vs 0,1243, n.z.; dobici u Q1-Q3 iz §3.7.2 se poništavaju sa gubitkom -0,196 u Q4). Vrednost je isključivo u KOMPLEMENTARNOSTI / fuziji — reći to ovde i u zaključku. -->


## 3.4. Model se oslanja na stabilan podskup dimenzija embeddinga

Sve analize u ovom odeljku sprovedene su na linearnom Hadamard modelu treniranom nad celim trening skupom (bez LOCO/pacijentskog holdout-a), preko 5 nezavisnih semena (42, 137, 271, 314, 500).

<!-- RECENZIJA: Kontradikcija — preambula kaže "bez LOCO/pacijentskog holdout-a", ali §3.4.3 koristi "LOCO (40 folda)". Uskladiti (verovatno: analize stabilnosti/interpretacije na celom skupu, orezivanje 3.4.3 pod LOCO — to eksplicitno reći). -->


### 3.4.1. Stabilnost najbolje rangiranih dimenzija kroz slučajna semena (cross-seed stability)

| Skup | Prosečan Jaccard indeks preko 5 semena |
|---|---:|
| Top-20 dimenzija po \|težini\| | 0,80 |
| Top-50 dimenzija po \|težini\| | 0,77 |

17/20, odnosno 42/50 dimenzija pojavljuje se u top-skupu kod ≥4 od 5 semena — model dosledno koristi skoro isti mali podskup dimenzija, ne nasumičan izbor pri svakom treningu.

<!-- RECENZIJA: Dodati null-model: očekivani slučajni Jaccard za top-20 od 1280 je ~1,6%, za top-50 ~4%. Jaccard 0,80/0,77 je jasno iznad slučajnog, ali to treba pokazati (permutacioni test ili analitički bazni nivo). Obrazložiti zašto se baš top-50 (→ 42 stabilne dimenzije) uzima kao "relevantni skup" za §3.5-3.6, a ne top-20. -->


### 3.4.2. Pojedinačna diskriminativnost ne objašnjava u potpunosti upotrebu dimenzija u modelu

Preklapanje top-20/top-50 dimenzija po \|težini\| sa top-20/top-50 dimenzija po Cohenovom $d$ (unutar istog semena) iznosi svega 6–12%, dosledno preko svih 5 semena. Ovo ne protivreči umerenoj globalnoj korelaciji \|težina\|↔Cohen's $d$ (Pearson ≈0,41, Spearman ≈0,51, p<10⁻⁵⁸) — globalna korelacija preko 1280 dimenzija ne garantuje poklapanje u samom vrhu raspodele.

### 3.4.3. Orezivanje pojedinačno diskriminativnih dimenzija ne reprodukuje performanse

| Odsečena varijanta | Protokol | Δ (5 semena, upareno) | Pobeda odsečene varijante |
|---|---|---|---|
| Zadrži top 50% dimenzija po train-fold Cohen's $d$, ostatak nuliran | LOCO (40 folda) | mean Δ = +0,0003 (std 0,0037) | 1/5 semena |

Prosečna razlika je blizu nule, ali to nije "nema efekta" — u 4 od 5 semena je odsecanje blago pogoršalo rezultat; jedino seme u kome je "pobedilo" imalo je neobično nizak baseline u tom konkretnom semenu (regresija ka sredini, ne sistematsko poboljšanje). Odsecanje po Cohen's $d$ uklanja i deo dimenzija koje model stvarno koristi (3.4.2), pa ne uspeva da odvoji šum od signala.

<!-- RECENZIJA: Sa n=5 semena i std 0,0037, mean Δ=+0,0003 je prosto "nema detektabilnog efekta". Post-hoc objašnjenje da je 1 "pobedničko" seme imalo nizak baseline deluje kao odbacivanje neugodnog rezultata — pošteniji iskaz: efekat nije detektabilan pri ovoj snazi. Za jaču tvrdnju: više semena ili formalni test ekvivalencije (TOST). -->


## 3.5. Većina ključnih dimenzija prati merljiva svojstva proteina

Za svih ~1535 proteina u pool-u izračunata su realna biofizička i strukturna svojstva direktno iz FASTA sekvenci (dužina, GRAVY hidrofobnost, naboj na pH 7, aromatičnost, izoelektrična tačka, indeks nestabilnosti, udeo sekundarne strukture, pun aminokiselinski sastav), i korelisana (Spearman) sa vrednošću svake od 42 dimenzije (3.4.1) preko celog pool-a.

<!-- RECENZIJA: "~1535" vs 1.536 u §2.1.1 — navesti tačan broj. §3.5.3 kaže "21 deskriptor", ali ovde nabrojano: 6 biohemijskih + sekundarna struktura (1 ili 3?) + 20 AA frekvencija >> 21 — uskladiti tačan broj i listu. Dodati FDR korekciju (42 dim × broj deskriptora) i korigovane p-vrednosti. -->


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

Ukupno **40/42 (95%) relevantnih dimenzija** korelira sa bar jednim od 21 testiranih realnih deskriptora. Preostale dve (nazvane po formalnom indeksu dimenzije) ne prate nijedno kontinuirano svojstvo, ali formalna provera (Mann–Whitney) pokazuje da svaka kodira kategorijsku pripadnost proteinskoj familiji (PR-10, p=1,4×10⁻¹⁷; Tropomyosin, p=4,1×10⁻⁹) — objašnjenje zašto ih korelacija sa kontinuiranim svojstvima nije uhvatila. Nijedna dimenzija ne prelazi \|r\|=0,5; ni jedna nije "čist" enkoder jedne osobine. Ovo su korelacione, ne uzročne asocijacije — ne dokazuju da navedena svojstva pokreću prediktivni signal modela.

<!-- RECENZIJA: "Nijedna dimenzija ne prelazi |r|=0,5" vs tabela §3.5.1 koja navodi raspon "0,33–0,50" — granični slučaj, preformulisati u "|r| <= 0,5". Mann–Whitney za familijsku pripadnost: koliko familija je testirano? Ako sve, navesti korekciju (p=1,4e-17 preživljava). Kauzalni disklejmer je dobar i konzistentan sa §2.6.2 i §4.8. -->


## 3.6. Eksplicitne međudimenzionalne interakcije ne pružaju merljiv dodatni signal

Za svih $\binom{42}{2}=861$ parova relevantnih dimenzija, poređen je aditivni logistički model (dve dimenzije) sa modelom koji dodaje eksplicitan proizvod (interakcioni član), 5-strukom unakrsnom validacijom nad celim trening skupom (jedno seme, 42).

<!-- RECENZIJA: Samo 1 seme ovde, a §3.4 koristi 5 — nedosledna strogost; proširiti na 5 ili obrazložiti. Nalaz je negativan (max ΔAUC 0,0006), pa nekorigovano višestruko testiranje ide u prilog robusnosti — reći to eksplicitno. ΔAUC je možda pogrešna mera za mali interakcioni efekat u rangiranju; razmotriti ΔMRR/Δrecall@k na held-out foldu. -->


| Mera | Vrednost preko 861 parova |
|---|---:|
| Maksimalni dobitak od interakcionog člana (ΔAUC) | 0,0006 |
| Medijalni dobitak | ≈0,000004 |

Nijedan par ne pokazuje merljiv dobitak od eksplicitne interakcije — aditivna kombinacija dve dimenzije već sadrži skoro svu njihovu zajedničku prediktivnu informaciju. Ovo je nezavisna potvrda nalaza iz 3.3.3: interakcija koju Hadamard produkt nosi je ona ugrađena po konstrukciji (ista dimenzija, dva proteina), ne interakcija između različitih dimenzija unutar predstave.

## 3.7. MLP 

<!-- RECENZIJA: Naslov odeljka 3.7 je nedovršen ("## 3.7. MLP ") — dopuniti, npr. "MLP naspram BLAST-a: gde nastaje razlika". -->

### 3.7.1. Ukupno LOCO poređenje sa BLAST-om

Videti tabelu u 3.3.1: MLP(Hadamard) 650M i BLAST su statistički izjednačeni pod LOCO-om (0,1259 vs. 0,1243, CI uključuje nulu).

### 3.7.2. Dobitak MLP-a raste kako se performanse BLAST-a smanjuju

| Kvartil BLAST rr (LOCO, ne-crowded upiti, n=1928) | mean BLAST rr | mean MLP rr | Δ (MLP−BLAST) |
|---|---:|---:|---:|
| Q1 (najslabiji) | 0,026 | 0,055 | +0,029 |
| Q2 | 0,075 | 0,136 | +0,061 |
| Q3 | 0,175 | 0,200 | +0,025 |
| Q4 (najjači) | 0,607 | 0,411 | −0,196 |

Spearman(BLAST rr, Δ) = −0,404 (p=1,5×10⁻⁷⁶). MLP nadmašuje BLAST u donja tri kvartila (75% upita); u gornjem kvartilu, gde je BLAST već blizu maksimuma, MLP zaostaje. Familije sa dijagnostikovanim "zagušenjem" kandidata (nsLTP/Profilin/PR-10) imaju sistemski nizak BLAST rr (mean 0,052 naspram 0,193 za ostale familije) — crowding je najekstremniji, ne poseban, slučaj ovog istog obrasca.

<!-- RECENZIJA: (1) Naslov preuveličava monotonost: nije monotono — Q1=+0,029, Q2=+0,061 (MAKSIMUM), Q3=+0,025, Q4=−0,196. Preciznije: "MLP dobija u donja tri kvartila, gubi izrazito u gornjem". (2) Nema CI ni testova po kvartilu — dodati. (3) n=1928: jedinice — proteini-upiti ili (upit, partner) probe? Uskladiti sa definicijom LOCO evaluacije. (4) Gubitak −0,196 u Q4 je VEĆI od svakog pojedinačnog dobitka — istaći i povezati sa agregatom (MLP sam ≈ BLAST). (5) "zagušenje" još nedefinisano operativno (v. §2.7.3). (6) Obrazac je bar delom artefakt NEFILTRIRANOG ranga — gusto povezani čvorovi imaju mnogo poznatih partnera koji se međusobno guraju naniže; proveriti sa FILTERED rangom. -->


### 3.7.3. Upitima sa slabim BLAST-om dominira konkurencija kandidata, a ne nužno nizak identitet sekvence

| Grupa (LOCO) | mean sequence identity % | % upita koji dodiruju crowded familiju |
|---|---:|---:|
| BLAST_jak (rang iznad medijane) | 60–62% | 15,6–27,1% |
| BLAST_slab (rang ispod medijane) | 48–53% | 73,8–81,9% |

BLAST_slab populacija ima umeren, ne nizak, sirov identitet — pada u srednji tercil sekvencijalne sličnosti definisan u 3.7.4/pacijentskoj stratifikaciji, ne u niski. Nizak *rang* BLAST-a je posledica konkurencije mnogo sličnih kandidata unutar iste familije, ne odsustva homologije.

<!-- RECENZIJA: Rasponi ("60–62%", "15,6–27,1%") — verovatno preko semena/foldova; označiti šta raspon predstavlja. Ako je §3.7.2 isključio "crowded upite" (n=1928 "ne-crowded"), kako BLAST_slab ovde ima 73–82% upita "koji dodiruju crowded familiju"? Razjasniti "crowded upit" vs "upit koji dodiruje crowded familiju". Referenca "tercil ... definisan u 3.7.4" — §3.7.4 ne definiše tercile (pacijentska stratifikacija po tercilima je u RAD.md §4.5); popraviti unakrsnu referencu. -->


### 3.7.4. Nijedan jednostavan biohemijski potpis ne razlikuje pobede MLP-a od njegovih poraza

| Svojstvo (unutar BLAST_slab, n=1086 pobeda MLP-a / 714 poraza) | p (Mann–Whitney) | Rank-biserial efekat |
|---|---:|---:|
| BLAST skor (sirov) | 2,8×10⁻⁵ | 0,117 (mali) |
| Naboj, hidrofobnost, aromatičnost, instabilnost, heliks (razlika para) | 0,37–0,80 | 0,01–0,03 (zanemarljivo) |

Ni sa velikom statističkom snagom (n=1800) nijedno testirano biofizičko svojstvo para ne razdvaja pobede od poraza MLP-a unutar BLAST-slabe zone; jedini (mali) signal je da MLP dodatno dobija kada je BLAST skor i unutar te zone niži — isti mehanizam iz 3.7.2, na finijoj rezoluciji, ne nov nezavisan signal.

<!-- RECENZIJA: n=1086+714=1800 "BLAST_slab" naspram n=1928 "ne-crowded" u §3.7.2: ako je BLAST_slab "ispod medijane ranga" (~50%), trebalo bi ~964, ne 1800. Ukupan N i način dobijanja svakog podskupa mora biti jasno naveden — brojevi se trenutno ne uklapaju. -->


## 3.8. Nezavisna validacija na nivou pacijenata

### 3.8.1. MLP(Hadamard) naspram BLAST-a

Skriveni nalaz (pozitivan ili negativan) evaluira se povratkom njegovog ranga; pošto je "uspeh" suprotno definisan za dva tipa proba (nizak rang za pozitiv, visok rang za negativ), izveštavaju se **dve odvojene, uparene metrike**, ne jedan kombinovan MRR:

$$\mathrm{MRR}_{+} = \mathrm{mean}(1/\mathrm{rang}), \quad \mathrm{NR}_{-} = \mathrm{mean}\left(\frac{\mathrm{rang}-1}{N-1}\right)\ (\text{veće} = \text{bolje potisnuto})$$

| Metrika | MLP | BLAST | Δ | Wilcoxon p | Cluster-perm. p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---|
| MRR₊ (n=100 proba, 34 uparena pac.) | 0,203 | 0,174 | +0,021–0,029 | 0,0156 | 0,0227 | [+0,0045, +0,0423] |
| NR₋ (n=76 proba, 37 uparenih pac.) | 0,441 | 0,599 | −0,142–0,158 | 0,0006 | <0,0001 | [−0,2135, −0,0787] |

Oba efekta su statistički značajna, u suprotnim smerovima, sva tri testa. MLP(Hadamard) značajno bolje prioritizuje prave unakrsno reaktivne partnere; BLAST značajno bolje potiskuje prave negativne kandidate, efektom veće apsolutne i relativne veličine. Modeli su komplementarni specijalisti, ne jedan univerzalno superioran.

<!-- RECENZIJA: Dobra i poštena reformulacija — ali sa posledicama kroz ceo rad:
(1) Protokol (skrivanje pozitivnog/negativnog nalaza, dve metrike) PRIPADA u §2.5.2, ne u Rezultate.
(2) Ista logika ruši kombinovanu-MRR metriku korišćenu u §3.1, §3.2, §3.3.3 (ablacije na pacijentima) i §3.8.2 (cosine) — ti brojevi se moraju preračunati ili eksplicitno ograditi.
(3) Δ dat kao raspon ("+0,021–0,029", "−0,142–0,158") — objasniti odakle raspon (point-estimate = prosta razlika sredina: +0,029 i −0,158).
(4) Mali uzorci: MRR+ na 100 proba / 34 pacijenta, NR- na 76 / 37; "34 uparena" vs "54 pac." drugde — definisati "upareno". Diskutovati snagu (CI za MRR+ [+0,0045,+0,0423] je širok u odnosu na efekat).
(5) Za pozitive dodati recall@k (klinički interpretabilnije); za negative razmotriti "1 − recall@k za negative".
(6) Apstrakt i Zaključak MORAJU reći "komplementarno / senzitivnost naspram specifičnosti", NE "MLP bolji" / "najbolje rangiranje". -->


### 3.8.2. ESM kosinusna sličnost naspram MLP(Hadamard)

<!-- NAPOMENA: brojevi u ovoj tabeli su preneti direktno iz RAD.md 4.5 i NISU nezavisno ponovo provereni pod ispravljenom (pravac-svesnom) metrikom primenjenom u 3.8.1 -- videti pre finalizacije. -->
<!-- RECENZIJA — BLOKATOR ZA SLANJE: ovaj pododeljak koristi upravo kombinovanu MRR metriku koju §3.8.1 proglašava neispravnom za pacijentski skup. Pre slanja: (a) preračunati cosine vs BLAST i cosine vs MLP po MRR+/NR-, ili (b) ukloniti §3.8.2 i zaključak o cosine-u izvesti samo iz LOCO rezultata (§3.3.1: cosine 3B prostor Δ −0,0007 vs cosine; cosine < BLAST na LOCO-u). Ostavljanje "videti pre finalizacije" komentara u tekstu je znak da rad nije spreman. -->

| Poređenje | Wilcoxon p | Cluster-permutacija p | Bootstrap 95% CI |
|---|---:|---:|---|
| Cosine vs. BLAST (svi upiti) | 0,7994 | 0,0205 | [−0,0921, −0,0121] |
| Cosine vs. MLP(Hadamard) (svi upiti) | 0,0172 | 0,0026 | [−0,1042, −0,0214] |

Cosine (netreniran signal nad istom reprezentacijom) je najslabiji od sva tri signala — prednost MLP(Hadamard)-a nad BLAST-om ne potiče iz same ESM-2 reprezentacije, već iz naučene Hadamard transformacije nad njom.

## 3.9. Sažetak nalaza

| Istraživačko pitanje | Eksperiment | Glavni nalaz |
|---|---|---|
| RQ1 | ESM naspram sastava (3.1) / cosine naspram MLP (3.8.2) | ESM reprezentacija nosi dominantan signal; sam trening (ne sama reprezentacija) daje prednost nad BLAST-om |
| RQ2 | Apsolutna razlika naspram Hadamard produkta (3.2) | Hadamard značajno bolji, potvrđeno na LOCO i pacijentima |
| RQ3 | Backbone/bilinear/MLP naspram linearnog (3.3, 3.6) | Veći kapacitet ne pomaže ni na jednom od tri testirana nivoa; interakcije između dimenzija ne postoje merljivo |
| RQ4 | Stabilnost i interpretacija dimenzija (3.4–3.5) + BLAST-komplementarnost (3.7–3.8) | Model koristi stabilan, delom biohemijski interpretabilan podskup dimenzija; prednost nad BLAST-om raste kontinuirano kako BLAST slabi, i statistički je značajna u oba smera na pacijentima (senzitivnost naspram specifičnosti) |

<!-- RECENZIJA: (1) RQ4: "raste kontinuirano kako BLAST slabi" — nije monotono (Q2>Q1, §3.7.2). (2) "značajna u oba smera na pacijentima" može zavarati da MLP dobija oba — jedan smer FAVORIZUJE BLAST. Preformulisati: "MLP dominira u senzitivnosti, BLAST u specifičnosti (većim efektom)". (3) RQ1: "sam trening (ne sama reprezentacija) daje prednost nad BLAST-om" — kvalifikovati: na LOCO-u nema prednosti, na pacijentima je podeljeno. (4) Dodati red: MLP samostalno ≈ BLAST u agregatu, vrednost je u fuziji. -->

---
# 4. Diskusija

## 4.1. Kvalitet reprezentacije i način formiranja para važniji su od povećavanja složenosti modela

<!-- RECENZIJA: Diskusija (4.1-4.9) je uglavnom dobro odmerena i poštena (naročito 4.8). Uskladiti sa §3.8.1: gde god se pominje "prednost MLP-a nad BLAST-om", dodati "u senzitivnosti; BLAST je bolji u specifičnosti". Dodati kratak pasus o ograničenju metrike (nefiltriran MRR / bez gradirane relevantnosti) i kako bi filtered MRR + Hits@k + nDCG + recall@k kriva promenili/učvrstili nalaze. -->

Rezultati pokazuju da povećavanje složenosti modela nije samo po sebi dovelo do boljeg predviđanja unakrsne reaktivnosti. ESM-2 reprezentacije sadržale su informaciju korisnu za razlikovanje cross-reactive parova koja se ne može objasniti samo sastavom aminokiselina. Istovremeno, veći ESM-2 model nije doneo poboljšanje u odnosu na model sa 650 miliona parametara. Slično tome, eksplicitno povećavanje prostora interakcija pomoću bilinearne reprezentacije nije poboljšalo rezultat, dok poređenje MLP-a sa linearnim klasifikatorom nije pokazalo jasnu prednost dodatne nelinearne složenosti.

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

Na taj način svaka latentna dimenzija jednog proteina ulazi u model zajedno sa odgovarajućom dimenzijom drugog proteina. Ovo omogućava modelu da detektuje obrasce zajedničke aktivacije ili suprotne aktivacije određenih latentnih osobina. Kod apsolutne razlike, s druge strane, informacija je zasnovana na udaljenosti između odgovarajućih komponenti i ne zadržava njihov zajednički znak.

Dodatna analiza interakcija između dimenzija pokazala je da eksplicitno uvođenje proizvoda između različitih dimenzija nije donelo značajno poboljšanje. To je važno za interpretaciju Hadamardovog rezultata: prednost ovog enkodiranja ne može se jednostavno pripisati tome što model koristi sve moguće parove latentnih dimenzija. Njegova uloga je uža — omogućava **interakciju između odgovarajućih dimenzija reprezentacija dva proteina**.

Stabilnost izabranih dimenzija kroz različite inicijalizacije dodatno pokazuje da model ne koristi potpuno proizvoljne komponente embeddinga. Ipak, relativno mala podudarnost između dimenzija sa najvećim klasifikacionim težinama i dimenzija sa najvećom pojedinačnom diskriminativnošću pokazuje da doprinos pojedinačne dimenzije ne treba posmatrati izolovano. Model koristi kombinaciju više komponenti reprezentacije, čiji doprinos zavisi od odnosa između dva proteina.

Zbog toga se Hadamardov proizvod može posmatrati kao jednostavan način da se iz dva pojedinačna ESM-2 embeddinga formira reprezentacija njihovog odnosa. Rezultati ne pokazuju da ova reprezentacija eksplicitno modeluje sve moguće proteinske interakcije. Pokazuju da je upravo ova ograničena, dimenzijski usklađena forma interakcije bila pogodnija za zadatak predviđanja unakrsne reaktivnosti od testiranih alternativnih enkodiranja.

## 4.3. Zašto linearna predikcija može biti dovoljna nakon učenja reprezentacije

Poređenje MLP klasifikatora sa logističkom regresijom nad istom Hadamardovom reprezentacijom pokazalo je da dodatna nelinearnost klasifikatora nije donela jasno poboljšanje performansi. Ovaj rezultat sugeriše da značajan deo složenosti problema nije nužno potrebno učiti na nivou završnog klasifikatora. ESM-2 je prethodno transformisao proteinsku sekvencu u visokodimenzionalnu reprezentaciju, dok je Hadamardov proizvod omogućio da se informacije iz dve takve reprezentacije kombinuju u reprezentaciju proteinskog para.

U tom kontekstu, uloga klasifikatora može biti pre svega da kombinuje već postojeće signale iz pairwise reprezentacije. Ako su relevantni obrasci nakon ovog postupka dovoljno separabilni, linearni model može da ih iskoristi bez potrebe za dodatnim slojevima nelinearnih transformacija.

Ovaj nalaz ne znači da je odnos između proteina inherentno linearan. ESM-2 reprezentacija je rezultat nelinearnog procesa učenja, a Hadamardov proizvod uvodi multiplicativnu interakciju između odgovarajućih dimenzija dva proteina. Linearna priroda završnog klasifikatora zato ne treba da se tumači kao odsustvo nelinearnosti u celom modelu. Preciznije, rezultat pokazuje da **nakon formiranja odgovarajuće reprezentacije para nije pronađena potreba za dodatnom nelinearnom transformacijom na nivou klasifikatora**.

Ovakva interpretacija je u skladu sa poređenjem različitih nivoa modela: povećavanje veličine ESM-2 modela, proširivanje prostora interakcija i dodavanje nelinearnosti završnom klasifikatoru nisu dali stabilno poboljšanje. Najizraženija razlika pojavila se pri izboru načina na koji se pojedinačne proteinske reprezentacije pretvaraju u reprezentaciju para.

## 4.4. Signal je raspoređen širom embedding prostora

Analiza latentnih dimenzija pokazuje da informacija koju model koristi nije koncentrisana u malom broju izolovanih komponenti embeddinga. Pet različitih inicijalizacija pokazalo je visoku stabilnost skupa najvažnijih dimenzija: 17 od 20 najčešće odabranih dimenzija i 42 od 50 najčešće odabranih dimenzija pojavili su se u odgovarajućem skupu u najmanje četiri od pet pokretanja. To ukazuje da model kroz različite inicijalizacije dolazi do sličnog podskupa relevantnih komponenti.

Međutim, sama stabilnost najvažnijih dimenzija nije dovoljna da pokaže da je signal koncentrisan upravo u njima. Kada su dimenzije sa najvećom pojedinačnom diskriminativnošću uklonjene na osnovu Cohenovog \(d\), njihovo uklanjanje nije dovelo do sistematskog poboljšanja. Još važnije, zadržavanje samo najdiskriminativnijih dimenzija nije pouzdano reprodukovalo performanse pune reprezentacije. Ovo pokazuje da pojedinačna diskriminativnost dimenzije nije isto što i njen doprinos konačnoj predikciji.

Dodatnu podršku ovoj interpretaciji daje analiza bioloških svojstava. Od 42 izdvojene ključne dimenzije, 40 je pokazalo povezanost sa najmanje jednim od ispitivanih biohemijskih ili strukturnih deskriptora. Povezanosti su obuhvatale osobine kao što su naelektrisanje, izoelektrična tačka, dužina proteina, hidrofobnost, aromatičnost, nestabilnost, sekundarna struktura i učestalost pojedinačnih aminokiselina. Međutim, nijedna pojedinačna korelacija nije bila dovoljno jaka da objasni ponašanje dimenzije sama za sebe.

Ovi rezultati ukazuju da ESM-2 embedding ne treba posmatrati kao skup potpuno nezavisnih i lako interpretabilnih osobina. Latentne dimenzije mogu istovremeno nositi više povezanih informacija, dok njihov značaj zavisi od kombinacije sa odgovarajućom reprezentacijom drugog proteina. Zbog toga analiza pojedinačnih dimenzija daje samo delimičnu sliku načina na koji model donosi odluku.

Važan dodatni rezultat dolazi iz testa eksplicitnih interakcija između različitih latentnih dimenzija. Uvođenje proizvoda između svih parova od 42 ključne dimenzije nije donelo praktično poboljšanje u odnosu na aditivni model. To sugeriše da se korisna interakcija u Hadamardovoj reprezentaciji ne mora proširivati na sve moguće kombinacije dimenzija. Dovoljno je da se odgovarajuće dimenzije dva proteina kombinuju, dok dodatno modelovanje interakcija između različitih latentnih dimenzija nije pokazalo novu informaciju.

Zajedno, ovi nalazi podržavaju sliku **distribuiranog signala**. Model koristi stabilan skup latentnih komponenti, ali njihov doprinos nije moguće svesti na nekoliko izolovanih dimenzija. Biološke osobine povezane sa tim komponentama takođe nisu dovoljne da pojedinačno objasne predikciju. Relevantna informacija je raspoređena kroz embedding prostor i postaje korisna prvenstveno kroz kombinovanje reprezentacija dva proteina.

## 4.5. Šta model uči što jednostavni deskriptori ne obuhvataju

Poređenje ESM-2 reprezentacija sa jednostavnim deskriptorima pokazalo je da informacija korisna za predikciju unakrsne reaktivnosti nije obuhvaćena samo osnovnim biohemijskim osobinama proteina. ESM-2 reprezentacija je na nezavisnoj evaluaciji pokazala znatno bolje ponašanje od reprezentacije zasnovane na sastavu aminokiselina. Istovremeno, analiza pojedinačnih latentnih dimenzija pokazala je da mnoge od njih jesu povezane sa merljivim biohemijskim i strukturnim svojstvima.

Ova dva rezultata nisu međusobno kontradiktorna. ESM-2 embedding može sadržati informacije povezane sa poznatim biohemijskim osobinama, ali ih organizovati u višedimenzionalnu reprezentaciju koju jednostavni zbirni deskriptori ne mogu u potpunosti opisati. Sama činjenica da je dimenzija korelisana sa određenim svojstvom ne pokazuje da je upravo to svojstvo mehanizam koji određuje predikciju.

Dodatno, analiza parova na kojima je MLP nadmašio BLAST u odnosu na parove na kojima je BLAST bio bolji nije otkrila jednostavan biohemijski obrazac koji bi razdvojio ove dve grupe. Razlike u ispitivanim deskriptorima nisu pružile stabilan kriterijum za predviđanje toga kada će model biti uspešniji od klasičnog poravnanja sekvenci.

Najopreznija interpretacija ovih rezultata jeste da model koristi **finiju organizaciju informacija u latentnom prostoru ESM-2** koja nije direktno predstavljena pojedinačnim deskriptorima. Takva interpretacija je u skladu sa činjenicom da je signal distribuiran kroz više dimenzija i da njihova pojedinačna svojstva ne objašnjavaju u potpunosti ponašanje modela. Međutim, ovi rezultati ne predstavljaju dokaz da je upravo geometrija latentnog prostora uzrok boljih predikcija. Za potvrdu takvog mehanizma bile bi potrebne dodatne intervencione i reprezentacione analize.

## 4.6. Komplementarnost sa poravnanjem sekvenci

Poređenje MLP-a sa BLAST-om pokazuje da ESM-2 reprezentacija ne predstavlja jednostavnu zamenu za klasično poravnanje sekvenci. Njihova relativna uspešnost zavisi od informacija koje su dostupne iz samih sekvenci i od strukture prostora kandidata.

Na LOCO evaluaciji ukupna razlika između MLP-a i BLAST-a bila je mala, što pokazuje da MLP nije univerzalno bolji prediktor. Međutim, analiza po jačini BLAST signala pokazala je izraženu promenu odnosa između dva pristupa. U najnižim kvartilima BLAST ranga MLP je ostvarivao veći recipročni rang, dok je u najjačem kvartilu BLAST imao jasnu prednost. Razlika između performansi MLP-a i BLAST-a pritom je negativno korelisala sa jačinom BLAST signala.

Ovaj obrazac pokazuje da vrednost ESM-2 reprezentacije nije ravnomerno raspoređena kroz sve slučajeve. Kada sekvencijalno poravnanje već daje snažan signal, dodatna informacija koju MLP izvlači iz embeddinga ima manji doprinos. Kada je BLAST signal slabiji, ESM-2 reprezentacija može pružiti informaciju koja nije dovoljno izražena kroz direktnu sekvencijalnu sličnost.

Analiza BLAST-slabih slučajeva dodatno pokazuje da slab BLAST rang ne znači nužno i nisku sekvencijalnu sličnost. U značajnom broju takvih slučajeva problem nastaje zbog konkurencije između više kandidata sa sličnim BLAST rezultatima. MLP u tim situacijama može drugačije rangirati kandidate koristeći informacije iz njihove latentne reprezentacije.

Zbog toga se najprirodnije tumačenje odnosa između ova dva pristupa ne zasniva na pitanju koji je model „bolji“. BLAST i MLP predstavljaju **dva različita izvora informacije o odnosu između proteina**. BLAST direktno koristi sekvencijalnu sličnost, dok MLP koristi obrasce prisutne u ESM-2 embedding prostoru. Njihova komplementarnost je naročito izražena kada sekvencijalni signal nije dovoljan da jednoznačno rangira kandidate.

Ovaj rezultat ima i praktičnu implikaciju za razvoj prediktivnih sistema. Umesto da se ESM-2 model posmatra kao zamena za postojeće metode poravnanja, prirodnije je posmatrati ga kao dopunski izvor informacije koji može biti najkorisniji upravo u slučajevima u kojima klasična sekvencijalna sličnost daje slab ili neodlučan signal.

## 4.7. Zašto „slab BLAST“ ne znači „nizak identitet sekvence“

Rezultati pokazuju da slab BLAST signal ne treba automatski tumačiti kao nisku sekvencijalnu sličnost. U grupi kandidata sa slabijim BLAST rangom nalazio se značajan broj proteina sa umerenim procentom identiteta. Problem je često bio u tome što je više kandidata istovremeno imalo relativno slične sekvencijalne rezultate.

Ovaj nalaz je važan za interpretaciju komplementarnosti MLP-a i BLAST-a. MLP nije nužno najkorisniji onda kada između dva proteina ne postoji nikakva sekvencijalna sličnost. Njegova prednost može da se pojavi i u situacijama u kojima sekvencijalna sličnost postoji, ali BLAST na osnovu nje ne može dovoljno dobro da izdvoji relevantnog kandidata iz grupe sličnih proteina.

Analiza BLAST-slabih slučajeva podržava upravo takvo tumačenje. U tim slučajevima nije pronađen jednostavan biohemijski obrazac koji bi razlikovao parove na kojima MLP dobija od onih na kojima gubi. Umesto toga, rezultati ukazuju da je važan deo problema povezan sa načinom rangiranja konkurentskih kandidata.

Zbog toga termin „slab BLAST“ u ovom radu treba razumeti prvenstveno kao **slabiji rang kandidata prema BLAST signalu**, a ne kao sinonim za nisku sekvencijalnu sličnost.

## 4.8. Šta nam model ne govori?

Iako analiza embedding prostora pokazuje povezanost latentnih dimenzija sa različitim biohemijskim i strukturnim svojstvima, ovi rezultati ne omogućavaju zaključak da ta svojstva uzrokuju unakrsnu reaktivnost. Korelacija latentne dimenzije sa naelektrisanjem, izoelektričnom tačkom ili sekundarnom strukturom pokazuje samo da su informacije povezane u reprezentaciji.

Model takođe ne daje direktan dokaz konkretnog imunološkog mehanizma. Predviđanje proteinskog para kao cross-reactive ne pokazuje da je model naučio specifičan način vezivanja IgE antitela, određenu epitopsku interakciju ili drugi pojedinačni molekularni mehanizam.

Slično tome, pojedinačna dimenzija ESM-2 embeddinga ne treba da se tumači kao nosilac jedne jedine biološke osobine. Iako su neke dimenzije povezane sa merljivim svojstvima, rezultati o distribuiranom signalu pokazuju da se korisna informacija verovatnije formira kroz kombinaciju više komponenti.

Zbog toga interpretaciju modela treba ograničiti na ono što je direktno podržano eksperimentima: ESM-2 embedding sadrži informaciju korisnu za ovaj zadatak, Hadamardovo enkodiranje omogućava njeno korišćenje na nivou proteinskog para, a deo te informacije povezan je sa poznatim osobinama proteina. Precizan biološki mehanizam koji stoji iza tih obrazaca ostaje otvoreno pitanje.

## 4.9. Metodološke implikacije

Izbor evaluacionog protokola značajno utiče na zaključke o sposobnosti modela da predviđa unakrsnu reaktivnost. Zbog povezanosti proteinskih parova kroz zajedničke komponente, nasumična podela parova može dovesti do curenja informacija između treninga i testa. Zbog toga je u radu korišćena LOCO validacija, pri kojoj se čitave povezane komponente proteinskog grafa izdvajaju iz treninga i koriste za testiranje. Na taj način se procenjuje generalizacija na proteinske odnose koji nisu direktno prisutni u trening skupu.

Rezultati dobijeni na nivou pojedinačnih proteinskih parova ne predstavljaju nužno ponašanje modela u nezavisnom kliničkom okruženju. Zbog toga je pacijentska evaluacija tretirana kao zaseban nivo validacije, pri čemu podaci pacijenata nisu korišćeni za treniranje modela. Ovakva evaluacija omogućava ispitivanje da li obrasci naučeni na kuriranom skupu proteinskih odnosa imaju vrednost i kada se model primeni na nezavisne nalaze.

Statistička analiza je takođe prilagođena strukturi podataka. Kada više proteinskih parova potiče iz istog literaturnog izvora ili kada više posmatranja pripada istom pacijentu, ta posmatranja nisu nužno nezavisna. Zbog toga je pri proceni razlika između modela potrebno uzeti u obzir nivo na kojem nastaje zavisnost, a ne tretirati svaki par kao potpuno nezavisnu jedinicu.

Ove metodološke odluke menjaju način na koji treba interpretirati rezultate. Poboljšanje na slučajno podeljenom skupu nije dovoljno da pokaže generalizaciju na nove proteinske odnose, dok rezultat na jednom proteinskom paru nije dovoljan da pokaže korisnost za nezavisne pacijente. Kombinovanje LOCO evaluacije, nezavisne pacijentske evaluacije i statističkih testova na odgovarajućem nivou zavisnosti daje strožu procenu toga šta model zaista nauči i gde se njegova prednost pojavljuje.


# 4.10. Ograničenja

<!-- RECENZIJA: Numeracija — "4.10" je označeno kao glavni odeljak (#) ali je pododeljak Diskusije; a §5 se pojavljuje DVA puta ("# 5. Budući pravci" i "# 5. Zaključak"). Preurediti: 4.10 Ograničenja (##), 5. Budući pravci, 6. Zaključak. Dodati ograničenja: (a) metrika — nefiltriran rang, bez gradirane relevantnosti, mikro-prosek preteže velike familije; (b) deduplikacija pool-a proizvoljnim pravilom; (c) samo BLAST kao baseline, bez domenskih prediktora; (d) mala pacijentska kohorta (34/37 uparenih pacijenata). -->

## 4.10.1. Pozitivno-neobeležena priroda skupa podataka

Skup podataka ne omogućava pouzdano razlikovanje svih negativnih parova od neobeleženih parova. Odsustvo dokumentovane unakrsne reaktivnosti ne znači nužno da ona ne postoji, zbog čega negativni primeri mogu sadržati neotkrivene pozitivne odnose. Ograničenje je posebno važno pri tumačenju performansi klasifikatora.

## 4.10.2. Ograničen broj nezavisnih bioloških primera

Iako skup sadrži veliki broj proteinskih parova, broj međusobno nezavisnih bioloških primera je ograničen njihovom povezanošću kroz zajedničke proteine i literaturne izvore. Zbog toga je korišćena LOCO validacija kako bi se smanjio uticaj ove zavisnosti. Ipak, rezultati treba da se tumače kao procena generalizacije na nove povezane komponente, a ne kao potpuno nezavisna procena na velikom broju nezavisnih eksperimenata.

## 4.10.3. Ograničena nezavisna kohorta pacijenata

Pacijentska evaluacija predstavlja nezavisnu proveru modela, ali je njen obim ograničen brojem dostupnih pacijenata i kompletnim nalazima. Rezultate stoga treba posmatrati kao potvrdu potencijala modela na nezavisnim podacima, a ne kao konačnu procenu kliničke primenljivosti.

## 4.10.4. Post-hoc interpretacija dimenzija embeddinga

Povezivanje latentnih dimenzija ESM-2 embeddinga sa biohemijskim i strukturnim osobinama izvršeno je nakon treniranja modela. Takve analize mogu ukazati na moguće značenje naučenih reprezentacija, ali ne dokazuju da određena osobina uzrokuje predikciju niti da latentna dimenzija ima jednu jasno definisanu biološku funkciju.

# 5. Budući pravci istraživanja

<!-- RECENZIJA: (1) Ovaj i sledeći odeljak su oba "# 5" — prenumerisati. (2) Najvažniji predlog — adaptivna fuzija BLAST+MLP — trebalo bi bar u minimalnoj formi (gating po BLAST rr) URADITI u ovom radu: ako pokaže dobitak nad samim BLAST-om na LOCO-u I na pacijentima (uz study/patient-level test), to je "glavni rezultat" koji radu trenutno nedostaje. Bez toga rad ostaje "MLP ≈ BLAST, ali drugačije". -->

Rezultati ovog rada otvaraju nekoliko pravaca za dalja istraživanja. Prvi je razvoj adaptivne fuzije BLAST-a i MLP-a, pri kojoj bi doprinos MLP-a mogao biti veći u slučajevima kada BLAST daje slab ili neodlučan signal. Ova ideja predstavlja hipotezu zasnovanu na uočenoj komplementarnosti dva pristupa i nije validirana u okviru ovog rada.

Dalji rad treba da uključi veće i potpuno nezavisne kohorte pacijenata kako bi se proverila reproduktivnost rezultata na različitim populacijama i eksperimentalnim protokolima. Takođe, analiza zasnovana na strukturi proteina i poznatim epitopima mogla bi pomoći u povezivanju obrazaca naučenih u embedding prostoru sa konkretnijim molekularnim svojstvima. Ove analize treba sprovoditi kao nezavisnu validaciju, a ne kao pretpostavljeni mehanizam modela.

Konačno, proširenje skupa podataka većim brojem nezavisno i eksperimentalno potvrđenih unakrsno reaktivnih parova predstavljalo bi jedan od najvažnijih narednih koraka. Takav skup bi smanjio neizvesnost izazvanu pozitivno-neobeleženom prirodom postojećih podataka i omogućio pouzdaniju procenu sposobnosti modela da generalizuje na nove biološke primere.

---

# 5. Zaključak 

Ovaj rad pokazuje da potencijalna unakrsna reaktivnost proteinskih alergena ne može biti pouzdano opisana jednom merom sličnosti. Dok sama ESM-2 cosine sličnost nije nadmašila BLAST, nadgledano kombinovanje ESM-2 reprezentacija putem Hadamard proizvoda pokazalo je znatno korisniji signal i na nezavisnim pacijentskim slučajevima ostvarilo najbolje rangiranje. Ablacione analize ukazuju da ključ nije u samoj kompleksnosti modela, već u kvalitetu proteinske reprezentacije i načinu na koji se dve reprezentacije povezuju.

<!-- RECENZIJA — PRETERANE TVRDNJE naspram §3.8.1:
- "znatno korisniji signal" — na LOCO-u je MLP ≈ BLAST (Δ +0,0016, n.z.). Ukloniti "znatno".
- "ostvarilo najbolje rangiranje" na pacijentima — NETAČNO: §3.8.1 pokazuje podeljen ishod (BLAST značajno bolji u potiskivanju negativa, VEĆIM efektom). Zameniti: "pokazalo komplementaran profil: bolju senzitivnost uz slabiju specifičnost od BLAST-a".
- Dodati: MLP samostalno ne zamenjuje BLAST; doprinos je kao dopunski signal, najkorisniji kada je BLAST rang slab zbog konkurencije kandidata. -->


Rezultati zato ne ukazuju na jednostavnu zamenu BLAST-a novim modelom, već na **nov način korišćenja proteinskih reprezentacija kao dopune postojećim signalima**. Najvažniji nalaz ovog rada nije da je jedan algoritam univerzalno najbolji, već da se informacija relevantna za unakrsnu reaktivnost može nalaziti u odnosu između dva proteinska zapisa, a ne samo u njihovoj pojedinačnoj sličnosti. To predstavlja osnovu za razvoj budućih sistema koji bi sekvencijalne, reprezentacione, strukturne i kliničke informacije povezivali u jedinstven, strogo nezavisno validiran model.

---

# Dostupnost podataka i koda

Kod i svi korisceni resursi dostupni na: https://github.com/tardigrafika/Allergorithm

<!-- RECENZIJA: Za časopis nije dovoljan GitHub link. Potrebno: (1) Zenodo DOI za zamrznut release koda + podataka; (2) objavljen kurirani skup parova (par, UniProt ID, nivo dokaza, izvor/PMID, familija) — vredan resurs, povećava citiranost; (3) tačne verzije (ESM-2 checkpoint, BLAST verzija + E-value + matrica, Foldseek verzija, RRF konstanta K, seed-ovi, hiperparametri); (4) skript koji reprodukuje svaku tabelu/figuru; (5) izjava o licenci. -->


# Zahvalnice

Želim da se zahvalim svom mentoru Stefanu Nožiniću na stručnom vođstvu, savetima i kontinuiranoj podršci tokom razvoja ovog istraživanja.

Posebnu zahvalnost dugujem Mariji Stefanović na pomoći u razumevanju biološke pozadine problema, savetima u vezi sa alergenima i korisnim komentarima tokom rada.

<!-- RECENZIJA — ETIKA (v. i komentar u §2.5.2): rad koristi pacijentske alergološke nalaze za nezavisnu validaciju. Pre slanja obavezno: izjava o odobrenju etičke komisije / IRB (ili obrazloženje zašto nije potrebno ako su podaci isključivo iz objavljene literature), izjava o informisanom pristanku, izjava o anonimizaciji. RAD.md Zahvalnice pominju "osobe koje su ustupile svoje rezultate alergoloških testiranja" — ako je tako, ovo je human-subjects istraživanje i traži formalno odobrenje. -->



# Literatura
Bibliografija 

<!-- RECENZIJA — BLOKATOR: bibliografija je prazna, a Uvod nema nijedan citat. Potrebno: (1) poglavlje "Srodni radovi" (postojeći prediktori unakrsne reaktivnosti/alergenosti: AllerCatPro 2.0, AlgPred 2.0, AllergenOnline, SDAP; strukturni pristupi); (2) citati za ESM-2 (Lin et al. 2023, Science), Foldseek (van Kempen et al. 2024), RRF (Cormack et al. 2009), BLAST, graf-split/LOCO validaciju, PU learning (Bekker & Davis), link-prediction metrike (filtered MRR/Hits@k); (3) 25-40 referenci ukupno, popuniti references.bib. -->




