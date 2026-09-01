"""
D3: Dodaje directionality_note kolonu -- SAMO za parove gde IZVOR eksplicitno
navodi asimetriju (primarni senzitajzer vs sekundarni/cross-reaktivni
partner, npr. preko IgE inhibicionog eseja gde X potpuno inhibira Y ali ne
obrnuto). Ne popunjava se za sve parove -- vecina izvora testira/izvestava
simetricno (ili ne testira smer uopste), pa "prazno" ovde ZNACI "smer nije
dokumentovan", ne "simetricno potvrdjeno".

VAZNA ARHITEKTONSKA NAPOMENA (ne samo podatak, vec ogranicenje modela):
cak i da je svaki par imao poznat smer, TRENUTNI modeli (cosine, BLAST
identity, Hadamard produkt, abs-diff MLP) su svi SIMETRICNE funkcije od
(A,B) -- cosine(A,B)=cosine(B,A), abs(A-B)=abs(B-A), A elementwise* B =
B*A -- nijedan trenutni model NE MOZE da predstavi "A->B ali ne B->A" bez
fundamentalne promene arhitekture (npr. konkatenacija [A;B] umesto
elementwise operacije, asimetrican klasifikator). Ovo je vece pitanje od
dodavanja kolone -- dokumentovano ovde, ne resavano u ovom prolazu.

Izlaz: dodaje directionality_note kolonu u cross_reactive_1548.csv.
"""

from pathlib import Path

import pandas as pd

GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")

# pair_id -> directionality note (izvor citiran u samom tekstu note-a)
DIRECTIONALITY_NOTES = {
    "CR001": "Bet v 1 je primarni senzitajzer (birch-apple OAS); Mal d 1 sekundaran/cross-reaktivan.",
    "CR056": "Vec dokumentovano u notes: Jug r 1 primarni senzitajzer -- jako inhibira Cor a 14-specificni IgE, "
             "ali Cor a 14 samo delimicno inhibira Jug r 1-specificni IgE. Asimetrija potvrdjena inhibicionim esejom.",
    "CR264": "MAUG 2.0, klinicki slucaj: IgE na Sus s 1 potpuno inhibiran prethodnom inkubacijom sa Fel d 2, "
             "potvrdjujuci primarnu senzitizaciju na macku. Fel d 2 primaran, Sus s 1 sekundaran (pork-cat syndrome).",
    "INF951": "MAUG 2.0, klinicki slucaj: inhibicioni eseji potvrdili primarnu senzitizaciju na Can f 3 i "
              "IgE-cross-reaktivnost na Equ c 3 (horse-meat allergy mediated by dog-allergy). Can f 3 primaran.",
    "INF941": "MAUG 2.0, klinicki slucaj 1: IgE-prepoznavanje Can f 6 potpuno inhibirano niskim dozama Equ c 1 -- "
              "Equ c 1 primaran, Can f 6 sekundaran u ovom pacijentu.",
    "CR258": "MAUG 2.0, klinicki slucaj 2: IgE na Can f 6 potpuno inhibirano sa Fel d 4 (macka primarni izvor). "
             "NAPOMENA: drugi klinicki slucaj (3) u istom izvoru pokazao SLABU (ne potpunu) inhibiciju izmedju "
             "istog para -- interpretirano kao ko-senzitizacija, ne cross-reaktivnost -- smer je pacijent-zavisan.",
    "CR1172": "MAUG 2.0, klinicki slucaj: primarna senzitizacija na riblji parvalbumin (Gad c 1) sa "
              "cross-reaktivnoscu ka krokodilskom homologu Cro p 1. Gad c 1 primaran.",
    "CR1169": "MAUG 2.0: BIDIREKCIONA inhibicija pokazana (Cry j 7 inhibira Cap a 7 reaktivnost I obrnuto) -- "
              "ovo je primer POTVRDJENE SIMETRICNOSTI, ne asimetrije, ukljuceno radi kontrasta sa gornjim primerima.",
}

df = pd.read_csv(GOLD)
df["directionality_note"] = df["pair_id"].map(DIRECTIONALITY_NOTES)

df.to_csv(GOLD, index=False)
print(f"directionality_note popunjeno za {df['directionality_note'].notna().sum()}/{len(df)} parova.")
for pid, note in DIRECTIONALITY_NOTES.items():
    print(f"  {pid}: {note[:80]}...")
