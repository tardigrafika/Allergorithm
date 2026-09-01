"""
E3: Alergen-vs-ljudski-homolog negativni parovi -- prvi PRAVI, literaturom
potvrdjen negativan skup u projektu (za razliku od nasumicno uzorkovanih
"negativa" koje koriste RF/MLP/XGBoost/Hadamard klasifikatori, koji su samo
odsustvo dokumentacije, ne potvrdjena odsutnost reaktivnosti).

Zasto BAS tropomiozin (ne profilin, ne parvalbumin): WebSearch verifikacija
(2026-08-29) pokazala je da je profilin OBRNUT slucaj -- ljudski profilin
(PFN1) je POTVRDJEN IgE autoantigen kod alergijskih pacijenata (molekularni
mimikrija, "prvi identifikovan IgE autoantigen") -- NE bi bio validan
negativ. Parvalbumin nema jasnu direktnu potvrdu u dostupnoj literaturi.
Tropomiozin JE cvrsto potvrdjen: IgE epitop regioni beskicmenjackog
tropomiozina uporedjeni DIREKTNO sa ljudskim skeletnim tropomiozinom
potvrdjuju odsustvo alergenske unakrsne reaktivnosti (Reese et al. 1999,
PMID 10474029; Ayuso et al., IgE-binding epitope mapping serija).

Metod:
1. Fetch ljudski TPM1 (P09493, skeletni alfa-tropomiozin -- referenca
   koriscena u Reese/Ayuso poredjenjima) direktno sa UniProt REST API.
2. Dodaj kao NOVI protein u clean_allergens.csv (allergen_id prefiks
   "HUMAN_" -- namerno RAZLICIT od "WHO_..." seme, da se nikad ne pomesa
   sa pravim WHO/IUIS alergenom).
3. Generisi ESM-2 embedding (ISTI model/mean-pool kao make_emmbedings.py --
   facebook/esm2_t33_650M_UR50D) -- dodaj u embeddings.pkl/.parquet.
4. Dodaj 18 negativnih parova (svaki postojeci tropomiozin alergen vs
   ljudski TPM1) u cross_reactive_1548.csv, isti format kao E1/E2.

Izlaz:
    output/clean_allergens.csv (izmenjen, +1 red)
    embeddings/embeddings.pkl, embeddings/embeddings.parquet (izmenjeni, +1 protein)
    output/cross_reactive_1548.csv (izmenjen, +18 redova)
"""

import pickle
from pathlib import Path

import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, EsmModel

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
EMBEDDINGS_PKL = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
EMBEDDINGS_PARQUET = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")

HUMAN_ID = "HUMAN_TPM1_P09493"
HUMAN_NAME = "Human tropomyosin alpha-1 chain (TPM1, self-protein)"
HUMAN_UNIPROT = "P09493"

CITATION = (
    "UniProt P09493 (TPM1_HUMAN); Reese G, Ayuso R, Lehrer SB. \"Tropomyosin: an invertebrate "
    "pan-allergen.\" Int Arch Allergy Immunol. 1999;119(4):247-58, PMID 10474029 -- vertebrate "
    "tropomyosins nonallergenic, IgE responses against vertebrate tropomyosin very rare despite "
    "conservation; Ayuso R et al., IgE-binding epitope mapping (Pen a 1/shrimp tropomyosin, PMID "
    "10224419) -- epitope regions (residues 50-66, 153-161) compared DIRECTLY to human skeletal "
    "muscle tropomyosin confirm lack of allergenic cross-reactivity between phylogenetically "
    "distinct species."
)
NOTE = ("E3 (allergen vs human homolog negative): invertebrate tropomyosin panallergen carries "
        "invertebrate-specific IgE epitopes; human/vertebrate tropomyosin not recognized as foreign "
        "(self-tolerance) despite conserved core structure.")

TROPOMYOSIN_ALLERGENS = [
    "Ani s 3.0101", "Bla g 7.0101", "Blo t 10.0101", "Cha f 1.0101", "Cra c 1.0101",
    "Cra g 1.0101", "Cra g 1.0102", "Der p 10.0101", "Hom a 1.0101", "Hom a 1.0102",
    "Lit v 1.0101", "Met e 1.0101", "Pan b 1.0101", "Per a 7.0101", "Per a 7.0102",
    "Scy p 1.0101", "Tod p 1.0101", "Tyr p 10.0101",
]

# -------------------------------------------------------
# 1) Fetch ljudsku sekvencu
# -------------------------------------------------------
print(f"Fetching {HUMAN_UNIPROT} sa UniProt-a...")
resp = requests.get(f"https://rest.uniprot.org/uniprotkb/{HUMAN_UNIPROT}.fasta", timeout=30)
resp.raise_for_status()
lines = resp.text.strip().split("\n")
sequence = "".join(lines[1:])
print(f"  Sekvenca: {len(sequence)} aa")
assert len(sequence) > 200, "Neocekivano kratka sekvenca -- proveri fetch"

# -------------------------------------------------------
# 2) Dodaj u clean_allergens.csv (ako vec nije dodato)
# -------------------------------------------------------
allergens = pd.read_csv(CLEAN_ALLERGENS)
if HUMAN_ID in set(allergens["allergen_id"]):
    print(f"{HUMAN_ID} vec postoji u clean_allergens.csv -- preskacem dodavanje proteina.")
else:
    new_row = pd.DataFrame([{
        "allergen_id": HUMAN_ID,
        "official_name": HUMAN_NAME,
        "source_food": "N/A (ljudski self-protein, negativna kontrola)",
        "organism": "Homo sapiens",
        "protein_family": "Tropomyosin",
        "uniprot_id": HUMAN_UNIPROT,
        "fasta_sequence": sequence,
        "sequence_length": len(sequence),
        "reference": CITATION,
    }])
    allergens = pd.concat([allergens, new_row], ignore_index=True)
    allergens.to_csv(CLEAN_ALLERGENS, index=False)
    print(f"Dodato u {CLEAN_ALLERGENS}: {HUMAN_ID}")

# -------------------------------------------------------
# 3) Generisi ESM-2 embedding (identican model/pooling kao make_emmbedings.py)
# -------------------------------------------------------
with open(EMBEDDINGS_PKL, "rb") as f:
    embeddings_dict = pickle.load(f)

if HUMAN_ID in embeddings_dict:
    print(f"{HUMAN_ID} vec ima embedding -- preskacem generisanje.")
else:
    print("Generisem ESM-2 embedding (facebook/esm2_t33_650M_UR50D, CPU)...")
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    with torch.no_grad():
        tokens = tokenizer([sequence], padding=True, truncation=True, max_length=1022, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(**tokens)
        pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"]).cpu().numpy()[0]

    embeddings_dict[HUMAN_ID] = pooled
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = pd.read_parquet(EMBEDDINGS_PARQUET)
    if HUMAN_ID not in set(metadata["allergen_id"]):
        new_meta_row = pd.DataFrame([{
            "allergen_id": HUMAN_ID, "official_name": HUMAN_NAME,
            "source_food": "N/A (ljudski self-protein, negativna kontrola)",
            "organism": "Homo sapiens", "protein_family": "Tropomyosin",
            "sequence_length": len(sequence), "embedding": pooled.tolist(),
        }])
        metadata = pd.concat([metadata, new_meta_row], ignore_index=True)
        metadata.to_parquet(EMBEDDINGS_PARQUET, index=False)
    print(f"Embedding generisan i sacuvan (dim={pooled.shape[0]}).")

# -------------------------------------------------------
# 4) Dodaj negativne parove u cross_reactive_1548.csv
# -------------------------------------------------------
gold = pd.read_csv(GOLD)
existing_pairs = {frozenset([str(r["allergen_id_1"]), str(r["allergen_id_2"])]) for _, r in gold.iterrows()}

allergen_source_lookup = dict(zip(allergens["official_name"], allergens["source_food"]))
# NAPOMENA: clean_allergens.csv["protein_family"] je PRAZNA kolona (0/1535 popunjeno) --
# familija se ovde uzima direktno iz TROPOMYOSIN_ALLERGENS liste (svi su vec poznati
# tropomiozin alergeni, po definiciji kako je lista i sastavljena), ne iz clean_allergens.csv.

next_num = max([int(pid[3:]) for pid in gold["pair_id"] if str(pid).startswith("NEG") and pid[3:].isdigit()],
               default=0) + 1

new_rows = []
for name in TROPOMYOSIN_ALLERGENS:
    if frozenset([HUMAN_NAME, name]) in existing_pairs:
        print(f"Preskoceno (vec postoji): {name}")
        continue
    fam1 = "Tropomyosin"
    src1 = allergen_source_lookup.get(name)
    new_rows.append({
        "pair_id": f"NEG{next_num:03d}",
        "allergen_id_1": name, "source_food_1": src1, "family_1": fam1,
        "allergen_id_2": HUMAN_NAME, "source_food_2": "N/A (ljudski self-protein)", "family_2": "Tropomyosin",
        "evidence_type": "E3: allergen vs ljudski homolog (self-tolerance, literaturom potvrdjeno)",
        "evidence_level": "Reported negative", "sequence_identity_pct": None,
        "reference": CITATION, "isoform_note": None, "notes": NOTE,
    })
    existing_pairs.add(frozenset([HUMAN_NAME, name]))
    next_num += 1

new_df = pd.DataFrame(new_rows)
combined = pd.concat([gold, new_df], ignore_index=True)
combined.to_csv(GOLD, index=False)

print(f"\nDodato {len(new_rows)} E3 negativnih parova (tropomiozin vs ljudski TPM1).")
print(f"Novi ukupan broj redova u gold datasetu: {len(combined)}")
