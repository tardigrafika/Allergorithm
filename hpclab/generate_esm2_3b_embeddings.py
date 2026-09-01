"""
Generise ESM-2 3B (facebook/esm2_t36_3B_UR50D) embeddinge za WHO/IUIS
alergene -- namenjeno za pokretanje na klasteru (ne Google Colab).

Zasto 3B: ne menja se model-familija (ostaje ESM-2, ista arhitektura/trening
recipe kao trenutni glavni embeddings.pkl), samo SKALA -- ~4.6x vise
parametara od trenutnog facebook/esm2_t33_650M_UR50D (650M -> 3B), dim
1280 -> 2560. Test da li je "representation ceiling" nalaz (cosine/RF/MLP/
bilinear svi pogadjaju isti plafon na 650M embeddinzima -- videti README.md)
plafon SAME ARHITEKTURE/SKALE, ili nesto sto veci kapacitet moze da probije.
NAPOMENA: postojeci embeddings/embeddings_maxpool.pkl NIJE 3B model uprkos
imenu generate_embedidngs.py skripta koji ga generise -- proveren dim=1280
(650M velicina), ne 2560 -- taj fajl je 650M+maxpool, ne 3B. Ovaj skript je
prvi stvaran pokusaj 3B skale u projektu.

Ista mean-pooling konvencija kao glavni embeddings.pkl
(embeddings/make_emmbedings.py) i embeddings_esm1b.pkl
(embeddings/generate_esm1b_embeddings.py) -- prosek preko validnih tokena,
padding maskiran -- direktno uporedivo, ista pool-metoda, samo veci encoder.

VAZNA napomena o resursima: 3B parametara u FP16 = ~6GB samo za tezine
modela, PLUS aktivacije koje skaliraju sa BATCH_SIZE x sequence_length x
2560 x 36 slojeva -- mnogo vece memorijsko opterecenje od 650M/ESM-1b
skripti u ovom folderu. BATCH_SIZE=2 ispod je namerno konzervativan
pocetak -- ako klaster GPU ima >=24GB VRAM, probaj 4; ako OOM na 2,
spusti na 1. Nema potrebe za multi-GPU/model-parallelism za 3B (za razliku
od 15B varijante facebook/esm2_t48_15B_UR50D, koja NIJE ovde koriscena
namerno -- overkill/nepraktican memorijski zahtev za probni test skale).

Ulaz (transferuj na klaster):
    clean_allergens.csv   (kopija output/clean_allergens.csv iz repo-a)

Izlaz:
    embeddings_esm2_3b.pkl
    embeddings_esm2_3b.parquet

Pokretanje (primer za SLURM, prilagodi klasteru):
    python3 generate_esm2_3b_embeddings.py
"""

# ======================================================
# Imports
# ======================================================

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel


# ======================================================
# Configuration -- relativne putanje, pokreni skript IZ foldera gde je
# clean_allergens.csv transferovan
# ======================================================

INPUT_CSV = Path("clean_allergens.csv")

OUTPUT_PICKLE = Path("embeddings_esm2_3b.pkl")
OUTPUT_PARQUET = Path("embeddings_esm2_3b.parquet")

MODEL_NAME = "facebook/esm2_t36_3B_UR50D"

MAX_LENGTH = 1022
BATCH_SIZE = 2  # konzervativan pocetak -- videti VRAM napomenu iznad, podesi po GPU-u


# ======================================================
# Device
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==============================")
print("DEVICE")
print("==============================")
print(device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("UPOZORENJE: nema GPU-a -- 3B model na CPU-u ce biti VRLO spor "
          "(sati/dani za ceo dataset), prakticno zahteva GPU klaster node.")
print()


# ======================================================
# Load model
# ======================================================

print(f"Loading tokenizer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading ESM-2 3B model ({MODEL_NAME}) -- ovo je ~6GB download pri prvom pokretanju...")
model = EsmModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

# FP16 samo na GPU -- ESM (BERT-stil, encoder-only) NEMA numericku
# nestabilnost koju ima T5/Ankh u FP16 (videti generate_ankh_embeddings.py
# napomenu), ista pretpostavka kao ostale ESM skripte u ovom folderu.
# Na 3B skali FP16 nije samo brzina -- praktican preduslov da tezine uopste
# stanu u razumnu kolicinu VRAM-a (FP32 bi trazio ~12GB SAMO za tezine).
use_fp16 = False
if device.type == "cuda":
    model.half()
    use_fp16 = True
    print("Using FP16 acceleration (takodje smanjuje VRAM za tezine sa ~12GB na ~6GB)")
print()


# ======================================================
# Load dataset
# ======================================================

df = pd.read_csv(INPUT_CSV)
df = df[df["fasta_sequence"].notna()]
df = df[df["fasta_sequence"] != ""]
df = df.reset_index(drop=True)

print("==============================")
print("DATASET")
print("==============================")
print(f"Loaded {len(df)} allergens")
print()


# ======================================================
# Mean pooling (identicna konvencija kao make_emmbedings.py i
# generate_esm1b_embeddings.py -- prosek preko validnih tokena, padding
# maskiran)
# ======================================================

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


# ======================================================
# Generate embeddings
# ======================================================

embeddings_dict = {}
embedding_rows = []

print("==============================")
print("GENERATING EMBEDDINGS")
print("==============================")
print()

with torch.no_grad():
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch = df.iloc[start:start + BATCH_SIZE]
        sequences = batch["fasta_sequence"].tolist()

        tokens = tokenizer(sequences, padding=True, truncation=True,
                             max_length=MAX_LENGTH, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}

        if use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(**tokens)
        else:
            outputs = model(**tokens)

        pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        pooled = pooled.float().cpu().numpy()

        for row, vector in zip(batch.itertuples(index=False), pooled):
            embeddings_dict[row.allergen_id] = vector
            embedding_rows.append({
                "allergen_id": row.allergen_id,
                "official_name": row.official_name,
                "source_food": row.source_food,
                "organism": row.organism,
                "protein_family": row.protein_family,
                "sequence_length": row.sequence_length,
                "embedding": vector.tolist(),
            })


# ======================================================
# Save
# ======================================================

print()
print("==============================")
print("SAVING")
print("==============================")

with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

embedding_df = pd.DataFrame(embedding_rows)
embedding_df.to_parquet(OUTPUT_PARQUET, index=False)

n_nan = sum(1 for v in embeddings_dict.values() if np.isnan(v).any())
print()
print("==============================")
print("DONE")
print("==============================")
print("Proteins embedded:", len(embedding_df))
print("Embedding size:", len(embedding_rows[0]["embedding"]), "(ocekivano 2560 za 3B model)")
print(f"Proteins with a NaN embedding: {n_nan}/{len(embeddings_dict)}"
      + ("  <<< PROBLEM, proveri FP16/precision" if n_nan else "  (clean)"))
print("Saved:", OUTPUT_PICKLE)
print("Saved:", OUTPUT_PARQUET)
print()
print("Prebaci embeddings_esm2_3b.pkl i embeddings_esm2_3b.parquet nazad u lokalni "
      "embeddings/ folder -- isti obrazac poredjenja kao ESM-1b "
      "(ml/loco_esm1b_vs_esm2_mlp_hadamard_1548.py, samo zameni embedding izvor).")
