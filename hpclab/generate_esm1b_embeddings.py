"""
Generise ESM-1b (facebook/esm1b_t33_650M_UR50S) embeddinge za WHO/IUIS
alergene -- namenjeno za pokretanje na klasteru (ne Google Colab, za
razliku od ostalih generate_*_embeddings.py skripti u ovom folderu).

Zasto ESM-1b: stariji model (2019, pre ESM-2 arhitekturnih/trening
poboljsanja), ali TRENIRAN NA MANJE/DRUGACIJE FILTRIRANOM UniRef50 skupu i
BEZ ESM-2-specificnih izmena (relative position embeddings, veci training
recipe) -- test da li nezavisno trenirana reprezentacija istog velicinskog
reda (650M, ISTA velicina kao trenutni glavni embeddings.pkl iz
facebook/esm2_t33_650M_UR50D) probija "representation ceiling" nalaz koji
drzi kroz ceo projekat (cosine/RF/MLP/bilinear svi pogadjaju isti plafon na
ESM-2 embeddinzima -- videti README.md).

Isti mean-pooling konvencija kao glavni embeddings.pkl
(embeddings/make_emmbedings.py) -- prosek preko validnih tokena, padding
maskiran -- da rezultat bude direktno uporediv (ista pool-metoda, samo
drugi encoder).

VAZNA razlika od ESM-2 tokenizacije: ESM-1b tokenizer/model OCEKUJE da
sekvence budu <=1022 rezidue (isti limit kao ESM-2 u ovom projektu, ali
proveri upozorenja pri ucitavanju -- ESM-1b je treniran sa max_position_
embeddings=1026, sto ukljucuje <cls>/<eos>, otud MAX_LENGTH=1022 ispod).

Ulaz (transferuj na klaster):
    clean_allergens.csv   (kopija output/clean_allergens.csv iz repo-a)

Izlaz:
    embeddings_esm1b.pkl
    embeddings_esm1b.parquet

Pokretanje (primer za SLURM, prilagodi klasteru):
    python3 generate_esm1b_embeddings.py
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
# clean_allergens.csv transferovan (vidi transfer checklist u odgovoru)
# ======================================================

INPUT_CSV = Path("clean_allergens.csv")

OUTPUT_PICKLE = Path("embeddings_esm1b.pkl")
OUTPUT_PARQUET = Path("embeddings_esm1b.parquet")

MODEL_NAME = "facebook/esm1b_t33_650M_UR50S"

MAX_LENGTH = 1022
BATCH_SIZE = 8  # smanji na 4 ili 2 ako GPU VRAM ne dozvoljava (isti red velicine kao ESM-2 650M)


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
print()


# ======================================================
# Load model
# ======================================================

print(f"Loading tokenizer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading ESM-1b model ({MODEL_NAME})...")
model = EsmModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

# FP16 samo na GPU -- ESM (BERT-stil, encoder-only) NEMA numericku
# nestabilnost koju ima T5/Ankh u FP16 (videti generate_ankh_embeddings.py
# napomenu), ista pretpostavka kao glavni ESM-2 max-pool skript.
use_fp16 = False
if device.type == "cuda":
    model.half()
    use_fp16 = True
    print("Using FP16 acceleration")
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
# Mean pooling (identicna konvencija kao make_emmbedings.py -- prosek
# preko validnih tokena, padding maskiran)
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
print("Embedding size:", len(embedding_rows[0]["embedding"]))
print(f"Proteins with a NaN embedding: {n_nan}/{len(embeddings_dict)}"
      + ("  <<< PROBLEM, proveri FP16/precision" if n_nan else "  (clean)"))
print("Saved:", OUTPUT_PICKLE)
print("Saved:", OUTPUT_PARQUET)
print()
print("Prebaci embeddings_esm1b.pkl i embeddings_esm1b.parquet nazad u lokalni "
      "embeddings/ folder, pa nastavljamo isti RRF/cosine-comparison pattern "
      "kao za Ankh (embeddings/generate_ankh_embeddings.py).")
