"""
Generise Ankh (ElnaggarLab/ankh-base) embeddinge za WHO/IUIS alergene
(Google Colab GPU) - mean pooling, isti pristup kao glavni ESM-2 embeddings.pkl.

Zasto Ankh: T5-stil encoder-decoder arhitektura (koristi se samo enkoder deo
za embeddinge), arhitekturno DALJE od ESM-2 (encoder-only, BERT-stil) nego
npr. ProtBERT - veca sansa da nauci genuinski drugaciju reprezentaciju.
Test hipoteze: da li Ankh cosine similarity nosi nezavisan signal koji
pomaze RRF fuziji (cosine_ESM + BLAST + FoldseekTM), slicno kao sto je
FoldseekTM pomogao dodavanjem STRUKTURNE (ne sekvencijalne) informacije.

VAZNO o Ankh tokenizaciji: T5-stil sentencepiece tokenizer OCEKUJE sekvencu
kao LISTU pojedinacnih karaktera (ne string), sa is_split_into_words=True -
drugacije od ESM tokenizera koji uzima string direktno. Videti tokenize_batch()
nize.

Ulaz:
    /content/clean_allergens.csv

Izlaz:
    /content/embeddings_ankh.pkl
    /content/embeddings_ankh.parquet
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
from transformers import AutoTokenizer, T5EncoderModel


# ======================================================
# Configuration
# ======================================================

INPUT_CSV = Path("/content/clean_allergens.csv")

OUTPUT_PICKLE = Path("/content/embeddings_ankh.pkl")
OUTPUT_PARQUET = Path("/content/embeddings_ankh.parquet")

MODEL_NAME = "ElnaggarLab/ankh-base"  # ~450M -- uporediva velicina sa ESM-2 650M
# MODEL_NAME = "ElnaggarLab/ankh-large"  # ~1.9B -- probaj ovo tek ako ankh-base pokaze signal

MAX_LENGTH = 1022
BATCH_SIZE = 8


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

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Ankh model (encoder only -- we don't need/want the decoder for embeddings)...")
model = T5EncoderModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

# NAPOMENA: FP16 namerno NIJE koriscen ovde (za razliku od ESM skripte).
# T5-stil modeli (Ankh je T5 arhitektura) su poznato numericki nestabilni u
# FP16 - aktivacije mogu da predju FP16 opseg i postanu NaN (potvrdjeno na
# prvom pokusaju: SVI embeddinzi su izasli kao NaN). FP32 je sporije ali
# ispravno - ESM (BERT-stil, encoder-only) nema ovaj problem, otud razlika
# u odnosu na generate_embedidngs.py.
print("Running in FP32 (T5/Ankh is numerically unstable in FP16)")
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
# Tokenization (Ankh/T5-style: list-of-characters, not raw string)
# ======================================================

def tokenize_batch(sequences):
    truncated = [s[:MAX_LENGTH] for s in sequences]
    char_lists = [list(s) for s in truncated]
    # tokenizer(...) instead of the older .batch_encode_plus(...) -- same
    # kwargs/output, but not at risk of being removed in newer transformers
    return tokenizer(
        char_lists,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=True,
        return_tensors="pt",
    )


# ======================================================
# Mean pooling (identical convention to the main ESM embeddings.pkl:
# average over real tokens only, padding/special tokens masked out)
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

        tokens = tokenize_batch(sequences)
        tokens = {k: v.to(device) for k, v in tokens.items()}

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
      + ("  <<< PROBLEM, do not use these -- check FP16/precision settings" if n_nan else "  (clean)"))
print("Saved:", OUTPUT_PICKLE)
print("Saved:", OUTPUT_PARQUET)
print()
print("Download both files back to your local embeddings/ folder, "
      "then we'll build the Ankh cosine RRF voter (same pattern as FoldseekTM/kmer).")
