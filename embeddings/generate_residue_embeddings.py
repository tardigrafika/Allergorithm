"""
Generise RESIDUE-LEVEL (po amino-kiselini) ESM-2 embeddinge - Eksperiment 2, korak 1.

Za razliku od mean/max pooling verzija, cuva CELU L x 1280 matricu po proteinu
(bez BOS/EOS/padding), ne jedan pooled vektor - priprema za residue-level
top-k similarity (korak 2). Isti 650M model kao mean/max pooling (kontrolisano
poredjenje, menja se samo reprezentacija).

Pokrece se na VM-u (GPU). Fajl je veliki (~2GB+, gzip kompresovan).

Ulaz:
    /content/clean_allergens.csv

Izlaz:
    /content/residue_embeddings.pkl.gz
    /content/residue_embeddings_metadata.parquet
"""


# ======================================================
# Imports
# ======================================================

import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel


# ======================================================
# Configuration
# ======================================================

INPUT_CSV = Path("/content/clean_allergens.csv")

OUTPUT_PICKLE_GZ = Path("/content/residue_embeddings.pkl.gz")
OUTPUT_METADATA_PARQUET = Path("/content/residue_embeddings_metadata.parquet")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # same 650M model as mean/max pooling -- controlled comparison

MAX_LENGTH = 1022
BATCH_SIZE = 8


# ======================================================
# Device
# ======================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading ESM-2 model...")

model = EsmModel.from_pretrained(
    MODEL_NAME
)


model.to(device)

model.eval()


# FP16 only on GPU
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


df = df[
    df["fasta_sequence"].notna()
]


df = df[
    df["fasta_sequence"] != ""
]


df = df.reset_index(drop=True)


print("==============================")
print("DATASET")
print("==============================")

print(
    f"Loaded {len(df)} allergens"
)

print()


# ======================================================
# Trim to real residues only (strip BOS/EOS/padding)
# ======================================================
# ESM's tokenizer format is: <cls> residue_1 ... residue_L <eos> <pad> ...
# attention_mask.sum() = 1 (cls) + L (residues) + 1 (eos), so the real
# per-residue rows are last_hidden_state[1 : attention_mask.sum() - 1].

def extract_residue_matrix(last_hidden_state_row, attention_mask_row):

    valid_length = int(attention_mask_row.sum().item())

    # valid_length includes <cls> and <eos>; strip both, keep only residues
    residue_matrix = last_hidden_state_row[1:valid_length - 1]

    return residue_matrix.float().cpu().numpy().astype(np.float32)


# ======================================================
# Generate residue-level embeddings
# ======================================================

residue_embeddings = {}

metadata_rows = []


print("==============================")
print("GENERATING RESIDUE-LEVEL EMBEDDINGS")
print("==============================")
print()


with torch.no_grad():

    for start in tqdm(
        range(
            0,
            len(df),
            BATCH_SIZE
        )
    ):

        batch = df.iloc[
            start:start + BATCH_SIZE
        ]

        sequences = batch[
            "fasta_sequence"
        ].tolist()

        tokens = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        tokens = {
            k: v.to(device)
            for k, v in tokens.items()
        }

        if use_fp16:

            with torch.cuda.amp.autocast():

                outputs = model(
                    **tokens
                )

        else:

            outputs = model(
                **tokens
            )

        last_hidden_state = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"]

        for i, row in enumerate(batch.itertuples(index=False)):

            residue_matrix = extract_residue_matrix(
                last_hidden_state[i],
                attention_mask[i]
            )

            residue_embeddings[row.allergen_id] = residue_matrix

            metadata_rows.append({
                "allergen_id": row.allergen_id,
                "official_name": row.official_name,
                "source_food": row.source_food,
                "organism": row.organism,
                "protein_family": row.protein_family,
                "sequence_length": residue_matrix.shape[0],
            })


# ======================================================
# Save
# ======================================================

print()
print("==============================")
print("SAVING")
print("==============================")

print("Compressing + saving residue embeddings (this can take a while)...")

with gzip.open(OUTPUT_PICKLE_GZ, "wb", compresslevel=4) as f:

    pickle.dump(
        residue_embeddings,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_parquet(
    OUTPUT_METADATA_PARQUET,
    index=False
)


print()
print("==============================")
print("DONE")
print("==============================")

print(
    "Proteins embedded:",
    len(residue_embeddings)
)

lengths = [m.shape[0] for m in residue_embeddings.values()]
print(
    f"Residue count per protein -- min={min(lengths)}, "
    f"mean={np.mean(lengths):.1f}, max={max(lengths)}"
)

print(
    "Saved:",
    OUTPUT_PICKLE_GZ
)

print(
    "Saved:",
    OUTPUT_METADATA_PARQUET
)

print(
    "\nDownload both files back to your local embeddings/ folder, "
    "then upload output/cross_reactive_1443.csv to /content/ as well "
    "before running the retrieval evaluation script (step 2)."
)
