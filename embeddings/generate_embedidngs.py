"""
Generise ESM-2 embeddinge za WHO/IUIS alergene (Google Colab GPU) - MAX pooling varijanta.

Umesto mean pooling-a (prosek preko cele sekvence), radi max pooling
(najjaca aktivacija po dimenziji) - test da li usrednjavanje razblazuje
epitope-nivo signal bitan za cross-reactivity.

Ulaz:
    /content/clean_allergens.csv

Izlaz:
    /content/embeddings3B_maxpool.pkl
    /content/embeddings3B_maxpool.parquet
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
# Configuration
# ======================================================

INPUT_CSV = Path("/content/clean_allergens.csv")

OUTPUT_PICKLE = Path("/content/embeddings3B_maxpool.pkl")
OUTPUT_PARQUET = Path("/content/embeddings3B_maxpool.parquet")


MODEL_NAME = "facebook/esm2_t36_3B_UR50D"

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
# Max pooling
# ======================================================
# Element-wise max over the sequence dimension, instead of the mean.
# Padding positions are masked out to -inf first so they can never win
# the max (a padded position is otherwise just another token position
# in last_hidden_state and would silently distort the pooled vector
# for shorter sequences in the batch if left unmasked).

def max_pool(
    last_hidden_state,
    attention_mask
):

    mask = (
        attention_mask
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .bool()
    )


    masked_hidden_state = last_hidden_state.masked_fill(
        ~mask,
        float("-inf")
    )


    pooled, _ = torch.max(
        masked_hidden_state,
        dim=1
    )


    return pooled



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

    for start in tqdm(
        range(
            0,
            len(df),
            BATCH_SIZE
        )
    ):


        batch = df.iloc[
            start:start+BATCH_SIZE
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
            k:v.to(device)
            for k,v in tokens.items()
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


        pooled = max_pool(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        )


        pooled = (
            pooled
            .float()
            .cpu()
            .numpy()
        )


        for row, vector in zip(
            batch.itertuples(index=False),
            pooled
        ):


            embeddings_dict[
                row.allergen_id
            ] = vector


            embedding_rows.append({

                "allergen_id":
                    row.allergen_id,

                "official_name":
                    row.official_name,

                "source_food":
                    row.source_food,

                "organism":
                    row.organism,

                "protein_family":
                    row.protein_family,

                "sequence_length":
                    row.sequence_length,

                "embedding":
                    vector.tolist()

            })



# ======================================================
# Save
# ======================================================


print()
print("==============================")
print("SAVING")
print("==============================")


with open(
    OUTPUT_PICKLE,
    "wb"
) as f:

    pickle.dump(
        embeddings_dict,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )



embedding_df = pd.DataFrame(
    embedding_rows
)


embedding_df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)



print()
print("==============================")
print("DONE")
print("==============================")

print(
    "Proteins embedded:",
    len(embedding_df)
)

print(
    "Embedding size:",
    len(
        embedding_rows[0]["embedding"]
    )
)

print(
    "Saved:",
    OUTPUT_PICKLE
)

print(
    "Saved:",
    OUTPUT_PARQUET
)