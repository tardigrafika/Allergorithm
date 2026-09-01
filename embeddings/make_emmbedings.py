"""
Generate ESM-2 embeddings for the cleaned WHO/IUIS allergen dataset.

Input:
    data/clean_allergens.csv

Outputs:
    data/embeddings.pkl
    data/embeddings.parquet
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

INPUT_CSV = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")

OUTPUT_PICKLE = Path("data/embeddings.pkl")
OUTPUT_PARQUET = Path("data/embeddings.parquet")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

MAX_LENGTH = 1022
BATCH_SIZE = 4


# -------------------------------------------------------
# Device
# -------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# -------------------------------------------------------
# Load model
# -------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading ESM-2 model...")

model = EsmModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

df = df[df["fasta_sequence"].notna()]
df = df[df["fasta_sequence"] != ""]

df = df.reset_index(drop=True)

print(f"\nLoaded {len(df)} allergens.")


# -------------------------------------------------------
# Mean pooling
# -------------------------------------------------------

def mean_pool(last_hidden_state, attention_mask):
    """
    Compute mean embedding over valid tokens only.
    """

    mask = attention_mask.unsqueeze(-1).expand(
        last_hidden_state.size()
    ).float()

    summed = torch.sum(last_hidden_state * mask, dim=1)

    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


# -------------------------------------------------------
# Embedding generation
# -------------------------------------------------------

embeddings_dict = {}

embedding_rows = []

print("\nGenerating embeddings...\n")

with torch.no_grad():

    for start in tqdm(
        range(0, len(df), BATCH_SIZE)
    ):

        batch = df.iloc[start:start+BATCH_SIZE]

        sequences = batch["fasta_sequence"].tolist()

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

        outputs = model(**tokens)

        pooled = mean_pool(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        )

        pooled = pooled.cpu().numpy()

        for row, vector in zip(
            batch.itertuples(index=False),
            pooled
        ):

            embeddings_dict[row.allergen_id] = vector

            embedding_rows.append({

                "allergen_id": row.allergen_id,

                "official_name": row.official_name,

                "source_food": row.source_food,

                "organism": row.organism,

                "protein_family": row.protein_family,

                "sequence_length": row.sequence_length,

                "embedding": vector.tolist()

            })


# -------------------------------------------------------
# Save
# -------------------------------------------------------

print("\nSaving pickle...")

with open(
    OUTPUT_PICKLE,
    "wb"
) as f:

    pickle.dump(
        embeddings_dict,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

*
print("Saving parquet...")

embedding_df = pd.DataFrame(
    embedding_rows
)

embedding_df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

print("\n======================================")
print("Embedding generation complete!")
print("======================================")

print(f"Proteins embedded : {len(embedding_df)}")
print(f"Embedding size    : {len(embedding_rows[0]['embedding'])}")
print(f"Pickle saved      : {OUTPUT_PICKLE}")
print(f"Parquet saved     : {OUTPUT_PARQUET}")