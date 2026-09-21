"""
Mean-pool residue_embeddings_af_trunk.pkl (per-residue, (L,384) po proteinu)
u embeddings_af_trunk_mean.pkl (pooled, (384,) po proteinu) -- prosek SAMO
preko rezidua (axis=0), isti oblik/dtype/kljuc-format kao postojeci
embeddings_esm2_3b.pkl ({allergen_id: np.ndarray(dim,), dtype=float32}).

TAKODJE pise embeddings_af_trunk_mean.parquet -- NUZAN pratilac fajl za
STVARNU drop-in zamenu. ml.pipeline.common.data.load_dataset() (koriscen u
SVIM MLP(hadamard) trening/LOCO skriptovima ove sesije) zahteva I .pkl I
.parquet (metadata: allergen_id/official_name/source_food/organism/
protein_family/sequence_length) -- bez parquet-a, "samo promeni putanju"
ne bi stvarno radilo. Metadata kolone se povlace iz clean_allergens.csv,
isti sablon kao postojeci embeddings_esm2_3b.parquet.

Ulaz:
    residue_embeddings_af_trunk.pkl
    clean_allergens.csv (za metadata kolone)

Izlaz:
    embeddings_af_trunk_mean.pkl
    embeddings_af_trunk_mean.parquet

Pokretanje:
    python3 mean_pool_af_trunk.py
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

INPUT_PICKLE = Path("residue_embeddings_af_trunk.pkl")
INPUT_CSV = Path("clean_allergens.csv")
OUTPUT_PICKLE = Path("embeddings_af_trunk_mean.pkl")
OUTPUT_PARQUET = Path("embeddings_af_trunk_mean.parquet")

with open(INPUT_PICKLE, "rb") as f:
    residue_embeddings = pickle.load(f)

print(f"Loaded {len(residue_embeddings)} proteina iz {INPUT_PICKLE}")

pooled = {}
n_nan = 0
for allergen_id, arr in residue_embeddings.items():
    assert arr.ndim == 2 and arr.shape[1] == 384, \
        f"{allergen_id}: ocekivan oblik (L,384), dobijeno {arr.shape}"
    mean_vec = arr.mean(axis=0).astype(np.float32)
    if np.isnan(mean_vec).any():
        n_nan += 1
    pooled[allergen_id] = mean_vec

with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(pooled, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Proteins pooled: {len(pooled)}")
print(f"Embedding size: {next(iter(pooled.values())).shape}")
print(f"Proteins with a NaN embedding: {n_nan}/{len(pooled)}" + ("  <<< PROBLEM" if n_nan else "  (clean)"))
print(f"Saved: {OUTPUT_PICKLE}")

# --- metadata parquet (nuzan pratilac za load_dataset()) ---
meta_df = pd.read_csv(INPUT_CSV)
meta_df = meta_df[meta_df["allergen_id"].isin(pooled.keys())].copy()
rows = []
for row in meta_df.itertuples(index=False):
    if row.allergen_id not in pooled:
        continue
    rows.append({
        "allergen_id": row.allergen_id,
        "official_name": row.official_name,
        "source_food": row.source_food,
        "organism": row.organism,
        "protein_family": row.protein_family,
        "sequence_length": row.sequence_length,
        "embedding": pooled[row.allergen_id].tolist(),
    })
embedding_df = pd.DataFrame(rows)
embedding_df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Saved: {OUTPUT_PARQUET} ({len(embedding_df)} redova)")
n_missing_meta = len(pooled) - len(embedding_df)
if n_missing_meta:
    print(f"UPOZORENJE: {n_missing_meta} proteina iz pkl-a nema odgovarajuci red u {INPUT_CSV} "
          f"(izbaceni iz parquet-a, proveri allergen_id poklapanje)")
