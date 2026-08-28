"""
Ucitavanje embeddinga/metapodataka/gold parova -- identicna logika koja se
ponavlja u SVIM ml/*.py skriptovima (random_forest_baseline.py, mlp_baseline.py,
xgboost_blast_kfold_1443.py, itd.). Preuzeto 1:1, bez izmena ponasanja.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    embeddings_dict: dict            # allergen_id -> np.ndarray (embedding)
    all_ids: list                    # candidate universe, fixed order
    id_to_index: dict                # allergen_id -> index in all_ids / embedding_matrix
    embedding_matrix: np.ndarray     # (n_candidates, embedding_dim)
    name_to_id: dict                 # official_name -> allergen_id
    gold_pairs: list                 # [{"id_1","id_2","pair_id","family_1","family_2"}, ...]
    positive_pair_set: set           # {(id_1,id_2) sorted, ...} -- za bezbedno negative sampling


def load_embeddings(embeddings_path: Path) -> dict:
    with open(embeddings_path, "rb") as f:
        return pickle.load(f)


def build_name_to_id(metadata: pd.DataFrame) -> dict:
    """Identicno svuda: prvi official_name pobedjuje, duplikati se preskacu."""
    name_to_id = {}
    for _, row in metadata.iterrows():
        official_name = str(row["official_name"]).strip()
        if official_name == "" or official_name.lower() == "nan":
            continue
        if official_name in name_to_id:
            continue
        name_to_id[official_name] = row["allergen_id"]
    return name_to_id


def map_gold_pairs(gold: pd.DataFrame, name_to_id: dict, id_to_index: dict) -> list:
    """Identicno svuda: mapira allergen_id_1/2 (official_name stringovi u gold
    fajlu) na stvarne allergen_id vrednosti, preskace nepoznate/self-parove."""
    gold_pairs = []
    for _, row in gold.iterrows():
        name_1 = str(row["allergen_id_1"]).strip()
        name_2 = str(row["allergen_id_2"]).strip()
        if name_1 not in name_to_id or name_2 not in name_to_id:
            continue
        id_1, id_2 = name_to_id[name_1], name_to_id[name_2]
        if id_1 not in id_to_index or id_2 not in id_to_index or id_1 == id_2:
            continue
        gold_pairs.append({
            "pair_id": row.get("pair_id"),
            "id_1": id_1, "id_2": id_2,
            "name_1": name_1, "name_2": name_2,
            "family_1": row.get("family_1"), "family_2": row.get("family_2"),
        })
    return gold_pairs


def filter_negative_evidence(gold_raw: pd.DataFrame) -> pd.DataFrame:
    """Identicno svuda (skriptovi sa evidence_level kolonom, npr. *_1443/*_1548):
    izbacuje redove ciji evidence_level oznacava da NIJE pravi pozitivan par."""
    if "evidence_level" not in gold_raw.columns:
        return gold_raw
    negative_mask = gold_raw["evidence_level"].str.contains(
        "negative|Contested|Risky|NO cross", case=False, na=False
    )
    return gold_raw.loc[~negative_mask].copy()


def load_dataset(embeddings_path: Path, metadata_path: Path, gold_path: Path,
                  filter_negatives: bool = True) -> Dataset:
    embeddings_dict = load_embeddings(embeddings_path)

    metadata = pd.read_parquet(metadata_path)
    metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

    all_ids = metadata["allergen_id"].tolist()
    id_to_index = {aid: i for i, aid in enumerate(all_ids)}
    embedding_matrix = np.array([embeddings_dict[aid] for aid in all_ids], dtype=np.float64)

    name_to_id = build_name_to_id(metadata)

    gold_raw = pd.read_csv(gold_path)
    gold = filter_negative_evidence(gold_raw) if filter_negatives else gold_raw

    gold_pairs = map_gold_pairs(gold, name_to_id, id_to_index)
    positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}

    return Dataset(
        embeddings_dict=embeddings_dict, all_ids=all_ids, id_to_index=id_to_index,
        embedding_matrix=embedding_matrix, name_to_id=name_to_id,
        gold_pairs=gold_pairs, positive_pair_set=positive_pair_set,
    )
