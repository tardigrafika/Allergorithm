"""
HITS@K и MRR benchmark za ESM-2 embeddinge
Cilj: Proveriti da li cosine similarity ESM-2 embeddinga može da rangira poznate cross-reactive alergene visoko.

Za svaki poznati cross-reactive par:

1. Uzme se alergen A.
2. Poredi se sa svim ostalim alergenima.
3. Kandidati se rangiraju prema cosine similarity.
4. Proverava se pozicija poznatog cross-reactive partnera.
5. Računaju se Hits@K i Reciprocal Rank.

Evaluacija se radi u oba smera: A → B i B → A.

Mapiranje: Pošto gold standard koristi zvanična imena alergena, a embeddingi WHO/IUIS ID-jeve, vrši se mapiranje:

official name → allergen ID → ESM-2 embedding

Metrike: Hits@1, Hits@5, Hits@10, Hits@20 i MRR

Ulazi: `embeddings.pkl`, `embeddings.parquet`, `gold_standard_cross_reactivity.csv`

Izlaz: `hits_mrr_results.csv`

"""


import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# PATHS
# =====================================================

EMBEDDINGS = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
)

METADATA = Path(
    "/home/lana/ALERGRAF/embeddings/embeddings.parquet"
)

GOLD = Path(
    "/home/lana/ALERGRAF/output/cross_reactive_combined.csv"
)

OUTPUT = Path(
    "/home/lana/ALERGRAF/output/hits_mrr_results.csv"
)


# =====================================================
# CONFIGURATION
# =====================================================

TOP_K = [1, 5, 10, 20]


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

print("\n==============================")
print("LOADING DATA")
print("==============================")

print("Loading embeddings...")

with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)

print(
    f"Proteins in embeddings: "
    f"{len(embeddings_dict)}"
)


# =====================================================
# LOAD METADATA
# =====================================================

print("Loading metadata...")

metadata = pd.read_parquet(
    METADATA
)

print(
    f"Metadata rows: "
    f"{len(metadata)}"
)


# =====================================================
# CHECK REQUIRED METADATA COLUMNS
# =====================================================

required_metadata_columns = [
    "allergen_id",
    "official_name"
]

for column in required_metadata_columns:

    if column not in metadata.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in embeddings.parquet.\n"
            f"Available columns: "
            f"{metadata.columns.tolist()}"
        )


# =====================================================
# LOAD GOLD STANDARD
# =====================================================

print("Loading gold standard...")

gold = pd.read_csv(
    GOLD
)

print(
    f"Gold standard pairs: "
    f"{len(gold)}"
)


# =====================================================
# CHECK REQUIRED GOLD STANDARD COLUMNS
# =====================================================

required_gold_columns = [
    "pair_id",
    "allergen_id_1",
    "allergen_id_2",
    "family_1",
    "family_2"
]

for column in required_gold_columns:

    if column not in gold.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in gold standard.\n"
            f"Available columns: "
            f"{gold.columns.tolist()}"
        )


# =====================================================
# FILTER METADATA
# =====================================================

print("\nFiltering metadata...")

# Keep only allergens that actually have embeddings

metadata = metadata[
    metadata["allergen_id"].isin(
        embeddings_dict.keys()
    )
].copy()

print(
    f"Metadata rows with embeddings: "
    f"{len(metadata)}"
)


# =====================================================
# CREATE OFFICIAL NAME -> ALLERGEN ID MAPPING
# =====================================================

print(
    "\nCreating official name -> "
    "WHO/IUIS ID mapping..."
)


name_to_id = {}

duplicate_names = 0


for _, row in metadata.iterrows():

    official_name = str(
        row["official_name"]
    ).strip()

    allergen_id = row[
        "allergen_id"
    ]


    # Skip missing names

    if (
        official_name == ""
        or
        official_name.lower() == "nan"
    ):

        continue


    # Check for duplicate official names

    if official_name in name_to_id:

        duplicate_names += 1

        continue


    name_to_id[
        official_name
    ] = allergen_id


print(
    f"Official names mapped: "
    f"{len(name_to_id)}"
)


print(
    f"Duplicate official names skipped: "
    f"{duplicate_names}"
)


# =====================================================
# CREATE ORDERED LIST OF ALLERGEN IDs
# =====================================================

all_ids = metadata[
    "allergen_id"
].tolist()


# =====================================================
# CREATE ID -> MATRIX INDEX MAPPING
# =====================================================

id_to_index = {
    allergen_id: i
    for i, allergen_id in enumerate(
        all_ids
    )
}


# =====================================================
# CREATE EMBEDDING MATRIX
# =====================================================

print(
    "\nCreating embedding matrix..."
)


embedding_matrix = np.array(
    [
        embeddings_dict[
            allergen_id
        ]
        for allergen_id in all_ids
    ]
)


print(
    f"Embedding matrix shape: "
    f"{embedding_matrix.shape}"
)


# =====================================================
# PRECOMPUTE COSINE SIMILARITY MATRIX
# =====================================================

print(
    "\nCalculating cosine similarities..."
)


similarity_matrix = cosine_similarity(
    embedding_matrix
)


print(
    "Cosine similarity matrix created."
)


# =====================================================
# GOLD STANDARD EVALUATION
# =====================================================

print(
    "\n=============================="
)

print(
    "RUNNING HITS@K / MRR"
)

print(
    "=============================="
)


results = []

missing_pairs = 0

evaluated_pairs = 0

evaluated_queries = 0


# =====================================================
# LOOP THROUGH GOLD STANDARD PAIRS
# =====================================================

for _, row in gold.iterrows():

    pair_id = row[
        "pair_id"
    ]


    # Gold standard columns are actually
    # official allergen names.
    #
    # Example:
    #
    # allergen_id_1 = Bet v 1.0101
    # allergen_id_2 = Mal d 1.0101

    name_1 = str(
        row["allergen_id_1"]
    ).strip()

    name_2 = str(
        row["allergen_id_2"]
    ).strip()


    # =================================================
    # MAP OFFICIAL NAMES TO WHO/IUIS IDS
    # =================================================

    if (
        name_1 not in name_to_id
        or
        name_2 not in name_to_id
    ):

        missing_pairs += 1


        missing_1 = (
            name_1
            not in name_to_id
        )

        missing_2 = (
            name_2
            not in name_to_id
        )


        print(
            f"WARNING: Missing mapping "
            f"for pair {pair_id}"
        )


        if missing_1:

            print(
                f"    Missing allergen 1: "
                f"{name_1}"
            )


        if missing_2:

            print(
                f"    Missing allergen 2: "
                f"{name_2}"
            )


        continue


    # =================================================
    # GET WHO/IUIS IDS
    # =================================================

    allergen_1 = name_to_id[
        name_1
    ]

    allergen_2 = name_to_id[
        name_2
    ]


    # =================================================
    # CHECK IDS EXIST IN MATRIX
    # =================================================

    if (
        allergen_1 not in id_to_index
        or
        allergen_2 not in id_to_index
    ):

        missing_pairs += 1

        print(
            f"WARNING: Mapped IDs not "
            f"found in embedding matrix "
            f"for pair {pair_id}"
        )

        continue


    # =================================================
    # GET MATRIX INDICES
    # =================================================

    index_1 = id_to_index[
        allergen_1
    ]

    index_2 = id_to_index[
        allergen_2
    ]


    evaluated_pairs += 1


    # =================================================
    # EVALUATE BOTH DIRECTIONS
    # =================================================

    directions = [

        {
            "query_name":
                name_1,

            "target_name":
                name_2,

            "query_id":
                allergen_1,

            "target_id":
                allergen_2,

            "query_index":
                index_1,

            "target_index":
                index_2,

            "family_query":
                row["family_1"],

            "family_target":
                row["family_2"]
        },


        {
            "query_name":
                name_2,

            "target_name":
                name_1,

            "query_id":
                allergen_2,

            "target_id":
                allergen_1,

            "query_index":
                index_2,

            "target_index":
                index_1,

            "family_query":
                row["family_2"],

            "family_target":
                row["family_1"]
        }

    ]


    # =================================================
    # RUN RETRIEVAL FOR EACH DIRECTION
    # =================================================

    for direction in directions:


        query_name = direction[
            "query_name"
        ]

        target_name = direction[
            "target_name"
        ]

        query_id = direction[
            "query_id"
        ]

        target_id = direction[
            "target_id"
        ]

        query_index = direction[
            "query_index"
        ]

        target_index = direction[
            "target_index"
        ]

        family_query = direction[
            "family_query"
        ]

        family_target = direction[
            "family_target"
        ]


        # =============================================
        # GET SIMILARITY SCORES
        # =============================================

        similarities = similarity_matrix[
            query_index
        ].copy()


        # =============================================
        # REMOVE SELF SIMILARITY
        # =============================================

        similarities[
            query_index
        ] = -np.inf


        # =============================================
        # RANK ALL OTHER PROTEINS
        # =============================================

        ranked_indices = np.argsort(
            similarities
        )[::-1]


        # =============================================
        # FIND RANK OF TRUE PARTNER
        # =============================================

        positions = np.where(
            ranked_indices
            ==
            target_index
        )[0]


        if len(positions) == 0:

            print(
                f"WARNING: Could not rank "
                f"{target_name} "
                f"for query {query_name}"
            )

            continue


        # Python index starts at 0.
        # Ranking starts at 1.

        rank = int(
            positions[0]
        ) + 1


        # =============================================
        # TRUE PAIR COSINE SIMILARITY
        # =============================================

        true_similarity = similarity_matrix[
            query_index,
            target_index
        ]


        # =============================================
        # RECIPROCAL RANK
        # =============================================

        reciprocal_rank = (
            1.0 / rank
        )


        # =============================================
        # HITS@K
        # =============================================

        hit_1 = int(
            rank <= 1
        )

        hit_5 = int(
            rank <= 5
        )

        hit_10 = int(
            rank <= 10
        )

        hit_20 = int(
            rank <= 20
        )


        # =============================================
        # SAVE RESULT
        # =============================================

        results.append({

            "pair_id":
                pair_id,

            "query_allergen":
                query_name,

            "target_allergen":
                target_name,

            "query_allergen_id":
                query_id,

            "target_allergen_id":
                target_id,

            "query_family":
                family_query,

            "target_family":
                family_target,

            "cosine_similarity":
                true_similarity,

            "rank":
                rank,

            "reciprocal_rank":
                reciprocal_rank,

            "hits_at_1":
                hit_1,

            "hits_at_5":
                hit_5,

            "hits_at_10":
                hit_10,

            "hits_at_20":
                hit_20

        })


        evaluated_queries += 1


# =====================================================
# CREATE RESULTS DATAFRAME
# =====================================================

result_df = pd.DataFrame(
    results
)


# =====================================================
# CHECK RESULTS
# =====================================================

print(
    "\n=============================="
)

print(
    "EVALUATION SUMMARY"
)

print(
    "=============================="
)


print(
    f"Gold standard pairs: "
    f"{len(gold)}"
)


print(
    f"Valid pairs evaluated: "
    f"{evaluated_pairs}"
)


print(
    f"Missing pairs: "
    f"{missing_pairs}"
)


print(
    f"Retrieval queries evaluated: "
    f"{evaluated_queries}"
)


if len(result_df) == 0:

    print(
        "\nERROR:"
    )

    print(
        "No valid pairs were found."
    )

    print(
        "Check the official_name mapping "
        "between gold standard and "
        "embeddings.parquet."
    )

    raise SystemExit(1)


# =====================================================
# CALCULATE METRICS
# =====================================================

hits_at_1 = result_df[
    "hits_at_1"
].mean()


hits_at_5 = result_df[
    "hits_at_5"
].mean()


hits_at_10 = result_df[
    "hits_at_10"
].mean()


hits_at_20 = result_df[
    "hits_at_20"
].mean()


mrr = result_df[
    "reciprocal_rank"
].mean()


# =====================================================
# PRINT HITS@K
# =====================================================

print(
    "\n=============================="
)

print(
    "HITS@K"
)

print(
    "=============================="
)


print(
    f"Hits@1  : "
    f"{hits_at_1:.4f}"
)


print(
    f"Hits@5  : "
    f"{hits_at_5:.4f}"
)


print(
    f"Hits@10 : "
    f"{hits_at_10:.4f}"
)


print(
    f"Hits@20 : "
    f"{hits_at_20:.4f}"
)


# =====================================================
# PRINT MRR
# =====================================================

print(
    "\n=============================="
)

print(
    "MRR"
)

print(
    "=============================="
)


print(
    f"MRR : "
    f"{mrr:.4f}"
)


# =====================================================
# RANK STATISTICS
# =====================================================

print(
    "\n=============================="
)

print(
    "RANK STATISTICS"
)

print(
    "=============================="
)


print(
    result_df[
        "rank"
    ].describe()
)


# =====================================================
# SAVE DETAILED RESULTS
# =====================================================

print(
    "\nSaving detailed results..."
)


# Create output directory if necessary

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


result_df.to_csv(
    OUTPUT,
    index=False
)


# =====================================================
# FINAL OUTPUT
# =====================================================

print(
    "\n=============================="
)

print(
    "SAVED"
)

print(
    "=============================="
)


print(
    f"Detailed results saved to:"
)

print(
    OUTPUT
)


print(
    "\nBenchmark completed successfully."
)