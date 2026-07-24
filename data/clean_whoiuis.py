### ova skripta služi da napravi čistiju verziju WHO/IUIS dataseta, izbacue nepotrebne karakteristike i beleske


import pandas as pd
import re


INPUT = "/home/lana/ALERGRAF/data/jointable.csv"
OUTPUT = "clean_allergens.csv"


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).replace("\xa0", " ").strip()


def clean_sequence(seq):
    if pd.isna(seq):
        return ""

    return re.sub(
        r"[^A-Za-z]",
        "",
        str(seq)
    ).upper()


def make_id(row):

    allergen = clean_text(row["AllergenID"])
    iso = clean_text(row["IsoAllergenID"])

    if iso:
        return f"WHO_{allergen}_ISO_{iso}"

    return f"WHO_{allergen}"


print("Loading WHO/IUIS...")

df = pd.read_csv(
    INPUT,
    low_memory=False
)


# clean column names

df.columns = [
    clean_text(c)
    for c in df.columns
]


rows = []


for _, row in df.iterrows():

    # prefer isoallergen name
    name = clean_text(row["IsoName"])

    if not name:
        name = clean_text(row["Name"])


    sequence = clean_sequence(
        row["Sequence"]
    )


    reference = "; ".join(
        filter(
            None,
            [
                clean_text(row["AllergenicityRef"]),
                clean_text(row["SequenceRef"])
            ]
        )
    )


    rows.append({

        "allergen_id":
            make_id(row),

        "official_name":
            name,

        "source_food":
            clean_text(row["Common"]),

        "organism":
            clean_text(row["Species"]),

        "protein_family":
            "",

        "uniprot_id":
            clean_text(row["AccUniProt"]),

        "sequence_available":
            "Yes" if sequence else "No",

        "fasta_sequence":
            sequence,

        "sequence_length":
            len(sequence) if sequence else "",

        "reference":
            reference
    })


out = pd.DataFrame(rows)


# remove exact duplicates
out = out.drop_duplicates()


# save

out.to_csv(
    OUTPUT,
    index=False
)


print("Finished!")
print("Number of allergens:", len(out))

print(
    "Missing UniProt:",
    (out["uniprot_id"] == "").sum()
)

print(
    "Missing sequences:",
    (out["sequence_available"] == "No").sum()
)