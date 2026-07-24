#ova skripta sluzi da napravi kombinaciju dva csv fajla WHO/IUIS i ALLERGENONLINE, vezane su imenima

import pandas as pd
import re
import os


WHO_FILE = "/home/lana/ALERGRAF/data/jointable.csv"
AO_FILE = "/home/lana/ALERGRAF/data/allergenonline.csv"

OUTPUT_DIR = "output"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------

def clean_text(x):
    if pd.isna(x):
        return ""

    return str(x).replace("\xa0", " ").strip()


def normalize_name(x):
    """
    Converts:
    Act c 10.0101 -> Act c 10
    Act d 8.0101 -> Act d 8
    """

    x = clean_text(x)

    if x == "":
        return ""

    return re.sub(r"\.\d+$", "", x)


def clean_sequence(seq):

    if pd.isna(seq):
        return ""

    seq = str(seq)

    seq = re.sub(
        r"[^A-Za-z]",
        "",
        seq
    )

    return seq.upper()


# -----------------------------
# Load WHO/IUIS
# -----------------------------

print("Loading WHO/IUIS...")

who = pd.read_csv(
    WHO_FILE,
    low_memory=False
)


who.columns = [
    clean_text(c)
    for c in who.columns
]


# create matching key

who["match_name"] = (
    who["IsoName"]
    .fillna(who["Name"])
    .apply(normalize_name)
)


# sequence cleaning

who["Sequence"] = (
    who["Sequence"]
    .apply(clean_sequence)
)


# -----------------------------
# Load AllergenOnline
# -----------------------------

print("Loading AllergenOnline...")


ao = pd.read_csv(
    AO_FILE,
    low_memory=False
)


ao.columns = [
    clean_text(c)
    for c in ao.columns
]


ao["match_name"] = (
    ao["IUIS Allergen"]
    .apply(normalize_name)
)


# -----------------------------
# Create output
# -----------------------------

print("Combining datasets...")


rows = []


for idx, row in who.iterrows():

    allergen_name = clean_text(
        row.get("IsoName")
    )

    if allergen_name == "":
        allergen_name = clean_text(
            row.get("Name")
        )


    match = normalize_name(
        allergen_name
    )


    # find AllergenOnline match

    ao_match = ao[
        ao["match_name"] == match
    ]


    protein_family = ""

    ao_reference = ""


    if len(ao_match) > 0:

        first = ao_match.iloc[0]

        protein_family = clean_text(
            first.get("Group")
        )

        ao_reference = clean_text(
            first.get("Allergenicity")
        )


    sequence = clean_sequence(
        row.get("Sequence")
    )


    if sequence:

        seq_available = "Yes"
        seq_length = len(sequence)

    else:

        seq_available = "No"
        seq_length = ""


    references = "; ".join(
        filter(
            None,
            [
                clean_text(row.get("AllergenicityRef")),
                clean_text(row.get("SequenceRef")),
                ao_reference
            ]
        )
    )


    notes = "; ".join(
        filter(
            None,
            [
                clean_text(row.get("Allergenicity")),
                clean_text(row.get("SeqFeatures"))
            ]
        )
    )


    output = {

        "allergen_id":
            f'{clean_text(row.get("AllergenID"))}_'
            f'{clean_text(row.get("IsoAllergenID"))}',


        "official_name":
            allergen_name,


        "source_food":
            clean_text(row.get("Common")),


        "organism":
            clean_text(row.get("Species")),


        "protein_family":
            protein_family,


        "uniprot_id":
            clean_text(row.get("AccUniProt")),


        "who_iuis_name":
            match,


        "sequence_available":
            seq_available,


        "fasta_sequence":
            sequence,


        "sequence_length":
            seq_length,


        "database_source":
            "WHO/IUIS; AllergenOnline",


        "reference":
            references,


        "notes":
            notes

    }


    rows.append(output)



# -----------------------------
# Save
# -----------------------------


df = pd.DataFrame(rows)


# remove exact duplicates

before = len(df)


df = df.drop_duplicates(
    subset=[
        "official_name",
        "uniprot_id",
        "fasta_sequence"
    ]
)


removed = before - len(df)



outfile = os.path.join(
    OUTPUT_DIR,
    "allergens.csv"
)


df.to_csv(
    outfile,
    index=False
)



# -----------------------------
# Validation report
# -----------------------------


report = []


report.append(
    f"Total allergens: {len(df)}"
)

report.append(
    f"Duplicates removed: {removed}"
)


report.append(
    f"Missing UniProt IDs: "
    f"{(df.uniprot_id=='').sum()}"
)


report.append(
    f"Missing FASTA sequences: "
    f"{(df.sequence_available=='No').sum()}"
)


report.append(
    f"Missing protein family: "
    f"{(df.protein_family=='').sum()}"
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "validation_report.txt"
    ),
    "w"
) as f:

    f.write(
        "\n".join(report)
    )


print("\nDONE")
print(outfile)

for r in report:
    print(r)