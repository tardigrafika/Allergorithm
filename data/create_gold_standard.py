import pandas as pd


CLEAN = "/home/lana/ALERGRAF/output/clean_allergens.csv"
CROSS = "/home/lana/ALERGRAF/output/cross_reactive.csv"

OUTPUT = "output/gold_standard_cross_reactivity.csv"


clean = pd.read_csv(CLEAN)
cross = pd.read_csv(CROSS)


# base name bez isoforme
clean["base_name"] = (
    clean["official_name"]
    .str.split(".")
    .str[0]
)


# uzmi prvu dostupnu sekvencu za svaki base name
mapping = (
    clean
    .sort_values("official_name")
    .groupby("base_name")
    .first()["official_name"]
    .to_dict()
)


rows = []


for _, row in cross.iterrows():

    a = row["allergen_1"]
    b = row["allergen_2"]


    if a in mapping and b in mapping:

        rows.append({

            "pair_id": row["pair_id"],

            "allergen_1_original": a,

            "allergen_2_original": b,

            "allergen_1":
                mapping[a],

            "allergen_2":
                mapping[b],

            "evidence_level":
                row["evidence_level"],

            "reference":
                row["reference"]

        })


gold = pd.DataFrame(rows)


gold.to_csv(
    OUTPUT,
    index=False
)


print("Original pairs:", len(cross))
print("Gold standard pairs:", len(gold))
print("Removed:", len(cross)-len(gold))


print(gold.head())