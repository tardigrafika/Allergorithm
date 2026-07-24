import pandas as pd


clean = pd.read_csv("/home/lana/ALERGRAF/output/clean_allergens.csv")
cross = pd.read_csv("/home/lana/ALERGRAF/output/cross_reactive.csv")


# uklanja verziju .0101
def base_name(x):
    return str(x).split(".")[0]


clean["base_name"] = clean["official_name"].apply(base_name)


available = set(clean["base_name"])


missing = []

for allergen in pd.concat([
    cross["allergen_1"],
    cross["allergen_2"]
]).unique():

    if allergen not in available:
        missing.append(allergen)


print("Missing after normalization:")
for x in missing:
    print(x)


print("\nTotal missing:", len(missing))


print("\nExample matches:")

for x in cross["allergen_1"].head(10):

    match = clean[
        clean["base_name"] == x
    ]["official_name"].tolist()

    print(
        x,
        "->",
        match[:3]
    )