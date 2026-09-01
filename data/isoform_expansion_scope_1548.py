"""
Obim provere za D2 (izoform ekspanzija): koliko Confirmed/Strong parova ima
bar jednog clana sa dodatnim izoformama u pool-u, i kolika je BLAST
identicnost izmedju izoformi (da filtriramo na >80% kako mentorka trazi).
Samo izvestava, ne upisuje.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

allergens = pd.read_csv("/home/lana/ALERGRAF/output/clean_allergens.csv")
names = sorted(set(allergens["official_name"].astype(str)))


def base_name(n):
    return n.rsplit(".", 1)[0] if "." in n else n


groups = defaultdict(list)
for n in names:
    groups[base_name(n)].append(n)

multi_isoform = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
print("Proteini sa >1 izoformom u pool-u:", len(multi_isoform))
print("Ukupno dodatnih izoformi (preko prve po grupi):", sum(len(v) - 1 for v in multi_isoform.values()))

gold = pd.read_csv("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
strong = gold[gold["evidence_level"].isin(
    ["Confirmed", "Strong evidence", "Strong evidence (within-species paralogs)", "Strong evidence (congeneric species)"]
)]


def has_extra(n):
    return len(multi_isoform.get(base_name(str(n)), [])) > 1


has_extra_mask = strong["allergen_id_1"].apply(has_extra) | strong["allergen_id_2"].apply(has_extra)
print(f"Confirmed+Strong parova gde BAR JEDAN clan ima dodatne izoforme: {has_extra_mask.sum()}/{len(strong)}")

with open("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl", "rb") as f:
    blast = pickle.load(f)
blast_ids = blast["ids"]
identity_matrix = blast["identity_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id

# Za svaku grupu izoformi, min/max identicnost naspram PRVE (najniza oznaka = "primarna")
low_identity_examples = []
high_identity_count = 0
total_extra = 0
for base, isoforms in multi_isoform.items():
    primary = isoforms[0]
    pid = name_to_id.get(primary)
    if pid not in blast_id_to_index:
        continue
    pidx = blast_id_to_index[pid]
    for iso in isoforms[1:]:
        iid = name_to_id.get(iso)
        if iid not in blast_id_to_index:
            continue
        iidx = blast_id_to_index[iid]
        pct = identity_matrix[pidx, iidx]
        total_extra += 1
        if pct >= 80.0:
            high_identity_count += 1
        elif pct < 50.0:
            low_identity_examples.append((primary, iso, round(float(pct), 1)))

print(f"\nOd {total_extra} dodatnih izoformi (naspram primarne u grupi): {high_identity_count} ima >=80% BLAST identicnosti")
print(f"\nPrimeri NISKE identicnosti (<50%, zanimljivo -- 'izoforme' koje su zapravo vrlo razlicite):")
for p, i, pct in low_identity_examples[:15]:
    print(f"  {p} vs {i}: {pct}%")
