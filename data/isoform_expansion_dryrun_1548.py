"""
Dry-run (samo brojanje, ne upisuje) za D2 izoform ekspanziju: za svaki
Confirmed/Strong par A-B, nadji izoforme A1 (i B1) sa >=80% BLAST
identicnosti naspram vec koriscene izoforme, koje JOS NISU uparene sa
drugim clanom para. Racuna koliko NOVIH kandidata bi ovo generisalo, PRE
nego sto se bilo sta upise.
"""

import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

ISOFORM_IDENTITY_THRESHOLD = 80.0

allergens = pd.read_csv("/home/lana/ALERGRAF/output/clean_allergens.csv")
names = sorted(set(allergens["official_name"].astype(str)))
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id


def base_name(n):
    return n.rsplit(".", 1)[0] if "." in n else n


groups = defaultdict(list)
for n in names:
    groups[base_name(n)].append(n)
isoform_groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

with open("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl", "rb") as f:
    blast = pickle.load(f)
blast_ids = blast["ids"]
identity_matrix = blast["identity_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}


def high_identity_isoforms(name):
    """Vraca listu (isoform_name, pct) za sve dodatne izoforme sa >=threshold
    identicnoscu naspram DATOG imena (ne nuzno 'primarne', naspram TOG konkretnog clana para)."""
    base = base_name(name)
    if base not in isoform_groups:
        return []
    pid = name_to_id.get(name)
    if pid not in blast_id_to_index:
        return []
    pidx = blast_id_to_index[pid]
    results = []
    for iso in isoform_groups[base]:
        if iso == name:
            continue
        iid = name_to_id.get(iso)
        if iid not in blast_id_to_index:
            continue
        iidx = blast_id_to_index[iid]
        pct = float(identity_matrix[pidx, iidx])
        if pct >= ISOFORM_IDENTITY_THRESHOLD:
            results.append((iso, pct))
    return results


gold = pd.read_csv("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
strong = gold[gold["evidence_level"].isin(
    ["Confirmed", "Strong evidence", "Strong evidence (within-species paralogs)", "Strong evidence (congeneric species)"]
)]

existing_pairs = set()
for _, row in gold.iterrows():
    a, b = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    existing_pairs.add(frozenset([a, b]))

new_candidates = []
for _, row in strong.iterrows():
    a, b = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    for iso, pct in high_identity_isoforms(a):
        if frozenset([iso, b]) not in existing_pairs and frozenset([iso, b]) not in {frozenset([x[0], x[1]]) for x in new_candidates}:
            new_candidates.append((iso, b, a, pct, row["evidence_level"]))
    for iso, pct in high_identity_isoforms(b):
        if frozenset([a, iso]) not in existing_pairs and frozenset([a, iso]) not in {frozenset([x[0], x[1]]) for x in new_candidates}:
            new_candidates.append((a, iso, b, pct, row["evidence_level"]))

print(f"Ukupno NOVIH kandidat-parova (izoform ekspanzija, prag >={ISOFORM_IDENTITY_THRESHOLD}%): {len(new_candidates)}")
print()
print("Raspodela po baznom evidence_level (od kog se par nasledjuje):")
print(pd.Series([c[4] for c in new_candidates]).value_counts())
print()
print("Prvih 15 primera:")
for iso_or_a, b_or_iso, orig, pct, base_level in new_candidates[:15]:
    print(f"  {iso_or_a}  <->  {b_or_iso}   (izvedeno iz {orig}, {pct:.1f}% identicnost, baza={base_level})")
