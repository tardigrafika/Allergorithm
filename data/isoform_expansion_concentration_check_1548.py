import sys
from collections import Counter

sys.path.insert(0, "/home/lana/ALERGRAF/data")
import isoform_expansion_dryrun_1548 as dryrun  # noqa: E402

base_protein_counter = Counter()
for iso_or_a, b_or_iso, orig, pct, base_level in dryrun.new_candidates:
    base_protein_counter[dryrun.base_name(orig)] += 1

print("Top 20 izvor-proteina po broju generisanih kandidata:")
for base, n in base_protein_counter.most_common(20):
    print(f"  {base:20s} {n}")
print()
print("Ukupno razlicitih izvornih (baznih) proteina koji generisu kandidate:", len(base_protein_counter))
top5_share = sum(n for _, n in base_protein_counter.most_common(5)) / len(dryrun.new_candidates) * 100
print(f"Top-5 izvor-proteina generise {top5_share:.1f}% svih 950 kandidata.")
