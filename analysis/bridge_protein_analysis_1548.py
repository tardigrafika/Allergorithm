"""
Bridge protein analiza: da li proteini koji POVEZUJU razlicite delove gold
grafa (razlicite familije unutar iste connected komponente, ili prave
artikulacione tacke izmedju komponenti) objasnjavaju neproporcionalan udeo
gresaka cosine baseline-a.

Motivacija: RRF-4/cosine su simetricne funkcije para -- ako je gold graf
lokalno "zvezdast" oko odredjenih proteina (jedan protein povezuje vise
razlicitih familija/klastera), ti proteini strukturno imaju najvise
konkurentskih kandidata sa RAZLICitim tacnim odgovorima u razlicitim
kontekstima -- moglo bi delimicno objasniti gresaka koncentraciju
nezavisno od familijske "guzve" vec dijagnostikovane.

Dve definicije "bridge" proteina:
  1. Familijski-bridging: protein ciji SUSEDI u gold grafu pripadaju
     VISE OD JEDNE razlicite familije (on sam most izmedju klastera).
  2. Artikulaciona tacka (pravi graph-theory bridge/cut vertex): protein
     ciji bi uklanjanje POVECALO broj connected komponenti gold grafa.

Za oba tipa: poredi prosecan cosine rank/MRR upita KOJI UKLJUCUJU bridge
protein (kao query ili target) vs upita koji ne ukljucuju nijedan.

Izlaz:
    output/bridge_protein_analysis_1548_summary.txt
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/bridge_protein_analysis_1548_summary.txt")
PER_QUERY_OUTPUT = Path("/home/lana/ALERGRAF/output/bridge_protein_analysis_1548_per_query.csv")

print("Loading dataset...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

# gold graf: cvorovi = allergen_id, ivice = gold parovi
G = nx.Graph()
G.add_nodes_from(dataset.all_ids)
family_of = {}
for p in dataset.gold_pairs:
    G.add_edge(p["id_1"], p["id_2"])
    if p.get("family_1"):
        family_of[p["id_1"]] = p["family_1"]
    if p.get("family_2"):
        family_of[p["id_2"]] = p["family_2"]

print(f"Graf: {G.number_of_nodes()} cvorova, {G.number_of_edges()} ivica, "
      f"{nx.number_connected_components(G)} connected komponenti")

# 1) Familijski-bridging cvorovi: susedi pripadaju >1 razlicitoj familiji
family_bridge_nodes = set()
for node in G.nodes():
    if G.degree(node) < 2:
        continue
    neighbor_families = {family_of.get(n) for n in G.neighbors(node) if family_of.get(n)}
    if len(neighbor_families) > 1:
        family_bridge_nodes.add(node)

# 2) Prave artikulacione tacke (cut vertices)
articulation_nodes = set(nx.articulation_points(G))

print(f"Familijski-bridging proteina: {len(family_bridge_nodes)}")
print(f"Artikulacionih tacaka (cut vertices): {len(articulation_nodes)}")
print(f"Preklapanje: {len(family_bridge_nodes & articulation_nodes)}")

id_to_name = {v: k for k, v in dataset.name_to_id.items()}

# evaluacija: za SVAKI gold par, oba smera, cosine rank
results = []
for p in dataset.gold_pairs:
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qi = dataset.id_to_index[query_id]
        ti = dataset.id_to_index[target_id]
        cos_scores = cosine_matrix[qi].copy()
        cos_scores[qi] = -np.inf
        cos_rank = int(np.argsort(np.argsort(-cos_scores))[ti]) + 1
        results.append({
            "pair_id": p["pair_id"],
            "query": query_id, "target": target_id,
            "query_name": id_to_name.get(query_id, query_id),
            "family": p.get("family_1"),
            "cosine_rank": cos_rank,
            "cosine_rr": 1.0 / cos_rank,
            "query_is_family_bridge": query_id in family_bridge_nodes,
            "query_is_articulation": query_id in articulation_nodes,
            "pair_involves_family_bridge": query_id in family_bridge_nodes or target_id in family_bridge_nodes,
            "pair_involves_articulation": query_id in articulation_nodes or target_id in articulation_nodes,
        })

results_df = pd.DataFrame(results)
results_df.to_csv(PER_QUERY_OUTPUT, index=False)

summary_lines = ["=" * 80, "Bridge protein analiza -- gold graf, cosine baseline", "=" * 80, "",
                  f"Graf: {G.number_of_nodes()} cvorova, {G.number_of_edges()} ivica, "
                  f"{nx.number_connected_components(G)} connected komponenti",
                  f"Familijski-bridging proteina: {len(family_bridge_nodes)}",
                  f"Artikulacionih tacaka: {len(articulation_nodes)}", ""]

for flag_col, label in [("pair_involves_family_bridge", "Familijski-bridging"),
                         ("pair_involves_articulation", "Artikulaciona tacka")]:
    involved = results_df[results_df[flag_col]]
    not_involved = results_df[~results_df[flag_col]]
    if len(involved) == 0:
        summary_lines.append(f"{label}: nema upita, preskoceno")
        continue
    mrr_involved = involved["cosine_rr"].mean()
    mrr_not = not_involved["cosine_rr"].mean()
    summary_lines.append(
        f"{label}: n={len(involved)} upita (od {len(results_df)}), "
        f"MRR={mrr_involved:.4f} vs MRR (ostali)={mrr_not:.4f}, "
        f"delta={mrr_involved - mrr_not:+.4f}"
    )

summary_lines.append("")
summary_lines.append("Top 15 familijski-bridging proteina po broju upita i prosecnom rangu:")
bridge_sub = results_df[results_df["query_is_family_bridge"]]
if len(bridge_sub) > 0:
    agg = bridge_sub.groupby("query_name").agg(
        n_queries=("cosine_rank", "size"),
        mean_rank=("cosine_rank", "mean"),
        mrr=("cosine_rr", "mean"),
    ).sort_values("mean_rank", ascending=False).head(15)
    summary_lines.append(agg.to_string())

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
