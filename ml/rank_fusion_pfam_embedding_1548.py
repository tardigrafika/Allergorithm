"""
Trenira malu MLP klasifikator (zamrznut ESM embedding -> Pfam familija),
izvlaci naucenu reprezentaciju (sloj pre klasifikacije) kao NOV embedding
prostor, testira cosine slicnost u tom prostoru kao RRF glas - 1548 dataset.

Zasto ovo, a ne direktno Pfam Jaccard (probano ranije, nije pomoglo): sirov
Pfam overlap je verovatno redundantan sa BLAST/cosine (Pfam dodela je i sama
sekvencijalno izvedena). Hipoteza ovde: reprezentacija NAUCENA da razlikuje
Pfam familije (multi-class klasifikacija, 48 familija sa >=5 primera, 970
proteina) moze da organizuje prostor DRUGACIJE od sirovog ESM-a ili prostog
Jaccard-a na skupovima domena - i, kljucno, uci iz mnogo vise podataka
(970 proteina, per-protein labela) nego bilo sta drugo probano ovu sesiju
(sve ostalo je zavisilo od ~1500 PAROVA u 44-50 nezavisnih komponenti).

Metodoloska napomena: Pfam labela je NEZAVISNA od naseg cross-reactivity
grafa (HMM sekvencijalno skeniranje, ne cirkularno kao stari same_family),
pa trening na SVIM Pfam-labeliranim proteinima (bez obzira da li se
pojavljuju u gold parovima) nije leakage u smislu cross-reactivity zadatka -
isti princip kao koriscenje BLAST/Foldseek matrica bez fold-podele.

Izlaz:
    output/rank_fusion_pfam_embedding_1548_summary.txt
    output/rank_fusion_pfam_embedding_1548_per_query.csv
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")
PFAM_DOMAINS = Path("/home/lana/ALERGRAF/output/pfam_domains_1548.csv")

OUTPUT_DIR = Path("/home/lana/ALERGRAF/output")
SUMMARY_OUTPUT = OUTPUT_DIR / "rank_fusion_pfam_embedding_1548_summary.txt"
PER_QUERY_OUTPUT = OUTPUT_DIR / "rank_fusion_pfam_embedding_1548_per_query.csv"

RRF_K = 60
TOP_K = [1, 5, 10, 20]
MIN_CLASS_SIZE = 5
HIDDEN_DIM = 256
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")
with open(EMBEDDINGS, "rb") as f:
    embeddings_dict = pickle.load(f)
metadata = pd.read_parquet(METADATA)
metadata = metadata[metadata["allergen_id"].isin(embeddings_dict.keys())].copy()

with open(BLAST_MATRIX, "rb") as f:
    blast_data = pickle.load(f)
blast_ids = blast_data["ids"]
blast_score_matrix = blast_data["score_matrix"]
blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}

with open(FOLDSEEK_LOOKUP, "rb") as f:
    foldseek_lookup = pickle.load(f)

all_ids = metadata["allergen_id"].tolist()
id_to_index = {aid: i for i, aid in enumerate(all_ids)}
n_candidates = len(all_ids)
embedding_matrix = np.array([embeddings_dict[aid] for aid in all_ids], dtype=np.float64)
cosine_matrix = cosine_similarity(embedding_matrix)


# =====================================================
# TRAIN Pfam-FAMILY CLASSIFIER (frozen ESM embedding -> Pfam class)
# =====================================================

pfam_df = pd.read_csv(PFAM_DOMAINS)
pfam_df = pfam_df[pfam_df["pfam_accessions"].notna() & (pfam_df["pfam_accessions"] != "")].copy()
pfam_df["primary_pfam"] = pfam_df["pfam_accessions"].str.split(";").str[0]
class_counts = pfam_df["primary_pfam"].value_counts()
keep_classes = set(class_counts[class_counts >= MIN_CLASS_SIZE].index)
pfam_df = pfam_df[pfam_df["primary_pfam"].isin(keep_classes)]
pfam_df = pfam_df[pfam_df["allergen_id"].isin(embeddings_dict.keys())]
print(f"Classifier training set: {len(pfam_df)} proteins, {pfam_df['primary_pfam'].nunique()} classes")

class_list = sorted(pfam_df["primary_pfam"].unique())
class_to_idx = {c: i for i, c in enumerate(class_list)}
n_classes = len(class_list)

X = np.array([embeddings_dict[aid] for aid in pfam_df["allergen_id"]], dtype=np.float32)
y = np.array([class_to_idx[c] for c in pfam_df["primary_pfam"]], dtype=np.int64)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val)


class PfamClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        h = self.embed(x)
        return self.classify(h), h


model = PfamClassifier(X.shape[1], HIDDEN_DIM, n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

best_val_acc = 0.0
best_state = None
patience, patience_counter = 15, 0

print("\nTraining Pfam classifier...")
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(X_train_t)
    loss = loss_fn(logits, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits, _ = model(X_val_t)
        val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch}, best val acc so far: {best_val_acc:.4f}")
            break

    if epoch % 20 == 0:
        print(f"  epoch {epoch}: train_loss={loss.item():.4f}, val_acc={val_acc:.4f}", flush=True)

model.load_state_dict(best_state)
print(f"Best validation accuracy: {best_val_acc:.4f}  "
      f"(chance level with {n_classes} classes: {1/n_classes:.4f})")

# extract the learned representation for ALL 1534 proteins (forward pass only)
model.eval()
with torch.no_grad():
    all_X = torch.tensor(embedding_matrix.astype(np.float32))
    _, learned_repr = model(all_X)
    learned_repr = learned_repr.numpy()

pfam_embed_cosine_matrix = cosine_similarity(learned_repr)


# =====================================================
# REST OF RRF SETUP (identical pattern to prior scripts)
# =====================================================

perm = np.array([blast_id_to_index.get(aid, -1) for aid in all_ids])
valid = perm >= 0
blast_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
valid_idx = np.where(valid)[0]
blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

print("Building dense Foldseek TM-score matrix...")
foldseek_matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
for key, score in foldseek_lookup.items():
    if len(key) != 2:
        continue
    a, b = tuple(key)
    if a in id_to_index and b in id_to_index:
        i, j = id_to_index[a], id_to_index[b]
        foldseek_matrix[i, j] = score
        foldseek_matrix[j, i] = score

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

name_to_id = {}
for _, row in metadata.iterrows():
    n = str(row["official_name"]).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row["allergen_id"]

gold_pairs = []
for _, row in gold.iterrows():
    n1, n2 = str(row["allergen_id_1"]).strip(), str(row["allergen_id_2"]).strip()
    if n1 not in name_to_id or n2 not in name_to_id:
        continue
    id1, id2 = name_to_id[n1], name_to_id[n2]
    if id1 == id2 or id1 not in id_to_index or id2 not in id_to_index:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2, "pair_id": row["pair_id"]})
print(f"Gold pairs: {len(gold_pairs)}")


def ranks_from_scores(scores, self_index):
    s = scores.astype(np.float64, copy=True)
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    ranks = np.empty(len(s), dtype=np.int64)
    ranks[order] = np.arange(1, len(s) + 1)
    return ranks


def rank_of(ranks, target_index):
    return int(ranks[target_index])


print("\nScoring all queries...")
start = time.time()
records = []

for qi, p in enumerate(gold_pairs):
    for query_id, target_id in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
        qidx = id_to_index[query_id]
        tidx = id_to_index[target_id]

        cos_ranks = ranks_from_scores(cosine_matrix[qidx], qidx)
        blast_ranks = ranks_from_scores(blast_matrix[qidx], qidx)
        fs_ranks = ranks_from_scores(foldseek_matrix[qidx], qidx)

        rrf3_score = 1.0 / (RRF_K + cos_ranks) + 1.0 / (RRF_K + blast_ranks) + 1.0 / (RRF_K + fs_ranks)

        pfam_embed_ranks = ranks_from_scores(pfam_embed_cosine_matrix[qidx], qidx)
        pfam_embed_contrib = 1.0 / (RRF_K + pfam_embed_ranks)

        rrf_pe_score = rrf3_score + pfam_embed_contrib

        rrf3_ranks = ranks_from_scores(rrf3_score, qidx)
        rrf_pe_ranks = ranks_from_scores(rrf_pe_score, qidx)
        pfam_embed_only_ranks = ranks_from_scores(pfam_embed_cosine_matrix[qidx], qidx)

        records.append({
            "pair_id": p["pair_id"],
            "pfam_embed_only_rank": rank_of(pfam_embed_only_ranks, tidx),
            "cosine_rank": rank_of(cos_ranks, tidx),
            "blast_rank": rank_of(blast_ranks, tidx),
            "foldseektm_rank": rank_of(fs_ranks, tidx),
            "rrf3_rank": rank_of(rrf3_ranks, tidx),
            "rrf_pfam_embed_rank": rank_of(rrf_pe_ranks, tidx),
        })

    if (qi + 1) % 200 == 0 or (qi + 1) == len(gold_pairs):
        elapsed = time.time() - start
        print(f"  {qi+1}/{len(gold_pairs)} pairs ({elapsed/60:.1f} min elapsed)", flush=True)

total_elapsed = time.time() - start
print(f"\nDone: {len(records)} queries in {total_elapsed/60:.1f} min")

df = pd.DataFrame(records)
df.to_csv(PER_QUERY_OUTPUT, index=False)
print(f"Saved: {PER_QUERY_OUTPUT}")


# =====================================================
# AGGREGATE
# =====================================================

pfam_embed_only_mrr = (1.0 / df["pfam_embed_only_rank"]).mean()
rrf3_mrr = (1.0 / df["rrf3_rank"]).mean()
rrf_pe_mrr = (1.0 / df["rrf_pfam_embed_rank"]).mean()

summary_lines = [
    "=" * 70,
    f"RRF: does a Pfam-family-trained embedding help as a voter? ({len(df)} queries, 1548 dataset)",
    "=" * 70,
    f"Pfam classifier: {n_classes} classes, {len(X_train)} train / {len(X_val)} val, "
    f"best val accuracy = {best_val_acc:.4f} (chance = {1/n_classes:.4f})",
    "",
    f"Pfam-embedding cosine ALONE (individual signal) MRR = {pfam_embed_only_mrr:.4f}",
    "",
    f"RRF-3 (cosine+BLAST+FoldseekTM)  MRR = {rrf3_mrr:.4f}",
    f"RRF-3 + Pfam-embedding cosine    MRR = {rrf_pe_mrr:.4f}",
    f"Delta: {rrf_pe_mrr - rrf3_mrr:+.4f}",
    "",
]
for k in TOP_K:
    h3 = (df["rrf3_rank"] <= k).mean()
    hp = (df["rrf_pfam_embed_rank"] <= k).mean()
    summary_lines.append(f"Hits@{k}: RRF-3={h3:.4f}  RRF-3+PfamEmb={hp:.4f}")

rng = np.random.default_rng(42)
pair_ids = df["pair_id"].unique()
deltas = []
for _ in range(2000):
    sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
    counts = pd.Series(sampled).value_counts()
    sub = df.merge(counts.rename("w"), left_on="pair_id", right_index=True)
    w = sub["w"].to_numpy()
    d = np.average(1.0 / sub["rrf_pfam_embed_rank"], weights=w) - np.average(1.0 / sub["rrf3_rank"], weights=w)
    deltas.append(d)
deltas = np.array(deltas)
summary_lines.append("")
summary_lines.append(f"Bootstrap 95% CI (RRF-3+PfamEmb - RRF-3): [{np.percentile(deltas,2.5):+.4f}, {np.percentile(deltas,97.5):+.4f}]")
summary_lines.append(f"Fraction of bootstrap resamples favoring RRF-3+PfamEmb: {(deltas>0).mean():.3f}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {SUMMARY_OUTPUT}")
