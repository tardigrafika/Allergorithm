"""
LoRA fine-tuning ESM-2 (650M) direktno na cross-reactivity zadatku - VM/GPU.

POSTEN RIZIK (postavljen unapred, ne posle): dataset ima samo 44-46
nezavisnih povezanih komponenti. I sa LoRA (nizak broj trenabilnih
parametara, ne pun fine-tune) postoji realna sansa da model "zapamti"
trening familije umesto da naucni generalizuje - isti fundamentalni problem
male kolicine nezavisnih podataka, samo u drugom obliku. Ovo je poslednja
preostala genuinski razlicita opcija, ne garantovano resenje.

Metod: triplet loss sa cosine distancom (dokazano bolje od MSE u ranijem
"Pristup B v2" eksperimentu na zamrznutim embeddinzima), safe hard-negative
mining (cross-family SAMO, "hardest-of-K") - identican princip kao
data/build_ml_dataset.py.

Protein-level split (Union-Find, isti algoritam kao svuda u projektu) da se
izbegne curenje - train/test na nivou PROTEINA, ne parova.

Ulaz (prebaciti na VM):
    output/clean_allergens.csv
    output/cross_reactive_1548.csv

Izlaz:
    /content/lora_adapter_1548/  (sacuvan LoRA adapter)
    /content/embeddings_lora_1548.pkl  (embeddinzi SVIH proteina sa fine-tuned modelom)
    Ispisuje MRR/Hits@K poredjenje (fine-tuned vs frozen baseline) na held-out test proteinima
"""

import os

# must be set BEFORE torch initializes CUDA -- reduces allocator fragmentation,
# which is what actually triggered the OOM (14.28/14.56 GiB "used" but a 20MB
# alloc still failed -- classic fragmentation signature, not genuine peak need)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, EsmModel

# =====================================================
# CONFIGURATION
# =====================================================

CLEAN_ALLERGENS = Path("/content/clean_allergens.csv")
GOLD = Path("/content/cross_reactive_1548.csv")

ADAPTER_OUTPUT = Path("/content/lora_adapter_1548")
EMBEDDINGS_OUTPUT = Path("/content/embeddings_lora_1548.pkl")

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH = 1022          # used for final embedding generation (inference only, cheap)
TRAIN_MAX_LENGTH = 512     # shorter cap during training -- backprop is much more memory-hungry
                           # than inference; covers the vast majority of sequences (mean ~245 aa)

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 2           # reduced from 5 -- with TRAIN_BATCH_SIZE=1 (forced by OOM), full config was ~8.75h
N_EPOCHS = 2               # reduced from 5 -- first pass is a directional check, not a fully converged run
LR = 1e-4
MARGIN = 0.3
LORA_R = 8
LORA_ALPHA = 16
TOP_K = [1, 5, 10, 20]
TRAIN_BATCH_SIZE = 1      # reduced from 2 -- still OOM'd (fragmentation, GPU was 14.28/14.56 GiB "full")

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))


# =====================================================
# LOAD DATA
# =====================================================

allergens = pd.read_csv(CLEAN_ALLERGENS)
allergens = allergens[allergens["fasta_sequence"].notna() & (allergens["fasta_sequence"] != "")].copy()
id_to_seq = dict(zip(allergens["allergen_id"], allergens["fasta_sequence"]))
name_to_id = {}
for row in allergens.itertuples(index=False):
    n = str(row.official_name).strip()
    if n and n not in name_to_id:
        name_to_id[n] = row.allergen_id
all_ids = sorted(id_to_seq.keys())
print(f"Proteins with a sequence: {len(all_ids)}")

gold_raw = pd.read_csv(GOLD)
negative_mask = gold_raw["evidence_level"].str.contains(
    "negative|Contested|Risky|NO cross", case=False, na=False
)
gold = gold_raw.loc[~negative_mask].copy()

gold_pairs = []
family_map = {}
for row in gold.itertuples(index=False):
    n1, n2 = str(row.allergen_id_1).strip(), str(row.allergen_id_2).strip()
    id1, id2 = name_to_id.get(n1), name_to_id.get(n2)
    if id1 is None or id2 is None or id1 == id2:
        continue
    if id1 not in id_to_seq or id2 not in id_to_seq:
        continue
    gold_pairs.append({"id_1": id1, "id_2": id2})
    f1, f2 = str(row.family_1).strip(), str(row.family_2).strip()
    if f1:
        family_map.setdefault(id1, f1)
    if f2:
        family_map.setdefault(id2, f2)

print(f"Mapped gold pairs: {len(gold_pairs)}")
positive_pair_set = {tuple(sorted((p["id_1"], p["id_2"]))) for p in gold_pairs}


# =====================================================
# PROTEIN-LEVEL SPLIT (Union-Find, identical algorithm to the rest of the project)
# =====================================================

parent = {}


def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for p in gold_pairs:
    union(p["id_1"], p["id_2"])

components = {}
for pid in parent:
    components.setdefault(find(pid), set()).add(pid)
component_list = list(components.values())

rng = np.random.default_rng(SEED)
order = rng.permutation(len(component_list))
gold_protein_count = sum(len(c) for c in component_list)
target_test = round(TEST_FRACTION * gold_protein_count)

train_ids, test_ids = set(), set()
running_test = 0
for idx in order:
    c = component_list[idx]
    if running_test < target_test:
        test_ids |= c
        running_test += len(c)
    else:
        train_ids |= c

free_proteins = [pid for pid in all_ids if pid not in train_ids and pid not in test_ids]
free_proteins = sorted(free_proteins)
rng.shuffle(free_proteins)
n_free_test = round(TEST_FRACTION * len(free_proteins))
test_ids |= set(free_proteins[:n_free_test])
train_ids |= set(free_proteins[n_free_test:])

train_positive_pairs = [p for p in gold_pairs if p["id_1"] in train_ids and p["id_2"] in train_ids]
test_positive_pairs = [p for p in gold_pairs if p["id_1"] in test_ids and p["id_2"] in test_ids]
print(f"Train proteins: {len(train_ids)}, Test proteins: {len(test_ids)}")
print(f"Train positive pairs: {len(train_positive_pairs)}, Test positive pairs: {len(test_positive_pairs)}")


# =====================================================
# SAFE HARD-NEGATIVE TRIPLET SAMPLING (cross-family only, identical
# principle to data/build_ml_dataset.py's sample_negatives)
# =====================================================

def sample_triplets(positive_pairs, protein_pool, family_map, seed):
    local_rng = np.random.default_rng(seed)
    pool = sorted(protein_pool)
    labeled = [a for a in pool if a in family_map]
    fam_buckets = {}
    for a in labeled:
        fam_buckets.setdefault(family_map[a], []).append(a)
    fam_names = list(fam_buckets.keys())

    triplets = []
    for p in positive_pairs:
        anchor, positive = p["id_1"], p["id_2"]
        negative = None
        anchor_fam = family_map.get(anchor)
        if anchor_fam is not None and len(fam_names) >= 2:
            for _ in range(20):
                fam_b = local_rng.choice([f for f in fam_names if f != anchor_fam]) \
                    if any(f != anchor_fam for f in fam_names) else None
                if fam_b is None:
                    break
                cand = local_rng.choice(fam_buckets[fam_b])
                if cand != anchor and cand != positive:
                    negative = cand
                    break
        if negative is None:
            cand = local_rng.choice(pool)
            if cand != anchor and cand != positive:
                negative = cand
        if negative is not None:
            triplets.append((anchor, positive, negative))
    return triplets


# =====================================================
# MODEL: ESM-2 + LoRA
# =====================================================

print("\nLoading tokenizer + ESM-2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = EsmModel.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["query", "value"],  # ESM-2 attention projections
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(base_model, lora_config)
model.to(device)
model.print_trainable_parameters()

# gradient checkpointing: trades compute for memory (recomputes activations
# during backward instead of storing them) -- essential for fine-tuning a
# 650M model on a 14-16GB GPU. Needs enable_input_require_grads() alongside
# it when the base weights are frozen (LoRA), otherwise checkpointing can
# silently produce a graph with no grad-requiring inputs.
model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def _make_inputs_require_grad(module, inp, out):
        out.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

use_fp16 = device.type == "cuda"  # safe for ESM (BERT-style) -- unlike the earlier Ankh/T5 NaN issue

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def embed_sequences(seqs, model_to_use, no_grad=True, max_length=MAX_LENGTH):
    tokens = tokenizer(seqs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model_to_use(**tokens)
        else:
            out = model_to_use(**tokens)
    mask = tokens["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
    summed = torch.sum(out.last_hidden_state.float() * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y)


triplet_loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=MARGIN)


# =====================================================
# TRAINING LOOP
# =====================================================

print(f"\nTraining for {N_EPOCHS} epochs...")
start = time.time()

for epoch in range(N_EPOCHS):
    epoch_triplets = []
    for _ in range(NEG_PER_POS):
        epoch_triplets.extend(sample_triplets(train_positive_pairs, train_ids, family_map, seed=SEED + epoch * 100 + _))
    rng.shuffle(epoch_triplets)

    model.train()
    total_loss = 0.0
    n_batches = 0
    batch_size = TRAIN_BATCH_SIZE
    for bi in range(0, len(epoch_triplets), batch_size):
        batch = epoch_triplets[bi:bi + batch_size]
        anchors = [id_to_seq[t[0]] for t in batch]
        positives = [id_to_seq[t[1]] for t in batch]
        negatives = [id_to_seq[t[2]] for t in batch]

        emb_a = embed_sequences(anchors, model, no_grad=False, max_length=TRAIN_MAX_LENGTH)
        emb_p = embed_sequences(positives, model, no_grad=False, max_length=TRAIN_MAX_LENGTH)
        emb_n = embed_sequences(negatives, model, no_grad=False, max_length=TRAIN_MAX_LENGTH)

        loss = triplet_loss_fn(emb_a, emb_p, emb_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        n_batches += 1
        del emb_a, emb_p, emb_n, loss  # drop references to the computation graph immediately
        if device.type == "cuda":
            torch.cuda.empty_cache()  # every batch -- fragmentation was the actual OOM cause, not peak size
        if n_batches % 20 == 0:
            elapsed = time.time() - start
            print(f"  epoch {epoch} batch {n_batches}/{len(epoch_triplets)//batch_size} "
                  f"loss={loss_value:.4f} ({elapsed/60:.1f} min elapsed)", flush=True)

    print(f"Epoch {epoch} done: mean loss = {total_loss/max(1,n_batches):.4f} "
          f"({(time.time()-start)/60:.1f} min elapsed)", flush=True)

    model.save_pretrained(str(ADAPTER_OUTPUT))
    print(f"  [checkpoint] adapter saved to {ADAPTER_OUTPUT}")


# =====================================================
# GENERATE EMBEDDINGS FOR ALL PROTEINS (fine-tuned model)
# =====================================================

print("\nGenerating embeddings for all proteins (fine-tuned model)...")
model.eval()
lora_embeddings = {}
batch_size = 8
for bi in range(0, len(all_ids), batch_size):
    batch_ids = all_ids[bi:bi + batch_size]
    seqs = [id_to_seq[aid] for aid in batch_ids]
    emb = embed_sequences(seqs, model, no_grad=True).cpu().numpy()
    for aid, vec in zip(batch_ids, emb):
        lora_embeddings[aid] = vec
    if bi % 200 == 0:
        print(f"  {bi}/{len(all_ids)} embedded", flush=True)

with open(EMBEDDINGS_OUTPUT, "wb") as f:
    pickle.dump(lora_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Saved: {EMBEDDINGS_OUTPUT}")


# =====================================================
# GENERATE BASELINE (FROZEN, NO LoRA) EMBEDDINGS -- for a fair, matched comparison
# =====================================================

print("\nGenerating baseline (frozen, no LoRA) embeddings for comparison...")
with model.disable_adapter():
    baseline_embeddings = {}
    for bi in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[bi:bi + batch_size]
        seqs = [id_to_seq[aid] for aid in batch_ids]
        emb = embed_sequences(seqs, model, no_grad=True).cpu().numpy()
        for aid, vec in zip(batch_ids, emb):
            baseline_embeddings[aid] = vec


# =====================================================
# EVALUATE ON HELD-OUT TEST PROTEINS (fine-tuned vs frozen baseline)
# =====================================================

def cosine_sim_matrix(emb_dict, ids):
    mat = np.array([emb_dict[i] for i in ids], dtype=np.float64)
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return norm @ norm.T


id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
lora_sim = cosine_sim_matrix(lora_embeddings, all_ids)
base_sim = cosine_sim_matrix(baseline_embeddings, all_ids)


def eval_mrr_hits(sim_matrix, test_pairs):
    rr, hits = [], {k: [] for k in TOP_K}
    for p in test_pairs:
        for qid, tid in [(p["id_1"], p["id_2"]), (p["id_2"], p["id_1"])]:
            qi, ti = id_to_idx[qid], id_to_idx[tid]
            scores = sim_matrix[qi].copy()
            scores[qi] = -np.inf
            rank = int(np.argsort(scores)[::-1].tolist().index(ti)) + 1
            rr.append(1.0 / rank)
            for k in TOP_K:
                hits[k].append(int(rank <= k))
    return float(np.mean(rr)), {k: float(np.mean(v)) for k, v in hits.items()}


lora_mrr, lora_hits = eval_mrr_hits(lora_sim, test_positive_pairs)
base_mrr, base_hits = eval_mrr_hits(base_sim, test_positive_pairs)

print("\n" + "=" * 60)
print("HELD-OUT TEST RESULTS (protein-level split, never seen in training)")
print("=" * 60)
print(f"Frozen ESM-2 (no LoRA)  MRR = {base_mrr:.4f}  " +
      "  ".join(f"Hits@{k}={base_hits[k]:.4f}" for k in TOP_K))
print(f"LoRA fine-tuned         MRR = {lora_mrr:.4f}  " +
      "  ".join(f"Hits@{k}={lora_hits[k]:.4f}" for k in TOP_K))
print(f"Delta: {lora_mrr - base_mrr:+.4f}")
print("\nDownload embeddings_lora_1548.pkl and the lora_adapter_1548/ folder back "
      "to local embeddings/, then we'll test it in the RRF fusion and against RRF-3.")
