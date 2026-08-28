"""
Izoluje da li je slab MLP rezultat (analysis/mlp_sensitivity_1548.py, svih 8
konfiguracija znacajno gore od cosine-a) posledica ENKODINGA ulaza (abs-diff
|u-v|) ili same nelinearnosti/kapaciteta mreze.

Hadamard bilinear (ml/pipeline/models/classifiers/hadamard.py) koristi
SIMETRICAN elementwise produkt u*v kao ulaz u JEDAN linearni sloj (~1280
parametara), i taj model je jedini koji je dostigao cosine (MRR 0.1761,
delta+0.0072). MLP koristi abs-diff |u-v| + nelinearnost (ReLU).

Ovaj skript trenira MALU MLP (ista velicina kao najbolje konfiguracije iz
mlp_sensitivity_1548.py: hidden=[16] i hidden=[32]+L2-u-loss-u) ali na
Hadamard produktu u*v kao ulazu umesto abs-diff -- ako nelinearna MLP na
Hadamard produktu dostigne/prevazidje cist linearni Hadamard bilinear,
enkoding NIJE problem (nelinearnost pomaze). Ako i dalje zaostaje slicno
abs-diff MLP-u, verovatnije je da je nelinearnost/MLP forma sama po sebi
losije uskladjena sa ovim problemom (ne samo enkoding ulaza).

Izlaz:
    output/mlp_hadamard_input_1548_summary.txt
    output/mlp_hadamard_input_1548_results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "/home/lana/ALERGRAF")
from ml.pipeline.common.data import load_dataset  # noqa: E402
from ml.pipeline.common.evaluation import bootstrap_ci, retrieval_evaluate, summarize_retrieval  # noqa: E402
from ml.pipeline.common.negatives import sample_negative_pairs  # noqa: E402
from ml.pipeline.common.splitting import group_aware_split, split_pairs  # noqa: E402
from ml.pipeline.models.classifiers.base import PairClassifier  # noqa: E402

EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
METADATA = Path("/home/lana/ALERGRAF/embeddings/embeddings.parquet")
GOLD = Path("/home/lana/ALERGRAF/output/cross_reactive_1548.csv")
SUMMARY_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_input_1548_summary.txt")
CSV_OUTPUT = Path("/home/lana/ALERGRAF/output/mlp_hadamard_input_1548_results.csv")

SEED = 42
TEST_FRACTION = 0.2
NEG_PER_POS = 10


class HadamardMLP(nn.Module):
    """MLP na Hadamard produktu u*v (simetrican ulaz, kao Hadamard bilinear, ali sa skrivenim slojem/nelinearnoscu)."""

    def __init__(self, dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = dim
        for h, d in zip(hidden_dims, dropout):
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(d)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u, v):
        return self.net(u * v).squeeze(-1)


class HadamardMLPClassifier(PairClassifier):
    def __init__(self, params, seed=42):
        super().__init__()
        self.params = params
        self.seed = seed
        torch.manual_seed(seed)
        self.model = None

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index, **kwargs):
        self.set_pool(embedding_matrix, id_to_index)
        p = self.params
        emb32 = embedding_matrix.astype(np.float32)

        rows_u, rows_v, labels = [], [], []
        for pos in positive_pairs:
            rows_u.append(emb32[id_to_index[pos["id_1"]]])
            rows_v.append(emb32[id_to_index[pos["id_2"]]])
            labels.append(1.0)
        for a, b in negative_pairs:
            rows_u.append(emb32[id_to_index[a]])
            rows_v.append(emb32[id_to_index[b]])
            labels.append(0.0)

        U = torch.from_numpy(np.array(rows_u, dtype=np.float32))
        V = torch.from_numpy(np.array(rows_v, dtype=np.float32))
        y = torch.tensor(labels, dtype=torch.float32)

        n = len(y)
        val_size = max(10, int(0.15 * n))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        val_idx, fit_idx = perm[:val_size], perm[val_size:]

        self.model = HadamardMLP(embedding_matrix.shape[1], p["hidden_dims"], p["dropout"])
        pos_weight = torch.tensor((y[fit_idx] == 0).sum().item() / max((y[fit_idx] == 1).sum().item(), 1))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])

        best_val_loss, best_state, no_improve = np.inf, None, 0
        for epoch in range(p["max_epochs"]):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(U[fit_idx], V[fit_idx])
            bce_loss = criterion(logits, y[fit_idx])
            if p["l2_lambda"] > 0:
                l2_term = p["l2_lambda"] * sum((param ** 2).sum() for param in self.model.parameters())
                loss = bce_loss + l2_term
            else:
                loss = bce_loss
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(U[val_idx], V[val_idx])
                val_loss = criterion(val_logits, y[val_idx]).item()
            if val_loss < best_val_loss:
                best_val_loss, best_state, no_improve = val_loss, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                no_improve += 1
            if no_improve >= p["patience"]:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.stopped_epoch = epoch
        self._embedding_tensor = torch.from_numpy(emb32)

    def score_all(self, query_id):
        qidx = self.id_to_index[query_id]
        q_vec = self._embedding_tensor[qidx].unsqueeze(0).expand(self._embedding_tensor.shape[0], -1)
        with torch.no_grad():
            return self.model(q_vec, self._embedding_tensor).numpy()


GRID = [
    ("hadamard_mlp_16",       dict(hidden_dims=[16], dropout=[0.2], learning_rate=1e-2, weight_decay=1e-4, l2_lambda=0.0, max_epochs=300, patience=20)),
    ("hadamard_mlp_32",       dict(hidden_dims=[32], dropout=[0.3], learning_rate=1e-2, weight_decay=1e-4, l2_lambda=0.0, max_epochs=300, patience=20)),
    ("hadamard_mlp_32_l2loss", dict(hidden_dims=[32], dropout=[0.3], learning_rate=1e-2, weight_decay=0.0, l2_lambda=1e-3, max_epochs=300, patience=20)),
]

print("Loading dataset (jednom)...")
dataset = load_dataset(EMBEDDINGS, METADATA, GOLD)
train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, TEST_FRACTION, SEED)
train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
n_train_neg = len(train_pairs) * NEG_PER_POS
train_negatives = sample_negative_pairs(train_ids, n_train_neg, SEED, dataset.positive_pair_set)
cosine_matrix = cosine_similarity(dataset.embedding_matrix)

print(f"\nProlazim kroz {len(GRID)} konfiguracija...\n")
results = []
for label, params in GRID:
    clf = HadamardMLPClassifier(params=params, seed=SEED)
    clf.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    retrieval_df = retrieval_evaluate(test_pairs, clf, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix)
    summary = summarize_retrieval(retrieval_df)
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=1000, seed=SEED, baseline_col="cosine_reciprocal_rank")

    results.append({
        "label": label, "hidden_dims": str(params["hidden_dims"]), "l2_lambda": params["l2_lambda"],
        "stopped_epoch": clf.stopped_epoch,
        "mrr": summary["mrr"], "cosine_mrr": summary["cosine_mrr"],
        "delta": delta_stats["mean"], "ci_lo": delta_stats["ci_lo"], "ci_hi": delta_stats["ci_hi"],
        "significant": delta_stats["significant"],
    })
    sig_marker = " <-- ZNACAJNO" if delta_stats["significant"] else ""
    print(f"  {label:22s}  epoch={clf.stopped_epoch:4d}  MRR={summary['mrr']:.4f}  delta={delta_stats['mean']:+.4f}  "
          f"CI=[{delta_stats['ci_lo']:+.4f},{delta_stats['ci_hi']:+.4f}]{sig_marker}", flush=True)

results_df = pd.DataFrame(results)
results_df.to_csv(CSV_OUTPUT, index=False)

summary_lines = ["=" * 80, "MLP na Hadamard produktu (izoluje enkoding od kapaciteta)", "=" * 80, "",
                  f"Cosine baseline MRR: {results_df['cosine_mrr'].iloc[0]:.4f}",
                  "Referenca -- cist linearan Hadamard bilinear (bez skrivenog sloja): MRR=0.1761, delta=+0.0072",
                  "Referenca -- najbolji abs-diff MLP (baseline_l2_in_loss, 100% podataka): MRR=0.1522, delta=-0.0165", ""]
for _, r in results_df.sort_values("mrr", ascending=False).iterrows():
    summary_lines.append(
        f"{r['label']:<22}{r['mrr']:<10.4f}delta={r['delta']:+.4f}  "
        f"CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]{'  ZNACAJNO' if r['significant'] else ''}"
    )
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(SUMMARY_OUTPUT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved: {CSV_OUTPUT}")
print(f"Saved: {SUMMARY_OUTPUT}")
