"""
Naucena projekcija embedding_dim -> proj_dim (deljena, Siamese-stil,
trenirana END-TO-END sa ostatkom mreze -- ne PCA/random projekcija kao
preprocessing korak) PRE pairwise kombinovanja. Testira Prioritet 2
korisnickog zahteva (analysis/mlp_hadamard_esm2_3b_richpair_sensitivity_1548.py):
da li se ESM-2 3B (2560-dim) reprezentacija bolje iskoristi kad se prvo
svede na manju, NAUCENU dimenziju (256/512) pre nego sto se paran par
kombinuje (hadamard ili richconcat), umesto da se ceo 2560-dim ulaz direktno
hadamard-uje kao u MLPPairClassifier.

Arhitektura:
    eA (dim) --\\                                    /-- combine(pA,pB) --> MLP head --> logit
                 shared Linear(dim, proj_dim) [+ opciono LayerNorm]
    eB (dim) --/

Deljena projekcija (ISTA tezinska matrica za A i B, ne dve odvojene) --
garantuje da projekcija sama ne unosi asimetriju; kombinovana sa
canonical_slots() (za "richconcat" combine mod, gde su sirovi projektovani
slotovi deo ulaza) score(A,B)==score(B,A) ostaje tacno.

score_all() projektuje CEO pool JEDNOM (posle fit-a, keširano) -- pairwise
kombinovanje i MLP head rade nad VEC projektovanim (manjim) vektorima, jeftino
za ponovljeno rangiranje (LOCO radi ovo mnogo puta).

Isti PairClassifier interfejs (fit/score_all) kao MLPPairClassifier -- moze
se ubaciti u iste screening/LOCO skripte bez izmene evaluacije.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from ...common.features import canonical_slots
from .base import PairClassifier

DEFAULT_PARAMS = dict(
    proj_dim=256,
    pre_layernorm=False,       # LayerNorm na PROJEKTOVANIM (proj_dim) vektorima, pre combine-a
    combine="hadamard",        # "hadamard" (proj_dim) ili "richconcat" (4*proj_dim: pA,pB,|pA-pB|,pA*pB)
    hidden_dims=[32],
    dropout=[0.3],
    val_fraction=0.15,
    batch_size=64,
    max_epochs=300,
    patience=20,
    learning_rate=1e-2,
    weight_decay=0.0,
    l2_lambda=1e-3,
)


class ProjectedPairNet(nn.Module):
    def __init__(self, input_dim, proj_dim, pre_layernorm, combine, hidden_dims, dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim, proj_dim)
        self.pre_norm = nn.LayerNorm(proj_dim) if pre_layernorm else None
        self.combine = combine

        combine_dim = proj_dim if combine == "hadamard" else 4 * proj_dim
        layers = []
        prev_dim = combine_dim
        for h, d in zip(hidden_dims, dropout):
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(d)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

    def project(self, emb):
        p = self.projection(emb)
        if self.pre_norm is not None:
            p = self.pre_norm(p)
        return p

    def combine_projected(self, pa, pb):
        if self.combine == "hadamard":
            return pa * pb
        return torch.cat([pa, pb, torch.abs(pa - pb), pa * pb], dim=-1)

    def forward(self, emb_a, emb_b):
        pa, pb = self.project(emb_a), self.project(emb_b)
        return self.head(self.combine_projected(pa, pb)).squeeze(-1)


class ProjectedMLPPairClassifier(PairClassifier):
    def __init__(self, params: dict | None = None, seed: int = 42, **kwargs):
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = None
        self._projected_pool_cache = None

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index, **kwargs):
        self.set_pool(embedding_matrix, id_to_index)
        p = self.params

        rows_a, rows_b, ids_a, ids_b, labels = [], [], [], [], []
        for pair in positive_pairs:
            rows_a.append(embedding_matrix[id_to_index[pair["id_1"]]])
            rows_b.append(embedding_matrix[id_to_index[pair["id_2"]]])
            ids_a.append(pair["id_1"])
            ids_b.append(pair["id_2"])
            labels.append(1.0)
        for a, b in negative_pairs:
            rows_a.append(embedding_matrix[id_to_index[a]])
            rows_b.append(embedding_matrix[id_to_index[b]])
            ids_a.append(a)
            ids_b.append(b)
            labels.append(0.0)

        rows_a, rows_b = np.array(rows_a, dtype=np.float32), np.array(rows_b, dtype=np.float32)
        # kanoniski poredak -- garantuje simetriju za "richconcat" combine (isti razlog kao
        # richconcat_features u common/features.py); za "hadamard" combine je vec simetrican
        # bez ovoga, ali sortiranje ovde ne menja hadamard rezultat (a*b==b*a), pa je bezbedno
        # primeniti uvek, jedna putanja koda za oba combine moda.
        slot1, slot2 = canonical_slots(rows_a, rows_b, ids_a, ids_b)
        y = np.array(labels, dtype=np.float32)

        idx = np.arange(len(y))
        idx_fit, idx_val = train_test_split(idx, test_size=p["val_fraction"], random_state=self.seed, stratify=y)

        self.model = ProjectedPairNet(embedding_matrix.shape[1], p["proj_dim"], p["pre_layernorm"],
                                        p["combine"], p["hidden_dims"], p["dropout"])

        a_t, b_t, y_t = torch.from_numpy(slot1), torch.from_numpy(slot2), torch.from_numpy(y)
        a_fit, b_fit, y_fit = a_t[idx_fit], b_t[idx_fit], y_t[idx_fit]
        a_val, b_val, y_val = a_t[idx_val], b_t[idx_val], y_t[idx_val]

        n_pos_fit, n_neg_fit = float(y_fit.sum()), float((y_fit == 0).sum())
        pos_weight = torch.tensor(n_neg_fit / max(n_pos_fit, 1.0), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])

        n_fit = len(idx_fit)
        batch_rng = np.random.default_rng(self.seed)
        best_val_auc, best_state, no_improve = -np.inf, None, 0
        from sklearn.metrics import roc_auc_score

        for epoch in range(1, p["max_epochs"] + 1):
            self.model.train()
            perm = batch_rng.permutation(n_fit)
            for start in range(0, n_fit, p["batch_size"]):
                batch_idx = perm[start:start + p["batch_size"]]
                optimizer.zero_grad()
                logits = self.model(a_fit[batch_idx], b_fit[batch_idx])
                bce_loss = criterion(logits, y_fit[batch_idx])
                if p["l2_lambda"] > 0:
                    l2_term = p["l2_lambda"] * sum((param ** 2).sum() for param in self.model.parameters())
                    loss = bce_loss + l2_term
                else:
                    loss = bce_loss
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(self.model(a_val, b_val)).numpy()
            val_auc = roc_auc_score(y_val.numpy(), val_probs)

            if val_auc > best_val_auc:
                best_val_auc, best_state, no_improve = val_auc, {k: v.clone() for k, v in self.model.state_dict().items()}, 0
            else:
                no_improve += 1
            if no_improve >= p["patience"]:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_val_auc = best_val_auc
        self.stopped_epoch = epoch

        # Projektuj CEO pool JEDNOM, kesiraj -- score_all onda samo combine+head
        # (jeftino, LOCO ovo zove mnogo puta).
        with torch.no_grad():
            self._projected_pool_cache = self.model.project(
                torch.from_numpy(embedding_matrix.astype(np.float32))).numpy()

    def score_all(self, query_id) -> np.ndarray:
        qi = self.id_to_index[query_id]
        proj = self._projected_pool_cache
        pa = torch.from_numpy(np.tile(proj[qi], (len(proj), 1)))
        pb = torch.from_numpy(proj)
        with torch.no_grad():
            logits = self.model.head(self.model.combine_projected(pa, pb)).squeeze(-1)
            return torch.sigmoid(logits).numpy()
