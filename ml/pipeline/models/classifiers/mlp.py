"""
MLP klasifikator -- preuzeto iz ml/mlp_baseline.py: arhitektura
1281->256->64->1 (ReLU, Dropout 0.3/0.2), BCEWithLogitsLoss sa pos_weight,
AdamW, standardizacija feature-a fit-ovana SAMO na train skupu, early
stopping preko validacionog ROC-AUC-a.

Opciono L2_LAMBDA (eksplicitan L2 clan U loss-u, ne AdamW weight_decay --
videti ml/hadamard_bilinear_1548.py komentar o razlici mehanizama) --
podrazumevano iskljuceno (0.0), da ostane verno originalnom mlp_baseline.py.

input_encoding="absdiff" (podrazumevano, verno originalu) ili "hadamard" --
dijagnostikovano (analysis/mlp_hadamard_input_1548.py, sesija avgust 2026)
da je abs_diff enkoding sam po sebi los izbor za MLP na ovom dataset-u/
velicini (SVIH 8 abs_diff konfiguracija znacajno gore od cosine-a u
analysis/mlp_sensitivity_1548.py), dok MLP na simetricnom Hadamard produktu
(u*v, isti ulaz kao Hadamard bilinear) dostize/blago prevazilazi cosine.
"hadamard" enkoding NEMA cosine kolonu niti BLAST prosirenje -- samo dim
(1280) ulaznih feature-a, ne dim*1+2/3 kao absdiff.

standardize=True (podrazumevano, verno originalu za absdiff) -- ali za
input_encoding="hadamard" OBAVEZNO standardize=False je bilo znacajno bolje
u testiranju (analysis/mlp_hadamard_pipeline_sensitivity_1548.py): z-score
standardizacija po dimenziji remeti prirodnu skalu Hadamard produkta na isti
nacin kao sto je L2-normalizacija ranije stetila Hadamard bilinear modelu
bez cosine_init-a (videti hadamard.py) -- skala produkta SAMA nosi signal.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from ...common.features import (
    build_feature_matrix,
    build_hadamard_matrix,
    build_richconcat_matrix,
    hadamard_batch_same_query,
    pairwise_features_batch_same_query,
    richconcat_batch_same_query,
)
from .base import PairClassifier

DEFAULT_PARAMS = dict(
    input_encoding="absdiff",   # "absdiff" (original) ili "hadamard" (dijagnostikovano bolje na ovom dataset-u)
    standardize=True,           # False preporuceno za input_encoding="hadamard" -- videti napomenu iznad
    hidden_dims=[256, 64],
    dropout=[0.3, 0.2],
    val_fraction=0.15,
    batch_size=64,
    max_epochs=200,
    patience=20,
    learning_rate=1e-3,
    weight_decay=1e-4,     # AdamW decoupled weight decay (podrazumevano, kao originalni skript)
    l2_lambda=0.0,          # eksplicitan L2 U loss-u, iskljuceno podrazumevano
    use_layernorm=False,    # LayerNorm na SKRIVENIM aktivacijama (Linear->LayerNorm->ReLU->Dropout),
                             # podrazumevano iskljuceno (verno originalu). NIJE isto sto i standardize=False
                             # za input_encoding="hadamard" -- ono je globalna z-score standardizacija
                             # SIROVIH ulaznih feature-a (dijagnostikovano da unistava signal), ovo je
                             # per-primer normalizacija SKRIVENIH aktivacija posle prvog linearnog sloja,
                             # razlicit mehanizam (analysis/mlp_layernorm_ablation_1548.py, avgust 2026).
)


class PairMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, use_layernorm=False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h, d in zip(hidden_dims, dropout):
            layers.append(nn.Linear(prev_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers += [nn.ReLU(), nn.Dropout(d)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPPairClassifier(PairClassifier):
    def __init__(self, params: dict | None = None, extra_features: list | None = None,
                 blast_matrix_path: str | None = None, seed: int = 42, **kwargs):
        super().__init__()
        params = dict(params or {})
        if params.get("input_encoding") == "hadamard" and "standardize" not in params:
            params["standardize"] = False  # dijagnostikovano bolje -- videti napomenu iznad DEFAULT_PARAMS
        self.params = {**DEFAULT_PARAMS, **params}
        self.seed = seed
        self.extra_features = extra_features or []
        self.blast_matrices = None
        if "blast_identity" in self.extra_features or "blast_score" in self.extra_features:
            assert self.params["input_encoding"] != "hadamard", \
                "extra_features (BLAST) nije podrzano sa input_encoding='hadamard' -- build_hadamard_matrix ne prima blast_matrices"
            from ...common.features import load_blast_matrices
            assert blast_matrix_path
            self.blast_matrices = load_blast_matrices(blast_matrix_path)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = None
        self.feature_mean = None
        self.feature_std = None

    def _scale(self, X):
        if not self.params["standardize"]:
            return X
        return (X - self.feature_mean) / self.feature_std

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index,
            positive_weights: np.ndarray | None = None, **kwargs):
        """positive_weights: opcioni niz iste duzine kao positive_pairs, per-primer
        tezina u loss-u (npr. manja tezina za slabije-pouzdane evidence_level tier-ove
        -- test/evaluate_weighted_evidence_mlp_patients_1548.py, 2026-09-02). Negativni
        parovi UVEK tezina 1.0 (nasumicno uzorkovani, nemaju evidence_level koncept).
        Podrazumevano None -> svi 1.0, matematicki IDENTICNO ranijem ne-tezinskom
        BCE-u (mean() tezinskog gubitka sa svim tezinama=1 jednak je obicnom mean()),
        pa ne menja ponasanje nijednog postojeceg pozivaoca."""
        self.set_pool(embedding_matrix, id_to_index)
        p = self.params

        if p["input_encoding"] == "hadamard":
            X, y = build_hadamard_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index,
                                           pre_l2_normalize=p.get("pre_l2_normalize", False))
        elif p["input_encoding"] == "absdiff":
            X, y = build_feature_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index,
                                          blast_matrices=self.blast_matrices)
        elif p["input_encoding"] == "richconcat":
            X, y = build_richconcat_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index,
                                             pre_l2_normalize=p.get("pre_l2_normalize", False))
        else:
            raise ValueError(f"Nepoznat input_encoding '{p['input_encoding']}' "
                              f"(ocekivano 'absdiff', 'hadamard' ili 'richconcat')")
        y = y.astype(np.float32)

        if positive_weights is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        else:
            positive_weights = np.asarray(positive_weights, dtype=np.float32)
            assert len(positive_weights) == len(positive_pairs), \
                f"positive_weights duzine {len(positive_weights)} != {len(positive_pairs)} positive_pairs"
            sample_weight = np.concatenate([positive_weights, np.ones(len(negative_pairs), dtype=np.float32)])

        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        X_scaled = self._scale(X).astype(np.float32)

        X_fit, X_val, y_fit, y_val, w_fit, w_val = train_test_split(
            X_scaled, y, sample_weight, test_size=p["val_fraction"], random_state=self.seed, stratify=y)

        self.model = PairMLP(X.shape[1], p["hidden_dims"], p["dropout"], use_layernorm=p["use_layernorm"])

        n_pos_fit, n_neg_fit = float(y_fit.sum()), float((y_fit == 0).sum())
        pos_weight = torch.tensor(n_neg_fit / n_pos_fit, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=p["learning_rate"],
                                        weight_decay=p["weight_decay"])

        X_fit_t, y_fit_t, w_fit_t = torch.from_numpy(X_fit), torch.from_numpy(y_fit), torch.from_numpy(w_fit)
        X_val_t, y_val_t, w_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(w_val)

        n_fit = X_fit_t.shape[0]
        batch_rng = np.random.default_rng(self.seed)

        best_val_auc, best_state, no_improve = -np.inf, None, 0
        from sklearn.metrics import roc_auc_score

        # Istorija po epohi (train/val loss + val AUC) -- UVEK se beleze (jeftino,
        # samo append float-ova), da se moze VIZUELNO proveriti overfitting umesto
        # verovati samo finalnom best_val_auc broju. Ranije se ovo NIGDE nije cuvalo
        # (samo najbolja vrednost, ostatak trajektorije se gubio) -- korisnicki
        # zahtev, 2026-09-01: "kako da znam da nije overfitovao... kako izgleda
        # train i val loss".
        self.history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}

        for epoch in range(1, p["max_epochs"] + 1):
            self.model.train()
            permutation = batch_rng.permutation(n_fit)
            batch_losses = []
            for start in range(0, n_fit, p["batch_size"]):
                batch_idx = permutation[start:start + p["batch_size"]]
                optimizer.zero_grad()
                logits = self.model(X_fit_t[batch_idx])
                bce_loss = (criterion(logits, y_fit_t[batch_idx]) * w_fit_t[batch_idx]).mean()
                if p["l2_lambda"] > 0:
                    l2_term = p["l2_lambda"] * sum((param ** 2).sum() for param in self.model.parameters())
                    loss = bce_loss + l2_term
                else:
                    loss = bce_loss
                loss.backward()
                optimizer.step()
                batch_losses.append(bce_loss.item())

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = (criterion(val_logits, y_val_t) * w_val_t).mean().item()
                val_probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(y_val, val_probs)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(float(np.mean(batch_losses)))
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= p["patience"]:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_val_auc = best_val_auc
        self.stopped_epoch = epoch

    def score_all(self, query_id) -> np.ndarray:
        query_vec = self.embedding_matrix[self.id_to_index[query_id]]
        p = self.params
        if p["input_encoding"] == "hadamard":
            X_candidates = hadamard_batch_same_query(query_vec, self.embedding_matrix,
                                                        pre_l2_normalize=p.get("pre_l2_normalize", False))
        elif p["input_encoding"] == "richconcat":
            X_candidates = richconcat_batch_same_query(query_vec, query_id, self.embedding_matrix, self.all_ids,
                                                          pre_l2_normalize=p.get("pre_l2_normalize", False))
        else:
            X_candidates = pairwise_features_batch_same_query(
                query_vec, query_id, self.embedding_matrix, self.all_ids, blast_matrices=self.blast_matrices)
        X_scaled = self._scale(X_candidates).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_scaled))
            return torch.sigmoid(logits).numpy()
