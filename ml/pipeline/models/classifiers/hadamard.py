"""
Hadamard bilinear -- preuzeto iz ml/hadamard_bilinear_1548.py: y = sigmoid(w
. (u*v) + b), gde je u*v Hadamard (elementwise) produkt embeddinga (vec
simetrican, ne treba abs-diff trik). Samo ~1280 parametara -- mnogo manje od
MLP-a, predlog mentora kao model bolje uskladjen sa velicinom podataka.

l2_lambda: eksplicitan L2 clan U loss-u (ne AdamW weight_decay -- razlicit
mehanizam, videti mentorovo pitanje/odgovor u istoriji sesije).

normalize="l2": L2-normalizuje svaki embedding (jedinicni vektor) PRE
Hadamard produkta -- dijagnostikovano da sirovi ESM embeddinzi imaju jako
razlicite skale po dimenziji (uzrok SGD nestabilnosti), ovo popravlja
uslovljenost optimizacionog problema za SVE optimizatore, ne samo SGD.

cosine_init=True (zahteva normalize="l2"): inicijalizuje w=1 (uniformno),
bias=0 -- sa jedinicnim vektorima w.(u*v)=cosine(u,v) TACNO, pa model
POCINJE sa identicnim rangiranjem kao cist cosine baseline i uci ODSTUPANJA/
poboljsanja od te tacke, umesto da krece od nasumicne inicijalizacije.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import PairClassifier

DEFAULT_PARAMS = dict(
    optimizer="adamw",     # "adamw" ili "sgd" -- mentorov zahtev: probati SGD umesto Adam-a
    max_epochs=300,
    patience=15,
    learning_rate=1e-2,
    momentum=0.9,           # samo za optimizer="sgd" (standardan SGD+momentum, ignorise se za adamw)
    weight_decay=0.0,   # AdamW decoupled weight decay -- podrazumevano iskljuceno
    l2_lambda=1e-3,       # eksplicitan L2 U loss-u (Ridge-stil)
    val_fraction=0.15,
    normalize="none",       # "none" ili "l2"
    cosine_init=False,       # zahteva normalize="l2"
)


def l2_normalize_rows(mat: torch.Tensor) -> torch.Tensor:
    return mat / (mat.norm(dim=-1, keepdim=True) + 1e-12)


def build_optimizer(params_iterable, p: dict):
    if p["optimizer"] == "sgd":
        return torch.optim.SGD(params_iterable, lr=p["learning_rate"], momentum=p["momentum"],
                                 weight_decay=p["weight_decay"])
    elif p["optimizer"] == "adamw":
        return torch.optim.AdamW(params_iterable, lr=p["learning_rate"], weight_decay=p["weight_decay"])
    else:
        raise ValueError(f"Nepoznat optimizer '{p['optimizer']}' (ocekivano 'adamw' ili 'sgd')")


class HadamardBilinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, u, v):
        return self.w(u * v).squeeze(-1)


class HadamardBilinearClassifier(PairClassifier):
    def __init__(self, params: dict | None = None, seed: int = 42, **kwargs):
        super().__init__()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.seed = seed
        torch.manual_seed(seed)
        self.model = None

    def fit(self, positive_pairs, negative_pairs, embedding_matrix, id_to_index, **kwargs):
        self.set_pool(embedding_matrix, id_to_index)
        p = self.params
        embedding_matrix_f32 = embedding_matrix.astype(np.float32)

        rows_u, rows_v, labels = [], [], []
        for pos in positive_pairs:
            rows_u.append(embedding_matrix_f32[id_to_index[pos["id_1"]]])
            rows_v.append(embedding_matrix_f32[id_to_index[pos["id_2"]]])
            labels.append(1.0)
        for a, b in negative_pairs:
            rows_u.append(embedding_matrix_f32[id_to_index[a]])
            rows_v.append(embedding_matrix_f32[id_to_index[b]])
            labels.append(0.0)

        U = torch.from_numpy(np.array(rows_u, dtype=np.float32))
        V = torch.from_numpy(np.array(rows_v, dtype=np.float32))
        y = torch.tensor(labels, dtype=torch.float32)

        if p["normalize"] == "l2":
            U = l2_normalize_rows(U)
            V = l2_normalize_rows(V)
        elif p["normalize"] != "none":
            raise ValueError(f"Nepoznat normalize '{p['normalize']}' (ocekivano 'none' ili 'l2')")

        n = len(y)
        val_size = max(10, int(p["val_fraction"] * n))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        val_idx, fit_idx = perm[:val_size], perm[val_size:]

        self.model = HadamardBilinear(embedding_matrix.shape[1])
        if p["cosine_init"]:
            assert p["normalize"] == "l2", "cosine_init zahteva normalize='l2' (inace w=1 ne odgovara cosine-u)"
            with torch.no_grad():
                self.model.w.weight.fill_(1.0)
                self.model.w.bias.fill_(0.0)
        pos_weight = torch.tensor((y[fit_idx] == 0).sum().item() / max((y[fit_idx] == 1).sum().item(), 1))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = build_optimizer(self.model.parameters(), p)

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
        self.best_val_loss = best_val_loss
        self.stopped_epoch = epoch
        embedding_tensor = torch.from_numpy(embedding_matrix_f32)
        if p["normalize"] == "l2":
            embedding_tensor = l2_normalize_rows(embedding_tensor)
        self._embedding_tensor = embedding_tensor

    def score_all(self, query_id) -> np.ndarray:
        qidx = self.id_to_index[query_id]
        q_vec = self._embedding_tensor[qidx].unsqueeze(0).expand(self._embedding_tensor.shape[0], -1)
        with torch.no_grad():
            return self.model(q_vec, self._embedding_tensor).numpy()
