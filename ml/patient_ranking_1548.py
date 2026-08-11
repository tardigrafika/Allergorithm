"""
Osnovna funkcija za "pacijent" upotrebu: ulaz = lista alergena na koje je
pacijent VEC potvrdjeno alergican (1 ili vise), izlaz = rangirana lista
SVIH ostalih proteina po prioritetu za dalje testiranje.

Metod: svaki poznati alergen tretira se kao posebna RRF-3 "upit" (cosine+
BLAST+FoldseekTM), pa se ti rangovi fuzionisu RECIPROCAL RANK FUSION-om
PREKO poznatih alergena -- ista RRF logika kao svuda u sesiji, samo
primenjena na "koliko poznatih" umesto na "koliko signala". Kad je poznat
samo 1 alergen, ovo je identicno cistom RRF-3 (validiranom rezultatu). Kad
ih je vise, ovo je isti mehanizam kao dokazani graph-propagation dobitak
(ml/graph_propagation_signal_1548.py) -- samo sto ovde "komsije" dolazi
DIREKTNO od pacijenta, ne iz gold grafa.

VAZNO -- ovo je molekularni cross-reactivity signal, NE klinicka predikcija
za pojedinacnog pacijenta (vidi real_world_case_validation_1548.py, Jug r 1
slucaj: visoka molekularna slicnost ne garantuje klinicku reaktivnost kod
svakog pacijenta).

Namerno BEZ CLI/UI poliranja -- samo funkcija + minimalan runnable primer.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_ALLERGENS = Path("/home/lana/ALERGRAF/output/clean_allergens.csv")
FROZEN_EMBEDDINGS = Path("/home/lana/ALERGRAF/embeddings/embeddings.pkl")
BLAST_MATRIX = Path("/home/lana/ALERGRAF/output/blast_identity_matrix_1443.pkl")
FOLDSEEK_LOOKUP = Path("/home/lana/ALERGRAF/output/foldseek_tmscore_lookup_1548.pkl")

RRF_K = 60

DISCLAIMER = (
    "This score indicates molecular evidence relevant to cross-reactivity. "
    "It does not predict whether an individual patient will experience a clinical reaction."
)


class CrossReactivityRanker:
    """Ucitava sve matrice jednom; poziv rank_for_patient() je jeftin."""

    def __init__(self):
        allergens = pd.read_csv(CLEAN_ALLERGENS)
        self.name_to_id = {}
        for row in allergens.itertuples(index=False):
            n = str(row.official_name).strip()
            if n and n not in self.name_to_id:
                self.name_to_id[n] = row.allergen_id
        self.id_to_name = {v: k for k, v in self.name_to_id.items()}

        with open(FROZEN_EMBEDDINGS, "rb") as f:
            embeddings_dict = pickle.load(f)
        self.pool = sorted(embeddings_dict.keys())
        self.id_to_index = {aid: i for i, aid in enumerate(self.pool)}
        n_pool = len(self.pool)

        embedding_matrix = np.array([embeddings_dict[aid] for aid in self.pool], dtype=np.float64)
        self.cosine_matrix = cosine_similarity(embedding_matrix)

        with open(BLAST_MATRIX, "rb") as f:
            blast_data = pickle.load(f)
        blast_ids = blast_data["ids"]
        blast_score_matrix = blast_data["score_matrix"]
        blast_id_to_index = {aid: i for i, aid in enumerate(blast_ids)}
        perm = np.array([blast_id_to_index.get(aid, -1) for aid in self.pool])
        valid_idx = np.where(perm >= 0)[0]
        self.blast_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
        self.blast_matrix[np.ix_(valid_idx, valid_idx)] = blast_score_matrix[np.ix_(perm[valid_idx], perm[valid_idx])]

        with open(FOLDSEEK_LOOKUP, "rb") as f:
            foldseek_lookup = pickle.load(f)
        self.foldseek_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
        for key, score in foldseek_lookup.items():
            if len(key) != 2:
                continue
            a, b = tuple(key)
            if a in self.id_to_index and b in self.id_to_index:
                i, j = self.id_to_index[a], self.id_to_index[b]
                self.foldseek_matrix[i, j] = score
                self.foldseek_matrix[j, i] = score

        self.n_pool = n_pool

    def _rrf3_score_vector(self, known_idx):
        def ranks_from_scores(scores, self_index):
            s = scores.astype(np.float64, copy=True)
            s[self_index] = -np.inf
            order = np.argsort(s)[::-1]
            ranks = np.empty(len(s), dtype=np.int64)
            ranks[order] = np.arange(1, len(s) + 1)
            return ranks

        cr = ranks_from_scores(self.cosine_matrix[known_idx], known_idx)
        br = ranks_from_scores(self.blast_matrix[known_idx], known_idx)
        fr = ranks_from_scores(self.foldseek_matrix[known_idx], known_idx)
        return 1.0 / (RRF_K + cr) + 1.0 / (RRF_K + br) + 1.0 / (RRF_K + fr)

    def rank_for_patient(self, known_positive_names: list[str]) -> pd.DataFrame:
        """known_positive_names: lista official_name vrednosti (WHO/IUIS oznake), 1 ili vise.
        Vraca DataFrame sortiran po prioritetu (najverovatniji cross-reactive prvi),
        iskljucujuci poznate alergene same iz kandidata."""
        known_ids = []
        for name in known_positive_names:
            aid = self.name_to_id.get(name)
            if aid is None or aid not in self.id_to_index:
                print(f"  [upozorenje] '{name}' nije nadjen u pool-u, preskacem")
                continue
            known_ids.append(aid)
        if not known_ids:
            raise ValueError("Nijedan poznati alergen nije nadjen u pool-u")

        known_idx_set = {self.id_to_index[aid] for aid in known_ids}
        combined = np.zeros(self.n_pool, dtype=np.float64)
        for aid in known_ids:
            idx = self.id_to_index[aid]
            score_vec = self._rrf3_score_vector(idx)
            order = np.argsort(score_vec)[::-1]
            ranks = np.empty(self.n_pool, dtype=np.int64)
            ranks[order] = np.arange(1, self.n_pool + 1)
            combined += 1.0 / (RRF_K + ranks)

        for idx in known_idx_set:
            combined[idx] = -np.inf  # ne predlazi ono sto je vec poznato

        order = np.argsort(combined)[::-1]
        result = pd.DataFrame({
            "candidate_id": [self.pool[i] for i in order],
            "candidate_name": [self.id_to_name.get(self.pool[i], self.pool[i]) for i in order],
            "priority_score": combined[order],
        })
        result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result


if __name__ == "__main__":
    print(DISCLAIMER)
    print()
    ranker = CrossReactivityRanker()

    print("--- Primer: samo Ara h 2 poznat (isto sto i cist RRF-3) ---")
    r1 = ranker.rank_for_patient(["Ara h 2.0101"])
    print(r1.head(10).to_string(index=False))

    print("\n--- Primer: Ara h 2 I Ana o 1 poznati (patient-provided propagation) ---")
    r2 = ranker.rank_for_patient(["Ara h 2.0101", "Ana o 1.0101"])
    print(r2.head(10).to_string(index=False))

    pisv_rank_1 = r1[r1["candidate_name"].str.startswith("Pis v")].iloc[0]
    pisv_rank_2 = r2[r2["candidate_name"].str.startswith("Pis v")].iloc[0]
    print(f"\nPis v 1 rang sa 1 poznatim alergenom: {pisv_rank_1['rank']} ({pisv_rank_1['candidate_name']})")
    print(f"Pis v 1 rang sa 2 poznata alergena:    {pisv_rank_2['rank']} ({pisv_rank_2['candidate_name']})")
