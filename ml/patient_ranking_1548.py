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

RRF_K = 20  # ml/rrf_k_sensitivity_1548.py nalaz (bolje od K=60, potvrdjeno split-half transferom)

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

        # pravi % sequence identity (0-100), odvojeno od blast_matrix iznad (koji je
        # alignment SCORE koriscen za RRF rangiranje, ne procenat) -- treba nam za
        # graduated_introduction_path, gde "50%" mora biti stvarni procenat identiteta
        blast_identity_pct = blast_data["identity_matrix"]
        self.identity_matrix = np.zeros((n_pool, n_pool), dtype=np.float32)
        self.identity_matrix[np.ix_(valid_idx, valid_idx)] = blast_identity_pct[np.ix_(perm[valid_idx], perm[valid_idx])]

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

    def rank_for_patient(self, known_positive_names: list[str],
                          known_negative_names: list[str] | None = None,
                          apply_negative_signal: bool = False) -> pd.DataFrame:
        """known_positive_names: alergeni na koje je pacijent VEC potvrdjeno
            pozitivan, 1 ili vise.
        known_negative_names: alergeni koji su VEC testirani i ispali
            NEGATIVNI (npr. testiran, tolerise) -- opciono. Uvek se iskljucuju
            iz predloga (vec testirano, ne treba ih ponovo predlagati).
        apply_negative_signal: EKSPERIMENTALNO, PODRAZUMEVANO ISKLJUCENO --
            test-suite (analysis/patient_ranking_test_suite_1548.py) je
            pokazao da ovo AKTIVNO KVARI dobre pozitivne predikcije (npr.
            Limao slucaj: Ana o 2 rang pao sa 274 na 1486/1532 kad se Jug r 1
            doda kao negativan, jer je Jug r 1 strukturno slican i pravim
            pozitivima iz iste storage-protein familije). Ne ukljucuj dok se
            ne nadje bolji mehanizam (npr. negativna propagacija ogranicena
            samo na VRLO bliske kandidate, ne na ceo RRF spektar).

        Vraca DataFrame sortiran po prioritetu (najverovatniji cross-reactive
        prvi), iskljucujuci sve vec poznate (pozitivne i negativne) alergene."""
        def resolve(names):
            ids = []
            for name in names or []:
                aid = self.name_to_id.get(name)
                if aid is None or aid not in self.id_to_index:
                    print(f"  [upozorenje] '{name}' nije nadjen u pool-u, preskacem")
                    continue
                ids.append(aid)
            return ids

        positive_ids = resolve(known_positive_names)
        negative_ids = resolve(known_negative_names)
        if not positive_ids:
            raise ValueError("Nijedan poznati pozitivan alergen nije nadjen u pool-u")

        exclude_idx = {self.id_to_index[aid] for aid in positive_ids + negative_ids}
        combined = np.zeros(self.n_pool, dtype=np.float64)

        for aid in positive_ids:
            idx = self.id_to_index[aid]
            score_vec = self._rrf3_score_vector(idx)
            order = np.argsort(score_vec)[::-1]
            ranks = np.empty(self.n_pool, dtype=np.int64)
            ranks[order] = np.arange(1, self.n_pool + 1)
            combined += 1.0 / (RRF_K + ranks)

        if apply_negative_signal:
            for aid in negative_ids:
                idx = self.id_to_index[aid]
                score_vec = self._rrf3_score_vector(idx)
                order = np.argsort(score_vec)[::-1]
                ranks = np.empty(self.n_pool, dtype=np.int64)
                ranks[order] = np.arange(1, self.n_pool + 1)
                combined -= 1.0 / (RRF_K + ranks)

        for idx in exclude_idx:
            combined[idx] = -np.inf  # ne predlazi ono sto je vec testirano (pozitivno ili negativno)

        order = np.argsort(combined)[::-1]
        result = pd.DataFrame({
            "candidate_id": [self.pool[i] for i in order],
            "candidate_name": [self.id_to_name.get(self.pool[i], self.pool[i]) for i in order],
            "priority_score": combined[order],
        })
        result = result[np.isfinite(result["priority_score"])].reset_index(drop=True)
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        return result

    # severity (1=najblaza reakcija, 5=najteza/anafilaksa) -> broj koraka puta.
    # Namerno JEDNOSTAVNO, LAKO PODESIVO mapiranje -- ne referenciramo tudju
    # gotovu skalu kao obavezujucu, ovo je nas prvi predlog za podesavanje.
    # Vise koraka = sitniji, opredzniji pomaci za tezu reakciju.
    SEVERITY_TO_STEPS = {1: 3, 2: 4, 3: 5, 4: 7, 5: 9}

    def graduated_introduction_path(self, allergen_name: str,
                                      severity: int | None = None,
                                      n_steps: int | None = None,
                                      start_percentile: float = 50.0,
                                      other_known_positives: list[str] | None = None) -> pd.DataFrame:
        """EKSPERIMENTALNO -- predlog za razmatranje od strane alergologa, NIJE gotov
        protokol i NIJE za primenu bez klinickog nadzora.

        Koristi RRF signal (cosine+BLAST+FoldseekTM, + graph propagation ako je
        prosledjen other_known_positives -- isti mehanizam kao rank_for_patient)
        kao meru srodnosti, NE sirov BLAST % identity -- ovaj je probni pristup
        pokazao da kratka/slaba poravnanja mogu slucajno dati visok % bez prave
        biološke srodnosti (npr. Ara h 2 vs Bos d 13 = 50% identity ali skor
        poravnanja svega 27, dok pravi homolog Pis v 1 ima samo 29% identity ali
        skor 145 -- dug, jak, biološki stvaran hit). RRF vec kombinuje vise
        signala i izbegava ovaj artefakt.

        allergen_name: potvrdjen alergen na koji pacijent reaguje (official_name).
        severity: 1-5, ozbiljnost PRETHODNE reakcije pacijenta na allergen_name
            (1=blaga, 5=anafilaksa) -- ODREDJUJE ALERGOLOG, ne izvodi se iz
            nasih podataka. Mapira se na broj koraka preko SEVERITY_TO_STEPS
            (podesivo, nije fiksirano na tudju skalu). Mora se dati JEDNO od
            severity ili n_steps.
        n_steps: direktan broj koraka, zaobilazi severity mapiranje ako je dat.
        start_percentile: pocetna tacka puta kao percentil RRF ranga u punom
            pool-u (podrazumevano 50 = srednje srodan protein, ne najsrodniji
            ni najmanje srodan).
        other_known_positives: opciono, drugi vec potvrdjeni alergeni istog
            pacijenta -- koriste se za graph-propagation obogacivanje signala,
            isto kao u rank_for_patient.

        Vraca korak-po-korak DataFrame: od proteina na ~start_percentile
        percentilu RRF ranga do samog poznatog alergena (poslednji korak) --
        realni proteini iz baze, NE simulirani/izmisljeni medjukoraci.
        """
        if n_steps is None:
            if severity is None:
                raise ValueError("Mora se dati severity (1-5) ili direktno n_steps")
            if severity not in self.SEVERITY_TO_STEPS:
                raise ValueError(f"severity mora biti 1-5, dobijeno {severity}")
            n_steps = self.SEVERITY_TO_STEPS[severity]
        if n_steps < 1:
            raise ValueError("n_steps mora biti >= 1")
        if not (0 <= start_percentile <= 100):
            raise ValueError("start_percentile mora biti izmedju 0 i 100")

        aid = self.name_to_id.get(allergen_name)
        if aid is None or aid not in self.id_to_index:
            raise ValueError(f"'{allergen_name}' nije nadjen u pool-u")
        idx = self.id_to_index[aid]

        combined = self._rrf3_score_vector(idx)
        exclude_idx = {idx}
        for name in (other_known_positives or []):
            other_aid = self.name_to_id.get(name)
            if other_aid is not None and other_aid in self.id_to_index:
                other_idx = self.id_to_index[other_aid]
                combined = combined + self._rrf3_score_vector(other_idx)
                exclude_idx.add(other_idx)

        for i in exclude_idx:
            combined[i] = -np.inf

        # rank 0 = najsrodniji (najvisi RRF skor) ... rank n-1 = najmanje srodan
        order_by_similarity_desc = np.argsort(combined)[::-1]
        valid = order_by_similarity_desc[np.isfinite(combined[order_by_similarity_desc])]
        n_valid = len(valid)

        # start_percentile=50 -> pocni od sredine ranga (umereno srodan protein)
        start_rank_pos = int(round((1 - start_percentile / 100.0) * (n_valid - 1)))
        path_ranks = valid[:start_rank_pos + 1]  # od pocetne pozicije do najsrodnijeg (rank 0)

        n_intermediate_needed = n_steps - 1
        if len(path_ranks) < n_intermediate_needed:
            print(f"  [upozorenje] samo {len(path_ranks)} realnih kandidata izmedju "
                  f"{start_percentile}. percentila i najsrodnijeg -- trazeno "
                  f"{n_intermediate_needed} medjukoraka. Ne izmisljam dodatne, "
                  f"vracam koliko stvarno postoji.")
            n_intermediate_needed = len(path_ranks)

        if n_intermediate_needed > 0:
            # rasporedi ravnomerno OD najmanje srodnog (pocetak puta) KA najsrodnijem
            reversed_path = path_ranks[::-1]  # sad je [0]=najmanje srodan (start), [-1]=najsrodniji
            positions = np.linspace(0, len(reversed_path) - 1, n_intermediate_needed).round().astype(int)
            positions = np.unique(positions)
            chosen = reversed_path[positions]
        else:
            chosen = np.array([], dtype=int)

        rows = []
        for step, cidx in enumerate(chosen, start=1):
            rows.append({
                "step": step,
                "candidate_id": self.pool[cidx],
                "candidate_name": self.id_to_name.get(self.pool[cidx], self.pool[cidx]),
                "rrf_score": round(float(combined[cidx]), 5),
                "is_confirmed_allergen": False,
            })
        rows.append({
            "step": len(rows) + 1,
            "candidate_id": aid,
            "candidate_name": allergen_name,
            "rrf_score": None,
            "is_confirmed_allergen": True,
        })
        return pd.DataFrame(rows)


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

    print("\n--- Primer: Ara h 2 poznat POZITIVAN, Jug r 1 poznat NEGATIVAN ---")
    r3 = ranker.rank_for_patient(["Ara h 2.0101"], known_negative_names=["Jug r 1.0101"])
    print(r3.head(10).to_string(index=False))
    jugr_in_r3 = r3[r3["candidate_name"] == "Jug r 1.0101"]
    print(f"Jug r 1 u listi (treba da NE bude, vec je testiran): "
          f"{'nema, ispravno iskljucen' if jugr_in_r3.empty else 'GRESKA, prisutan'}")

    print("\n--- EKSPERIMENTALNO: graduirani put uvodjenja ka Ara h 2, severity=3 (RRF signal) ---")
    print("(predlog za razmatranje od strane alergologa, nije gotov protokol)")
    path = ranker.graduated_introduction_path("Ara h 2.0101", severity=3, start_percentile=50.0)
    print(path.to_string(index=False))

    print("\n--- Isto, severity=5 (teza reakcija -> vise, sitnijih koraka) ---")
    path5 = ranker.graduated_introduction_path("Ara h 2.0101", severity=5, start_percentile=50.0)
    print(path5.to_string(index=False))
