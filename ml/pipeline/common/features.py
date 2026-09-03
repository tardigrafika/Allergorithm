"""
Parni feature vektor za klasifikatore (RF, MLP, XGBoost) -- osnovna verzija
(abs_diff + cosine, 1281 dim) identicna svuda (random_forest_baseline.py,
mlp_baseline.py). Opciono prosirenje sa BLAST identity/score kolonama
(ml/random_forest_blast_1443.py) -- 1283 dim kad je ukljuceno.

hadamard_product(): simetrican elementwise produkt u*v (dim, ne 2*dim+1) --
dijagnostikovano (analysis/mlp_hadamard_input_1548.py) da MLP na ovom ulazu
dostize Hadamard bilinear performanse, za razliku od abs_diff ulaza koji je
znacajno losiji za MLP na ovom dataset-u/velicini. Koristi ga MLPPairClassifier
kad je input_encoding="hadamard" (videti models/classifiers/mlp.py).

Napomena: cist Hadamard bilinear klasifikator (bez skrivenog sloja) i dalje
NE koristi ovaj modul -- radi direktno na sirovim embeddinzima. Videti
models/classifiers/hadamard.py i cosine.py.
"""

import numpy as np


def l2_normalize_rows(emb):
    """L2-normalizuje SVAKI red (protein embedding) nezavisno -- 'standardizacija
    PRE pairwise operacije' (analysis/mlp_hadamard_esm2_3b_richpair_sensitivity_1548.py),
    RAZLICITO od MLPPairClassifier standardize=True (koji z-score standardizuje
    SVAKU KOLONU finalnog feature vektora PREKO CELOG dataset-a POSLE pairwise
    kombinovanja -- vec dijagnostikovano stetno za hadamard enkoding). Ovo je
    fiksna, ne-naucena operacija na SIROVOM embedding vektoru, primenjena
    nezavisno na svaki protein pre nego sto se uopste kombinuje sa parnjakom."""
    emb = np.atleast_2d(emb)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-12, None)


def canonical_slots(emb_a, emb_b, ids_a, ids_b):
    """Deterministicki poredak (po ID stringu) tako da RAW (ne-simetricno
    kombinovan) feature vektor i dalje garantuje score(A,B)==score(B,A) --
    slot dodela zavisi SAMO od NEUREDJENOG para {A,B}, ne od toga koji je
    prosledjen kao 'a' a koji kao 'b'. Potrebno za richconcat enkoding
    (eA,eB su SIROVI slotovi, ne simetricna kombinacija kao abs_diff/hadamard,
    pa bi bez ovoga isti par mogao dobiti razlicit skor zavisno od smera
    upita tokom retrieval-a)."""
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    swap = ids_a > ids_b
    slot1 = np.where(swap[:, None], emb_b, emb_a)
    slot2 = np.where(swap[:, None], emb_a, emb_b)
    return slot1, slot2


def richconcat_features(emb_a, emb_b, ids_a, ids_b, pre_l2_normalize=False):
    """[eA, eB, |eA-eB|, eA*eB] -- eA/eB u KANONICNOM (ID-sortiranom) poretku
    da se garantuje simetrija (videti canonical_slots). dim = 4*embedding_dim.
    float32 (embedding_matrix dolazi kao float64 iz load_dataset -- 4 kolone
    x float64 na 2560-dim ESM-2 3B ulazu je znacajno skuplje nego na hadamard-u
    (samo 1 kolona), pa se ovde EKSPLICITNO baca na float32 da se to ne
    kumulira -- pola memorije/racunanja, bez merljivog gubitka preciznosti za
    ovu svrhu)."""
    emb_a = np.atleast_2d(emb_a).astype(np.float32, copy=False)
    emb_b = np.atleast_2d(emb_b).astype(np.float32, copy=False)
    if pre_l2_normalize:
        emb_a = l2_normalize_rows(emb_a)
        emb_b = l2_normalize_rows(emb_b)
    slot1, slot2 = canonical_slots(emb_a, emb_b, ids_a, ids_b)
    return np.hstack([slot1, slot2, np.abs(slot1 - slot2), slot1 * slot2])


def build_richconcat_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index,
                              pre_l2_normalize=False):
    rows_a, rows_b, ids_a, ids_b, labels = [], [], [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        ids_a.append(p["id_1"])
        ids_b.append(p["id_2"])
        labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        ids_a.append(a)
        ids_b.append(b)
        labels.append(0)
    X = richconcat_features(np.array(rows_a), np.array(rows_b), ids_a, ids_b, pre_l2_normalize)
    y = np.array(labels)
    return X, y


def richconcat_batch_same_query(query_emb, query_id, candidate_embs, candidate_ids, pre_l2_normalize=False):
    query_batch = np.tile(query_emb, (len(candidate_ids), 1))
    query_ids_batch = np.array([query_id] * len(candidate_ids))
    return richconcat_features(query_batch, candidate_embs, query_ids_batch, np.asarray(candidate_ids),
                                 pre_l2_normalize)


def hadamard_product(emb_a, emb_b, pre_l2_normalize=False):
    """Simetrican elementwise produkt u*v -- isti oblik ulaza kao Hadamard
    bilinear (models/classifiers/hadamard.py), bez abs_diff/cosine kolona.
    pre_l2_normalize: opciono (default False, ne menja postojece ponasanje),
    videti l2_normalize_rows() -- 'standardizacija PRE pairwise operacije'."""
    emb_a = np.atleast_2d(emb_a)
    emb_b = np.atleast_2d(emb_b)
    if pre_l2_normalize:
        emb_a = l2_normalize_rows(emb_a)
        emb_b = l2_normalize_rows(emb_b)
    return emb_a * emb_b


def build_hadamard_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index, pre_l2_normalize=False):
    rows_a, rows_b, labels = [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        labels.append(0)
    X = hadamard_product(np.array(rows_a), np.array(rows_b), pre_l2_normalize)
    y = np.array(labels)
    return X, y


def hadamard_batch_same_query(query_emb, candidate_embs, pre_l2_normalize=False):
    query_batch = np.tile(query_emb, (len(candidate_embs), 1))
    return hadamard_product(query_batch, candidate_embs, pre_l2_normalize)


def pairwise_features(emb_a, emb_b, ids_a=None, ids_b=None, blast_matrices=None):
    """
    Simetrican parni feature vektor za protein par(ove) (A, B) -- oba dela
    (abs razlika i cosine) su invarijantni na zamenu A i B, pa model tretira
    (A,B) i (B,A) identicno (odgovara simetricnoj prirodi "cross-reactive"
    relacije).

    emb_a, emb_b: (N, dim) ili (dim,), poravnati po redu.
    blast_matrices: opciono, dict {"identity_matrix":, "score_matrix":,
        "id_to_index": } -- ako je dat, dodaje 2 dodatne kolone
        (blast_identity, blast_score); tada ids_a/ids_b MORAJU biti dati.

    Vraca (N, 1281) ili (N, 1283) ako je blast_matrices dat.
    """
    emb_a = np.atleast_2d(emb_a)
    emb_b = np.atleast_2d(emb_b)

    abs_diff = np.abs(emb_a - emb_b)

    dot = np.sum(emb_a * emb_b, axis=1)
    norm_a = np.linalg.norm(emb_a, axis=1)
    norm_b = np.linalg.norm(emb_b, axis=1)
    cosine = dot / (norm_a * norm_b + 1e-12)

    features = np.hstack([abs_diff, cosine.reshape(-1, 1)])

    if blast_matrices is not None:
        assert ids_a is not None and ids_b is not None, "blast_matrices zahteva ids_a/ids_b"
        id_to_index = blast_matrices["id_to_index"]
        identity_matrix = blast_matrices["identity_matrix"]
        score_matrix = blast_matrices["score_matrix"]
        blast_id = np.array([identity_matrix[id_to_index[a], id_to_index[b]] for a, b in zip(ids_a, ids_b)])
        blast_sc = np.array([score_matrix[id_to_index[a], id_to_index[b]] for a, b in zip(ids_a, ids_b)])
        features = np.hstack([features, blast_id.reshape(-1, 1), blast_sc.reshape(-1, 1)])

    return features


def pairwise_features_batch_same_query(query_emb, query_id, candidate_embs, candidate_ids, blast_matrices=None):
    """Brza putanja za retrieval: jedan fiksiran upit protiv mnogo kandidata."""
    query_batch = np.tile(query_emb, (len(candidate_ids), 1))
    query_ids_batch = [query_id] * len(candidate_ids)
    return pairwise_features(query_batch, candidate_embs, query_ids_batch, candidate_ids, blast_matrices)


def build_feature_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index, blast_matrices=None):
    """positive_pairs: lista dict-ova sa id_1/id_2. negative_pairs: lista (a,b) torki."""
    rows_a, rows_b, ids_a, ids_b, labels = [], [], [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        ids_a.append(p["id_1"])
        ids_b.append(p["id_2"])
        labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        ids_a.append(a)
        ids_b.append(b)
        labels.append(0)

    X = pairwise_features(np.array(rows_a), np.array(rows_b), ids_a, ids_b, blast_matrices)
    y = np.array(labels)
    return X, y


def load_blast_matrices(blast_pkl_path):
    """Ucitava BLAST identity/score matricu (data/compute_blast_identity_1443.py
    izlaz) u oblik koji pairwise_features ocekuje."""
    import pickle
    with open(blast_pkl_path, "rb") as f:
        blast_data = pickle.load(f)
    blast_ids = blast_data["ids"]
    return {
        "identity_matrix": blast_data["identity_matrix"],
        "score_matrix": blast_data["score_matrix"],
        "id_to_index": {aid: i for i, aid in enumerate(blast_ids)},
    }
