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


def hadamard_product(emb_a, emb_b):
    """Simetrican elementwise produkt u*v -- isti oblik ulaza kao Hadamard
    bilinear (models/classifiers/hadamard.py), bez abs_diff/cosine kolona."""
    emb_a = np.atleast_2d(emb_a)
    emb_b = np.atleast_2d(emb_b)
    return emb_a * emb_b


def build_hadamard_matrix(positive_pairs, negative_pairs, embedding_matrix, id_to_index):
    rows_a, rows_b, labels = [], [], []
    for p in positive_pairs:
        rows_a.append(embedding_matrix[id_to_index[p["id_1"]]])
        rows_b.append(embedding_matrix[id_to_index[p["id_2"]]])
        labels.append(1)
    for a, b in negative_pairs:
        rows_a.append(embedding_matrix[id_to_index[a]])
        rows_b.append(embedding_matrix[id_to_index[b]])
        labels.append(0)
    X = hadamard_product(np.array(rows_a), np.array(rows_b))
    y = np.array(labels)
    return X, y


def hadamard_batch_same_query(query_emb, candidate_embs):
    query_batch = np.tile(query_emb, (len(candidate_embs), 1))
    return hadamard_product(query_batch, candidate_embs)


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
