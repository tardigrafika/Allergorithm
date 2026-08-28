"""
Evaluacija -- identicna logika u SVIM ml/*.py skriptovima:
  A) klasifikacione metrike (precision/recall/f1/ROC-AUC/PR-AUC/confusion matrix)
     na held-out test parovima vs sempl-ovanim negativima
  B) retrieval evaluacija (Hits@K, MRR) -- svaki test protein poredi se sa
     SVIM kandidatima u pool-u (isto sto i cosine baseline, radi fer poredjenja)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TOP_K_DEFAULT = [1, 5, 10, 20]


def classification_metrics(y_test, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


def rank_of(scores: np.ndarray, self_index: int, target_index: int) -> int:
    """1-indeksiran rang target_index-a u scores (opadajuce), self iskljucen."""
    s = scores.copy()
    s[self_index] = -np.inf
    order = np.argsort(s)[::-1]
    return int(np.where(order == target_index)[0][0]) + 1


def retrieval_evaluate(test_pairs, classifier, embedding_matrix, id_to_index,
                        cosine_matrix=None, top_k=TOP_K_DEFAULT) -> pd.DataFrame:
    """
    Za svaki test par (oba pravca), skoruje SVE kandidate preko classifier-a
    (mora imati .score_all(query_id) -> np.ndarray poravnat sa embedding_matrix
    redosledom), racuna rang pravog cilja, i (opciono) uporedjuje sa cosine
    baseline-om na ISTOM upitu -- identicno svuda ("Cosine (same test)" kolona).
    """
    records = []
    for p in test_pairs:
        directions = [
            (p["id_1"], p["id_2"], p.get("name_1"), p.get("name_2"), p.get("family_1"), p.get("family_2")),
            (p["id_2"], p["id_1"], p.get("name_2"), p.get("name_1"), p.get("family_2"), p.get("family_1")),
        ]
        for query_id, target_id, query_name, target_name, family_q, family_t in directions:
            query_index = id_to_index[query_id]
            target_index = id_to_index[target_id]

            scores = classifier.score_all(query_id)
            rank = rank_of(scores, query_index, target_index)

            row = {
                "pair_id": p.get("pair_id"), "query_allergen": query_name, "target_allergen": target_name,
                "query_allergen_id": query_id, "target_allergen_id": target_id,
                "query_family": family_q, "target_family": family_t,
                "model_score": float(scores[target_index]), "model_rank": rank,
                "model_reciprocal_rank": 1.0 / rank,
            }
            for k in top_k:
                row[f"model_hits_at_{k}"] = int(rank <= k)

            if cosine_matrix is not None:
                cos_scores = cosine_matrix[query_index].copy()
                cos_rank = rank_of(cos_scores, query_index, target_index)
                row["cosine_rank"] = cos_rank
                row["cosine_reciprocal_rank"] = 1.0 / cos_rank
                for k in top_k:
                    row[f"cosine_hits_at_{k}"] = int(cos_rank <= k)

            records.append(row)

    return pd.DataFrame(records)


def summarize_retrieval(df: pd.DataFrame, top_k=TOP_K_DEFAULT) -> dict:
    summary = {"mrr": df["model_reciprocal_rank"].mean()}
    for k in top_k:
        summary[f"hits_at_{k}"] = df[f"model_hits_at_{k}"].mean()
    if "cosine_reciprocal_rank" in df.columns:
        summary["cosine_mrr"] = df["cosine_reciprocal_rank"].mean()
        for k in top_k:
            summary[f"cosine_hits_at_{k}"] = df[f"cosine_hits_at_{k}"].mean()
    return summary


def bootstrap_ci(df: pd.DataFrame, value_col: str, group_col: str = "pair_id",
                  n_resamples: int = 2000, seed: int = 42, baseline_col: str | None = None):
    """
    Bootstrap 95% CI (resempluje po group_col, npr. pair_id -- ne po pojedinacnom
    upitu, da se izbegne pseudoreplikacija). Ako je baseline_col dat, racuna CI
    na RAZLICI (value_col - baseline_col), inace samo na value_col.
    Isti standard kao sav LOCO/rank-fusion rad u sesiji.
    """
    rng = np.random.default_rng(seed)
    group_ids = df[group_col].unique()
    deltas = []
    for _ in range(n_resamples):
        sampled = rng.choice(group_ids, size=len(group_ids), replace=True)
        counts = pd.Series(sampled).value_counts()
        resampled = df.merge(counts.rename("w"), left_on=group_col, right_index=True)
        w = resampled["w"].to_numpy()
        if baseline_col is not None:
            d = np.average(resampled[value_col], weights=w) - np.average(resampled[baseline_col], weights=w)
        else:
            d = np.average(resampled[value_col], weights=w)
        deltas.append(d)
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_lo > 0) or (ci_hi < 0)
    return {"mean": deltas.mean(), "ci_lo": ci_lo, "ci_hi": ci_hi, "significant": significant}
