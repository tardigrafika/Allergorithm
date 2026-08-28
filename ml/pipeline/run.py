"""
Univerzalni pipeline: pokreni BILO KOJI klasifikator (cosine/RF/MLP/XGBoost/
Hadamard) preko istog CLI-a i iste (group-aware split, negative sampling,
retrieval evaluacija) logike -- umesto ~15 zasebnih, skoro identicnih
skriptova.

Model + njegovi specificni hiperparametri se navode u JSON konfigu
(configs/*.json). Parametri koji su UNIVERZALNI (isti za sve modele --
dataset putanje, seed, test_fraction, neg_per_pos) su CLI argumenti.

Primer:
    python -m ml.pipeline.run --model-config configs/random_forest_blast.json \\
        --gold output/cross_reactive_1443.csv --output-dir output/pipeline_test

Ne menja nijedan postojeci skript -- ovo je NOVA, paralelna putanja.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .common.data import load_dataset
from .common.evaluation import bootstrap_ci, classification_metrics, retrieval_evaluate, summarize_retrieval
from .common.features import build_feature_matrix
from .common.negatives import sample_negative_pairs
from .common.splitting import group_aware_split, split_pairs
from .registry import build_classifier

DEFAULT_EMBEDDINGS = "/home/lana/ALERGRAF/embeddings/embeddings.pkl"
DEFAULT_METADATA = "/home/lana/ALERGRAF/embeddings/embeddings.parquet"


def parse_args():
    p = argparse.ArgumentParser(description="ALERGRAF univerzalni model pipeline")
    p.add_argument("--model-config", required=True, help="Putanja do JSON konfiga modela (configs/*.json)")
    p.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    p.add_argument("--metadata", default=DEFAULT_METADATA)
    p.add_argument("--gold", required=True, help="Putanja do gold-standard CSV-a")
    p.add_argument("--blast-matrix", default=None, help="Putanja do BLAST identity/score .pkl (ako model treba)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--neg-per-pos", type=int, default=10)
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--bootstrap-resamples", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model_config) as f:
        model_config = json.load(f)
    model_name = model_config.get("name", Path(args.model_config).stem)

    print(f"\n{'='*60}\nMODEL: {model_name} ({model_config['type']})\n{'='*60}")

    print("\nLoading dataset...")
    dataset = load_dataset(Path(args.embeddings), Path(args.metadata), Path(args.gold))
    print(f"Gold pairs: {len(dataset.gold_pairs)}  |  Candidate pool: {len(dataset.all_ids)}")

    print("\nGroup-aware protein-level split...")
    train_ids, test_ids = group_aware_split(dataset.gold_pairs, dataset.all_ids, args.test_fraction, args.seed)
    train_pairs, test_pairs = split_pairs(dataset.gold_pairs, train_ids, test_ids)
    print(f"Train proteins: {len(train_ids)}  Test proteins: {len(test_ids)}")
    print(f"Train pairs: {len(train_pairs)}  Test pairs: {len(test_pairs)}")

    print("\nSampling negatives...")
    n_train_neg = len(train_pairs) * args.neg_per_pos
    n_test_neg = len(test_pairs) * args.neg_per_pos
    train_negatives = sample_negative_pairs(train_ids, n_train_neg, args.seed, dataset.positive_pair_set)
    test_negatives = sample_negative_pairs(test_ids, n_test_neg, args.seed + 1, dataset.positive_pair_set)

    print(f"\nTraining {model_config['type']}...")
    classifier = build_classifier(model_config, seed=args.seed, blast_matrix_path=args.blast_matrix)
    classifier.fit(train_pairs, train_negatives, dataset.embedding_matrix, dataset.id_to_index)

    print("\nClassification metrics (held-out test pairs vs sampled negatives)...")
    X_test, y_test = build_feature_matrix(test_pairs, test_negatives, dataset.embedding_matrix,
                                            dataset.id_to_index,
                                            blast_matrices=getattr(classifier, "blast_matrices", None)) \
        if model_config["type"] not in ("cosine", "hadamard_bilinear") else (None, None)
    clf_metrics = None
    if X_test is not None:
        # RF/MLP/XGBoost izlazu predict_proba iz konkretnog modela; jednostavnosti
        # radi ovde koristimo score_all po upitu za dosledne klasifikacione metrike
        y_proba = np.array([
            classifier.score_all(pid)[dataset.id_to_index[tid]]
            for pid, tid in zip(
                [p["id_1"] for p in test_pairs] + [a for a, b in test_negatives],
                [p["id_2"] for p in test_pairs] + [b for a, b in test_negatives],
            )
        ])
        y_pred = (y_proba >= 0.5).astype(int)
        clf_metrics = classification_metrics(y_test, y_pred, y_proba)
        for k, v in clf_metrics.items():
            if k != "confusion_matrix":
                print(f"  {k:10s}: {v:.4f}")

    print("\nRetrieval evaluation (Hits@K / MRR)...")
    cosine_matrix = cosine_similarity(dataset.embedding_matrix)
    retrieval_df = retrieval_evaluate(test_pairs, classifier, dataset.embedding_matrix, dataset.id_to_index,
                                        cosine_matrix=cosine_matrix, top_k=args.top_k)
    retrieval_summary = summarize_retrieval(retrieval_df, top_k=args.top_k)
    print(f"  MRR   : model={retrieval_summary['mrr']:.4f}  cosine={retrieval_summary.get('cosine_mrr', float('nan')):.4f}")
    for k in args.top_k:
        print(f"  Hits@{k}: model={retrieval_summary[f'hits_at_{k}']:.4f}  "
              f"cosine={retrieval_summary.get(f'cosine_hits_at_{k}', float('nan')):.4f}")

    print("\nBootstrap CI (model MRR - cosine MRR, po pair_id)...")
    delta_stats = bootstrap_ci(retrieval_df, "model_reciprocal_rank", group_col="pair_id",
                                 n_resamples=args.bootstrap_resamples, seed=args.seed,
                                 baseline_col="cosine_reciprocal_rank")
    verdict = "ZNACAJNO" if delta_stats["significant"] else "nije znacajno"
    print(f"  delta={delta_stats['mean']:+.4f}  95% CI [{delta_stats['ci_lo']:+.4f}, {delta_stats['ci_hi']:+.4f}] -- {verdict}")

    retrieval_df.to_csv(output_dir / f"{model_name}_retrieval_results.csv", index=False)
    with open(output_dir / f"{model_name}_summary.json", "w") as f:
        json.dump({
            "model_name": model_name, "model_config": model_config,
            "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
            "classification_metrics": clf_metrics, "retrieval_summary": retrieval_summary,
            "bootstrap_delta_vs_cosine": {k: (float(v) if not isinstance(v, bool) else v)
                                            for k, v in delta_stats.items()},
        }, f, indent=2)

    print(f"\nSaved: {output_dir / f'{model_name}_retrieval_results.csv'}")
    print(f"Saved: {output_dir / f'{model_name}_summary.json'}")


if __name__ == "__main__":
    main()
