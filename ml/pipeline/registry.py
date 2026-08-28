"""
Mapira "type" string iz JSON konfiga na odgovarajucu klasu klasifikatora.
Dodavanje novog modela = nova klasa u models/classifiers/ + jedan red ovde
+ novi configs/*.json -- ne dira postojece.
"""

from .models.classifiers.cosine import CosineSimilarity
from .models.classifiers.hadamard import HadamardBilinearClassifier
from .models.classifiers.mlp import MLPPairClassifier
from .models.classifiers.random_forest import RandomForestPairClassifier
from .models.classifiers.xgboost_clf import XGBoostPairClassifier

CLASSIFIER_REGISTRY = {
    "cosine": CosineSimilarity,
    "random_forest": RandomForestPairClassifier,
    "mlp": MLPPairClassifier,
    "xgboost": XGBoostPairClassifier,
    "hadamard_bilinear": HadamardBilinearClassifier,
}


def build_classifier(config: dict, seed: int, blast_matrix_path: str | None = None):
    model_type = config["type"]
    if model_type not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Nepoznat model type '{model_type}'. Poznati: {list(CLASSIFIER_REGISTRY)}")
    cls = CLASSIFIER_REGISTRY[model_type]
    params = config.get("params", {})
    extra_features = config.get("extra_features")
    return cls(params=params, extra_features=extra_features, blast_matrix_path=blast_matrix_path, seed=seed)
