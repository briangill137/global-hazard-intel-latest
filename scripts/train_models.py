import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from ai_models.features import FEATURE_NAMES
from ai_models.predictor import train_ensemble, pack_and_save


def load_dataset(path: Path | None):
    # Placeholder: generate synthetic dataset if no file provided
    n = 800
    X = np.random.rand(n, len(FEATURE_NAMES))
    labels = {
        "flood": (X[:, 4] + X[:, 5] + X[:, 6] > 1.0).astype(int),
        "snow": (X[:, 6] + (1 - X[:, 0]) > 1.0).astype(int),
        "fire": (X[:, 0] + X[:, 2] - X[:, 1] > 1.0).astype(int),
        "storm": (X[:, 2] + X[:, 3] > 1.0).astype(int),
    }
    return X, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to historical dataset (CSV)")
    args = parser.parse_args()

    X, labels = load_dataset(Path(args.data) if args.data else None)

    scaler = MinMaxScaler(feature_range=(0, 1))
    Xn = scaler.fit_transform(X)

    trained_wrappers = {}
    for name, y in labels.items():
        X_train, X_val, y_train, y_val = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)
        trained = train_ensemble(X_train, {name: y_train})
        wrapper = trained[name]
        # Simple evaluation
        y_pred = np.mean([m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_val) for m in wrapper], axis=0)
        auc = roc_auc_score(y_val, y_pred)
        print(f"{name} AUC: {auc:.3f}")
        trained_wrappers[name] = wrapper

    pack_and_save(trained_wrappers)
    print("Models saved to models/*.pkl")


if __name__ == "__main__":
    main()
