"""Find and use saved model versions (models/<version>/)."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from pmodel.model.features import FEATURE_NAMES, logit
from pmodel.model.train import MODELS_DIR


class LoadedModel:
    def __init__(self, path):
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        self.version = self.meta["version"]
        self.booster = xgb.Booster()
        self.booster.load_model(path / "model.json")
        self.best_iteration = int(self.meta["best_iteration"])

    def predict(self, rows: list[dict]) -> np.ndarray:
        """P(YES) for feature dicts from features.features_at()."""
        df = pd.DataFrame(rows)[FEATURE_NAMES]
        margin = np.array([logit(p) for p in df["price"]])
        d = xgb.DMatrix(df, base_margin=margin, feature_names=FEATURE_NAMES)
        return self.booster.predict(d, iteration_range=(0, self.best_iteration + 1))


def versions(models_dir=MODELS_DIR):
    """General-model version folders (v<timestamp>/ with a meta.json), oldest first.
    Other folders (e.g. models/weather/) are ignored."""
    return sorted(p for p in models_dir.iterdir()
                  if p.is_dir() and p.name.startswith("v") and (p / "meta.json").exists())


def latest(models_dir=MODELS_DIR) -> LoadedModel:
    """Newest version that hasn't been marked superseded."""
    for path in reversed(versions(models_dir)):
        meta = json.loads((path / "meta.json").read_text())
        if not meta.get("superseded"):
            return LoadedModel(path)
    raise SystemExit("No usable model found — run `python -m pmodel train` first.")
