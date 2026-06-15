"""AI Studio MLflow pyfunc model wrapper."""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import pandas as pd


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Replace predict() with the real inference logic for your model."""

    def load_context(self, context):
        self.model_path = Path(context.artifacts["model"])
        self.config_path = Path(context.artifacts["config"])

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")
        return model_input
