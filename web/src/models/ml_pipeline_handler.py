import cloudpickle
import pandas as pd


class MlPipelineHandler:
    def __init__(self, model_path: str, categorical_features: list[str]):
        with open(model_path, "rb") as pipeline_file:
            self.pipeline = cloudpickle.load(pipeline_file)

        self.categorical_features = categorical_features
        self.expected_columns = (
            self.pipeline.named_steps["preprocessor"].transformers_[0][2]
            + self.pipeline.named_steps["preprocessor"].transformers_[1][2]
        )
        self.default_values = {
            col: ("unknown" if col in categorical_features else 0.0)
            for col in self.expected_columns
        }

    def preprocess(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        df = df.reindex(columns=self.expected_columns)

        for col in self.expected_columns:
            if col not in self.categorical_features:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.fillna(self.default_values)

    def predict(self, data: dict):
        df = self.preprocess(data)
        result = self.pipeline.predict(df)[0]
        if result == 1:
            return "Exoplanet candidate detected! 🚀"
        return "No exoplanet detected this time. Keep hunting!"
