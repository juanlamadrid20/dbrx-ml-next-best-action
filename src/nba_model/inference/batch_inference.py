"""Batch inference pipeline for NBA model."""

import mlflow.pyfunc
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config import ModelConfig


class BatchInference:
    """Batch inference for NBA recommendations."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()

    def run_inference(self, train_columns: list, limit: int = 5000) -> None:
        """Run batch inference and save recommendations."""
        # Load features for inference
        features_df = self.spark.table(self.config.feature_table_full).limit(limit).toPandas()

        # Prepare inference matrix
        X_infer = self._prepare_inference_matrix(features_df, train_columns)

        # Load model and predict
        loaded_model = mlflow.pyfunc.load_model(f"models:/{self.config.uc_model_name}@champion")
        pred_int = loaded_model.predict(X_infer)

        # Convert predictions to action strings
        inv_map = {0: "Email", 1: "SMS", 2: "Push"}
        pred_str = [inv_map[int(i)] for i in pred_int]

        # Create recommendations dataframe
        recs_pdf = pd.DataFrame(
            {
                "customer_id": features_df["customer_id"].values,
                "recommended_action": pred_str,
                "scored_at_ts": pd.Timestamp.utcnow(),
            }
        )

        # Save to Delta table
        recs_sdf = self.spark.createDataFrame(recs_pdf)
        recs_sdf.write.mode("overwrite").format("delta").saveAsTable(self.config.rec_table_full)

        print(f"Saved {len(recs_pdf)} recommendations to {self.config.rec_table_full}")

    def _prepare_inference_matrix(
        self, features_df: pd.DataFrame, train_columns: list
    ) -> pd.DataFrame:
        """Prepare inference matrix matching training format."""
        X_infer = features_df[
            self.config.numeric_features + self.config.categorical_features
        ].copy()

        # One-hot encode categoricals
        X_infer = pd.get_dummies(X_infer, columns=self.config.categorical_features, drop_first=True)

        # Align to training columns
        missing = set(train_columns) - set(X_infer.columns)
        for c in missing:
            X_infer[c] = 0

        return X_infer[train_columns]  # Exact order match

    def create_inference_log(self) -> None:
        """Create inference log joining predictions with features."""
        preds = self.spark.table(self.config.rec_table_full)
        feats = self.spark.table(self.config.feature_table_full).select(
            "customer_id", *self.config.numeric_features, *self.config.categorical_features
        )

        log_df = (
            preds.alias("p")
            .join(feats.alias("f"), on="customer_id", how="left")
            .withColumn("log_date", F.current_date())
        )

        log_df.write.mode("append").format("delta").saveAsTable(self.config.log_table_full)
        print(f"Created inference log in {self.config.log_table_full}")

    def predict_single(self, customer_features: dict, train_columns: list) -> str:
        """Predict action for a single customer (for serving endpoints)."""
        # Convert to DataFrame
        features_df = pd.DataFrame([customer_features])

        # Prepare inference matrix
        X_infer = self._prepare_inference_matrix(features_df, train_columns)

        # Load model and predict
        loaded_model = mlflow.pyfunc.load_model(f"models:/{self.config.uc_model_name}@champion")
        pred_int = loaded_model.predict(X_infer)[0]

        # Convert to action string
        inv_map = {0: "Email", 1: "SMS", 2: "Push"}
        return inv_map[int(pred_int)]
