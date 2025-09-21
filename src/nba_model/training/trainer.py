"""Model training pipeline for NBA model."""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pyspark.sql import SparkSession, functions as F
from databricks.feature_store import FeatureLookup
from ..config import ModelConfig


class ModelTrainer:
    """Train and evaluate NBA recommendation model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()
        self.train_columns = None

    def prepare_training_data(self, fs_mode: str = "feature_store"):
        """Prepare training dataset with features and labels."""
        # Create labels dataframe
        raw = self.spark.table(self.config.raw_table_full)
        labels_sdf = raw.select(
            "customer_id",
            F.when(F.col("best_action") == "Email", 0)
             .when(F.col("best_action") == "SMS", 1)
             .otherwise(2).alias("action_label"),
            "best_action"
        )

        # Get training data with features
        if fs_mode == "feature_store":
            from databricks.feature_store import FeatureStoreClient
            fs = FeatureStoreClient()

            feature_lookups = [
                FeatureLookup(
                    table_name=self.config.feature_table_full,
                    feature_names=self.config.numeric_features + self.config.categorical_features,
                    lookup_key="customer_id"
                )
            ]
            training_set = fs.create_training_set(
                df=labels_sdf,
                feature_lookups=feature_lookups,
                label="action_label"
            )
            train_sdf = training_set.load_df()
        else:
            # Delta-only mode
            feats = self.spark.table(self.config.feature_table_full)
            train_sdf = labels_sdf.alias("l").join(feats.alias("f"), on="customer_id") \
                .select("l.action_label", *self.config.numeric_features, *self.config.categorical_features, "f.customer_id")

        # Convert to pandas and prepare design matrix
        train_pdf = train_sdf.toPandas()
        y_all = train_pdf["action_label"].astype(int).values
        X_all = train_pdf[self.config.numeric_features + self.config.categorical_features].copy()
        X_all = pd.get_dummies(X_all, columns=self.config.categorical_features, drop_first=True)

        # Store training columns for inference alignment
        self.train_columns = list(X_all.columns)

        print(f"Training matrix shape: {X_all.shape}")
        print(f"Label shape: {y_all.shape}")

        return X_all, y_all

    def train_model(self, X_all, y_all):
        """Train RandomForest model with MLflow tracking."""
        # Set up MLflow
        if self.config.experiment_path:
            mlflow.set_experiment(self.config.experiment_path)

        try:
            mlflow.set_registry_uri("databricks-uc")
            print("MLflow registry set to Unity Catalog")
        except Exception as e:
            raise RuntimeError("Unity Catalog MLflow registry is required") from e

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
        )

        # Enable autologging
        mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

        with mlflow.start_run() as run:
            # Train model
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            test_accuracy = float(accuracy_score(y_test, y_pred))

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")

            # Create confusion matrix
            self._log_confusion_matrix(y_test, y_pred)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
            )

            run_id = run.info.run_id
            model_runs_uri = f"runs:/{run_id}/model"

        print(f"Training completed. Run ID: {run_id}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        return model_runs_uri, test_accuracy, run_id

    def _log_confusion_matrix(self, y_test, y_pred):
        """Create and log confusion matrix visualization."""
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix (0=Email, 1=SMS, 2=Push)")
        plt.colorbar()

        ticks = np.arange(3)
        labs = ["Email", "SMS", "Push"]
        plt.xticks(ticks, labs, rotation=45)
        plt.yticks(ticks, labs)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]:d}", ha="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()

    def register_model(self, model_runs_uri: str) -> int:
        """Register model to Unity Catalog."""
        try:
            mv = mlflow.register_model(model_uri=model_runs_uri, name=self.config.uc_model_name)
            new_version = int(mv.version)
            print(f"Registered to UC: {self.config.uc_model_name} v{new_version}")
            return new_version
        except Exception as e:
            raise RuntimeError("Unity Catalog model registration failed") from e

    def promote_model(self, new_version: int, test_accuracy: float) -> None:
        """Promote model based on performance."""
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        model_name = self.config.uc_model_name

        def get_alias_version_or_none(name, alias):
            try:
                mv = client.get_model_version_by_alias(name, alias)
                return int(mv.version)
            except Exception:
                return None

        def get_metric_for_version(name, version, metric_key="test_accuracy"):
            if version is None:
                return None
            mv = client.get_model_version(name, str(version))
            r = mlflow.get_run(mv.run_id)
            return r.data.metrics.get(metric_key)

        # Set staging alias
        try:
            client.set_registered_model_alias(model_name, "staging", str(new_version))
            print(f"Set @{model_name}@staging → v{new_version}")
        except Exception as e:
            print(f"Staging alias warning: {e}")

        # Champion vs Challenger promotion
        champ_v = get_alias_version_or_none(model_name, "champion")
        champ_acc = get_metric_for_version(model_name, champ_v) if champ_v else None

        if champ_v is None:
            client.set_registered_model_alias(model_name, "champion", str(new_version))
            print(f"Set initial champion: @{model_name}@champion → v{new_version}")
        else:
            if test_accuracy >= (champ_acc or float("-inf")):
                client.set_registered_model_alias(model_name, "prev_champion", str(champ_v))
                client.set_registered_model_alias(model_name, "champion", str(new_version))
                print(f"Promoted new champion: v{new_version} (prev: v{champ_v})")
            else:
                client.set_registered_model_alias(model_name, "challenger", str(new_version))
                print(f"Set challenger: v{new_version} (acc={test_accuracy:.4f} < champ={champ_acc:.4f})")

    def get_train_columns(self) -> list:
        """Get the training column order for inference alignment."""
        return self.train_columns