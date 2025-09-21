# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Model - Training
# MAGIC
# MAGIC This notebook trains the Next Best Action model using scikit-learn and MLflow.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.append('../src')

from nba_model.config import get_config
from nba_model.training.trainer import ModelTrainer

# COMMAND ----------

# Get configuration
config = get_config()

print("Configuration:")
print(f"  Feature table: {config.feature_table_full}")
print(f"  Model name: {config.uc_model_name}")
print(f"  Experiment: {config.experiment_path}")

# COMMAND ----------

# Initialize trainer
trainer = ModelTrainer(config)

# Prepare training data (try Feature Store first, fallback to Delta)
try:
    X_all, y_all = trainer.prepare_training_data(fs_mode="feature_store")
    print("Using Feature Store for training data")
except Exception as e:
    print(f"Feature Store unavailable, using Delta: {e}")
    X_all, y_all = trainer.prepare_training_data(fs_mode="delta")

# COMMAND ----------

# Train model with MLflow tracking
model_runs_uri, test_accuracy, run_id = trainer.train_model(X_all, y_all)

print(f"Training completed!")
print(f"Run ID: {run_id}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Model runs URI: {model_runs_uri}")

# COMMAND ----------

# Register model to Unity Catalog
new_version = trainer.register_model(model_runs_uri)
print(f"Model registered as version {new_version}")

# COMMAND ----------

# Promote model based on performance
trainer.promote_model(new_version, test_accuracy)

# COMMAND ----------

# Store training columns for inference alignment
train_columns = trainer.get_train_columns()
print(f"Training columns ({len(train_columns)}): {train_columns[:5]}...")

# Save training columns to a file for inference jobs to use
import json
train_columns_json = json.dumps(train_columns)

# Write to Unity Catalog Volume for persistence across jobs
volume_path = f"/Volumes/{config.catalog}/{config.schema}/data"
columns_file_path = f"{volume_path}/nba_model_train_columns.json"
dbutils.fs.put(columns_file_path, train_columns_json, overwrite=True)
print(f"Saved training columns to {columns_file_path}")

# Also save to widgets for immediate downstream use
dbutils.widgets.text("train_columns", str(train_columns), "Training columns")
dbutils.widgets.text("model_version", str(new_version), "Model version")