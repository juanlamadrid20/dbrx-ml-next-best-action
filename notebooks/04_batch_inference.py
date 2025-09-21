# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Model - Batch Inference
# MAGIC
# MAGIC This notebook runs batch inference using the trained NBA model.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.append('../src')
import ast

from nba_model.config import get_config
from nba_model.inference.batch_inference import BatchInference

# COMMAND ----------

# Get configuration
config = get_config()

print("Configuration:")
print(f"  Feature table: {config.feature_table_full}")
print(f"  Model name: {config.uc_model_name}")
print(f"  Recommendations table: {config.rec_table_full}")

# COMMAND ----------

# Get training columns from persistent storage or widgets
train_columns = None

# Try to load from Unity Catalog Volume first (cross-job compatibility)
try:
    import json
    volume_path = f"/Volumes/{config.catalog}/{config.schema}/data"
    columns_file_path = f"{volume_path}/nba_model_train_columns.json"
    train_columns_json = dbutils.fs.head(columns_file_path)
    train_columns = json.loads(train_columns_json)
    print(f"Loaded training columns from volume ({len(train_columns)} columns): {columns_file_path}")
except Exception as e:
    print(f"Could not load from volume: {e}")

# Fallback to widgets (same job execution)
if train_columns is None:
    try:
        train_columns_str = dbutils.widgets.get("train_columns")
        train_columns = ast.literal_eval(train_columns_str)
        print(f"Using training columns from widgets ({len(train_columns)} columns)")
    except Exception as e:
        print(f"Could not get from widgets: {e}")

# Last resort: construct from inference data (may cause schema mismatch)
if train_columns is None:
    print("⚠️  WARNING: Constructing training columns from inference data - this may cause schema mismatches!")
    import pandas as pd
    sample_features = spark.table(config.feature_table_full).limit(1).toPandas()
    X_sample = sample_features[config.numeric_features + config.categorical_features].copy()
    X_sample = pd.get_dummies(X_sample, columns=config.categorical_features, drop_first=True)
    train_columns = list(X_sample.columns)
    print(f"Constructed training columns ({len(train_columns)} columns)")

print("Training columns preview:", train_columns[:10])

# COMMAND ----------

# Initialize batch inference
batch_inference = BatchInference(config)

# Run batch inference
batch_inference.run_inference(train_columns, limit=5000)

# COMMAND ----------

# Display sample recommendations
recs = spark.table(config.rec_table_full)
display(recs.limit(10))

print("Recommendation distribution:")
display(recs.groupBy("recommended_action").count().orderBy("recommended_action"))

# COMMAND ----------

# Create inference log for monitoring
batch_inference.create_inference_log()

print(f"Inference log created: {config.log_table_full}")
display(spark.table(config.log_table_full).limit(10))