# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Model - Feature Engineering
# MAGIC
# MAGIC This notebook creates engineered features from raw customer data and sets up Feature Store integration.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.append('../src')

from nba_model.config import get_config
from nba_model.features.engineering import FeatureEngineer

# COMMAND ----------

# Get configuration
config = get_config()

print("Configuration:")
print(f"  Raw table: {config.raw_table_full}")
print(f"  Feature table: {config.feature_table_full}")

# COMMAND ----------

# Initialize feature engineer
feature_eng = FeatureEngineer(config)

# Create engineered features
feature_eng.create_features()

# COMMAND ----------

# Display sample of engineered features
display(spark.table(config.feature_table_full).limit(10))

# COMMAND ----------

# Set up Feature Store integration
fs_mode = feature_eng.setup_feature_store()
print(f"Feature Store mode: {fs_mode}")

# COMMAND ----------

# Feature summary statistics
features = spark.table(config.feature_table_full)

print("Feature table statistics:")
print(f"  Total rows: {features.count():,}")
print(f"  Unique customers: {features.select('customer_id').distinct().count():,}")

print("\nNumeric feature summary:")
display(features.select(config.numeric_features).summary())

print("\nCategorical feature distributions:")
for cat_feature in config.categorical_features:
    print(f"\n{cat_feature}:")
    display(features.groupBy(cat_feature).count().orderBy("count", ascending=False))