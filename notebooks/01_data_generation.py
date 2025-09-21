# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Model - Data Generation
# MAGIC
# MAGIC This notebook generates synthetic customer data for the Next Best Action model.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.append('../src')

from nba_model.config import get_config
from nba_model.data.generator import DataGenerator

# COMMAND ----------

# Get configuration
config = get_config()

print("Configuration:")
print(f"  Catalog: {config.catalog}")
print(f"  Schema: {config.schema}")
print(f"  Raw table: {config.raw_table_full}")

# COMMAND ----------

# Initialize data generator
data_gen = DataGenerator(config)

# Create schema
data_gen.create_schema()

# COMMAND ----------

# Generate synthetic data
data_gen.generate_raw_data(n_customers=10_000, random_seed=42)

# COMMAND ----------

# Display sample of generated data
display(spark.table(config.raw_table_full).limit(10))

# COMMAND ----------

# Basic data exploration
raw = spark.table(config.raw_table_full)

print(f"Total customers: {raw.count():,}")
print("\nAction distribution:")
display(raw.groupBy("best_action").count().orderBy("best_action"))

print("\nAge statistics:")
display(raw.select("age").summary())