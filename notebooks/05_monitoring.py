# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Model - Monitoring & Drift Detection
# MAGIC
# MAGIC This notebook monitors model performance and detects data drift.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.append('../src')

from nba_model.config import get_config
from nba_model.monitoring.drift_detection import ModelMonitor

# COMMAND ----------

# Get configuration
config = get_config()

print("Configuration:")
print(f"  Log table: {config.log_table_full}")
print(f"  Feature table: {config.feature_table_full}")

# COMMAND ----------

# Initialize model monitor
monitor = ModelMonitor(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Report

# COMMAND ----------

# Generate comprehensive performance report
performance_report = monitor.generate_model_performance_report()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Action Distribution Analysis

# COMMAND ----------

# Analyze action distribution over time
monitor.analyze_action_distribution()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check data quality metrics
quality_metrics = monitor.check_data_quality()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Drift Detection

# COMMAND ----------

# Detect feature drift
monitor.detect_feature_drift()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Dashboard

# COMMAND ----------

# Create summary visualization
log_table = spark.table(config.log_table_full)

print("=== MODEL MONITORING SUMMARY ===")
print(f"Total predictions: {log_table.count():,}")
print(f"Date range: {log_table.select('log_date').distinct().count()} days")

# Recent activity (last 7 days)
from pyspark.sql import functions as F
recent = log_table.filter(F.col("log_date") >= F.date_sub(F.current_date(), 7))
print(f"Recent predictions (7d): {recent.count():,}")

# Action distribution
print("\nOverall action distribution:")
action_dist = log_table.groupBy("recommended_action").count().collect()
for row in action_dist:
    pct = (row["count"] / log_table.count()) * 100
    print(f"  {row['recommended_action']}: {row['count']:,} ({pct:.1f}%)")

print("\n=== ALERTS ===")
# Check for any concerning patterns
total_logs = log_table.count()
if total_logs == 0:
    print("⚠️  No inference logs found")
elif recent.count() == 0:
    print("⚠️  No recent predictions (last 7 days)")
else:
    print("✅ System appears healthy")