# Databricks notebook source
# MAGIC %md
# MAGIC # NBA Model - Lakehouse Monitoring Setup & Analytics
# MAGIC
# MAGIC This notebook sets up Lakehouse Monitoring and provides basic model analytics.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

import sys
sys.path.append('../src')

from nba_model.config import get_config
from nba_model.monitoring.drift_detection import ModelMonitor
from nba_model.monitoring.lakehouse_monitor_sql import LakehouseMonitorSQL

# COMMAND ----------

# Get configuration
config = get_config()

print("Configuration:")
print(f"  Log table: {config.log_table_full}")
print(f"  Feature table: {config.feature_table_full}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Lakehouse Monitoring
# MAGIC
# MAGIC This section sets up automated monitoring for both feature drift and inference patterns.

# COMMAND ----------

# Initialize Lakehouse Monitor (SQL-based)
lakehouse_monitor = LakehouseMonitorSQL(config)

# Generate monitoring report first
monitoring_report = lakehouse_monitor.get_monitoring_status_report()

print("=== MONITORING READINESS REPORT ===")
print(f"Validation: {monitoring_report['validation']}")
print(f"Ready for monitoring: {monitoring_report['ready_for_monitoring']}")
print(f"Recommendations: {monitoring_report['recommendations']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create All Monitors
# MAGIC
# MAGIC Uncomment and run this cell to set up all monitoring (run only once):

# COMMAND ----------

# Uncomment to create monitors (run only once)
# try:
#     results = lakehouse_monitor.setup_all_monitors_sql(create_baseline=True)
#     print("✅ All monitors created successfully!")
#     print(results)
# except Exception as e:
#     print(f"❌ Monitor creation failed: {e}")
#     print("Showing manual setup instructions...")
#     lakehouse_monitor.show_manual_setup_instructions()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitor Status Check

# COMMAND ----------

# Show manual setup instructions if needed
print("=== MANUAL SETUP INSTRUCTIONS ===")
lakehouse_monitor.show_manual_setup_instructions()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Table Validation

# COMMAND ----------

# Validate tables are ready for monitoring
validation_report = lakehouse_monitor.validate_tables_for_monitoring()
print("=== TABLE VALIDATION ===")
print(f"Feature table ready: {validation_report['feature_table_ready']}")
if validation_report['feature_table_issues']:
    print(f"Feature table issues: {validation_report['feature_table_issues']}")

print(f"Inference table ready: {validation_report['inference_table_ready']}")
if validation_report['inference_table_issues']:
    print(f"Inference table issues: {validation_report['inference_table_issues']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Model Analytics
# MAGIC
# MAGIC Traditional analytics that complement Lakehouse Monitoring

# COMMAND ----------

# Initialize basic model monitor for analytics
basic_monitor = ModelMonitor(config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Report

# COMMAND ----------

# Generate comprehensive performance report
performance_report = basic_monitor.generate_model_performance_report()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Action Distribution Analysis

# COMMAND ----------

# Analyze action distribution over time
basic_monitor.analyze_action_distribution()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check data quality metrics
quality_metrics = basic_monitor.check_data_quality()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lakehouse Monitoring Integration

# COMMAND ----------

# Check Lakehouse Monitoring status
lakehouse_info = basic_monitor.get_lakehouse_monitoring_info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Dashboard

# COMMAND ----------

# Create summary visualization
log_table = spark.table(config.log_table_full)

print("=== MODEL MONITORING SUMMARY ===")
print(f"Total predictions: {log_table.count():,}")

# Check if table has the new columns
try:
    date_range = log_table.select('log_date').distinct().count()
    print(f"Date range: {date_range} days")

    # Recent activity (last 7 days)
    from pyspark.sql import functions as F
    recent = log_table.filter(F.col("log_date") >= F.date_sub(F.current_date(), 7))
    print(f"Recent predictions (7d): {recent.count():,}")

    # Action distribution
    print("\nOverall action distribution:")
    action_dist = log_table.groupBy("recommended_action").count().collect()
    total_count = log_table.count()
    for row in action_dist:
        pct = (row["count"] / total_count) * 100 if total_count > 0 else 0
        print(f"  {row['recommended_action']}: {row['count']:,} ({pct:.1f}%)")

    # Model version distribution (if available)
    try:
        version_dist = log_table.groupBy("model_version").count().collect()
        print("\nModel version distribution:")
        for row in version_dist:
            pct = (row["count"] / total_count) * 100 if total_count > 0 else 0
            print(f"  Version {row['model_version']}: {row['count']:,} ({pct:.1f}%)")
    except:
        print("\n⚠️  Model version column not found - run inference pipeline to update schema")

    print("\n=== MONITORING STATUS ===")
    # Check for any concerning patterns
    if total_count == 0:
        print("⚠️  No inference logs found")
    elif recent.count() == 0:
        print("⚠️  No recent predictions (last 7 days)")
    else:
        print("✅ Basic system health appears good")

    # Lakehouse Monitoring status
    if lakehouse_info.get("tables_ready_for_monitoring"):
        print("✅ Tables ready for Lakehouse Monitoring")
    else:
        print("⚠️  Tables need schema updates for Lakehouse Monitoring")

    if lakehouse_info.get("monitors_configured", 0) > 0:
        print(f"✅ {lakehouse_info['monitors_configured']} Lakehouse monitors configured")
    else:
        print("⚠️  No Lakehouse monitors configured yet")

except Exception as e:
    print(f"⚠️  Error accessing tables: {e}")
    print("Run the complete pipeline first to ensure all tables exist")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **First Run**: Uncomment the monitor creation cell above and run it once
# MAGIC 2. **Regular Monitoring**: Use the Lakehouse Monitoring dashboards for drift detection
# MAGIC 3. **Alerts**: Configure email notifications for significant drift or data quality issues
# MAGIC 4. **Integration**: Set up automated retraining triggers based on drift thresholds