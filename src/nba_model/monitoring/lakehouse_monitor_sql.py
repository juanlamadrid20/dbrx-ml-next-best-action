"""Lakehouse Monitoring setup using SQL commands for better compatibility."""

from typing import Optional
from pyspark.sql import SparkSession

from ..config import ModelConfig


class LakehouseMonitorSQL:
    """Lakehouse Monitoring setup using SQL commands."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()

    def create_inference_monitor_sql(self, baseline_table: Optional[str] = None) -> None:
        """Create Lakehouse Monitor for inference log table using SQL."""
        # Get current user for assets directory
        current_user = self.spark.sql("SELECT current_user()").collect()[0][0]
        assets_dir = f"/Workspace/Users/{current_user}/lakehouse_monitoring/{self.config.log_table}"

        # SQL command to create inference monitor
        sql_command = f"""
        CREATE OR REFRESH QUALITY MONITOR {self.config.log_table_full}
        USING INFERENCE_LOG
        OPTIONS (
          granularities => array("1 day"),
          timestamp_col => "inference_timestamp",
          prediction_col => "recommended_action",
          problem_type => "classification",
          model_id_col => "model_version"
        )
        ASSETS_DIR '{assets_dir}'
        OUTPUT_SCHEMA_NAME '{self.config.catalog}.{self.config.schema}'
        """

        if baseline_table:
            sql_command += f"\nBASELINE_TABLE '{baseline_table}'"

        try:
            self.spark.sql(sql_command)
            print(f"✅ Created inference monitor for {self.config.log_table_full}")
            print(f"   Assets directory: {assets_dir}")
        except Exception as e:
            print(f"❌ Failed to create inference monitor: {e}")
            print("Note: You may need to create monitors through the Databricks UI")
            print(f"Table: {self.config.log_table_full}")
            print("Monitor type: Inference Log")
            print("Timestamp column: inference_timestamp")
            print("Prediction column: recommended_action")
            print("Model ID column: model_version")

    def create_feature_monitor_sql(self, baseline_table: Optional[str] = None) -> None:
        """Create Lakehouse Monitor for feature table using SQL."""
        # Get current user for assets directory
        current_user = self.spark.sql("SELECT current_user()").collect()[0][0]
        assets_dir = f"/Workspace/Users/{current_user}/lakehouse_monitoring/{self.config.feature_table}"

        # SQL command to create time series monitor
        sql_command = f"""
        CREATE OR REFRESH QUALITY MONITOR {self.config.feature_table_full}
        USING TIME_SERIES
        OPTIONS (
          granularities => array("1 day", "1 week"),
          timestamp_col => "updated_at"
        )
        ASSETS_DIR '{assets_dir}'
        OUTPUT_SCHEMA_NAME '{self.config.catalog}.{self.config.schema}'
        """

        if baseline_table:
            sql_command += f"\nBASELINE_TABLE '{baseline_table}'"

        try:
            self.spark.sql(sql_command)
            print(f"✅ Created feature monitor for {self.config.feature_table_full}")
            print(f"   Assets directory: {assets_dir}")
        except Exception as e:
            print(f"❌ Failed to create feature monitor: {e}")
            print("Note: You may need to create monitors through the Databricks UI")
            print(f"Table: {self.config.feature_table_full}")
            print("Monitor type: Time Series")
            print("Timestamp column: updated_at")

    def setup_all_monitors_sql(self, create_baseline: bool = True) -> dict:
        """Set up all monitoring using SQL commands."""
        results = {}

        print("🚀 Setting up Lakehouse Monitoring for NBA model using SQL...")

        # Create baseline tables if requested
        baseline_inference = None
        baseline_features = None

        if create_baseline:
            baseline_inference = self._create_baseline_table(
                self.config.log_table_full, "inference_baseline"
            )
            baseline_features = self._create_baseline_table(
                self.config.feature_table_full, "feature_baseline"
            )

        # Create monitors using SQL
        try:
            # Feature monitoring (data drift)
            self.create_feature_monitor_sql(baseline_table=baseline_features)
            results["feature_monitor"] = "created"

            # Inference monitoring (prediction drift)
            self.create_inference_monitor_sql(baseline_table=baseline_inference)
            results["inference_monitor"] = "created"

            print("\n✅ Monitor setup completed!")
            print("📊 Dashboards will be available in the Databricks workspace")
            print("   Navigate to: Data -> [your table] -> Quality tab")

        except Exception as e:
            print(f"❌ Monitor setup failed: {e}")
            results["error"] = str(e)

        return results

    def _create_baseline_table(self, source_table: str, baseline_suffix: str) -> str:
        """Create a baseline snapshot table."""
        baseline_table = f"{self.config.catalog}.{self.config.schema}.{baseline_suffix}"

        try:
            # Check if source table exists and has data
            if self.spark.catalog.tableExists(source_table):
                count = self.spark.table(source_table).count()
                if count > 0:
                    # Create baseline as current snapshot
                    self.spark.sql(f"""
                        CREATE OR REPLACE TABLE {baseline_table}
                        AS SELECT * FROM {source_table}
                        LIMIT 1000  -- Limit baseline to reasonable size
                    """)
                    print(f"📸 Created baseline table: {baseline_table} ({count} total rows, 1000 in baseline)")
                    return baseline_table
                else:
                    print(f"⚠️  Source table {source_table} is empty, skipping baseline")
            else:
                print(f"⚠️  Source table {source_table} does not exist, skipping baseline")

        except Exception as e:
            print(f"⚠️  Baseline table creation failed: {e}")

        return None

    def show_manual_setup_instructions(self) -> None:
        """Show manual setup instructions if automated setup fails."""
        print("\n" + "="*60)
        print("MANUAL LAKEHOUSE MONITORING SETUP INSTRUCTIONS")
        print("="*60)

        print("\n1. Feature Table Monitor:")
        print(f"   - Table: {self.config.feature_table_full}")
        print("   - Monitor Type: Time Series")
        print("   - Timestamp Column: updated_at")
        print("   - Granularities: 1 day, 1 week")

        print("\n2. Inference Log Monitor:")
        print(f"   - Table: {self.config.log_table_full}")
        print("   - Monitor Type: Inference Log")
        print("   - Timestamp Column: inference_timestamp")
        print("   - Prediction Column: recommended_action")
        print("   - Model ID Column: model_version")
        print("   - Problem Type: Classification")
        print("   - Granularities: 1 day")

        print("\nSteps:")
        print("1. Go to Databricks workspace")
        print("2. Navigate to Data -> [table name] -> Quality tab")
        print("3. Click 'Enable monitoring'")
        print("4. Configure with the settings above")
        print("5. Set up email notifications if desired")

    def validate_tables_for_monitoring(self) -> dict:
        """Validate that tables have required columns for monitoring."""
        validation = {
            "feature_table_ready": False,
            "inference_table_ready": False,
            "feature_table_issues": [],
            "inference_table_issues": [],
        }

        # Check feature table
        try:
            if self.spark.catalog.tableExists(self.config.feature_table_full):
                feature_cols = [col.name for col in self.spark.table(self.config.feature_table_full).schema]
                if "updated_at" in feature_cols:
                    validation["feature_table_ready"] = True
                else:
                    validation["feature_table_issues"].append("Missing 'updated_at' column")

                count = self.spark.table(self.config.feature_table_full).count()
                if count == 0:
                    validation["feature_table_issues"].append("Table is empty")
            else:
                validation["feature_table_issues"].append("Table does not exist")
        except Exception as e:
            validation["feature_table_issues"].append(f"Error accessing table: {e}")

        # Check inference log table
        try:
            if self.spark.catalog.tableExists(self.config.log_table_full):
                log_cols = [col.name for col in self.spark.table(self.config.log_table_full).schema]
                required_cols = ["inference_timestamp", "model_version", "recommended_action"]
                missing_cols = [col for col in required_cols if col not in log_cols]

                if not missing_cols:
                    validation["inference_table_ready"] = True
                else:
                    validation["inference_table_issues"].append(f"Missing columns: {missing_cols}")

                count = self.spark.table(self.config.log_table_full).count()
                if count == 0:
                    validation["inference_table_issues"].append("Table is empty")
            else:
                validation["inference_table_issues"].append("Table does not exist")
        except Exception as e:
            validation["inference_table_issues"].append(f"Error accessing table: {e}")

        return validation

    def get_monitoring_status_report(self) -> dict:
        """Generate a comprehensive monitoring readiness report."""
        validation = self.validate_tables_for_monitoring()

        report = {
            "validation": validation,
            "ready_for_monitoring": validation["feature_table_ready"] and validation["inference_table_ready"],
            "recommendations": []
        }

        # Add recommendations
        if not validation["feature_table_ready"]:
            report["recommendations"].append("Run feature engineering pipeline to prepare feature table")
        if not validation["inference_table_ready"]:
            report["recommendations"].append("Run inference pipeline to prepare inference log table")
        if validation["feature_table_ready"] and validation["inference_table_ready"]:
            report["recommendations"].append("Tables are ready - proceed with monitor creation")

        return report