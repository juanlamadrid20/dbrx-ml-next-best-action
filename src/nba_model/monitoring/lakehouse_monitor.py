"""Lakehouse Monitoring setup and management for NBA model."""

import os
from typing import Optional

try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import (
        MonitorInferenceLog,
        MonitorTimeSeries,
    )
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"Databricks SDK import error: {e}")
    SDK_AVAILABLE = False

from pyspark.sql import SparkSession

from ..config import ModelConfig


class LakehouseMonitor:
    """Lakehouse Monitoring setup and management."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()
        if SDK_AVAILABLE:
            try:
                self.workspace_client = WorkspaceClient()
            except Exception as e:
                print(f"Warning: Could not initialize WorkspaceClient: {e}")
                self.workspace_client = None
        else:
            self.workspace_client = None

    def create_inference_monitor(
        self,
        baseline_table: Optional[str] = None,
        email_notifications: Optional[list] = None,
    ) -> str:
        """Create Lakehouse Monitor for inference log table."""
        if not SDK_AVAILABLE or not self.workspace_client:
            raise RuntimeError("Databricks SDK not available or WorkspaceClient not initialized")

        monitor_name = f"{self.config.catalog}.{self.config.schema}.nba_inference_monitor"

        # Get current user for assets directory
        current_user = self.spark.sql("SELECT current_user()").collect()[0][0]
        assets_dir = f"/Workspace/Users/{current_user}/lakehouse_monitoring/{self.config.log_table}"

        try:
            # Create inference log monitor
            monitor_info = self.workspace_client.quality_monitors.create(
                table_name=self.config.log_table_full,
                assets_dir=assets_dir,
                output_schema_name=f"{self.config.catalog}.{self.config.schema}",
                inference_log=MonitorInferenceLog(
                    granularities=["1 day"],
                    timestamp_col="inference_timestamp",
                    prediction_col="recommended_action",
                    problem_type="classification",
                    model_id_col="model_version",
                    label_col=None,  # No ground truth available initially
                ),
                baseline_table_name=baseline_table,
                # Note: Email notifications configuration may vary by SDK version
                # notifications can be configured through the UI after monitor creation
            )

            print(f"✅ Created inference monitor: {monitor_name}")
            print(f"   Monitor ID: {monitor_info.monitor_id}")
            print(f"   Assets directory: {assets_dir}")
            return monitor_info.monitor_id

        except Exception as e:
            print(f"❌ Failed to create inference monitor: {e}")
            raise

    def create_feature_monitor(
        self,
        baseline_table: Optional[str] = None,
        email_notifications: Optional[list] = None,
    ) -> str:
        """Create Lakehouse Monitor for feature table."""
        monitor_name = f"{self.config.catalog}.{self.config.schema}.nba_feature_monitor"

        # Get current user for assets directory
        current_user = self.spark.sql("SELECT current_user()").collect()[0][0]
        assets_dir = f"/Workspace/Users/{current_user}/lakehouse_monitoring/{self.config.feature_table}"

        try:
            # Create time series monitor for features
            monitor_info = self.workspace_client.quality_monitors.create(
                table_name=self.config.feature_table_full,
                assets_dir=assets_dir,
                output_schema_name=f"{self.config.catalog}.{self.config.schema}",
                time_series=MonitorTimeSeries(
                    granularities=["1 day", "1 week"],
                    timestamp_col="updated_at",
                ),
                baseline_table_name=baseline_table,
                # Note: Email notifications configuration may vary by SDK version
                # notifications can be configured through the UI after monitor creation
            )

            print(f"✅ Created feature monitor: {monitor_name}")
            print(f"   Monitor ID: {monitor_info.monitor_id}")
            print(f"   Assets directory: {assets_dir}")
            return monitor_info.monitor_id

        except Exception as e:
            print(f"❌ Failed to create feature monitor: {e}")
            raise

    def setup_all_monitors(
        self,
        email_notifications: Optional[list] = None,
        create_baseline: bool = True,
    ) -> dict:
        """Set up all monitoring for the NBA model."""
        results = {}

        print("🚀 Setting up Lakehouse Monitoring for NBA model...")

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

        # Create monitors
        try:
            # Feature monitoring (data drift)
            feature_monitor_id = self.create_feature_monitor(
                baseline_table=baseline_features,
                email_notifications=email_notifications,
            )
            results["feature_monitor_id"] = feature_monitor_id

            # Inference monitoring (prediction drift)
            inference_monitor_id = self.create_inference_monitor(
                baseline_table=baseline_inference,
                email_notifications=email_notifications,
            )
            results["inference_monitor_id"] = inference_monitor_id

            print("\n✅ All monitors created successfully!")
            print("📊 Dashboards will be available at:")
            current_user = self.spark.sql("SELECT current_user()").collect()[0][0]
            print(f"   Feature Monitor: /Workspace/Users/{current_user}/lakehouse_monitoring/{self.config.feature_table}")
            print(f"   Inference Monitor: /Workspace/Users/{current_user}/lakehouse_monitoring/{self.config.log_table}")

        except Exception as e:
            print(f"❌ Monitor setup failed: {e}")
            raise

        return results

    def _create_baseline_table(self, source_table: str, baseline_suffix: str) -> str:
        """Create a baseline snapshot table."""
        baseline_table = f"{self.config.catalog}.{self.config.schema}.{baseline_suffix}"

        try:
            # Create baseline as current snapshot
            self.spark.sql(f"""
                CREATE OR REPLACE TABLE {baseline_table}
                AS SELECT * FROM {source_table}
                WHERE 1=1  -- Take current data as baseline
            """)

            print(f"📸 Created baseline table: {baseline_table}")
            return baseline_table

        except Exception as e:
            print(f"⚠️  Baseline table creation failed: {e}")
            return None

    def refresh_monitors(self) -> None:
        """Manually refresh all monitors."""
        monitors = self.list_monitors()

        for monitor in monitors:
            try:
                self.workspace_client.quality_monitors.run_refresh(
                    table_name=monitor["table_name"]
                )
                print(f"🔄 Refreshed monitor for {monitor['table_name']}")
            except Exception as e:
                print(f"❌ Failed to refresh {monitor['table_name']}: {e}")

    def list_monitors(self) -> list:
        """List all monitors in the schema."""
        try:
            monitors = self.workspace_client.quality_monitors.list()
            schema_monitors = [
                {
                    "table_name": m.table_name,
                    "monitor_id": m.monitor_id,
                    "status": m.status,
                }
                for m in monitors
                if m.table_name.startswith(f"{self.config.catalog}.{self.config.schema}")
            ]
            return schema_monitors
        except Exception as e:
            print(f"❌ Failed to list monitors: {e}")
            return []

    def delete_monitor(self, table_name: str) -> None:
        """Delete a monitor by table name."""
        try:
            self.workspace_client.quality_monitors.delete(table_name=table_name)
            print(f"🗑️  Deleted monitor for {table_name}")
        except Exception as e:
            print(f"❌ Failed to delete monitor for {table_name}: {e}")

    def get_monitor_status(self) -> dict:
        """Get status of all NBA model monitors."""
        monitors = self.list_monitors()

        status = {
            "total_monitors": len(monitors),
            "monitors": monitors,
            "feature_monitor": None,
            "inference_monitor": None,
        }

        for monitor in monitors:
            if self.config.feature_table in monitor["table_name"]:
                status["feature_monitor"] = monitor
            elif self.config.log_table in monitor["table_name"]:
                status["inference_monitor"] = monitor

        return status

    def generate_monitoring_report(self) -> dict:
        """Generate a comprehensive monitoring status report."""
        status = self.get_monitor_status()

        # Check if tables have required columns
        table_checks = self._validate_monitoring_requirements()

        report = {
            "monitoring_status": status,
            "table_validation": table_checks,
            "recommendations": [],
        }

        # Add recommendations
        if not status["feature_monitor"]:
            report["recommendations"].append("Create feature drift monitor")
        if not status["inference_monitor"]:
            report["recommendations"].append("Create inference monitor")
        if not table_checks["feature_table_valid"]:
            report["recommendations"].append("Add updated_at column to feature table")
        if not table_checks["inference_table_valid"]:
            report["recommendations"].append("Add model_version and inference_timestamp to inference log")

        return report

    def _validate_monitoring_requirements(self) -> dict:
        """Validate that tables have required columns for monitoring."""
        validation = {
            "feature_table_valid": False,
            "inference_table_valid": False,
            "feature_table_columns": [],
            "inference_table_columns": [],
        }

        try:
            # Check feature table
            feature_cols = [col.name for col in self.spark.table(self.config.feature_table_full).schema]
            validation["feature_table_columns"] = feature_cols
            validation["feature_table_valid"] = "updated_at" in feature_cols

            # Check inference log table
            inference_cols = [col.name for col in self.spark.table(self.config.log_table_full).schema]
            validation["inference_table_columns"] = inference_cols
            validation["inference_table_valid"] = all(
                col in inference_cols
                for col in ["model_version", "inference_timestamp"]
            )

        except Exception as e:
            print(f"⚠️  Table validation error: {e}")

        return validation