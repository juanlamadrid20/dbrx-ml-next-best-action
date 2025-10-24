"""Model monitoring and basic analytics for NBA model."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config import ModelConfig


class ModelMonitor:
    """Monitor model performance and basic analytics.

    Note: Advanced drift detection is now handled by Lakehouse Monitoring.
    This class provides basic reporting and data quality checks.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()

    def analyze_action_distribution(self) -> None:
        """Analyze distribution of recommended actions over time."""
        log_table = self.spark.table(self.config.log_table_full)

        action_dist = (
            log_table.groupBy("log_date", "recommended_action")
            .count()
            .orderBy("log_date", "recommended_action")
        )

        print("Action distribution over time:")
        action_dist.show()

    def generate_model_performance_report(self) -> dict:
        """Generate comprehensive model performance report."""
        log_table = self.spark.table(self.config.log_table_full)

        # Basic statistics
        total_predictions = log_table.count()
        unique_customers = log_table.select("customer_id").distinct().count()

        # Action distribution
        action_counts = log_table.groupBy("recommended_action").count().collect()
        action_distribution = {row["recommended_action"]: row["count"] for row in action_counts}

        # Recent activity
        recent_predictions = log_table.filter(
            F.col("log_date") >= F.date_sub(F.current_date(), 7)
        ).count()

        # Model version distribution
        version_counts = log_table.groupBy("model_version").count().collect()
        version_distribution = {row["model_version"]: row["count"] for row in version_counts}

        report = {
            "total_predictions": total_predictions,
            "unique_customers": unique_customers,
            "action_distribution": action_distribution,
            "version_distribution": version_distribution,
            "recent_predictions_7d": recent_predictions,
            "monitoring_date": F.current_timestamp(),
        }

        print("Model Performance Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")

        return report

    def check_data_quality(self) -> dict:
        """Check basic data quality metrics."""
        features_table = self.spark.table(self.config.feature_table_full)

        quality_metrics = {}

        # Check for nulls in key features
        for feature in self.config.numeric_features:
            null_count = features_table.filter(F.col(feature).isNull()).count()
            total_count = features_table.count()
            null_rate = (null_count / total_count) * 100 if total_count > 0 else 0

            quality_metrics[f"{feature}_null_rate"] = null_rate

            if null_rate > 5:  # Flag if >5% nulls
                print(f"⚠️  High null rate in {feature}: {null_rate:.2f}%")

        # Check for duplicate customer IDs
        total_rows = features_table.count()
        unique_customers = features_table.select("customer_id").distinct().count()
        duplicate_rate = (
            ((total_rows - unique_customers) / total_rows) * 100 if total_rows > 0 else 0
        )

        quality_metrics["duplicate_customer_rate"] = duplicate_rate

        if duplicate_rate > 0:
            print(f"⚠️  Duplicate customers detected: {duplicate_rate:.2f}%")

        print("Data Quality Check:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value:.2f}%")

        return quality_metrics

    def get_lakehouse_monitoring_info(self) -> dict:
        """Get information about Lakehouse Monitoring setup."""
        try:
            from .lakehouse_monitor import LakehouseMonitor

            lh_monitor = LakehouseMonitor(self.config)
            status = lh_monitor.get_monitor_status()
            validation = lh_monitor._validate_monitoring_requirements()

            info = {
                "monitors_configured": status["total_monitors"],
                "feature_monitor_active": status["feature_monitor"] is not None,
                "inference_monitor_active": status["inference_monitor"] is not None,
                "tables_ready_for_monitoring": all([
                    validation["feature_table_valid"],
                    validation["inference_table_valid"]
                ])
            }

            print("Lakehouse Monitoring Status:")
            for key, value in info.items():
                status_emoji = "✅" if value else "❌"
                print(f"  {status_emoji} {key}: {value}")

            return info

        except Exception as e:
            print(f"⚠️  Could not check Lakehouse Monitoring status: {e}")
            return {"error": str(e)}
