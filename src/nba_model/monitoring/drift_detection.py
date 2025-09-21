"""Model monitoring and drift detection for NBA model."""

from pyspark.sql import SparkSession, functions as F
from ..config import ModelConfig


class ModelMonitor:
    """Monitor model performance and data drift."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()

    def analyze_action_distribution(self) -> None:
        """Analyze distribution of recommended actions over time."""
        log_table = self.spark.table(self.config.log_table_full)

        action_dist = (
            log_table
            .groupBy("log_date", "recommended_action")
            .count()
            .orderBy("log_date", "recommended_action")
        )

        print("Action distribution over time:")
        action_dist.show()

    def detect_feature_drift(self) -> None:
        """Detect feature drift by comparing means across time periods."""
        log_tbl = self.spark.table(self.config.log_table_full)

        # Get available dates
        dates = [
            r["log_date"] for r in
            log_tbl.select("log_date").distinct().orderBy("log_date").collect()
        ]

        if len(dates) < 2:
            print("Not enough days logged to compute drift deltas")
            return

        baseline_date = dates[0]
        current_date = dates[-1]

        # Calculate means for baseline and current periods
        baseline_means = self._calculate_feature_means(log_tbl, baseline_date)
        current_means = self._calculate_feature_means(log_tbl, current_date)

        # Compute drift metrics
        drift_rows = []
        for feature in self.config.numeric_features:
            baseline_val = float(baseline_means.get(feature, 0) or 0)
            current_val = float(current_means.get(feature, 0) or 0)
            delta = current_val - baseline_val
            pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

            drift_rows.append((feature, baseline_val, current_val, delta, pct_change))

        # Create drift summary
        drift_df = self.spark.createDataFrame(
            drift_rows,
            ["feature", "baseline_mean", "current_mean", "delta", "pct_change"]
        )

        print("Feature drift analysis:")
        drift_df.orderBy(F.abs(F.col("pct_change")).desc()).show()

        # Flag significant drift (>10% change)
        significant_drift = drift_df.filter(F.abs(F.col("pct_change")) > 10)
        if significant_drift.count() > 0:
            print("⚠️  Significant drift detected:")
            significant_drift.show()
        else:
            print("✅ No significant drift detected")

    def _calculate_feature_means(self, log_tbl, date_filter):
        """Calculate feature means for a specific date."""
        filtered_data = log_tbl.filter(F.col("log_date") == date_filter)
        means = (
            filtered_data
            .select([F.mean(c).alias(c) for c in self.config.numeric_features])
            .collect()[0]
            .asDict()
        )
        return means

    def generate_model_performance_report(self) -> dict:
        """Generate comprehensive model performance report."""
        log_table = self.spark.table(self.config.log_table_full)

        # Basic statistics
        total_predictions = log_table.count()
        unique_customers = log_table.select("customer_id").distinct().count()

        # Action distribution
        action_counts = (
            log_table
            .groupBy("recommended_action")
            .count()
            .collect()
        )

        action_distribution = {row["recommended_action"]: row["count"] for row in action_counts}

        # Recent activity
        recent_predictions = (
            log_table
            .filter(F.col("log_date") >= F.date_sub(F.current_date(), 7))
            .count()
        )

        report = {
            "total_predictions": total_predictions,
            "unique_customers": unique_customers,
            "action_distribution": action_distribution,
            "recent_predictions_7d": recent_predictions,
            "monitoring_date": F.current_timestamp()
        }

        print("Model Performance Report:")
        for key, value in report.items():
            print(f"  {key}: {value}")

        return report

    def check_data_quality(self) -> dict:
        """Check data quality metrics."""
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
        duplicate_rate = ((total_rows - unique_customers) / total_rows) * 100 if total_rows > 0 else 0

        quality_metrics["duplicate_customer_rate"] = duplicate_rate

        if duplicate_rate > 0:
            print(f"⚠️  Duplicate customers detected: {duplicate_rate:.2f}%")

        print("Data Quality Check:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value:.2f}%")

        return quality_metrics