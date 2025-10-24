"""Feature engineering for NBA model."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config import ModelConfig


class FeatureEngineer:
    """Feature engineering pipeline for NBA model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()

    def create_features(self) -> None:
        """Create engineered features from raw data."""
        raw = self.spark.table(self.config.raw_table_full)

        features_sdf = (
            raw.withColumn("gender_ix", F.when(F.col("gender") == "M", 1).otherwise(0))
            .withColumn(
                "age_bucket",
                F.when(F.col("age") < 28, "young")
                .when((F.col("age") >= 28) & (F.col("age") <= 50), "mid")
                .otherwise("senior"),
            )
            .withColumn(
                "avg_purchase_value",
                F.when(
                    F.col("num_purchases_last_month") > 0,
                    F.col("purchase_amount_last_month") / F.col("num_purchases_last_month"),
                ).otherwise(0.0),
            )
            .withColumn("is_high_spender", (F.col("purchase_amount_last_month") > 600).cast("int"))
            .select(
                "customer_id",
                "age",
                "gender_ix",
                "income",
                "num_purchases_last_month",
                "purchase_amount_last_month",
                "avg_purchase_value",
                "browsing_minutes_last_week",
                "is_high_spender",
                "region",
                "top_category",
                "age_bucket",
            )
        )

        # Add timestamp for monitoring
        final_features = features_sdf.withColumn("updated_at", F.current_timestamp())

        # Save engineered features with schema evolution enabled
        final_features.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(
            self.config.feature_table_full
        )
        print(f"Created features table: {self.config.feature_table_full}")

        # Ensure proper constraints for Feature Store
        self._ensure_feature_constraints()

    def _ensure_feature_constraints(self) -> None:
        """Ensure feature table has proper constraints for Feature Store."""
        feat_df = self.spark.table(self.config.feature_table_full)

        # Check for duplicates and nulls
        dups = feat_df.groupBy("customer_id").count().filter("count > 1").count()
        nulls = feat_df.filter(F.col("customer_id").isNull()).count()

        if dups > 0 or nulls > 0:
            print("Fixing duplicates/nulls...")
            fixed = feat_df.filter(F.col("customer_id").isNotNull()).dropDuplicates(["customer_id"])
            fixed.write.mode("overwrite").format("delta").saveAsTable(
                self.config.feature_table_full
            )

        # Set constraints
        try:
            self.spark.sql(
                f"ALTER TABLE {self.config.feature_table_full} ALTER COLUMN customer_id SET NOT NULL"
            )
            constraint_name = f"{self.config.feature_table}_pk"
            self.spark.sql(
                f"ALTER TABLE {self.config.feature_table_full} ADD CONSTRAINT {constraint_name} PRIMARY KEY (customer_id)"
            )
            print("Added constraints for Feature Store compatibility")
        except Exception as e:
            print(f"Constraint setup warning: {e}")

    def setup_feature_store(self) -> str:
        """Set up Feature Store integration."""
        fs_mode = "delta"
        try:
            from databricks.feature_store import FeatureStoreClient

            fs = FeatureStoreClient()

            # Create Feature Store table entry
            try:
                fs.create_table(
                    name=self.config.feature_table_full,
                    primary_keys=["customer_id"],
                    schema=self.spark.table(self.config.feature_table_full).schema,
                    description="Customer features for NBA model",
                )
                print("Feature Store table entry created")
            except Exception as e:
                print(f"Feature Store table may already exist: {e}")

            # Write to Feature Store (exclude updated_at for Feature Store compatibility)
            feature_store_df = self.spark.table(self.config.feature_table_full).drop("updated_at")
            fs.write_table(
                name=self.config.feature_table_full,
                df=feature_store_df,
                mode="overwrite",
            )
            fs_mode = "feature_store"
            print(f"Feature Store write successful: {self.config.feature_table_full}")

        except Exception as e:
            print(f"Feature Store not available, using Delta: {e}")

        return fs_mode
