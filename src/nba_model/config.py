"""Configuration management for NBA model pipeline."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the NBA model pipeline."""

    # Environment configuration
    catalog: str = "juan_dev"
    schema: str = "ml"

    # Table names
    feature_table: str = "nba_customer_features"
    raw_table: str = "nba_customers_raw"
    rec_table: str = "nba_recommendations"
    log_table: str = "nba_inference_log"

    # Model configuration
    model_base: str = "nba_model"
    experiment_path: Optional[str] = None

    # Feature configuration
    numeric_features: list = None
    categorical_features: list = None

    def __post_init__(self):
        """Initialize derived properties and default values."""
        if self.numeric_features is None:
            self.numeric_features = [
                "age",
                "gender_ix",
                "income",
                "num_purchases_last_month",
                "purchase_amount_last_month",
                "avg_purchase_value",
                "browsing_minutes_last_week",
                "is_high_spender",
            ]

        if self.categorical_features is None:
            self.categorical_features = ["region", "top_category", "age_bucket"]

        if self.experiment_path is None:
            self.experiment_path = "/Users/juan.lamadrid@databricks.com/experiments/nba-model"

    @property
    def feature_table_full(self) -> str:
        """Full table name for features."""
        return f"{self.catalog}.{self.schema}.{self.feature_table}"

    @property
    def raw_table_full(self) -> str:
        """Full table name for raw data."""
        return f"{self.catalog}.{self.schema}.{self.raw_table}"

    @property
    def rec_table_full(self) -> str:
        """Full table name for recommendations."""
        return f"{self.catalog}.{self.schema}.{self.rec_table}"

    @property
    def log_table_full(self) -> str:
        """Full table name for inference logs."""
        return f"{self.catalog}.{self.schema}.{self.log_table}"

    @property
    def uc_model_name(self) -> str:
        """Unity Catalog model name."""
        return f"{self.catalog}.{self.schema}.{self.model_base}"


def get_config() -> ModelConfig:
    """Get the model configuration."""
    return ModelConfig()
