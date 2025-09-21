"""Data generation utilities for NBA model."""

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from ..config import ModelConfig


class DataGenerator:
    """Generate synthetic customer data for NBA model training."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.spark = SparkSession.getActiveSession()

    def generate_raw_data(self, n_customers: int = 10_000, random_seed: int = 42) -> None:
        """Generate and save raw customer data."""
        np.random.seed(random_seed)

        # Generate customer data
        customer_id = np.arange(1, n_customers + 1)

        # Demographics
        age = np.random.randint(18, 71, size=n_customers)
        gender = np.random.choice(['M', 'F'], size=n_customers, p=[0.48, 0.52])
        region = np.random.choice(
            ['Northeast', 'Midwest', 'South', 'West'],
            size=n_customers,
            p=[0.20, 0.23, 0.35, 0.22]
        )
        income = np.random.normal(80_000, 25_000, size=n_customers).clip(20_000, 250_000)

        # Transactions (last month)
        num_purchases_last_month = np.random.poisson(lam=2.2, size=n_customers)
        purchase_amount_last_month = (
            np.random.gamma(shape=2.0, scale=120.0, size=n_customers) *
            (1 + (income / 250_000))
        )

        # Browsing (last week)
        browsing_minutes_last_week = np.random.gamma(shape=2.2, scale=25.0, size=n_customers)
        categories = ['Beauty', 'Electronics', 'Fashion', 'Home', 'Grocery', 'Sports', 'Toys']
        top_category = np.random.choice(categories, size=n_customers)

        # Ground-truth NEXT BEST ACTION (rule-based for education)
        best_action = self._generate_best_actions(
            age, purchase_amount_last_month, num_purchases_last_month, browsing_minutes_last_week
        )

        # Create DataFrame
        raw_pdf = pd.DataFrame({
            "customer_id": customer_id,
            "age": age,
            "gender": gender,
            "region": region,
            "income": income.round(2),
            "num_purchases_last_month": num_purchases_last_month,
            "purchase_amount_last_month": purchase_amount_last_month.round(2),
            "browsing_minutes_last_week": browsing_minutes_last_week.round(2),
            "top_category": top_category,
            "best_action": best_action
        })

        # Save to Delta table
        raw_sdf = self.spark.createDataFrame(raw_pdf)
        raw_sdf.write.mode("overwrite").format("delta").saveAsTable(self.config.raw_table_full)

        print(f"Generated {n_customers:,} customers and saved to {self.config.raw_table_full}")

    def _generate_best_actions(self, age, spend, purchases, browse):
        """Generate best action labels based on customer behavior."""
        best_action = []
        for a, s, p, b in zip(age, spend, purchases, browse):
            if s > 600 or p >= 6:
                best_action.append("Email")
            elif b > 200 or a < 28:
                best_action.append("Push")
            elif 28 <= a <= 50:
                best_action.append("SMS")
            else:
                best_action.append("Email")
        return best_action

    def create_schema(self) -> None:
        """Create the database schema if it doesn't exist."""
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.config.catalog}.{self.config.schema}")
        self.spark.sql(f"USE {self.config.catalog}.{self.config.schema}")
        print(f"Using schema: {self.config.catalog}.{self.config.schema}")