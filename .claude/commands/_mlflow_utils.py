#!/usr/bin/env python3
"""
Shared MLflow utilities for Databricks workspace commands.

Provides standardized MLflow client initialization for all slash commands
that interact with models and experiments in the Databricks workspace.
"""

def init_databricks_mlflow_client():
    """
    Initialize MLflow client configured for Databricks workspace.

    Returns:
        MlflowClient: Configured client for Databricks workspace

    Raises:
        Exception: If MLflow client initialization fails
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # Set tracking URI to Databricks workspace (not local MLflow server)
        # This ensures all experiment tracking, run queries, and metric retrieval
        # happen against the Databricks workspace MLflow instance
        mlflow.set_tracking_uri("databricks")

        # Set registry to Unity Catalog for model registry operations
        # This ensures model versioning, aliases, and metadata are stored in UC
        mlflow.set_registry_uri("databricks-uc")

        # Create client with the configured URIs
        client = MlflowClient()

        return client

    except ImportError as e:
        raise Exception(f"MLflow not available. Install with: pip install mlflow[databricks]: {e}")
    except Exception as e:
        raise Exception(f"Failed to initialize Databricks MLflow client: {e}")

def get_databricks_mlflow_config():
    """
    Get current MLflow configuration for Databricks workspace.

    Returns:
        dict: Current MLflow configuration
    """
    try:
        import mlflow

        return {
            "tracking_uri": mlflow.get_tracking_uri(),
            "registry_uri": mlflow.get_registry_uri(),
            "is_databricks_tracking": mlflow.get_tracking_uri() == "databricks",
            "is_uc_registry": mlflow.get_registry_uri() == "databricks-uc"
        }
    except Exception as e:
        return {"error": str(e)}

def validate_databricks_connection():
    """
    Validate connection to Databricks workspace MLflow.

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        client = init_databricks_mlflow_client()

        # Test basic connectivity by listing experiments
        # This will fail if not properly authenticated to Databricks
        experiments = client.search_experiments(max_results=1)

        return True, "Successfully connected to Databricks workspace MLflow"

    except Exception as e:
        return False, f"Failed to connect to Databricks workspace: {e}"