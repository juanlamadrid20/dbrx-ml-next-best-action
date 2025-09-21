# NBA Model MLOps Stack

A production-ready Next Best Action (NBA) recommendation system built with Databricks MLOps best practices.

## Project Structure

```
├── src/nba_model/           # Core ML pipeline code
│   ├── data/                # Data generation and processing
│   ├── features/            # Feature engineering
│   ├── training/            # Model training pipeline
│   ├── inference/           # Batch and real-time inference
│   ├── monitoring/          # Model monitoring and drift detection
│   └── config.py           # Configuration management
├── notebooks/               # Databricks pipeline notebooks
│   ├── 01_data_generation.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   ├── 04_batch_inference.py
│   └── 05_monitoring.py
├── tests/                  # Unit and integration tests
├── resources/              # Static resources
├── databricks.yml          # Asset Bundle configuration (includes all env configs)
├── pyproject.toml          # Python dependencies and project metadata
├── requirements.txt        # Auto-generated for Databricks compatibility
└── sync_requirements.sh    # Sync script for dependency management
```

## Phase 1: Project Setup ✅

This phase migrates the original notebook to a modular MLOps structure:

- ✅ Modular Python codebase with separation of concerns
- ✅ Environment-specific configurations
- ✅ Databricks Asset Bundle setup
- ✅ Pipeline notebooks using modular code
- ✅ Feature Store integration
- ✅ MLflow model management
- ✅ Monitoring and drift detection

## Quick Start

### 1. Local Development Setup

```bash
# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up local environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip sync pyproject.toml

# Sync requirements.txt for Databricks (when dependencies change)
./sync_requirements.sh
```

### 2. Deploy to Development Environment

```bash
# Install Databricks CLI
uv pip install databricks-cli

# Configure authentication
databricks configure

# Deploy to dev environment
dbb deploy -t dev

# Run the complete pipeline
dbb run data_pipeline_job -t dev
dbb run training_job -t dev
dbb run inference_job -t dev
```

### 3. Validate Implementation

Run the notebooks in sequence to validate the implementation:

1. **01_data_generation.py** - Creates synthetic customer data
2. **02_feature_engineering.py** - Engineers features and sets up Feature Store
3. **03_model_training.py** - Trains and registers the NBA model
4. **04_batch_inference.py** - Runs batch inference
5. **05_monitoring.py** - Monitors model performance and drift

## Key Features

### 🔧 Modular Architecture
- Separation of data, features, training, inference, and monitoring
- Environment-specific configurations
- Reusable code components

### ⚡ Modern Package Management
- UV for fast local development and dependency resolution
- Dual compatibility: pyproject.toml for local dev, requirements.txt for Databricks
- Lockfile support for reproducible builds

### 📊 Feature Store Integration
- Databricks Feature Store for feature management
- Automatic fallback to Delta tables
- Feature versioning and lineage

### 🚀 MLflow Integration
- Experiment tracking and model versioning
- Unity Catalog model registry
- Champion/challenger model promotion

### 📈 Model Monitoring
- Data drift detection
- Performance monitoring
- Quality checks and alerts

### 🔄 CI/CD Ready
- Asset bundle configuration
- Multi-environment deployment
- Scheduled job orchestration

## Environment Configuration

The project supports three environments defined in `databricks.yml`:

- **dev**: Development and experimentation (`juan_dev.ml_nba_demo_dev`)
- **staging**: Testing and validation (`juan_dev.ml_nba_demo_staging`)
- **prod**: Production deployment (`prod.ml_nba_demo`)

Each environment has isolated:
- Catalogs and schemas
- Model registries
- Experiment paths
- Job schedules

## Next Steps (Phase 2)

- [ ] Add comprehensive unit and integration tests
- [ ] Implement CI/CD with GitHub Actions
- [ ] Add model serving endpoints
- [ ] Enhance monitoring with custom metrics
- [ ] Add automated retraining triggers

## Validation Checklist

- [ ] Code runs in development workspace
- [ ] Asset bundle deploys successfully
- [ ] All pipeline notebooks execute without errors
- [ ] Model registers to Unity Catalog
- [ ] Batch inference generates recommendations
- [ ] Monitoring detects data quality issues

## Migration from Original Notebook

The original `next-best-action.ipynb` notebook has been fully migrated to this modular structure while preserving all functionality:

- Data generation → `src/nba_model/data/generator.py`
- Feature engineering → `src/nba_model/features/engineering.py`
- Model training → `src/nba_model/training/trainer.py`
- Batch inference → `src/nba_model/inference/batch_inference.py`
- Monitoring → `src/nba_model/monitoring/drift_detection.py`

All original functionality is preserved while gaining MLOps benefits like modularity, testability, and CI/CD readiness.

## Package Management

This project uses **UV** for fast local development and **requirements.txt** for Databricks compatibility:

### Local Development (UV)
```bash
# Install dependencies
uv pip sync pyproject.toml

# Add new dependency
uv add scikit-learn

# Development dependencies
uv pip install -e ".[dev]"
```

### Databricks Compatibility
```bash
# Generate requirements.txt from pyproject.toml
./sync_requirements.sh

# Notebooks use: %pip install -r ../requirements.txt
```

### Dependency Management Workflow
1. **Add/modify dependencies** in `pyproject.toml`
2. **Test locally** with `uv pip sync pyproject.toml`
3. **Sync for Databricks** with `./sync_requirements.sh`
4. **Commit both files** (pyproject.toml and requirements.txt)