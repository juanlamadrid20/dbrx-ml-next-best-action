# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Local Environment Setup
```bash
# Setup local Python environment with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip sync pyproject.toml

# Sync dependencies for Databricks compatibility
./sync_requirements.sh
```

### Code Quality
```bash
# Format code (Black formatter)
black src/ notebooks/ tests/ --line-length 100

# Lint code (Ruff)
ruff check src/ notebooks/ tests/

# Run tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m slow
```

### Databricks Asset Bundle Commands
```bash
# Deploy to development environment
dbb deploy -t dev

# Deploy to staging/production
dbb deploy -t staging
dbb deploy -t prod

# Run specific jobs
dbb run data_pipeline_job -t dev
dbb run training_job -t dev
dbb run inference_job -t dev
dbb run monitoring_job -t dev

# Validate bundle configuration
/validate-bundle [target] [--fix] [--verbose]
```

### Specialized Commands
```bash
# Compare model performance with statistical analysis
/compare-models model1 model2 [--metrics=LIST] [--recommendation]

# Promote models between aliases safely
/promote-model catalog.schema.model [--from=staging] [--to=champion] [--dry-run]
```

## Project Architecture

### Core Structure
This is a Next Best Action (NBA) ML system built with Databricks MLOps best practices:

- **src/nba_model/**: Modular Python codebase with separation of concerns
  - `data/`: Data generation and processing modules
  - `features/`: Feature engineering pipelines
  - `training/`: Model training logic
  - `inference/`: Batch and real-time inference
  - `monitoring/`: Model monitoring and drift detection
  - `config.py`: Centralized configuration management

- **notebooks/**: Databricks pipeline notebooks that orchestrate the src/ modules
  - Execute in sequence: 01_data_generation → 02_feature_engineering → 03_model_training → 04_batch_inference → 05_monitoring

### Environment Configuration
Three environments defined in `databricks.yml`:
- **dev**: `juan_dev.ml_nba_demo_dev` - Development and experimentation
- **staging**: `juan_dev.ml_nba_demo_staging` - Testing and validation
- **prod**: `prod.ml_nba_demo` - Production deployment

Each environment has isolated catalogs, schemas, model registries, and experiment paths.

### Package Management Strategy
Dual package management for local development and Databricks compatibility:
- **pyproject.toml**: Primary dependency specification for local development with UV
- **requirements.txt**: Auto-generated for Databricks notebooks via `./sync_requirements.sh`

Workflow: Modify pyproject.toml → Run `./sync_requirements.sh` → Commit both files

### MLOps Components
- **Feature Store**: Databricks Feature Store with Delta table fallback
- **MLflow**: Experiment tracking, model versioning, Unity Catalog registry
- **Asset Bundle**: Multi-environment deployment with scheduled jobs
- **Monitoring**: Data drift detection and performance monitoring

### Job Dependencies
Jobs are designed to run in sequence:
1. `data_pipeline_job`: Generates data and engineers features
2. `training_job`: Trains and registers models
3. `inference_job`: Runs batch inference (scheduled daily at 8 AM)
4. `monitoring_job`: Monitors performance and drift (scheduled daily at 9 AM)

### Model Promotion Workflow
Standard MLOps aliases: `@candidate` → `@staging` → `@champion`
Use `/promote-model` and `/compare-models` commands for safe model promotion with validation.

## Development Notes

### Testing Strategy
Test markers defined in pyproject.toml:
- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests with external services
- `@pytest.mark.slow`: Long-running tests

### Code Style
- Black formatter with 100-character line length
- Ruff linting with comprehensive rule set
- Python 3.9+ compatibility required