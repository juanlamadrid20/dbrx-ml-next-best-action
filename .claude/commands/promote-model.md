# promote-model

**Self-contained model promotion command - executes independently without additional tools**

Safely promotes models between aliases with validation, testing, and rollback capabilities. This command handles all MLflow operations, validation checks, and database queries internally.

## Usage

```
/promote-model [model-name] [options]
```

## Parameters

- `model-name` (required): Full Unity Catalog model name (catalog.schema.model)
- `--from=ALIAS`: Source alias to promote from (default: "staging")
- `--to=ALIAS`: Target alias to promote to (default: "champion")
- `--version=N`: Specific version to promote (overrides --from)
- `--force`: Skip safety checks and validation
- `--dry-run`: Show what would be promoted without making changes
- `--rollback`: Rollback the previous promotion
- `--compare-metrics`: Compare performance metrics before promotion
- `--run-tests`: Run validation tests before promotion

## Description

**⚠️ IMPORTANT: This is a self-contained command that requires no additional Claude actions or tool calls.**

This command manages safe model promotion workflows with comprehensive validation. All MLflow client operations, model registry queries, and validation checks are handled internally by the Python script.

### 🔄 **Promotion Workflow**
- Validates source and target model versions exist
- Compares performance metrics between versions
- Runs optional validation tests on staging model
- Creates backup of current champion before promotion
- Promotes model with proper alias management
- Provides rollback capability if issues arise

### 🛡️ **Internal Safety Checks**
- Verifies model registry permissions (via internal MLflow client)
- Checks for schema compatibility between versions (internal validation)
- Validates model signatures match (internal MLflow operations)
- Ensures no active serving endpoints would break (internal checks)
- Compares key performance metrics (internal metric comparison)

### 📊 **Performance Validation**
- Compares test metrics between source and target
- Checks for performance regressions
- Validates model serving readiness
- Runs smoke tests on model endpoints

### 🔙 **Rollback Support**
- Automatically tracks previous champion version
- Quick rollback to previous stable version
- Maintains promotion history for audit trails
- Preserves model lineage and metadata

## Aliases and Promotion Paths

### Standard MLOps Aliases
- `@staging` → `@champion` (default promotion path)
- `@challenger` → `@champion` (A/B test winner promotion)
- `@candidate` → `@staging` (pre-production promotion)

### Custom Promotion Paths
- Version-specific promotions
- Environment-based aliases
- Feature branch model promotion

## Examples

```bash
# Standard staging to champion promotion
/promote-model juan_dev.ml_nba_demo.nba_model

# Promote specific version to champion
/promote-model juan_dev.ml_nba_demo.nba_model --version=5 --to=champion

# A/B test winner promotion
/promote-model juan_dev.ml_nba_demo.nba_model --from=challenger --to=champion

# Safe promotion with validation
/promote-model juan_dev.ml_nba_demo.nba_model --compare-metrics --run-tests

# Dry run to see what would happen
/promote-model juan_dev.ml_nba_demo.nba_model --dry-run

# Emergency rollback
/promote-model juan_dev.ml_nba_demo.nba_model --rollback

# Cross-environment promotion (staging to prod)
/promote-model prod.ml_nba_demo.nba_model --from=staging --to=champion --compare-metrics
```

## Validation Checks

### 🔍 **Pre-Promotion Validation**
- Model version exists and is accessible
- Target alias is valid for the model
- Model schema compatibility
- Performance regression detection
- Serving endpoint compatibility

### 📈 **Metric Comparison**
- Accuracy/precision/recall comparisons
- Latency and throughput validation
- Model size and resource requirements
- Custom business metrics evaluation

### 🧪 **Optional Testing**
- Schema validation tests
- Inference smoke tests
- Load testing on serving endpoints
- Data drift validation

## Output

The command provides complete, detailed feedback (no additional Claude interaction needed):
- 🔄 **Promotion progress** with step-by-step updates
- 📊 **Metric comparisons** between source and target (computed internally)
- ✅ **Validation results** for all safety checks (performed internally)
- 🎯 **Promotion summary** with version details
- 🔙 **Rollback instructions** if needed

**Note**: All output comes directly from the Python script execution. Claude should not perform additional validation or queries.

## Safety Features

- **Backup Creation**: Automatically creates backup aliases
- **Gradual Rollout**: Supports canary deployments
- **Health Monitoring**: Tracks model performance post-promotion
- **Automatic Rollback**: Triggers rollback on critical failures
- **Audit Trail**: Logs all promotion activities

## Exit Codes

- `0`: Promotion successful
- `1`: Promotion completed with warnings
- `2`: Promotion failed validation checks
- `3`: Model or alias not found
- `4`: Permission or access errors