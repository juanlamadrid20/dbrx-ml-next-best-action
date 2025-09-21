# validate-bundle

Validates Databricks Asset Bundle configuration and checks for common ML project issues.

## Usage

```
/validate-bundle [target] [options]
```

## Parameters

- `target` (optional): Target environment to validate (dev, staging, prod). Defaults to "dev"
- `--fix`: Automatically fix common issues where possible
- `--verbose`: Show detailed validation output
- `--check-permissions`: Validate catalog/schema permissions (requires connection)
- `--check-volumes`: Validate Unity Catalog volume access

## Description

This command performs comprehensive validation of your Databricks Asset Bundle ML project:

### 🔧 **Bundle Configuration**
- Validates `databricks.yml` syntax and schema
- Checks environment-specific variable definitions
- Verifies job dependencies and cluster configurations
- Validates notebook paths and source references

### 📊 **ML-Specific Checks**
- Schema alignment between training/inference notebooks
- Feature Store table constraints and primary keys
- Model registry configuration and naming conventions
- Volume paths for ML artifacts storage

### 🔒 **Permissions & Access**
- Catalog and schema access permissions
- Unity Catalog volume read/write permissions
- MLflow experiment access rights
- Job execution permissions

### 🐛 **Common Issue Detection**
- Missing environment variables in targets
- Notebook dependency issues (import paths, package requirements)
- Model serving endpoint configuration problems
- Resource naming conflicts across environments

## Examples

```bash
# Basic validation for dev environment
/validate-bundle

# Validate production with permission checks
/validate-bundle prod --check-permissions

# Validate and auto-fix common issues
/validate-bundle dev --fix --verbose

# Full validation including volumes
/validate-bundle staging --check-volumes --verbose
```

## Output

The command provides:
- ✅ **Success indicators** for passing checks
- ⚠️  **Warnings** for potential issues
- ❌ **Errors** that will prevent deployment
- 💡 **Suggestions** for fixes and improvements
- 🔧 **Auto-fix** results when `--fix` is used

## Exit Codes

- `0`: All validations passed
- `1`: Warnings found (deployment likely to succeed)
- `2`: Errors found (deployment will fail)
- `3`: Critical configuration issues