# compare-models

**Self-contained model comparison command - executes independently without additional tools**

Compares model performance between versions/aliases with statistical analysis, A/B testing insights, and promotion recommendations. This command handles all MLflow operations, metric analysis, and statistical computations internally.

## Usage

```
/compare-models [model1] [model2] [options]
```

## Parameters

- `model1` (required): First model to compare (catalog.schema.model@alias or catalog.schema.model:version)
- `model2` (required): Second model to compare (catalog.schema.model@alias or catalog.schema.model:version)
- `--metrics=LIST`: Specific metrics to compare (comma-separated, default: all available)
- `--datasets=LIST`: Datasets to evaluate on (comma-separated, default: test dataset)
- `--confidence=LEVEL`: Confidence level for statistical tests (default: 0.95)
- `--format=TYPE`: Output format (table, json, detailed) (default: detailed)
- `--export=PATH`: Export results to file
- `--visualization`: Generate performance comparison charts
- `--recommendation`: Include promotion recommendation

## Description

**⚠️ IMPORTANT: This is a self-contained command that requires no additional Claude actions or tool calls.**

This command performs comprehensive model comparison analysis with statistical rigor. All MLflow client operations, metric retrieval, statistical testing, and analysis are handled internally by the Python script.

### 🔍 **Model Comparison Analysis**
- Retrieves model metadata and performance metrics from MLflow
- Compares training, validation, and test performance
- Analyzes model characteristics (size, latency, complexity)
- Evaluates feature importance differences
- Computes statistical significance of performance differences

### 📊 **Statistical Testing**
- Performs paired t-tests for metric comparisons
- Calculates confidence intervals for performance differences
- Runs McNemar's test for classification models
- Wilcoxon signed-rank test for non-parametric comparisons
- Effect size analysis (Cohen's d, etc.)

### 🎯 **A/B Testing Insights**
- Champion vs challenger analysis
- Sample size recommendations for A/B tests
- Power analysis for detecting meaningful differences
- Risk assessment for model swaps
- Business impact projections

### 📈 **Performance Metrics**
- **Accuracy Metrics**: Precision, recall, F1-score, accuracy, AUC-ROC
- **Regression Metrics**: MAE, MSE, RMSE, R², MAPE
- **Business Metrics**: Revenue impact, conversion rates, custom KPIs
- **Operational Metrics**: Latency, throughput, resource usage
- **Fairness Metrics**: Bias detection across demographic groups

### 🔬 **Advanced Analysis**
- Feature drift analysis between model versions
- Prediction correlation analysis
- Error pattern comparison
- Confidence score distributions
- Model uncertainty quantification

## Model Specification Formats

### Alias-based Comparison
```bash
# Compare current champion vs staging
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model@staging

# Cross-model comparison
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.alternative_model@candidate
```

### Version-based Comparison
```bash
# Compare specific versions
/compare-models juan_dev.ml_nba_demo.nba_model:5 juan_dev.ml_nba_demo.nba_model:7

# Mixed alias and version
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model:latest
```

## Examples

```bash
# Basic comparison with recommendation
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model@staging --recommendation

# Detailed statistical analysis
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model@challenger --confidence=0.99 --format=detailed

# Focus on specific metrics
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model:1 --metrics=accuracy,f1_score,precision,recall

# Generate visualization and export
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model@1 --visualization --export=model_comparison.json

# A/B testing analysis
/compare-models juan_dev.ml_nba_demo.nba_model@champion juan_dev.ml_nba_demo.nba_model@challenger --datasets=test,validation --format=table

# Cross-environment comparison
/compare-models dev.ml_nba_demo.nba_model@champion prod.ml_nba_demo.nba_model@champion --metrics=accuracy,latency
```

## Statistical Tests Performed

### 🧮 **Performance Comparison Tests**
- **Paired t-test**: For normally distributed metrics
- **Wilcoxon signed-rank**: For non-parametric metrics
- **McNemar's test**: For binary classification comparisons
- **Bootstrap confidence intervals**: For robust uncertainty estimates
- **Permutation tests**: For distribution-free comparisons

### 📊 **Effect Size Analysis**
- **Cohen's d**: Standardized difference between means
- **Glass's delta**: Effect size with control group standard deviation
- **Hedge's g**: Bias-corrected effect size for small samples
- **Cliff's delta**: Non-parametric effect size measure

### ⚖️ **Business Impact Analysis**
- Revenue impact projections
- Risk-adjusted returns
- Cost-benefit analysis
- Implementation effort assessment

## Output

The command provides comprehensive analysis (no additional Claude interaction needed):

- 🎯 **Executive Summary** with promotion recommendation
- 📊 **Metric Comparison Table** with statistical significance indicators
- 📈 **Performance Trends** and improvement/regression analysis
- 🔬 **Statistical Test Results** with p-values and effect sizes
- 💡 **Business Impact Assessment** with risk/reward analysis
- 📋 **Detailed Model Metadata** comparison
- 🎨 **Visualization Charts** (if requested)

**Note**: All output comes directly from the Python script execution. Claude should not perform additional analysis or queries.

## Recommendation Engine

### 🚦 **Promotion Recommendations**
- **PROMOTE**: Statistically significant improvement with acceptable risk
- **CAUTION**: Marginal improvement or mixed results - consider A/B testing
- **REJECT**: Performance regression or insufficient evidence
- **INVESTIGATE**: Unexpected results requiring further analysis

### 🎯 **Recommendation Factors**
- Statistical significance of improvements
- Business impact magnitude
- Risk tolerance and rollback complexity
- Model stability and reliability
- Operational considerations (latency, cost)

## Safety Features

- **Multi-metric Analysis**: Prevents gaming single metrics
- **Statistical Rigor**: Robust significance testing
- **Business Context**: Considers real-world impact
- **Risk Assessment**: Quantifies deployment risks
- **Confidence Intervals**: Uncertainty quantification

## Exit Codes

- `0`: Comparison completed successfully
- `1`: Comparison completed with warnings
- `2`: Model loading or data access errors
- `3`: Statistical analysis failures
- `4`: Invalid model specifications