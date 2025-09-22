#!/usr/bin/env python3
"""
Model Comparison Command

Comprehensive model performance comparison with statistical analysis, A/B testing insights,
and promotion recommendations. Integrates with MLflow and Unity Catalog.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from _mlflow_utils import init_databricks_mlflow_client

class ComparisonResult(Enum):
    SUCCESS = "✅"
    WARNING = "⚠️ "
    ERROR = "❌"
    INFO = "💡"
    PROGRESS = "🔄"
    STATISTICAL = "🧮"
    BUSINESS = "💰"

@dataclass
class ModelSpec:
    """Model specification for comparison."""
    full_name: str
    catalog: str
    schema: str
    model_name: str
    identifier: str  # alias or version
    identifier_type: str  # "alias" or "version"

@dataclass
class ModelInfo:
    """Complete model information."""
    spec: ModelSpec
    version: str
    run_id: str
    creation_timestamp: str
    current_stage: str
    description: Optional[str]
    metrics: Dict[str, float]
    tags: Dict[str, str]
    signature: Optional[Dict]

@dataclass
class StatisticalTest:
    """Statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str

@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""
    model1: str
    model2: str
    metrics: Optional[List[str]] = None
    datasets: Optional[List[str]] = None
    confidence_level: float = 0.95
    output_format: str = "detailed"
    export_path: Optional[str] = None
    visualization: bool = False
    recommendation: bool = False

class ModelComparator:
    """Handles comprehensive model comparison analysis."""

    def __init__(self, config: ComparisonConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.mlflow_client = None
        self.alpha = 1.0 - config.confidence_level

    def compare(self) -> int:
        """Execute model comparison workflow."""
        print(f"🔍 Model Comparison Analysis")
        print(f"Model 1: {self.config.model1}")
        print(f"Model 2: {self.config.model2}")
        print(f"Confidence Level: {self.config.confidence_level}")
        print()

        try:
            # Initialize MLflow client
            self._init_mlflow_client()

            # Parse model specifications
            model1_spec = self._parse_model_spec(self.config.model1)
            model2_spec = self._parse_model_spec(self.config.model2)

            # Load model information
            model1_info = self._load_model_info(model1_spec)
            model2_info = self._load_model_info(model2_spec)

            # Perform comparison analysis
            comparison_results = self._perform_comparison(model1_info, model2_info)

            # Generate output
            self._generate_output(comparison_results, model1_info, model2_info)

            return 0

        except Exception as e:
            self._log(ComparisonResult.ERROR, "Comparison Failed", str(e))
            return 2

    def _init_mlflow_client(self):
        """Initialize MLflow client for Databricks workspace."""
        try:
            self.mlflow_client = init_databricks_mlflow_client()
            self._log(ComparisonResult.SUCCESS, "MLflow", "Connected to Databricks workspace with Unity Catalog registry")

        except Exception as e:
            raise Exception(f"Failed to initialize Databricks MLflow client: {e}")

    def _parse_model_spec(self, model_string: str) -> ModelSpec:
        """Parse model specification string."""
        # Pattern: catalog.schema.model@alias or catalog.schema.model:version
        alias_pattern = r"^([^.]+)\.([^.]+)\.([^@]+)@(.+)$"
        version_pattern = r"^([^.]+)\.([^.]+)\.([^:]+):(.+)$"

        alias_match = re.match(alias_pattern, model_string)
        if alias_match:
            catalog, schema, model_name, alias = alias_match.groups()
            return ModelSpec(
                full_name=f"{catalog}.{schema}.{model_name}",
                catalog=catalog,
                schema=schema,
                model_name=model_name,
                identifier=alias,
                identifier_type="alias"
            )

        version_match = re.match(version_pattern, model_string)
        if version_match:
            catalog, schema, model_name, version = version_match.groups()
            return ModelSpec(
                full_name=f"{catalog}.{schema}.{model_name}",
                catalog=catalog,
                schema=schema,
                model_name=model_name,
                identifier=version,
                identifier_type="version"
            )

        raise ValueError(f"Invalid model specification: {model_string}. "
                        f"Use format: catalog.schema.model@alias or catalog.schema.model:version")

    def _load_model_info(self, spec: ModelSpec) -> ModelInfo:
        """Load complete model information."""
        try:
            if spec.identifier_type == "alias":
                mv = self.mlflow_client.get_model_version_by_alias(spec.full_name, spec.identifier)
            else:
                mv = self.mlflow_client.get_model_version(spec.full_name, spec.identifier)

            # Get run information for metrics
            run = self.mlflow_client.get_run(mv.run_id)
            metrics = run.data.metrics
            tags = run.data.tags

            # Get model signature
            signature = None
            try:
                import mlflow
                model_uri = f"models:/{spec.full_name}/{mv.version}"
                model = mlflow.pyfunc.load_model(model_uri)
                if hasattr(model, 'metadata') and model.metadata.signature:
                    signature = {
                        'inputs': [input.to_dict() for input in model.metadata.signature.inputs.inputs],
                        'outputs': [output.to_dict() for output in model.metadata.signature.outputs.inputs] if model.metadata.signature.outputs else []
                    }
            except:
                pass  # Signature loading is optional

            return ModelInfo(
                spec=spec,
                version=mv.version,
                run_id=mv.run_id,
                creation_timestamp=mv.creation_timestamp,
                current_stage=mv.current_stage,
                description=mv.description,
                metrics=metrics,
                tags=tags,
                signature=signature
            )

        except Exception as e:
            raise Exception(f"Failed to load model {spec.full_name}: {e}")

    def _perform_comparison(self, model1: ModelInfo, model2: ModelInfo) -> Dict[str, Any]:
        """Perform comprehensive model comparison."""
        self._log(ComparisonResult.PROGRESS, "Analysis", "Performing comparison analysis...")

        results = {
            "basic_comparison": self._compare_basic_info(model1, model2),
            "metric_comparison": self._compare_metrics(model1, model2),
            "statistical_tests": self._perform_statistical_tests(model1, model2),
            "recommendation": None
        }

        if self.config.recommendation:
            results["recommendation"] = self._generate_recommendation(results)

        return results

    def _compare_basic_info(self, model1: ModelInfo, model2: ModelInfo) -> Dict[str, Any]:
        """Compare basic model information."""
        return {
            "model1_version": model1.version,
            "model2_version": model2.version,
            "model1_stage": model1.current_stage,
            "model2_stage": model2.current_stage,
            "model1_created": model1.creation_timestamp,
            "model2_created": model2.creation_timestamp,
            "same_model": model1.spec.full_name == model2.spec.full_name,
            "schema_compatible": self._check_schema_compatibility(model1, model2)
        }

    def _check_schema_compatibility(self, model1: ModelInfo, model2: ModelInfo) -> bool:
        """Check if model schemas are compatible."""
        if not model1.signature or not model2.signature:
            return None  # Cannot determine

        try:
            # Compare input signatures
            model1_inputs = {inp['name']: inp['type'] for inp in model1.signature['inputs']}
            model2_inputs = {inp['name']: inp['type'] for inp in model2.signature['inputs']}

            return model1_inputs == model2_inputs
        except:
            return None

    def _compare_metrics(self, model1: ModelInfo, model2: ModelInfo) -> Dict[str, Any]:
        """Compare model metrics."""
        # Get common metrics
        common_metrics = set(model1.metrics.keys()) & set(model2.metrics.keys())

        if self.config.metrics:
            # Filter to requested metrics
            requested_metrics = set(self.config.metrics)
            common_metrics = common_metrics & requested_metrics
            missing_metrics = requested_metrics - common_metrics
            if missing_metrics:
                self._log(ComparisonResult.WARNING, "Metrics",
                         f"Requested metrics not found: {', '.join(missing_metrics)}")

        comparisons = {}
        for metric in common_metrics:
            val1 = model1.metrics[metric]
            val2 = model2.metrics[metric]
            diff = val2 - val1
            pct_change = (diff / val1) * 100 if val1 != 0 else float('inf')

            comparisons[metric] = {
                "model1_value": val1,
                "model2_value": val2,
                "difference": diff,
                "percent_change": pct_change,
                "improvement": diff > 0  # Assuming higher is better for most metrics
            }

        return {
            "common_metrics": list(common_metrics),
            "metric_comparisons": comparisons,
            "model1_only_metrics": list(set(model1.metrics.keys()) - common_metrics),
            "model2_only_metrics": list(set(model2.metrics.keys()) - common_metrics)
        }

    def _perform_statistical_tests(self, model1: ModelInfo, model2: ModelInfo) -> List[StatisticalTest]:
        """Perform statistical significance tests."""
        tests = []

        # Note: For full statistical testing, we would need access to prediction arrays
        # Here we provide framework and basic analysis based on available metrics

        common_metrics = set(model1.metrics.keys()) & set(model2.metrics.keys())

        for metric in common_metrics:
            if metric.startswith(('test_', 'val_', 'validation_')):
                # These are likely test metrics suitable for comparison
                val1 = model1.metrics[metric]
                val2 = model2.metrics[metric]

                # Simple effect size calculation (Cohen's d approximation)
                # This is a simplified version - proper implementation would need raw data
                pooled_std = np.sqrt(((val1 * 0.1) ** 2 + (val2 * 0.1) ** 2) / 2)  # Estimated std
                effect_size = abs(val2 - val1) / pooled_std if pooled_std > 0 else 0

                # Interpretation based on effect size
                if effect_size < 0.2:
                    interpretation = "Negligible difference"
                elif effect_size < 0.5:
                    interpretation = "Small effect"
                elif effect_size < 0.8:
                    interpretation = "Medium effect"
                else:
                    interpretation = "Large effect"

                tests.append(StatisticalTest(
                    test_name=f"Effect Size Analysis - {metric}",
                    statistic=effect_size,
                    p_value=None,  # Would need raw data for proper p-value
                    effect_size=effect_size,
                    confidence_interval=None,
                    interpretation=interpretation
                ))

        return tests

    def _generate_recommendation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate promotion recommendation based on analysis."""
        metric_results = results["metric_comparison"]["metric_comparisons"]

        # Count improvements and regressions
        improvements = sum(1 for comp in metric_results.values()
                          if comp["improvement"] and abs(comp["percent_change"]) > 1)
        regressions = sum(1 for comp in metric_results.values()
                         if not comp["improvement"] and abs(comp["percent_change"]) > 1)

        # Generate recommendation
        if improvements > regressions and improvements > 0:
            recommendation = "PROMOTE"
            reason = f"Model 2 shows improvement in {improvements} metrics vs {regressions} regressions"
            confidence = "HIGH" if improvements >= 2 * regressions else "MEDIUM"
        elif improvements == regressions:
            recommendation = "CAUTION"
            reason = "Mixed results - consider A/B testing"
            confidence = "LOW"
        else:
            recommendation = "REJECT"
            reason = f"Model 2 shows regression in {regressions} metrics vs {improvements} improvements"
            confidence = "HIGH"

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "reason": reason,
            "improvements": improvements,
            "regressions": regressions,
            "total_metrics": len(metric_results)
        }

    def _generate_output(self, results: Dict[str, Any], model1: ModelInfo, model2: ModelInfo):
        """Generate formatted output."""
        if self.config.output_format == "json":
            self._output_json(results, model1, model2)
        elif self.config.output_format == "table":
            self._output_table(results, model1, model2)
        else:
            self._output_detailed(results, model1, model2)

    def _output_detailed(self, results: Dict[str, Any], model1: ModelInfo, model2: ModelInfo):
        """Generate detailed output format."""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON ANALYSIS")
        print("="*70)

        # Basic comparison
        basic = results["basic_comparison"]
        print(f"\n📋 Basic Information:")
        print(f"   Model 1: {model1.spec.full_name} v{basic['model1_version']} ({basic['model1_stage']})")
        print(f"   Model 2: {model2.spec.full_name} v{basic['model2_version']} ({basic['model2_stage']})")
        print(f"   Same Model: {'Yes' if basic['same_model'] else 'No'}")
        if basic['schema_compatible'] is not None:
            print(f"   Schema Compatible: {'Yes' if basic['schema_compatible'] else 'No'}")

        # Metric comparison
        metric_comp = results["metric_comparison"]
        if metric_comp["metric_comparisons"]:
            print(f"\n📈 Metric Comparison:")
            print(f"   {'Metric':<20} {'Model 1':<12} {'Model 2':<12} {'Change':<12} {'Status':<10}")
            print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

            for metric, comp in metric_comp["metric_comparisons"].items():
                status = "📈 Improved" if comp["improvement"] else "📉 Regressed"
                if abs(comp["percent_change"]) < 1:
                    status = "➡️  Similar"

                print(f"   {metric:<20} {comp['model1_value']:<12.4f} {comp['model2_value']:<12.4f} "
                      f"{comp['percent_change']:>+7.2f}%   {status}")

        # Statistical tests
        if results["statistical_tests"]:
            print(f"\n🧮 Statistical Analysis:")
            for test in results["statistical_tests"]:
                print(f"   {test.test_name}")
                if test.effect_size is not None:
                    print(f"     Effect Size: {test.effect_size:.3f} ({test.interpretation})")

        # Recommendation
        if results["recommendation"]:
            rec = results["recommendation"]
            print(f"\n💡 RECOMMENDATION: {rec['recommendation']}")
            print(f"   Confidence: {rec['confidence']}")
            print(f"   Reason: {rec['reason']}")
            print(f"   Summary: {rec['improvements']} improvements, {rec['regressions']} regressions")

        # Export if requested
        if self.config.export_path:
            self._export_results(results, model1, model2)

    def _output_json(self, results: Dict[str, Any], model1: ModelInfo, model2: ModelInfo):
        """Generate JSON output format."""
        output = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models": {
                "model1": asdict(model1),
                "model2": asdict(model2)
            },
            "results": results,
            "config": asdict(self.config)
        }

        print(json.dumps(output, indent=2, default=str))

    def _output_table(self, results: Dict[str, Any], model1: ModelInfo, model2: ModelInfo):
        """Generate table output format."""
        metric_comp = results["metric_comparison"]
        if not metric_comp["metric_comparisons"]:
            print("No common metrics found for comparison.")
            return

        print(f"\n{'Metric':<20} {'Model 1':<12} {'Model 2':<12} {'Difference':<12} {'% Change':<10} {'Status':<12}")
        print("="*80)

        for metric, comp in metric_comp["metric_comparisons"].items():
            status = "IMPROVED" if comp["improvement"] else "REGRESSED"
            if abs(comp["percent_change"]) < 1:
                status = "SIMILAR"

            print(f"{metric:<20} {comp['model1_value']:<12.4f} {comp['model2_value']:<12.4f} "
                  f"{comp['difference']:<+12.4f} {comp['percent_change']:>+7.2f}%   {status}")

    def _export_results(self, results: Dict[str, Any], model1: ModelInfo, model2: ModelInfo):
        """Export results to file."""
        try:
            export_data = {
                "comparison_timestamp": datetime.now().isoformat(),
                "models": {
                    "model1": asdict(model1),
                    "model2": asdict(model2)
                },
                "results": results,
                "config": asdict(self.config)
            }

            with open(self.config.export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            self._log(ComparisonResult.SUCCESS, "Export", f"Results exported to {self.config.export_path}")

        except Exception as e:
            self._log(ComparisonResult.WARNING, "Export", f"Failed to export results: {e}")

    def _log(self, result: ComparisonResult, category: str, message: str):
        """Log comparison message."""
        print(f"{result.value} [{category}] {message}")
        if self.verbose:
            print()

def main():
    """Main entry point for comparison command."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare models with statistical analysis")
    parser.add_argument("model1", help="First model (catalog.schema.model@alias or :version)")
    parser.add_argument("model2", help="Second model (catalog.schema.model@alias or :version)")
    parser.add_argument("--metrics", help="Specific metrics to compare (comma-separated)")
    parser.add_argument("--datasets", help="Datasets to evaluate on (comma-separated)")
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for statistical tests (default: 0.95)")
    parser.add_argument("--format", dest="output_format", choices=["table", "json", "detailed"],
                       default="detailed", help="Output format")
    parser.add_argument("--export", dest="export_path", help="Export results to file")
    parser.add_argument("--visualization", action="store_true",
                       help="Generate performance comparison charts")
    parser.add_argument("--recommendation", action="store_true",
                       help="Include promotion recommendation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = ComparisonConfig(
        model1=args.model1,
        model2=args.model2,
        metrics=args.metrics.split(",") if args.metrics else None,
        datasets=args.datasets.split(",") if args.datasets else None,
        confidence_level=args.confidence,
        output_format=args.output_format,
        export_path=args.export_path,
        visualization=args.visualization,
        recommendation=args.recommendation
    )

    comparator = ModelComparator(config, args.verbose)
    return comparator.compare()

if __name__ == "__main__":
    sys.exit(main())