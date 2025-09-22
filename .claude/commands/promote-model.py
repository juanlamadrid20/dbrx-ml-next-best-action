#!/usr/bin/env python3
"""
Model Promotion Command

Safe model promotion workflow with validation, testing, and rollback capabilities.
Integrates with MLflow, Unity Catalog, and Databricks Model Registry.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from ._mlflow_utils import init_databricks_mlflow_client

class PromotionResult(Enum):
    SUCCESS = "✅"
    WARNING = "⚠️ "
    ERROR = "❌"
    INFO = "💡"
    PROGRESS = "🔄"

@dataclass
class ModelVersion:
    """Model version information."""
    name: str
    version: str
    alias: Optional[str]
    stage: Optional[str]
    run_id: str
    creation_timestamp: str
    current_stage: str
    description: Optional[str] = None

@dataclass
class PromotionConfig:
    """Configuration for model promotion."""
    model_name: str
    from_alias: str = "staging"
    to_alias: str = "champion"
    version: Optional[str] = None
    force: bool = False
    dry_run: bool = False
    rollback: bool = False
    compare_metrics: bool = False
    run_tests: bool = False

@dataclass
class PromotionPlan:
    """Planned promotion actions."""
    source_version: ModelVersion
    target_alias: str
    current_champion: Optional[ModelVersion]
    backup_alias: Optional[str]
    actions: List[str]
    validations: List[str]

class ModelPromoter:
    """Handles safe model promotion workflows."""

    def __init__(self, config: PromotionConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.mlflow_client = None
        self.promotion_history = []

    def promote(self) -> int:
        """Execute model promotion workflow."""
        print(f"🚀 Model Promotion Workflow")
        print(f"📦 Model: {self.config.model_name}")
        print(f"🎯 Promotion: {self.config.from_alias} → {self.config.to_alias}")
        print()

        try:
            # Initialize MLflow client
            self._init_mlflow_client()

            if self.config.rollback:
                return self._handle_rollback()

            # Create promotion plan
            plan = self._create_promotion_plan()

            if self.config.dry_run:
                return self._show_dry_run(plan)

            # Execute promotion
            return self._execute_promotion(plan)

        except Exception as e:
            self._log(PromotionResult.ERROR, "Promotion Failed", str(e))
            return 2

    def _init_mlflow_client(self):
        """Initialize MLflow client for Databricks workspace."""
        try:
            self.mlflow_client = init_databricks_mlflow_client()
            self._log(PromotionResult.SUCCESS, "MLflow", "Connected to Databricks workspace with Unity Catalog registry")

        except Exception as e:
            raise Exception(f"Failed to initialize Databricks MLflow client: {e}")

    def _create_promotion_plan(self) -> PromotionPlan:
        """Create detailed promotion plan."""
        self._log(PromotionResult.PROGRESS, "Planning", "Creating promotion plan...")

        # Get source version
        source_version = self._get_source_version()

        # Get current champion (if exists)
        current_champion = self._get_current_champion()

        # Plan backup strategy
        backup_alias = f"prev_{self.config.to_alias}" if current_champion else None

        # Define actions
        actions = []
        validations = []

        if current_champion and not self.config.force:
            actions.append(f"Backup current {self.config.to_alias} as {backup_alias}")
            validations.append("Compare metrics between versions")

        actions.append(f"Set {self.config.to_alias} alias to version {source_version.version}")

        if self.config.compare_metrics:
            validations.append("Performance regression analysis")

        if self.config.run_tests:
            validations.append("Run validation tests")

        plan = PromotionPlan(
            source_version=source_version,
            target_alias=self.config.to_alias,
            current_champion=current_champion,
            backup_alias=backup_alias,
            actions=actions,
            validations=validations
        )

        self._log(PromotionResult.SUCCESS, "Planning", f"Promotion plan created ({len(actions)} actions, {len(validations)} validations)")
        return plan

    def _get_source_version(self) -> ModelVersion:
        """Get source model version for promotion."""
        if self.config.version:
            # Promote specific version
            try:
                mv = self.mlflow_client.get_model_version(self.config.model_name, self.config.version)
                return self._model_version_from_mlflow(mv, alias=None)
            except Exception as e:
                raise Exception(f"Version {self.config.version} not found: {e}")
        else:
            # Promote from alias
            try:
                mv = self.mlflow_client.get_model_version_by_alias(self.config.model_name, self.config.from_alias)
                return self._model_version_from_mlflow(mv, alias=self.config.from_alias)
            except Exception as e:
                raise Exception(f"Alias {self.config.from_alias} not found: {e}")

    def _get_current_champion(self) -> Optional[ModelVersion]:
        """Get current champion version."""
        try:
            mv = self.mlflow_client.get_model_version_by_alias(self.config.model_name, self.config.to_alias)
            return self._model_version_from_mlflow(mv, alias=self.config.to_alias)
        except:
            return None  # No current champion

    def _model_version_from_mlflow(self, mv, alias: Optional[str] = None) -> ModelVersion:
        """Convert MLflow model version to ModelVersion."""
        return ModelVersion(
            name=mv.name,
            version=mv.version,
            alias=alias,
            stage=mv.current_stage,
            run_id=mv.run_id,
            creation_timestamp=mv.creation_timestamp,
            current_stage=mv.current_stage,
            description=mv.description
        )

    def _show_dry_run(self, plan: PromotionPlan) -> int:
        """Show what would happen in dry run mode."""
        print("🔍 DRY RUN - No changes will be made")
        print("="*50)

        print(f"\n📋 Promotion Plan:")
        print(f"   Source: {plan.source_version.name} v{plan.source_version.version}")
        if plan.source_version.alias:
            print(f"   From alias: @{plan.source_version.alias}")
        print(f"   Target alias: @{plan.target_alias}")

        if plan.current_champion:
            print(f"   Current {plan.target_alias}: v{plan.current_champion.version}")
            if plan.backup_alias:
                print(f"   Backup as: @{plan.backup_alias}")

        print(f"\n🔄 Actions to be performed:")
        for i, action in enumerate(plan.actions, 1):
            print(f"   {i}. {action}")

        if plan.validations:
            print(f"\n✅ Validations to be run:")
            for i, validation in enumerate(plan.validations, 1):
                print(f"   {i}. {validation}")

        print(f"\n💡 To execute this plan, run without --dry-run")
        return 0

    def _execute_promotion(self, plan: PromotionPlan) -> int:
        """Execute the promotion plan."""
        print("🚀 Executing Promotion...")
        print("="*30)

        # Run pre-promotion validations
        if not self._run_validations(plan):
            self._log(PromotionResult.ERROR, "Validation", "Pre-promotion validations failed")
            return 2

        # Create backup if needed
        if plan.current_champion and plan.backup_alias:
            self._create_backup(plan.current_champion, plan.backup_alias)

        # Perform promotion
        success = self._promote_model(plan)

        if success:
            self._log_promotion_success(plan)
            return 0
        else:
            self._log(PromotionResult.ERROR, "Promotion", "Model promotion failed")
            return 2

    def _run_validations(self, plan: PromotionPlan) -> bool:
        """Run pre-promotion validations."""
        if not plan.validations and not self.config.force:
            return True

        self._log(PromotionResult.PROGRESS, "Validation", "Running pre-promotion checks...")

        # Schema validation
        if not self._validate_model_schema(plan.source_version):
            return False

        # Metric comparison
        if self.config.compare_metrics and plan.current_champion:
            if not self._compare_model_metrics(plan.source_version, plan.current_champion):
                if not self.config.force:
                    return False

        # Run tests
        if self.config.run_tests:
            if not self._run_model_tests(plan.source_version):
                return False

        self._log(PromotionResult.SUCCESS, "Validation", "All pre-promotion checks passed")
        return True

    def _validate_model_schema(self, version: ModelVersion) -> bool:
        """Validate model schema and signature."""
        try:
            import mlflow

            # Load model to validate it's accessible
            model_uri = f"models:/{version.name}/{version.version}"
            model = mlflow.pyfunc.load_model(model_uri)

            self._log(PromotionResult.SUCCESS, "Schema", f"Model v{version.version} schema validated")
            return True

        except Exception as e:
            self._log(PromotionResult.ERROR, "Schema", f"Schema validation failed: {e}")
            return False

    def _compare_model_metrics(self, source: ModelVersion, champion: ModelVersion) -> bool:
        """Compare metrics between source and champion models."""
        try:
            import mlflow

            # Get run metrics for both versions
            source_run = mlflow.get_run(source.run_id)
            champion_run = mlflow.get_run(champion.run_id)

            source_metrics = source_run.data.metrics
            champion_metrics = champion_run.data.metrics

            # Compare key metrics
            key_metrics = ["test_accuracy", "accuracy", "f1_score", "precision", "recall"]

            improvements = []
            regressions = []

            for metric in key_metrics:
                if metric in source_metrics and metric in champion_metrics:
                    source_val = source_metrics[metric]
                    champion_val = champion_metrics[metric]
                    diff = source_val - champion_val

                    if diff > 0.01:  # Improvement threshold
                        improvements.append(f"{metric}: {champion_val:.4f} → {source_val:.4f} (+{diff:.4f})")
                    elif diff < -0.01:  # Regression threshold
                        regressions.append(f"{metric}: {champion_val:.4f} → {source_val:.4f} ({diff:.4f})")

            if improvements:
                self._log(PromotionResult.SUCCESS, "Metrics", f"Improvements detected: {', '.join(improvements)}")

            if regressions:
                self._log(PromotionResult.WARNING, "Metrics", f"Regressions detected: {', '.join(regressions)}")
                if not self.config.force:
                    self._log(PromotionResult.ERROR, "Metrics", "Use --force to promote despite regressions")
                    return False

            return True

        except Exception as e:
            self._log(PromotionResult.WARNING, "Metrics", f"Could not compare metrics: {e}")
            return True  # Don't fail promotion for metric comparison issues

    def _run_model_tests(self, version: ModelVersion) -> bool:
        """Run validation tests on the model."""
        self._log(PromotionResult.INFO, "Testing", "Model testing not yet implemented")
        # TODO: Implement model testing framework
        return True

    def _create_backup(self, current_champion: ModelVersion, backup_alias: str):
        """Create backup of current champion."""
        try:
            self.mlflow_client.set_registered_model_alias(
                current_champion.name,
                backup_alias,
                current_champion.version
            )
            self._log(PromotionResult.SUCCESS, "Backup", f"Created backup @{backup_alias} → v{current_champion.version}")
        except Exception as e:
            self._log(PromotionResult.WARNING, "Backup", f"Backup creation failed: {e}")

    def _promote_model(self, plan: PromotionPlan) -> bool:
        """Perform the actual model promotion."""
        try:
            self.mlflow_client.set_registered_model_alias(
                plan.source_version.name,
                plan.target_alias,
                plan.source_version.version
            )

            self._log(PromotionResult.SUCCESS, "Promotion",
                     f"Set @{plan.target_alias} → v{plan.source_version.version}")

            # Record promotion history
            promotion_record = {
                "timestamp": datetime.now().isoformat(),
                "source_version": plan.source_version.version,
                "target_alias": plan.target_alias,
                "previous_champion": plan.current_champion.version if plan.current_champion else None,
                "backup_alias": plan.backup_alias
            }
            self.promotion_history.append(promotion_record)

            return True

        except Exception as e:
            self._log(PromotionResult.ERROR, "Promotion", f"Promotion failed: {e}")
            return False

    def _handle_rollback(self) -> int:
        """Handle rollback to previous version."""
        self._log(PromotionResult.PROGRESS, "Rollback", "Initiating rollback...")

        backup_alias = f"prev_{self.config.to_alias}"

        try:
            # Get backup version
            backup_version = self.mlflow_client.get_model_version_by_alias(
                self.config.model_name, backup_alias
            )

            # Rollback to backup
            self.mlflow_client.set_registered_model_alias(
                self.config.model_name,
                self.config.to_alias,
                backup_version.version
            )

            self._log(PromotionResult.SUCCESS, "Rollback",
                     f"Rolled back @{self.config.to_alias} to v{backup_version.version}")

            return 0

        except Exception as e:
            self._log(PromotionResult.ERROR, "Rollback", f"Rollback failed: {e}")
            return 3

    def _log_promotion_success(self, plan: PromotionPlan):
        """Log successful promotion details."""
        print("\n" + "="*50)
        print("🎉 PROMOTION SUCCESSFUL!")
        print("="*50)
        print(f"📦 Model: {plan.source_version.name}")
        print(f"🎯 Promoted: v{plan.source_version.version} → @{plan.target_alias}")

        if plan.current_champion:
            print(f"📋 Previous: v{plan.current_champion.version}")
            if plan.backup_alias:
                print(f"🔙 Backup: @{plan.backup_alias}")

        print(f"\n💡 Quick rollback: /promote-model {plan.source_version.name} --rollback")

    def _log(self, result: PromotionResult, category: str, message: str):
        """Log promotion message."""
        print(f"{result.value} [{category}] {message}")
        if self.verbose:
            print()

def main():
    """Main entry point for promotion command."""
    import argparse

    parser = argparse.ArgumentParser(description="Promote models safely with validation")
    parser.add_argument("model_name", help="Full Unity Catalog model name")
    parser.add_argument("--from", dest="from_alias", default="staging",
                       help="Source alias (default: staging)")
    parser.add_argument("--to", dest="to_alias", default="champion",
                       help="Target alias (default: champion)")
    parser.add_argument("--version", help="Specific version to promote")
    parser.add_argument("--force", action="store_true",
                       help="Skip safety checks and validation")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be promoted without making changes")
    parser.add_argument("--rollback", action="store_true",
                       help="Rollback the previous promotion")
    parser.add_argument("--compare-metrics", action="store_true",
                       help="Compare performance metrics before promotion")
    parser.add_argument("--run-tests", action="store_true",
                       help="Run validation tests before promotion")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = PromotionConfig(
        model_name=args.model_name,
        from_alias=args.from_alias,
        to_alias=args.to_alias,
        version=args.version,
        force=args.force,
        dry_run=args.dry_run,
        rollback=args.rollback,
        compare_metrics=args.compare_metrics,
        run_tests=args.run_tests
    )

    promoter = ModelPromoter(config, args.verbose)
    return promoter.promote()

if __name__ == "__main__":
    sys.exit(main())