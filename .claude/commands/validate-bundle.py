#!/usr/bin/env python3
"""
Databricks Asset Bundle ML Project Validator

Comprehensive validation for ML projects using Databricks Asset Bundles.
Catches common configuration issues, schema mismatches, and deployment problems.
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    SUCCESS = "✅"
    WARNING = "⚠️ "
    ERROR = "❌"
    INFO = "💡"
    FIX = "🔧"

@dataclass
class ValidationResult:
    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None
    auto_fixable: bool = False

class MLBundleValidator:
    """Validates Databricks Asset Bundle ML projects."""

    def __init__(self, project_root: Path, target: str = "dev", verbose: bool = False):
        self.project_root = project_root
        self.target = target
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def validate_all(self, check_permissions: bool = False, check_volumes: bool = False) -> int:
        """Run all validations and return exit code."""
        print(f"🔍 Validating Databricks Asset Bundle ML Project")
        print(f"📁 Project: {self.project_root}")
        print(f"🎯 Target: {self.target}")
        print()

        # Core validations
        self._validate_bundle_config()
        self._validate_project_structure()
        self._validate_ml_notebooks()
        self._validate_package_management()
        self._validate_job_configurations()

        # Optional validations
        if check_permissions:
            self._validate_permissions()
        if check_volumes:
            self._validate_volumes()

        return self._report_results()

    def _validate_bundle_config(self):
        """Validate databricks.yml configuration."""
        print("📋 Validating Bundle Configuration...")

        bundle_file = self.project_root / "databricks.yml"
        if not bundle_file.exists():
            self._add_result(ValidationLevel.ERROR, "Bundle Config",
                           "databricks.yml not found",
                           "Asset Bundle requires databricks.yml in project root")
            return

        try:
            with open(bundle_file) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self._add_result(ValidationLevel.ERROR, "Bundle Config",
                           f"Invalid YAML syntax: {e}")
            return

        # Check required sections
        required_sections = ["bundle", "targets"]
        for section in required_sections:
            if section not in config:
                self._add_result(ValidationLevel.ERROR, "Bundle Config",
                               f"Missing required section: {section}")

        # Check target exists
        if "targets" in config and self.target not in config["targets"]:
            available = list(config["targets"].keys())
            self._add_result(ValidationLevel.ERROR, "Bundle Config",
                           f"Target '{self.target}' not found",
                           f"Available targets: {available}")

        # Check variables are defined for resources that use them
        self._validate_variables(config)

        # Check for ML-specific configurations
        self._validate_ml_resources(config)

    def _validate_variables(self, config: Dict):
        """Validate variable definitions and usage."""
        variables = config.get("variables", {})
        target_config = config.get("targets", {}).get(self.target, {})
        target_vars = target_config.get("variables", {})

        # Check for undefined variables used in resources
        resources = config.get("resources", {})

        # Common variable patterns in ML projects
        ml_variables = ["catalog", "schema", "model_base", "experiment_path"]
        for var in ml_variables:
            if var not in variables and var not in target_vars:
                # Check if variable is used in resources
                if self._variable_used_in_resources(var, resources):
                    self._add_result(ValidationLevel.WARNING, "Variables",
                                   f"Variable '{var}' used but not defined",
                                   f"Define in variables section or target.{self.target}.variables")

    def _validate_ml_resources(self, config: Dict):
        """Validate ML-specific resource configurations."""
        resources = config.get("resources", {})

        # Check experiments
        if "experiments" in resources:
            for exp_name, exp_config in resources["experiments"].items():
                if "name" not in exp_config:
                    self._add_result(ValidationLevel.ERROR, "ML Resources",
                                   f"Experiment '{exp_name}' missing name")

        # Check model serving endpoints
        if "model_serving_endpoints" in resources:
            for endpoint_name, endpoint_config in resources["model_serving_endpoints"].items():
                self._validate_serving_endpoint(endpoint_name, endpoint_config)

        # Check jobs for ML workflows
        if "jobs" in resources:
            self._validate_ml_jobs(resources["jobs"])

    def _validate_ml_jobs(self, jobs: Dict):
        """Validate ML job configurations."""
        ml_job_patterns = ["data", "train", "inference", "monitor"]

        for job_name, job_config in jobs.items():
            # Check if job follows ML naming patterns
            job_type = None
            for pattern in ml_job_patterns:
                if pattern in job_name.lower():
                    job_type = pattern
                    break

            if job_type:
                self._add_result(ValidationLevel.SUCCESS, "ML Jobs",
                               f"ML job detected: {job_name} ({job_type})")

            # Check for proper task dependencies
            if "tasks" in job_config:
                self._validate_job_tasks(job_name, job_config["tasks"])

    def _validate_job_tasks(self, job_name: str, tasks: List):
        """Validate job task configurations."""
        for task in tasks:
            if "notebook_task" in task:
                notebook_path = task["notebook_task"].get("notebook_path", "")
                if not notebook_path.startswith("./"):
                    self._add_result(ValidationLevel.WARNING, "Job Tasks",
                                   f"Job '{job_name}' notebook path should be relative: {notebook_path}")

            if "depends_on" in task:
                self._add_result(ValidationLevel.SUCCESS, "Job Tasks",
                               f"Job '{job_name}' has proper task dependencies")

    def _validate_serving_endpoint(self, name: str, config: Dict):
        """Validate model serving endpoint configuration."""
        if "config" not in config:
            self._add_result(ValidationLevel.ERROR, "Model Serving",
                           f"Endpoint '{name}' missing config section")
            return

        endpoint_config = config["config"]
        if "served_entities" in endpoint_config:
            for entity in endpoint_config["served_entities"]:
                if "entity_version" in entity:
                    version = entity["entity_version"]
                    if version == "latest":
                        self._add_result(ValidationLevel.ERROR, "Model Serving",
                                       f"Endpoint '{name}' uses 'latest' version",
                                       "Use numeric version or omit entity_version",
                                       "Replace 'latest' with specific version number")

    def _validate_project_structure(self):
        """Validate expected ML project structure."""
        print("📁 Validating Project Structure...")

        expected_dirs = {
            "src": "Source code directory",
            "notebooks": "Pipeline notebooks directory",
            "tests": "Test directory"
        }

        for dir_name, description in expected_dirs.items():
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self._add_result(ValidationLevel.WARNING, "Project Structure",
                               f"Missing {description}: {dir_name}/")

        # Check for package management files
        package_files = {
            "pyproject.toml": "UV/modern Python packaging",
            "requirements.txt": "Databricks notebook compatibility"
        }

        for file_name, description in package_files.items():
            file_path = self.project_root / file_name
            if not file_path.exists():
                self._add_result(ValidationLevel.WARNING, "Package Management",
                               f"Missing {description}: {file_name}")

    def _validate_ml_notebooks(self):
        """Validate ML notebook configurations."""
        print("📓 Validating ML Notebooks...")

        notebooks_dir = self.project_root / "notebooks"
        if not notebooks_dir.exists():
            return

        # Expected notebook patterns for ML pipelines
        expected_notebooks = [
            ("*data*", "Data generation/ingestion"),
            ("*feature*", "Feature engineering"),
            ("*train*", "Model training"),
            ("*inference*", "Batch inference"),
            ("*monitor*", "Model monitoring")
        ]

        notebook_files = list(notebooks_dir.glob("*.py"))

        for pattern, description in expected_notebooks:
            matching = [f for f in notebook_files if pattern.replace("*", "") in f.name.lower()]
            if not matching:
                self._add_result(ValidationLevel.INFO, "ML Pipeline",
                               f"No {description} notebook found (pattern: {pattern})")

        # Check notebook imports and dependencies
        for notebook in notebook_files:
            self._validate_notebook_content(notebook)

    def _validate_notebook_content(self, notebook_path: Path):
        """Validate individual notebook content."""
        try:
            with open(notebook_path) as f:
                content = f.read()
        except Exception as e:
            self._add_result(ValidationLevel.WARNING, "Notebooks",
                           f"Cannot read {notebook_path.name}: {e}")
            return

        # Check for requirements.txt installation
        if "%pip install -r" in content and "requirements.txt" in content:
            self._add_result(ValidationLevel.SUCCESS, "Notebooks",
                           f"{notebook_path.name} uses requirements.txt")

        # Check for schema persistence (training notebooks)
        if "train" in notebook_path.name.lower():
            if "train_columns" in content and ("dbutils.fs.put" in content or "Volume" in content):
                self._add_result(ValidationLevel.SUCCESS, "Schema Management",
                               f"{notebook_path.name} persists training columns")
            else:
                self._add_result(ValidationLevel.WARNING, "Schema Management",
                               f"{notebook_path.name} may not persist training schema",
                               "Consider saving training columns for inference alignment")

        # Check for inference notebooks loading schema
        if "inference" in notebook_path.name.lower():
            if "train_columns" in content and ("dbutils.fs.head" in content or "Volume" in content):
                self._add_result(ValidationLevel.SUCCESS, "Schema Management",
                               f"{notebook_path.name} loads training columns")
            else:
                self._add_result(ValidationLevel.WARNING, "Schema Management",
                               f"{notebook_path.name} may not load training schema",
                               "Load training columns to avoid schema mismatches")

    def _validate_package_management(self):
        """Validate package management setup."""
        print("📦 Validating Package Management...")

        pyproject_file = self.project_root / "pyproject.toml"
        requirements_file = self.project_root / "requirements.txt"
        sync_script = self.project_root / "sync_requirements.sh"

        if pyproject_file.exists() and requirements_file.exists():
            self._add_result(ValidationLevel.SUCCESS, "Package Management",
                           "Dual package management detected (UV + requirements.txt)")

            if sync_script.exists():
                self._add_result(ValidationLevel.SUCCESS, "Package Management",
                               "Sync script available for dependency management")
            else:
                self._add_result(ValidationLevel.WARNING, "Package Management",
                               "Missing sync script for requirements.txt generation")

        elif pyproject_file.exists():
            self._add_result(ValidationLevel.WARNING, "Package Management",
                           "Missing requirements.txt for Databricks compatibility")

        elif requirements_file.exists():
            self._add_result(ValidationLevel.INFO, "Package Management",
                           "Using requirements.txt only (consider upgrading to UV)")

    def _validate_job_configurations(self):
        """Validate job configurations for ML workflows."""
        print("⚙️  Validating Job Configurations...")

        try:
            result = subprocess.run(["databricks", "bundle", "validate", "--target", self.target],
                                  capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                self._add_result(ValidationLevel.SUCCESS, "Bundle Validation",
                               f"Asset bundle validates successfully for {self.target}")
            else:
                self._add_result(ValidationLevel.ERROR, "Bundle Validation",
                               f"Asset bundle validation failed: {result.stderr}")

        except FileNotFoundError:
            self._add_result(ValidationLevel.WARNING, "Bundle Validation",
                           "Databricks CLI not found - cannot validate bundle")

    def _validate_permissions(self):
        """Validate catalog and schema permissions."""
        print("🔒 Validating Permissions...")
        # This would require MCP integration to check actual permissions
        self._add_result(ValidationLevel.INFO, "Permissions",
                       "Permission validation requires MCP integration (future enhancement)")

    def _validate_volumes(self):
        """Validate Unity Catalog volume access."""
        print("💾 Validating Volume Access...")
        # This would require actual connection to validate volume access
        self._add_result(ValidationLevel.INFO, "Volumes",
                       "Volume validation requires connection (future enhancement)")

    def _variable_used_in_resources(self, var_name: str, resources: Dict) -> bool:
        """Check if variable is used in resources configuration."""
        resources_str = json.dumps(resources)
        return f"${{{var_name}}}" in resources_str or f"${{var.{var_name}}}" in resources_str

    def _add_result(self, level: ValidationLevel, category: str, message: str,
                   details: str = None, fix_suggestion: str = None, auto_fixable: bool = False):
        """Add validation result."""
        result = ValidationResult(level, category, message, details, fix_suggestion, auto_fixable)
        self.results.append(result)

        if self.verbose or level in [ValidationLevel.ERROR, ValidationLevel.WARNING]:
            self._print_result(result)

    def _print_result(self, result: ValidationResult):
        """Print individual validation result."""
        print(f"{result.level.value} [{result.category}] {result.message}")
        if result.details and self.verbose:
            print(f"    Details: {result.details}")
        if result.fix_suggestion:
            print(f"    💡 Fix: {result.fix_suggestion}")

    def _report_results(self) -> int:
        """Generate final validation report and return exit code."""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)

        # Count results by level
        counts = {level: 0 for level in ValidationLevel}
        for result in self.results:
            counts[result.level] += 1

        print(f"✅ Successes: {counts[ValidationLevel.SUCCESS]}")
        print(f"💡 Info:      {counts[ValidationLevel.INFO]}")
        print(f"⚠️  Warnings:  {counts[ValidationLevel.WARNING]}")
        print(f"❌ Errors:    {counts[ValidationLevel.ERROR]}")

        # Determine exit code
        if counts[ValidationLevel.ERROR] > 0:
            print(f"\n❌ VALIDATION FAILED - {counts[ValidationLevel.ERROR]} error(s) found")
            print("🔧 Fix errors before deployment")
            return 2
        elif counts[ValidationLevel.WARNING] > 0:
            print(f"\n⚠️  VALIDATION PASSED WITH WARNINGS - {counts[ValidationLevel.WARNING]} warning(s)")
            print("💡 Consider addressing warnings for best practices")
            return 1
        else:
            print(f"\n✅ VALIDATION PASSED - Project ready for deployment!")
            return 0

def main():
    """Main entry point for validation command."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Databricks Asset Bundle ML Project")
    parser.add_argument("target", nargs="?", default="dev", help="Target environment (default: dev)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix common issues")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--check-permissions", action="store_true", help="Check permissions")
    parser.add_argument("--check-volumes", action="store_true", help="Check volume access")

    args = parser.parse_args()

    # Find project root (directory containing databricks.yml)
    current_dir = Path.cwd()
    project_root = current_dir

    while project_root != project_root.parent:
        if (project_root / "databricks.yml").exists():
            break
        project_root = project_root.parent
    else:
        print("❌ Error: No databricks.yml found in current directory or parents")
        return 3

    validator = MLBundleValidator(project_root, args.target, args.verbose)
    exit_code = validator.validate_all(args.check_permissions, args.check_volumes)

    return exit_code

if __name__ == "__main__":
    sys.exit(main())