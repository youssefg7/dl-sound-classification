#!/usr/bin/env python
"""
Optuna study analysis script.

This script provides analysis and visualization tools for Optuna studies,
including hyperparameter importance, optimization history, and trial comparisons.

Examples:
- Analyze a study:
  ```bash
  python scripts/analyze_study.py --study-name "envnet_esc50_optimization"
  ```

- Generate plots:
  ```bash
  python scripts/analyze_study.py --study-name "my_study" --plots
  ```

- Export results:
  ```bash
  python scripts/analyze_study.py --study-name "my_study" --export results.csv
  ```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import optuna
from omegaconf import OmegaConf

from src.optimization import StudyManager


def analyze_study(
    study_name: str,
    storage_path: str = "optuna_studies.db",
    generate_plots: bool = False,
    export_path: Optional[str] = None,
) -> None:
    """
    Analyze an Optuna study.
    
    Args:
        study_name: Name of the study to analyze
        storage_path: Path to the SQLite database
        generate_plots: Whether to generate visualization plots
        export_path: Path to export results (optional)
    """
    print(f"📊 Analyzing Optuna Study: {study_name}")
    print("=" * 60)
    
    # Create study manager and load study
    study_manager = StudyManager(storage_path=storage_path)
    
    try:
        study = study_manager.load_study(study_name)
    except ValueError as e:
        print(f"❌ Error loading study: {e}")
        return
    
    # Print basic study information
    print_study_info(study)
    
    # Analyze trials
    analyze_trials(study)
    
    # Print best trial details
    print_best_trial(study)
    
    # Analyze hyperparameter importance
    if len(study.trials) > 10:  # Need sufficient trials for importance analysis
        analyze_hyperparameter_importance(study)
    
    # Generate plots if requested
    if generate_plots:
        generate_study_plots(study, study_name)
    
    # Export results if requested
    if export_path:
        export_study_results(study, export_path)
    
    print("\n✅ Analysis completed!")


def print_study_info(study: optuna.Study) -> None:
    """Print basic study information."""
    print("📋 Study Information:")
    print(f"   Name: {study.study_name}")
    print(f"   Direction: {study.direction.name}")
    print(f"   Total trials: {len(study.trials)}")
    
    # Count trial states
    from collections import Counter
    trial_states = Counter(trial.state.name for trial in study.trials)
    
    print(f"   Completed trials: {trial_states.get('COMPLETE', 0)}")
    print(f"   Pruned trials: {trial_states.get('PRUNED', 0)}")
    print(f"   Failed trials: {trial_states.get('FAIL', 0)}")
    print(f"   Running trials: {trial_states.get('RUNNING', 0)}")
    
    if study.best_trial:
        print(f"   Best value: {study.best_value:.4f}")
        print(f"   Best trial: #{study.best_trial.number}")


def analyze_trials(study: optuna.Study) -> None:
    """Analyze trial statistics."""
    print("\n📈 Trial Analysis:")
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    if completed_trials:
        values = [t.value for t in completed_trials if t.value is not None]
        print(f"   Completed trials: {len(completed_trials)}")
        if values:
            print(f"   Value statistics:")
            print(f"     Mean: {sum(values) / len(values):.4f}")
            print(f"     Std:  {pd.Series(values).std():.4f}")
            print(f"     Min:  {min(values):.4f}")
            print(f"     Max:  {max(values):.4f}")
    
    if pruned_trials:
        print(f"   Pruned trials: {len(pruned_trials)}")
        # Analyze pruning epochs
        pruning_epochs = []
        for trial in pruned_trials:
            if trial.intermediate_values:
                pruning_epochs.append(max(trial.intermediate_values.keys()))
        
        if pruning_epochs:
            print(f"   Pruning statistics:")
            print(f"     Mean pruning epoch: {sum(pruning_epochs) / len(pruning_epochs):.1f}")
            print(f"     Earliest pruning: {min(pruning_epochs)}")
            print(f"     Latest pruning: {max(pruning_epochs)}")


def print_best_trial(study: optuna.Study) -> None:
    """Print details of the best trial."""
    if not study.best_trial:
        print("\n❌ No best trial found")
        return
    
    print(f"\n🏆 Best Trial (#{study.best_trial.number}):")
    print(f"   Value: {study.best_value:.4f}")
    print(f"   Parameters:")
    
    for key, value in study.best_trial.params.items():
        print(f"     {key}: {value}")
    
    # Print learning curve if available
    if study.best_trial.intermediate_values:
        print(f"   Learning curve:")
        for epoch, value in study.best_trial.intermediate_values.items():
            print(f"     Epoch {epoch}: {value:.4f}")


def analyze_hyperparameter_importance(study: optuna.Study) -> None:
    """Analyze hyperparameter importance."""
    print("\n🔍 Hyperparameter Importance:")
    
    try:
        # Calculate importance using fANOVA
        importance = optuna.importance.get_param_importances(study)
        
        print("   Parameter importances:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"     {param}: {imp:.4f}")
            
    except Exception as e:
        print(f"   ⚠ Could not calculate importance: {e}")


def generate_study_plots(study: optuna.Study, study_name: str) -> None:
    """Generate visualization plots for the study."""
    print(f"\n📊 Generating plots for study: {study_name}")
    
    try:
        import optuna.visualization as vis
        import plotly.io as pio
        
        # Create output directory
        plot_dir = Path(f"outputs/plots/{study_name}")
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        plots = {
            "optimization_history": vis.plot_optimization_history(study),
            "param_importances": vis.plot_param_importances(study),
            "parallel_coordinate": vis.plot_parallel_coordinate(study),
            "contour": vis.plot_contour(study),
            "slice": vis.plot_slice(study),
        }
        
        # Save plots
        for plot_name, fig in plots.items():
            if fig is not None:
                output_path = plot_dir / f"{plot_name}.html"
                pio.write_html(fig, str(output_path))
                print(f"   ✓ Saved: {output_path}")
        
        print(f"   📁 Plots saved to: {plot_dir}")
        
    except ImportError:
        print("   ⚠ Plotly not available. Install with: pip install plotly")
    except Exception as e:
        print(f"   ⚠ Error generating plots: {e}")


def export_study_results(study: optuna.Study, export_path: str) -> None:
    """Export study results to CSV."""
    print(f"\n💾 Exporting results to: {export_path}")
    
    try:
        # Create DataFrame with trial data
        trial_data = []
        
        for trial in study.trials:
            row = {
                "trial_number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
                "duration": trial.duration.total_seconds() if trial.duration else None,
            }
            
            # Add parameters
            for key, value in trial.params.items():
                row[f"param_{key}"] = value
            
            trial_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(trial_data)
        df.to_csv(export_path, index=False)
        
        print(f"   ✓ Exported {len(trial_data)} trials to {export_path}")
        
    except Exception as e:
        print(f"   ❌ Error exporting results: {e}")


def list_studies(storage_path: str = "optuna_studies.db") -> None:
    """List all studies in the storage."""
    print("📚 Available Studies:")
    
    study_manager = StudyManager(storage_path=storage_path)
    studies = study_manager.list_studies()
    
    if not studies:
        print("   No studies found")
        return
    
    for study_name in studies:
        try:
            summary = study_manager.get_study_summary(study_name)
            print(f"   • {study_name}")
            print(f"     Trials: {summary['n_trials']}")
            print(f"     Best value: {summary['best_value']}")
            print(f"     Direction: {summary['direction']}")
        except Exception as e:
            print(f"   • {study_name} (error: {e})")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze Optuna studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--study-name",
        "-s",
        type=str,
        help="Name of the study to analyze",
    )
    
    parser.add_argument(
        "--storage-path",
        "-p",
        type=str,
        default="optuna_studies.db",
        help="Path to the SQLite database (default: optuna_studies.db)",
    )
    
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots",
    )
    
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to CSV file",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available studies",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_studies(args.storage_path)
        return
    
    if not args.study_name:
        print("❌ Please provide a study name with --study-name")
        parser.print_help()
        return
    
    # Run analysis
    analyze_study(
        study_name=args.study_name,
        storage_path=args.storage_path,
        generate_plots=args.plots,
        export_path=args.export,
    )


if __name__ == "__main__":
    main() 