"""
Sentinel-X: Deep-Pulse Predictive Failure Modeling
Principal ML Engineer Implementation
NASA C-MAPSS Dataset - Turbofan Engine Degradation Simulation
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, classification_report
)

# Handle XGBoost import with helpful error message
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    error_msg = str(e)
    if "libomp" in error_msg or "XGBoost" in error_msg:
        print("\n" + "="*70)
        print("ERROR: XGBoost could not be loaded.")
        print("="*70)
        print("\nOn macOS, XGBoost requires the OpenMP runtime library (libomp).")
        print("\nTo fix this, please run:")
        print("  1. Install Homebrew (if not installed):")
        print('     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        print("  2. Install libomp:")
        print("     brew install libomp")
        print("  3. Reinstall xgboost:")
        print("     pip uninstall xgboost && pip install xgboost")
        print("\nAlternatively, run the setup script:")
        print("     ./setup_xgboost.sh")
        print("\n" + "="*70 + "\n")
    raise

import shap
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SentinelXEngine:
    """
    Sentinel-X Engine: Deep-Pulse Predictive Failure Modeling System
    
    This class implements a comprehensive predictive maintenance solution
    for turbofan engines using the NASA C-MAPSS dataset.
    """
    
    def __init__(self, data_dir: str = "CMaps"):
        """
        Initialize Sentinel-X Engine.
        
        Args:
            data_dir: Directory containing NASA C-MAPSS dataset files
        """
        self.data_dir = data_dir
        self.console = Console()
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_names = None
        self.train_data = None
        self.test_data = None
        self.train_features = None
        self.test_features = None
        self.train_target = None
        self.test_target = None
        
        # Column names for NASA C-MAPSS dataset
        self.columns = (
            ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] +
            [f's_{i}' for i in range(1, 22)]
        )
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Load NASA C-MAPSS dataset files.
        
        Returns:
            Tuple of (train_df, test_df, rul_test)
        """
        self.console.print("[bold cyan]Loading NASA C-MAPSS Dataset...[/bold cyan]")
        
        # Load training data
        train_path = os.path.join(self.data_dir, 'train_FD001.txt')
        train_df = pd.read_csv(
            train_path,
            sep='\s+',
            header=None,
            names=self.columns
        )
        
        # Load test data
        test_path = os.path.join(self.data_dir, 'test_FD001.txt')
        test_df = pd.read_csv(
            test_path,
            sep='\s+',
            header=None,
            names=self.columns
        )
        
        # Load RUL ground truth for test set
        rul_path = os.path.join(self.data_dir, 'RUL_FD001.txt')
        rul_test = pd.read_csv(
            rul_path,
            header=None,
            names=['RUL']
        ).values.flatten()
        
        self.console.print(f"[green][OK][/green] Training data: {train_df.shape}")
        self.console.print(f"[green][OK][/green] Test data: {test_df.shape}")
        self.console.print(f"[green][OK][/green] RUL test labels: {len(rul_test)}")
        
        return train_df, test_df, rul_test
    
    def engineer_targets(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_test: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Engineer target variables (RUL and failure_imminent).
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            rul_test: Ground truth RUL for test set
            
        Returns:
            Tuple of (train_df_with_targets, test_df_with_targets)
        """
        self.console.print("[bold cyan]Engineering Target Variables...[/bold cyan]")
        
        # Training set: RUL = Max_Cycle - Current_Cycle
        train_df = train_df.copy()
        max_cycles = train_df.groupby('unit_nr')['time_cycles'].max()
        train_df['max_cycle'] = train_df['unit_nr'].map(max_cycles)
        train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']
        
        # Testing set: RUL = RUL_FD001.txt + cycles remaining from last point
        test_df = test_df.copy()
        last_cycles = test_df.groupby('unit_nr')['time_cycles'].max()
        test_df['last_cycle'] = test_df['unit_nr'].map(last_cycles)
        
        # Map RUL values to each unit
        unit_rul_map = dict(zip(range(1, len(rul_test) + 1), rul_test))
        test_df['RUL_from_file'] = test_df['unit_nr'].map(unit_rul_map)
        test_df['RUL'] = test_df['RUL_from_file'] + (
            test_df['last_cycle'] - test_df['time_cycles']
        )
        
        # Classification target: failure_imminent (1 if RUL <= 30, else 0)
        train_df['failure_imminent'] = (train_df['RUL'] <= 30).astype(int)
        test_df['failure_imminent'] = (test_df['RUL'] <= 30).astype(int)
        
        self.console.print(
            f"[green][OK][/green] Training: "
            f"{train_df['failure_imminent'].sum()} failure cases, "
            f"{len(train_df) - train_df['failure_imminent'].sum()} healthy cases"
        )
        self.console.print(
            f"[green][OK][/green] Test: "
            f"{test_df['failure_imminent'].sum()} failure cases, "
            f"{len(test_df) - test_df['failure_imminent'].sum()} healthy cases"
        )
        
        return train_df, test_df
    
    def engineer_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Deep-Pulse feature engineering: rolling statistics and normalization.
        
        Args:
            train_df: Training dataframe with targets
            test_df: Test dataframe with targets
            
        Returns:
            Tuple of (train_features, test_features)
        """
        self.console.print("[bold cyan]Deep-Pulse Feature Engineering...[/bold cyan]")
        
        # Identify constant sensors (sensors with no variance)
        sensor_cols = [f's_{i}' for i in range(1, 22)]
        constant_sensors = []
        
        for col in sensor_cols:
            if train_df[col].nunique() <= 1:
                constant_sensors.append(col)
        
        self.console.print(
            f"[yellow][WARNING][/yellow] Dropping constant sensors: {constant_sensors}"
        )
        
        # Drop constant sensors
        remaining_sensors = [s for s in sensor_cols if s not in constant_sensors]
        
        # Create feature dataframe
        train_features = train_df[['unit_nr', 'time_cycles'] + remaining_sensors].copy()
        test_features = test_df[['unit_nr', 'time_cycles'] + remaining_sensors].copy()
        
        # Rolling statistics (mean and std) for windows 10 and 25
        windows = [10, 25]
        
        for window in windows:
            for sensor in remaining_sensors:
                # Rolling mean
                train_features[f'{sensor}_rolling_mean_{window}'] = (
                    train_df.groupby('unit_nr')[sensor]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                test_features[f'{sensor}_rolling_mean_{window}'] = (
                    test_df.groupby('unit_nr')[sensor]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Rolling std
                train_features[f'{sensor}_rolling_std_{window}'] = (
                    train_df.groupby('unit_nr')[sensor]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .fillna(0)
                    .reset_index(0, drop=True)
                )
                test_features[f'{sensor}_rolling_std_{window}'] = (
                    test_df.groupby('unit_nr')[sensor]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .fillna(0)
                    .reset_index(0, drop=True)
                )
        
        # Feature columns (excluding unit_nr and time_cycles)
        feature_cols = (
            remaining_sensors +
            [f'{s}_rolling_mean_{w}' for s in remaining_sensors for w in windows] +
            [f'{s}_rolling_std_{w}' for s in remaining_sensors for w in windows]
        )
        
        self.console.print(f"[green][OK][/green] Total features: {len(feature_cols)}")
        
        # Normalization using MinMaxScaler
        train_feature_values = train_features[feature_cols].values
        test_feature_values = test_features[feature_cols].values
        
        train_feature_values_scaled = self.scaler.fit_transform(train_feature_values)
        test_feature_values_scaled = self.scaler.transform(test_feature_values)
        
        train_features_scaled = pd.DataFrame(
            train_feature_values_scaled,
            columns=feature_cols,
            index=train_features.index
        )
        test_features_scaled = pd.DataFrame(
            test_feature_values_scaled,
            columns=feature_cols,
            index=test_features.index
        )
        
        # Store feature names
        self.feature_names = feature_cols
        
        return train_features_scaled, test_features_scaled
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups: pd.Series
    ) -> xgb.XGBClassifier:
        """
        Train XGBoost model with GroupKFold cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            groups: Group labels for GroupKFold (unit_nr)
            
        Returns:
            Trained XGBoost classifier
        """
        self.console.print("[bold cyan]Training Sentinel-X Model...[/bold cyan]")
        
        # Calculate scale_pos_weight
        healthy_count = (y_train == 0).sum()
        failure_count = (y_train == 1).sum()
        scale_pos_weight = healthy_count / failure_count if failure_count > 0 else 1.0
        
        self.console.print(
            f"[green][OK][/green] Class balance: "
            f"{healthy_count} healthy, {failure_count} failure"
        )
        self.console.print(
            f"[green][OK][/green] scale_pos_weight: {scale_pos_weight:.2f}"
        )
        
        # GroupKFold for time-series integrity
        group_kfold = GroupKFold(n_splits=5)
        
        # Base model
        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Grid search
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        
        self.console.print("[yellow][SEARCH][/yellow] Performing grid search...")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=group_kfold,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train, groups=groups)
        
        self.console.print(
            f"[green][OK][/green] Best parameters: {grid_search.best_params_}"
        )
        self.console.print(
            f"[green][OK][/green] Best CV F1-score: {grid_search.best_score_:.4f}"
        )
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return metrics
    
    def display_dashboard(self, metrics: Dict[str, float]):
        """
        Display Sentinel-X Status Dashboard using rich library.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        # Create dashboard table
        table = Table(title="Sentinel-X Status Dashboard", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("F1-Score", f"{metrics['f1_score']:.4f}")
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall (Critical)", f"{metrics['recall']:.4f}")
        
        panel = Panel(
            table,
            title="[bold green]Sentinel-X: Deep-Pulse Predictive Failure Modeling[/bold green]",
            border_style="green"
        )
        
        self.console.print("\n")
        self.console.print(panel)
        self.console.print("\n")
    
    def plot_precision_recall_curve(
        self,
        y_test: pd.Series,
        y_pred_proba: np.ndarray,
        save_path: str
    ):
        """
        Generate and save Precision-Recall curve.
        
        Args:
            y_test: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(
            y_test, y_pred_proba
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2, label='Sentinel-X')
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve: Sentinel-X Failure Prediction', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green][OK][/green] Saved PR curve: {save_path}")
    
    def plot_shap_summary(
        self,
        X_test: pd.DataFrame,
        save_path: str,
        max_display: int = 20
    ):
        """
        Generate and save SHAP summary plot.
        
        Args:
            X_test: Test features
            save_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Sample for SHAP (to speed up computation)
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        # Ensure column order matches feature_names
        if self.feature_names:
            X_sample = X_sample[self.feature_names]
        
        self.console.print("[yellow][COMPUTING][/yellow] Computing SHAP values...")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification: use positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (failure) SHAP values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot: Primary "Pulses" of Failure', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green][OK][/green] Saved SHAP plot: {save_path}")
    
    def plot_rul_decay(
        self,
        train_df: pd.DataFrame,
        unit_nr: int = 10,
        save_path: str = None
    ):
        """
        Plot RUL decay for a sample engine unit.
        
        Args:
            train_df: Training dataframe with RUL
            unit_nr: Unit number to plot
            save_path: Path to save the plot
        """
        unit_data = train_df[train_df['unit_nr'] == unit_nr].copy()
        
        if len(unit_data) == 0:
            self.console.print(f"[red][WARNING][/red] Unit {unit_nr} not found in training data")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(
            unit_data['time_cycles'],
            unit_data['RUL'],
            linewidth=2,
            label=f'Unit #{unit_nr}',
            color='#2E86AB'
        )
        plt.axhline(y=30, color='r', linestyle='--', linewidth=2, label='Failure Threshold (RUL=30)')
        plt.xlabel('Time Cycles', fontsize=12)
        plt.ylabel('Remaining Useful Life (RUL)', fontsize=12)
        plt.title(f'RUL Decay Plot: Engine Unit #{unit_nr}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.console.print(f"[green][OK][/green] Saved RUL decay plot: {save_path}")
        
        plt.close()
    
    def run_pipeline(self):
        """
        Execute the complete Sentinel-X pipeline.
        """
        self.console.print(
            Panel(
                "[bold green]Sentinel-X: Deep-Pulse Predictive Failure Modeling[/bold green]\n"
                "[cyan]NASA C-MAPSS Dataset - Turbofan Engine Degradation[/cyan]",
                border_style="green"
            )
        )
        
        # 1. Load data
        train_df, test_df, rul_test = self.load_data()
        self.train_data = train_df
        self.test_data = test_df
        
        # 2. Engineer targets
        train_df, test_df = self.engineer_targets(train_df, test_df, rul_test)
        
        # 3. Engineer features
        train_features, test_features = self.engineer_features(train_df, test_df)
        self.train_features = train_features
        self.test_features = test_features
        self.train_target = train_df['failure_imminent']
        self.test_target = test_df['failure_imminent']
        
        # 4. Train model
        self.train_model(
            train_features,
            train_df['failure_imminent'],
            train_df['unit_nr']
        )
        
        # 5. Evaluate
        metrics = self.evaluate_model(test_features, test_df['failure_imminent'])
        
        # 6. Display dashboard
        self.display_dashboard(metrics)
        
        # 7. Generate visualizations
        output_dir = "/Users/wijeratne/Sentinel-X"
        
        self.plot_precision_recall_curve(
            test_df['failure_imminent'],
            metrics['y_pred_proba'],
            os.path.join(output_dir, 'precision_recall_curve.png')
        )
        
        self.plot_shap_summary(
            test_features,
            os.path.join(output_dir, 'shap_summary_plot.png')
        )
        
        self.plot_rul_decay(
            train_df,
            unit_nr=10,
            save_path=os.path.join(output_dir, 'rul_decay_plot.png')
        )
        
        return metrics


def sentinel_economic_impact(
    metrics: Dict[str, float],
    test_target: pd.Series,
    n_engines: int = 100
) -> Dict[str, float]:
    """
    Calculate economic impact of Sentinel-X vs. Run-to-Failure strategy.
    
    Args:
        metrics: Model evaluation metrics
        test_target: True test labels
        n_engines: Number of engines in fleet
        
    Returns:
        Dictionary of economic metrics
    """
    console = Console()
    
    # Cost parameters
    COST_UNPLANNED_FAILURE = 50000  # $50,000
    COST_PROACTIVE_MAINTENANCE = 12000  # $12,000
    
    # Calculate predictions
    y_pred = metrics['y_pred']
    y_true = test_target.values
    
    # True Positives: Correctly predicted failures (proactive maintenance)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    
    # False Negatives: Missed failures (unplanned failures)
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # False Positives: False alarms (unnecessary maintenance)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    
    # True Negatives: Correctly predicted healthy (no action needed)
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    # Economic calculations
    # Sentinel-X strategy
    sentinel_cost = (
        tp * COST_PROACTIVE_MAINTENANCE +  # Proactive maintenance for true failures
        fn * COST_UNPLANNED_FAILURE +  # Unplanned failures (missed)
        fp * COST_PROACTIVE_MAINTENANCE  # Unnecessary maintenance (false alarms)
    )
    
    # Run-to-Failure strategy (all failures are unplanned)
    total_failures = (y_true == 1).sum()
    run_to_failure_cost = total_failures * COST_UNPLANNED_FAILURE
    
    # ROI calculation
    cost_saved = run_to_failure_cost - sentinel_cost
    roi_percentage = (cost_saved / run_to_failure_cost * 100) if run_to_failure_cost > 0 else 0
    
    results = {
        'sentinel_cost': sentinel_cost,
        'run_to_failure_cost': run_to_failure_cost,
        'cost_saved': cost_saved,
        'roi_percentage': roi_percentage,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn
    }
    
    # Display results
    table = Table(title="Sentinel-X Economic Impact Analysis", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("True Positives (Caught Failures)", f"{tp}")
    table.add_row("False Negatives (Missed Failures)", f"{fn}")
    table.add_row("False Positives (False Alarms)", f"{fp}")
    table.add_row("", "")
    table.add_row("Sentinel-X Total Cost", f"${sentinel_cost:,.2f}")
    table.add_row("Run-to-Failure Total Cost", f"${run_to_failure_cost:,.2f}")
    table.add_row("Cost Saved (ROI)", f"${cost_saved:,.2f}")
    table.add_row("ROI Percentage", f"{roi_percentage:.2f}%")
    
    panel = Panel(
        table,
        title="[bold green]Economic Impact Module[/bold green]",
        border_style="green"
    )
    
    console.print("\n")
    console.print(panel)
    console.print("\n")
    
    return results


if __name__ == "__main__":
    # Initialize and run Sentinel-X
    sentinel = SentinelXEngine(data_dir="CMaps")
    metrics = sentinel.run_pipeline()
    
    # Economic impact analysis
    economic_results = sentinel_economic_impact(
        metrics,
        sentinel.test_target,
        n_engines=100
    )
