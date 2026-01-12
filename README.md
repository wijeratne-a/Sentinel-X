# Sentinel-X: Deep-Pulse Predictive Failure Modeling

## Overview

**Sentinel-X** is an advanced predictive maintenance system designed for turbofan engine failure prediction using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. This production-grade ML solution implements state-of-the-art time-series feature engineering, group-based cross-validation, and interpretable machine learning to predict imminent engine failures with high precision and recall.

## Technical Architecture

### Dataset: NASA C-MAPSS FD001

The system utilizes the NASA C-MAPSS FD001 dataset, which contains:
- **Training trajectories**: 100 engine units
- **Test trajectories**: 100 engine units
- **Operating conditions**: Single (Sea Level)
- **Fault modes**: Single (HPC Degradation)
- **Features**: 26 columns per cycle (unit number, time cycles, 3 operational settings, 21 sensor measurements)

### Core Components

#### 1. Advanced Data Loading
- NASA-specific format parsing using `pd.read_csv` with `sep='\s+'`
- Structured column mapping: `['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3', 's_1', ..., 's_21']`
- Ground truth RUL loading from `RUL_FD001.txt`

#### 2. Target Engineering ("Sentinel" Logic)
- **Training Set**: RUL = Max_Cycle - Current_Cycle (per engine unit)
- **Testing Set**: RUL = RUL_FD001.txt value + cycles remaining from last data point
- **Classification Target**: `failure_imminent` (binary: 1 if RUL ≤ 30 cycles, else 0)

#### 3. Deep-Pulse Feature Engineering
- **Constant Sensor Removal**: Identifies and drops sensors with zero variance (e.g., s_1, s_5, s_10, s_16, s_18, s_19)
- **Rolling Statistics**: Computes rolling `mean` and `std` for remaining sensors using windows of 10 and 25 cycles, grouped by `unit_nr` to preserve time-series integrity
- **Normalization**: Applies `MinMaxScaler` to ensure feature scale consistency for XGBoost

#### 4. ML Pipeline & Leakage Protection
- **GroupKFold Cross-Validation**: Uses `GroupKFold(n_splits=5)` on `unit_nr` to prevent data leakage—ensuring no engine unit appears in both training and validation sets simultaneously
- **Model**: XGBoost Classifier with `scale_pos_weight` automatically calculated from class imbalance
- **Hyperparameter Tuning**: Grid search over `max_depth` [3, 5, 7] and `learning_rate` [0.01, 0.1] with F1-score optimization

#### 5. Interpretability & Evaluation
- **SHAP (SHapley Additive exPlanations)**: TreeExplainer-based feature importance analysis to identify primary "pulses" of failure
- **Metrics**: F1-Score, Precision, and Recall (critical for maintenance applications)
- **Visualizations**:
  - Precision-Recall Curve: Trade-off analysis between false alarms and missed failures
  - SHAP Summary Plot: Feature-level interpretability
  - RUL Decay Plot: Sample engine unit degradation visualization

#### 6. Economic Impact Module
- **Cost Parameters**:
  - Unplanned Engine Failure: $50,000
  - Proactive Maintenance: $12,000
- **ROI Calculation**: Compares Sentinel-X strategy vs. traditional "Run-to-Failure" approach
- **Output**: Total cost savings, ROI percentage, and breakdown of true/false positives/negatives

## Installation

### Prerequisites
```bash
Python 3.8+
```

### macOS Setup (Required for XGBoost)

**Important**: On macOS, XGBoost requires the OpenMP runtime library. If you encounter an error about `libomp.dylib`, follow these steps:

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install libomp**:
   ```bash
   brew install libomp
   ```

3. **Reinstall XGBoost**:
   ```bash
   pip uninstall xgboost
   pip install xgboost
   ```

**Or use the automated setup script**:
```bash
chmod +x setup_xgboost.sh
./setup_xgboost.sh
```

### Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn rich
```

## Usage

### Basic Execution
```bash
python sentinel_x.py
```

### Programmatic Usage
```python
from sentinel_x import SentinelXEngine, sentinel_economic_impact

# Initialize Sentinel-X
sentinel = SentinelXEngine(data_dir="CMaps")

# Run complete pipeline
metrics = sentinel.run_pipeline()

# Economic impact analysis
economic_results = sentinel_economic_impact(
    metrics,
    sentinel.test_target,
    n_engines=100
)
```

## Output

The system generates:

1. **Terminal Dashboard**: Real-time status updates using the `rich` library, displaying:
   - Data loading progress
   - Feature engineering statistics
   - Model training metrics
   - Evaluation results (F1-Score, Precision, Recall)
   - Economic impact analysis

2. **Visualizations** (saved as PNG files):
   - `precision_recall_curve.png`: Precision-Recall trade-off analysis
   - `shap_summary_plot.png`: Feature importance and interpretability
   - `rul_decay_plot.png`: Sample engine unit RUL degradation over time

## Key Technical Highlights

### Time-Series Integrity
- **GroupKFold Strategy**: Ensures temporal and unit-level data integrity by preventing cross-contamination between training and validation sets
- **Rolling Statistics**: Computed per-unit to maintain engine-specific degradation patterns

### Interpretability
- **SHAP Integration**: Provides model-agnostic feature importance, enabling maintenance teams to understand which sensor readings are primary indicators of failure
- **Visual Analytics**: Multiple visualization layers for both technical and business stakeholders

### Production Readiness
- **Class-Based Architecture**: Modular `SentinelXEngine` class for easy integration and extension
- **PEP-8 Compliance**: Clean, maintainable code structure
- **Absolute Path Handling**: Robust file I/O for cross-platform compatibility

## Performance Considerations

- **Scalability**: Efficient pandas operations with vectorized rolling statistics
- **Memory Management**: Feature engineering optimized for large-scale time-series data
- **Computation**: SHAP values computed on sampled test set (100 instances) for performance

## Dataset Reference

**NASA C-MAPSS Dataset**
- Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

## License

This project is designed for research and educational purposes. Please refer to NASA's dataset licensing terms for commercial use.

## Author

Principal ML Engineer - Sentinel-X Development Team

---

**Sentinel-X**: *Predicting failures before they happen. Deep insights. Economic impact.*
