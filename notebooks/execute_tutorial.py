#!/usr/bin/env python3
"""Execute notebook tutorial cells sequentially."""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Set plotting style (compatible with different matplotlib versions)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

print("=" * 70)
print("🚀 Execute Notebook Tutorial - Complete Workflow")
print("=" * 70)

# Cell 1: Import libraries (already done above)
print("\n✅ Cell 1: Libraries imported successfully!")
print(f"📁 Project root directory: {project_root}")
print(f"🐍 Python version: {sys.version.split()[0]}")

# Cell 3: Load raw data
print("\n" + "=" * 70)
print("📂 Cell 3: Load Raw Data")
print("=" * 70)

from src.data.loaders import DataLoader

data_path = project_root / "data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz"

if not data_path.exists():
    print(f"❌ Data file not found: {data_path}")
    sys.exit(1)

print(f"📂 Loading data: {data_path}")
loader = DataLoader()
df_raw = loader.load_raw_data(data_path)
print(f"✅ Data loaded successfully!")
print(f"   Shape: {df_raw.shape}")
print(f"   Columns: {len(df_raw.columns)}")
print(f"   Time range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
print(f"   Number of stations: {df_raw['Stn Id'].nunique()}")

# Cell 9: Configure data pipeline
print("\n" + "=" * 70)
print("⚙️  Cell 9: Configure Data Processing Pipeline")
print("=" * 70)

from src.data import DataPipeline

config = {
    "cleaning": {
        "config_path": str(project_root / "config/data_cleaning.yaml")
    },
    "labels": {
        "threshold": 0.0
    },
    "feature_engineering": {
        "enabled": True,
        "feature_selection": {
            "method": "top_k",
            "top_k": 175
        }
    },
    "random_state": 42
}

pipeline = DataPipeline(config=config)
print("✅ Data pipeline created successfully!")

# Cell 10: Process data
print("\n" + "=" * 70)
print("🔄 Cell 10: Process Data (Using Sampling)")
print("=" * 70)

print("   ⚠️  Note: For demonstration speed, we use sampled data (100,000 rows)")
print("   💡 For actual training, remove sample_size parameter to use full data")

dataset_bundle = pipeline.run(
    data_path=data_path,
    horizons=[12],
    use_feature_engineering=True,
    sample_size=100000,
    random_state=42
)

df_processed = dataset_bundle.data
print(f"✅ Data processing complete!")
print(f"   Processed shape: {df_processed.shape}")
print(f"   Number of features: {len(dataset_bundle.feature_columns)}")
print(f"   Number of labels: {len(dataset_bundle.label_columns)}")

# Cell 12: Prepare training data
print("\n" + "=" * 70)
print("📊 Cell 12: Prepare Training Data")
print("=" * 70)

from src.training.data_preparation import prepare_features_and_targets
from src.evaluation.validators import CrossValidator
from src.models.registry import get_model_class

print("📊 Performing time series split...")
train_df, val_df, test_df = CrossValidator.time_split(
    df=df_processed,
    train_ratio=0.7,
    val_ratio=0.15,
    date_col="Date"
)

print(f"   Training set: {len(train_df)} samples")
print(f"   Validation set: {len(val_df)} samples")
print(f"   Test set: {len(test_df)} samples")

print("\n🔧 Preparing training set features and labels...")
X_train, y_frost_train, y_temp_train = prepare_features_and_targets(
    df=train_df,
    horizon=12,
    track="top175_features"
)

print("🔧 Preparing validation set features and labels...")
X_val, y_frost_val, y_temp_val = prepare_features_and_targets(
    df=val_df,
    horizon=12,
    track="top175_features"
)

print("🔧 Preparing test set features and labels...")
X_test, y_frost_test, y_temp_test = prepare_features_and_targets(
    df=test_df,
    horizon=12,
    track="top175_features"
)

print("\n✅ Data preparation complete!")
print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   Validation set: {X_val.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Frost events (training set): {y_frost_train.sum()} ({y_frost_train.mean()*100:.2f}%)")
print(f"   Average temperature (training set): {y_temp_train.mean():.2f}°C")

# Cell 13: Train models
print("\n" + "=" * 70)
print("🤖 Cell 13: Train Models")
print("=" * 70)

ModelClass = get_model_class('lightgbm')

print("🤖 Training frost classification model (LightGBM)...")
frost_model = ModelClass(
    config={
        'task_type': 'classification',
        'model_params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'random_state': 42,
            'verbosity': -1
        }
    }
)

frost_model.fit(
    X=X_train,
    y=y_frost_train,
    eval_set=[(X_val, y_frost_val)]
)
print("✅ Classification model training complete!")

print("🤖 Training temperature regression model (LightGBM)...")
temp_model = ModelClass(
    config={
        'task_type': 'regression',
        'model_params': {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 7,
            'random_state': 42,
            'verbosity': -1
        }
    }
)

temp_model.fit(
    X=X_train,
    y=y_temp_train,
    eval_set=[(X_val, y_temp_val)]
)
print("✅ Regression model training complete!")

# Cell 15: Evaluate classification model
print("\n" + "=" * 70)
print("📊 Cell 15: Evaluate Classification Model")
print("=" * 70)

from src.evaluation.metrics import MetricsCalculator

y_frost_pred = frost_model.predict(X_test)
y_frost_proba = frost_model.predict_proba(X_test)

metrics_calc = MetricsCalculator()
class_metrics = metrics_calc.calculate_classification_metrics(
    y_true=y_frost_test,
    y_pred=y_frost_pred,
    y_proba=y_frost_proba
)

print("📊 Classification Model Performance (Test Set):")
print(f"   ROC-AUC: {class_metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in class_metrics else "   ROC-AUC: N/A")
print(f"   PR-AUC: {class_metrics.get('pr_auc', 'N/A'):.4f}" if 'pr_auc' in class_metrics else "   PR-AUC: N/A")
print(f"   Brier Score: {class_metrics.get('brier_score', 'N/A'):.4f}" if 'brier_score' in class_metrics else "   Brier Score: N/A")
if 'ece' in class_metrics:
    print(f"   ECE: {class_metrics['ece']:.4f}")
print(f"   Accuracy: {class_metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in class_metrics else "   Accuracy: N/A")
print(f"   Precision: {class_metrics.get('precision', 'N/A'):.4f}" if 'precision' in class_metrics else "   Precision: N/A")
print(f"   Recall: {class_metrics.get('recall', 'N/A'):.4f}" if 'recall' in class_metrics else "   Recall: N/A")
print(f"   F1 Score: {class_metrics.get('f1_score', 'N/A'):.4f}" if 'f1_score' in class_metrics else "   F1 Score: N/A")

# Cell 16: Evaluate regression model
print("\n" + "=" * 70)
print("📊 Cell 16: Evaluate Regression Model")
print("=" * 70)

y_temp_pred = temp_model.predict(X_test)

reg_metrics = metrics_calc.calculate_regression_metrics(
    y_true=y_temp_test,
    y_pred=y_temp_pred
)

print("📊 Regression Model Performance (Test Set):")
print(f"   MAE: {reg_metrics['mae']:.4f}°C")
print(f"   RMSE: {reg_metrics['rmse']:.4f}°C")
print(f"   R²: {reg_metrics['r2']:.4f}")
print(f"   MAPE: {reg_metrics.get('mape', 'N/A')}")

# Cell 21: Generate predictions
print("\n" + "=" * 70)
print("🔮 Cell 21: Generate Predictions")
print("=" * 70)

new_data = X_test[:100].copy()

frost_proba_predictions = frost_model.predict_proba(new_data)
temp_predictions = temp_model.predict(new_data)

predictions_df = pd.DataFrame({
    'Frost_Probability': frost_proba_predictions,
    'Temperature_Prediction_C': temp_predictions,
    'Frost_Risk': ['Low' if p < 0.1 else 'Medium' if p < 0.5 else 'High' for p in frost_proba_predictions]
})

print("📊 Prediction Results Example (First 20):")
print(predictions_df.head(20).to_string(index=True))

high_risk = (predictions_df['Frost_Probability'] > 0.5).sum()
print(f"\n⚠️  High-risk predictions (probability > 0.5): {high_risk} / {len(predictions_df)} ({high_risk/len(predictions_df)*100:.1f}%)")

print("\n" + "=" * 70)
print("🎉 Notebook Tutorial Execution Complete!")
print("=" * 70)
print("\n✅ Completed:")
print("   • Data loading and exploration")
print("   • Data processing pipeline")
print("   • Model training (classification + regression)")
print("   • Model evaluation")
print("   • Prediction generation")
print("\n📊 Model Performance Summary:")
print(f"   • Classification ROC-AUC: {class_metrics['roc_auc']:.4f}")
print(f"   • Regression R²: {reg_metrics['r2']:.4f}")
print(f"   • Regression MAE: {reg_metrics['mae']:.4f}°C")
print("\n" + "=" * 70)

