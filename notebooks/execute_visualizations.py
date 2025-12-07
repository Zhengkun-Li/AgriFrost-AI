#!/usr/bin/env python3
"""Execute visualization cells from notebook tutorial."""

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

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

# Create output directory
output_dir = project_root / "notebooks" / "outputs" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🎨 执行 Notebook 可视化单元格")
print("=" * 70)

# Load data
print("\n📂 加载数据...")
from src.data.loaders import DataLoader

data_path = project_root / "data/raw/frost-risk-forecast-challenge/cimis_all_stations.csv.gz"
loader = DataLoader()
df_raw = loader.load_raw_data(data_path)
print(f"✅ 数据加载成功: {df_raw.shape}")

# Process data (using pipeline)
print("\n🔄 处理数据...")
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
dataset_bundle = pipeline.run(
    data_path=data_path,
    horizons=[12],
    use_feature_engineering=True,
    sample_size=50000,  # Use smaller sample for faster visualization
    random_state=42
)

df_processed = dataset_bundle.data
print(f"✅ 数据处理完成: {df_processed.shape}")

# Prepare training data
print("\n📊 准备训练数据...")
from src.training.data_preparation import prepare_features_and_targets
from src.evaluation.validators import CrossValidator
from src.models.registry import get_model_class
from src.evaluation.metrics import MetricsCalculator

train_df, val_df, test_df = CrossValidator.time_split(
    df=df_processed,
    train_ratio=0.7,
    val_ratio=0.15,
    date_col="Date"
)

X_train, y_frost_train, y_temp_train = prepare_features_and_targets(
    df=train_df,
    horizon=12,
    track="top175_features"
)

X_val, y_frost_val, y_temp_val = prepare_features_and_targets(
    df=val_df,
    horizon=12,
    track="top175_features"
)

X_test, y_frost_test, y_temp_test = prepare_features_and_targets(
    df=test_df,
    horizon=12,
    track="top175_features"
)

print(f"✅ 数据准备完成: {X_train.shape}")

# Train models (quick training)
print("\n🤖 训练模型（快速版本）...")
ModelClass = get_model_class('lightgbm')

frost_model = ModelClass(
    config={
        'task_type': 'classification',
        'model_params': {
            'objective': 'binary',
            'n_estimators': 50,  # Fewer trees for faster training
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42,
            'verbosity': -1
        }
    }
)

frost_model.fit(X=X_train, y=y_frost_train, eval_set=[(X_val, y_frost_val)])
print("✅ 分类模型训练完成")

temp_model = ModelClass(
    config={
        'task_type': 'regression',
        'model_params': {
            'objective': 'regression',
            'n_estimators': 50,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42,
            'verbosity': -1
        }
    }
)

temp_model.fit(X=X_train, y=y_temp_train, eval_set=[(X_val, y_temp_val)])
print("✅ 回归模型训练完成")

# Generate predictions
y_frost_pred = frost_model.predict(X_test)
y_frost_proba = frost_model.predict_proba(X_test)
y_temp_pred = temp_model.predict(X_test)

# Cell 6: Time series visualization
print("\n" + "=" * 70)
print("📊 Cell 6: 时间序列可视化")
print("=" * 70)

df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_station = df_raw[df_raw['Stn Id'] == 2].copy()
df_station = df_station.sort_values('Date')
df_sample = df_station.tail(1000)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(df_sample['Date'], df_sample['Air Temp (C)'], label='Air Temperature', linewidth=1)
axes[0].axhline(y=0, color='r', linestyle='--', label='Frost Threshold (0°C)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Air Temperature Time Series (Station 2, Last 1000 Hours)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_sample['Date'], df_sample['Rel Hum (%)'], label='Relative Humidity', color='green', linewidth=1)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Relative Humidity (%)')
axes[1].set_title('Relative Humidity Time Series (Station 2, Last 1000 Hours)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = output_dir / "time_series.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 时间序列图已保存: {output_path}")

# Cell 7: Frost event statistics
print("\n" + "=" * 70)
print("📊 Cell 7: 霜冻事件统计")
print("=" * 70)

df_raw['is_frost'] = (df_raw['Air Temp (C)'] <= 0.0).astype(int)
df_raw['Month'] = pd.to_datetime(df_raw['Date']).dt.month
frost_by_month = df_raw.groupby('Month')['is_frost'].agg(['sum', 'count', 'mean'])
frost_by_month.columns = ['Frost Events', 'Total Observations', 'Frost Rate']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(frost_by_month.index, frost_by_month['Frost Rate'] * 100, color='steelblue')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Frost Rate (%)')
axes[0].set_title('Frost Rate by Month')
axes[0].set_xticks(range(1, 13))
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(frost_by_month.index, frost_by_month['Frost Events'], color='coral')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Number of Frost Events')
axes[1].set_title('Total Frost Events by Month')
axes[1].set_xticks(range(1, 13))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = output_dir / "frost_statistics.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 霜冻统计图已保存: {output_path}")
print(f"   总霜冻事件: {df_raw['is_frost'].sum():,}")
print(f"   霜冻率: {df_raw['is_frost'].mean()*100:.2f}%")

# Cell 17: Prediction visualization
print("\n" + "=" * 70)
print("📊 Cell 17: 预测结果可视化")
print("=" * 70)

metrics_calc = MetricsCalculator()
reg_metrics = metrics_calc.calculate_regression_metrics(
    y_true=y_temp_test,
    y_pred=y_temp_pred
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Temperature prediction vs true
axes[0, 0].scatter(y_temp_test, y_temp_pred, alpha=0.5, s=10)
axes[0, 0].plot([y_temp_test.min(), y_temp_test.max()], 
                [y_temp_test.min(), y_temp_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('True Temperature (°C)')
axes[0, 0].set_ylabel('Predicted Temperature (°C)')
axes[0, 0].set_title(f'Temperature Prediction (R² = {reg_metrics["r2"]:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Temperature prediction error distribution
temp_errors = y_temp_pred - y_temp_test
axes[0, 1].hist(temp_errors, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Prediction Error (°C)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Temperature Prediction Error Distribution (MAE = {reg_metrics["mae"]:.4f}°C)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_frost_test, y_frost_proba)
roc_auc = auc(fpr, tpr)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve for Frost Classification')
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(True, alpha=0.3)

# Frost probability distribution
axes[1, 1].hist(y_frost_proba[y_frost_test == 0], bins=50, alpha=0.7, label='No Frost', color='blue')
axes[1, 1].hist(y_frost_proba[y_frost_test == 1], bins=50, alpha=0.7, label='Frost', color='red')
axes[1, 1].set_xlabel('Predicted Frost Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Frost Probability Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = output_dir / "prediction_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 预测结果图已保存: {output_path}")

# Cell 19: Feature importance
print("\n" + "=" * 70)
print("📊 Cell 19: 特征重要性分析")
print("=" * 70)

try:
    # Get feature importance from LightGBM model
    feature_importance = frost_model.model.booster_.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 20 最重要特征:")
    print(importance_df.head(20).to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = importance_df.head(20)
    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('Top 20 Feature Importance (Frost Classification Model)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    output_path = output_dir / "feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存: {output_path}")
except Exception as e:
    print(f"⚠️  特征重要性分析失败: {e}")

# Cell 22: Prediction distribution
print("\n" + "=" * 70)
print("📊 Cell 22: 预测分布可视化")
print("=" * 70)

predictions_df = pd.DataFrame({
    'Frost_Probability': y_frost_proba[:100],
    'Temperature_Prediction_C': y_temp_pred[:100],
    'Frost_Risk': ['Low' if p < 0.1 else 'Medium' if p < 0.5 else 'High' for p in y_frost_proba[:100]]
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(predictions_df['Frost_Probability'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='High Risk Threshold (0.5)')
axes[0].set_xlabel('Frost Probability')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Frost Probability Predictions')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].hist(predictions_df['Temperature_Prediction_C'], bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Frost Threshold (0°C)')
axes[1].set_xlabel('Predicted Temperature (°C)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Temperature Predictions')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = output_dir / "prediction_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 预测分布图已保存: {output_path}")

print("\n" + "=" * 70)
print("🎉 所有可视化完成！")
print("=" * 70)
print(f"\n📁 图表保存位置: {output_dir}")
print("\n📊 生成的图表:")
for fig_file in sorted(output_dir.glob("*.png")):
    print(f"   • {fig_file.name} ({fig_file.stat().st_size / 1024:.1f} KB)")

print("\n💡 提示:")
print("   • 可以在浏览器中打开 Jupyter Notebook 查看交互式可视化")
print("   • 或在图像查看器中打开生成的 PNG 文件")
print("\n" + "=" * 70)

