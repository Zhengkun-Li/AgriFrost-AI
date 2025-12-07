# AgriFrost-AI Documentation Center

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="200"/>

## AI-Powered Frost Risk Prediction System for California Agriculture

**A Comprehensive Framework for Multi-Horizon, Multi-Station Frost Forecasting**

*F3 Innovate Frost Risk Forecasting Challenge (2025)*

</div>

---

## 📚 Documentation Navigation

### 🚀 Quick Start

- **[Quick Start Guide](guides/QUICK_START.md)** - Get started quickly, begin using in 5 minutes
- **[User Guide](guides/USER_GUIDE.md)** - Complete usage instructions and examples

### 📖 Core Guides

- **[Implementation Guide](guides/IMPLEMENTATION_GUIDE.md)** - High-level implementation guide (English)
  - [Chinese Version](guides/IMPLEMENTATION_GUIDE_CN.md) - High-level implementation guide (中文/Chinese)

### 🔬 Feature Related

- **[Feature Engineering Guide](features/FEATURE_GUIDE.md)** - Complete feature engineering guide
  - Feature categories and implementation
  - QC field processing
  - Jul feature details
  - Feature selection strategies
  - Feature engineering recommendations

- **[Feature Importance Guide](features/FEATURE_IMPORTANCE.md)** - Feature importance analysis and selection
  - Feature importance evaluation methods
  - Model-specific vs dataset-specific
  - Numerical vs percentage representation
  - Feature selection strategies

### 📊 Experiment Analysis Reports

- **[B Experiment Feature Importance Analysis](features/experiments/B_EXPERIMENT_FEATURE_IMPORTANCE_ANALYSIS.md)** - Single horizon analysis
- **[B Experiment Feature Importance Comprehensive Analysis](features/experiments/B_EXPERIMENT_FEATURE_IMPORTANCE_ALL_HORIZONS.md)** - All horizons analysis
- **[LightGBM A vs B Comparison](features/experiments/LIGHTGBM_A_VS_B_COMPARISON.md)** - Model comparison report

### 🎓 Training and Evaluation

- **[Training Guide](training/TRAINING_GUIDE.md)** - Complete training and evaluation guide
  - Training configuration
  - LOSO evaluation
  - Training monitoring
  - Performance comparison
  - Command details

### 🤖 Model Related

- **[Models Guide](models/MODELS_GUIDE.md)** - Detailed descriptions of all models
  - Model principles
  - Advantages and disadvantages analysis
  - Use cases

### 🔮 Inference Related

- **[Inference Guide](inference/INFERENCE_GUIDE.md)** - Model inference and usage guide

### 🔧 Technical Documentation

- **[Technical Documentation](technical/TECHNICAL_DOCUMENTATION.md)** - Technical details and API reference
- **[Data Documentation](technical/DATA_DOCUMENTATION.md)** - Data processing documentation

### 📄 Reference Materials

- **[Challenge Brief](reference/F3-Innovate-Frost-Risk-Forecasting-Challenge-Brief.pdf)** - PDF
- **[Challenge Slides](reference/F3-Innovate-Frost-Risk-Forecast-Data-Challenge-slides.pdf)** - PDF

---

## 📋 Documentation Structure

```
docs/
├── README.md                    # Main entry (this file)
├── guides/                      # Core guides
│   ├── QUICK_START.md
│   ├── USER_GUIDE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── IMPLEMENTATION_GUIDE_CN.md
├── features/                    # Feature related
│   ├── FEATURE_GUIDE.md
│   ├── FEATURE_IMPORTANCE.md
│   └── experiments/             # Experiment reports
│       ├── B_EXPERIMENT_FEATURE_IMPORTANCE_ANALYSIS.md
│       ├── B_EXPERIMENT_FEATURE_IMPORTANCE_ALL_HORIZONS.md
│       └── LIGHTGBM_A_VS_B_COMPARISON.md
├── training/                    # Training related
│   └── TRAINING_GUIDE.md
├── models/                      # Model related
│   └── MODELS_GUIDE.md
├── inference/                   # Inference related
│   └── INFERENCE_GUIDE.md
├── technical/                   # Technical documentation
│   ├── TECHNICAL_DOCUMENTATION.md
│   └── DATA_DOCUMENTATION.md
└── reference/                   # Reference materials
    └── *.pdf
```

---

## 🎯 Quick Navigation

### I want to...

- **Quickly start using the project** → [Quick Start Guide](guides/QUICK_START.md)
- **Learn how to use training commands** → [Training Guide](training/TRAINING_GUIDE.md)
- **Understand feature engineering** → [Feature Engineering Guide](features/FEATURE_GUIDE.md)
- **Analyze feature importance** → [Feature Importance Guide](features/FEATURE_IMPORTANCE.md)
- **View experiment analysis** → [Experiment Reports](features/experiments/)
- **Learn model principles** → [Models Guide](models/MODELS_GUIDE.md)
- **Use model inference** → [Inference Guide](inference/INFERENCE_GUIDE.md)
- **View technical details** → [Technical Documentation](technical/TECHNICAL_DOCUMENTATION.md)

---

## 📝 Documentation Update History

- **2025-11-20**: Documentation reorganization, streamlined from 23 files to ~12-15 files, organized by function
- **2025-11-19**: Added feature importance analysis and experiment reports
- **2025-11-12**: Initial documentation creation

---

**Last Updated**: 2025-12-06  
**Document Version**: 3.0
