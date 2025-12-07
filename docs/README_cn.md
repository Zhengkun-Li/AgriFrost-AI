# AgriFrost-AI 文档中心

<div align="center">

<img src="logo/AgriFrost-AI-transparent.png" alt="AgriFrost-AI Logo" width="200"/>

## 加州农业AI驱动的霜冻风险预测系统

**多时间范围、多站点霜冻预测的综合框架**

*F3 Innovate 霜冻风险预测挑战赛 (2025)*

</div>

---

## 📚 文档导航

### 🚀 快速开始

- **[快速开始指南](guides/QUICK_START.md)** - 快速上手项目，5分钟开始使用
- **[用户指南](guides/USER_GUIDE.md)** - 完整的使用说明和示例

### 📖 核心指南

- **[实现指南](guides/IMPLEMENTATION_GUIDE.md)** - 高层实现指南（英文）
  - [中文版本](guides/IMPLEMENTATION_GUIDE_CN.md) - 高层实现指南（中文）

### 🔬 特征相关

- **[特征工程指南](features/FEATURE_GUIDE.md)** - 完整的特征工程指南
  - 特征类别和实现
  - QC字段处理
  - Jul特征详解
  - 特征选择策略
  - 特征工程建议

- **[特征重要性指南](features/FEATURE_IMPORTANCE.md)** - 特征重要性分析和选择
  - 特征重要性评估方法
  - 模型特定 vs 数据集特定
  - 数值 vs 百分比表示
  - 特征选择策略

### 📊 实验分析报告

- **[B实验特征重要性分析](features/experiments/B_EXPERIMENT_FEATURE_IMPORTANCE_ANALYSIS.md)** - 单horizon分析
- **[B实验特征重要性综合分析](features/experiments/B_EXPERIMENT_FEATURE_IMPORTANCE_ALL_HORIZONS.md)** - 所有horizon分析
- **[LightGBM A vs B 对比](features/experiments/LIGHTGBM_A_VS_B_COMPARISON.md)** - 模型对比报告

### 🎓 训练和评估

- **[训练指南](training/TRAINING_GUIDE.md)** - 完整的训练和评估指南
  - 训练配置
  - LOSO评估
  - 训练监控
  - 性能对比
  - 命令详解

### 🤖 模型相关

- **[模型指南](models/MODELS_GUIDE.md)** - 所有模型的详细说明
  - 模型原理
  - 优缺点分析
  - 使用场景

### 🔮 推理相关

- **[推理指南](inference/INFERENCE_GUIDE.md)** - 模型推理和使用指南

### 🔧 技术文档

- **[技术文档](technical/TECHNICAL_DOCUMENTATION.md)** - 技术细节和API参考
- **[数据文档](technical/DATA_DOCUMENTATION.md)** - 数据处理文档

### 📄 参考资料

- **[挑战赛简介](reference/F3-Innovate-Frost-Risk-Forecasting-Challenge-Brief.pdf)** - PDF
- **[挑战赛幻灯片](reference/F3-Innovate-Frost-Risk-Forecast-Data-Challenge-slides.pdf)** - PDF

---

## 📋 文档结构

```
docs/
├── README.md                    # 主入口（本文件）
├── guides/                      # 核心指南
│   ├── QUICK_START.md
│   ├── USER_GUIDE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── IMPLEMENTATION_GUIDE_CN.md
├── features/                    # 特征相关
│   ├── FEATURE_GUIDE.md
│   ├── FEATURE_IMPORTANCE.md
│   └── experiments/             # 实验报告
│       ├── B_EXPERIMENT_FEATURE_IMPORTANCE_ANALYSIS.md
│       ├── B_EXPERIMENT_FEATURE_IMPORTANCE_ALL_HORIZONS.md
│       └── LIGHTGBM_A_VS_B_COMPARISON.md
├── training/                    # 训练相关
│   └── TRAINING_GUIDE.md
├── models/                      # 模型相关
│   └── MODELS_GUIDE.md
├── inference/                   # 推理相关
│   └── INFERENCE_GUIDE.md
├── technical/                   # 技术文档
│   ├── TECHNICAL_DOCUMENTATION.md
│   └── DATA_DOCUMENTATION.md
└── reference/                   # 参考资料
    └── *.pdf
```

---

## 🎯 快速导航

### 我想...

- **快速开始使用项目** → [快速开始指南](guides/QUICK_START.md)
- **了解如何使用训练命令** → [训练指南](training/TRAINING_GUIDE.md)
- **理解特征工程** → [特征工程指南](features/FEATURE_GUIDE.md)
- **分析特征重要性** → [特征重要性指南](features/FEATURE_IMPORTANCE.md)
- **查看实验分析** → [实验报告](features/experiments/)
- **了解模型原理** → [模型指南](models/MODELS_GUIDE.md)
- **使用模型推理** → [推理指南](inference/INFERENCE_GUIDE.md)
- **查看技术细节** → [技术文档](technical/TECHNICAL_DOCUMENTATION.md)

---

## 📝 文档更新记录

- **2025-11-20**: 文档重组，从23个文件精简到~12-15个文件，按功能分类组织
- **2025-11-19**: 添加特征重要性分析和实验报告
- **2025-11-12**: 初始文档创建

---

**最后更新**: 2025-11-20  
**文档版本**: 3.0

