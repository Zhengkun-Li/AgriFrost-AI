#!/bin/bash
# Run all scripts to reproduce manuscript results
# Usage: ./run_all.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=" * 70
echo "论文结果重现脚本 - 完整流程"
echo "=" * 70
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "脚本目录: $SCRIPT_DIR"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ 已激活虚拟环境"
    echo "   Python: $(which python3)"
    echo ""
fi

# Step 1: Collect data
echo "=" * 70
echo "步骤1: 收集数据"
echo "=" * 70
echo ""

echo "1.1 收集所有指标数据..."
python3 "$SCRIPT_DIR/collect_all_metrics_for_supplementary.py"
echo ""

echo "1.2 收集所有特征重要性数据..."
python3 "$SCRIPT_DIR/collect_all_feature_importance_for_supplementary.py"
echo ""

echo "1.3 运行LOSO评估（如果还没有运行）..."
echo "   注意: LOSO评估可能需要较长时间，可以跳过或单独运行"
read -p "   是否运行LOSO评估? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 "$SCRIPT_DIR/run_loso_for_abc_matrices.py"
else
    echo "   跳过LOSO评估"
fi
echo ""

# Step 2: Generate figures
echo "=" * 70
echo "步骤2: 生成图表"
echo "=" * 70
echo ""

echo "2.1 生成Figure 6: 类别平衡训练影响分析..."
python3 "$SCRIPT_DIR/generate_class_balance_comparison_figure.py"
echo ""

echo "2.2 生成Figure 7: 矩阵A vs 矩阵B性能对比..."
python3 "$SCRIPT_DIR/generate_matrix_ab_comparison_figure.py"
echo ""

echo "2.3 生成Figure 8: 矩阵A特征重要性分析..."
python3 "$SCRIPT_DIR/generate_matrix_a_feature_importance_figure.py"
echo ""

echo "2.4 生成Figure 9: 矩阵B特征类别重要性分析..."
python3 "$SCRIPT_DIR/generate_matrix_b_feature_category_importance_cumulative.py"
echo ""

echo "2.5 生成Figure 10: 矩阵A vs 矩阵C性能对比..."
python3 "$SCRIPT_DIR/generate_matrix_ac_comparison_figure.py"
echo ""

echo "2.6 生成Figure 11: 矩阵C特征类别重要性分析..."
python3 "$SCRIPT_DIR/generate_matrix_c_feature_category_importance_cumulative.py"
echo ""

echo "2.7 生成Figure 12: 矩阵C半径敏感性分析..."
if [ -f "$SCRIPT_DIR/generate_matrix_c_radius_sensitivity_plot.py" ]; then
    python3 "$SCRIPT_DIR/generate_matrix_c_radius_sensitivity_plot.py"
else
    echo "   ⚠️  脚本不存在，跳过"
fi
echo ""

echo "=" * 70
echo "✅ 所有脚本执行完成！"
echo "=" * 70
echo ""
echo "输出位置:"
echo "  - 数据文件: docs/manuscript/Supplementary_lighgbm_abc/*.csv"
echo "  - 图表文件: docs/manuscript/Supplementary_lighgbm_abc/figures/*.png"
echo ""

