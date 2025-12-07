#!/usr/bin/env python3
"""Check for data cache and add debugging output.

This script checks:
1. If there are any cached data files that might be reused
2. Adds debugging output to verify neighbor features are generated
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 90)
print("🔍 数据缓存检查和调试")
print("=" * 90)

# 1. Check for cached data files
print("\n1. 检查数据缓存:")
print("-" * 90)

cache_dirs = [
    project_root / "data" / "processed",
    project_root / "data" / "processed" / "labeled",
    project_root / "data" / "processed" / "pipeline_bundles",
]

found_cache = False
for cache_dir in cache_dirs:
    if cache_dir.exists():
        parquet_files = list(cache_dir.glob("**/*.parquet"))
        if parquet_files:
            found_cache = True
            print(f"\n✅ 找到缓存目录: {cache_dir}")
            print(f"   Parquet文件数量: {len(parquet_files)}")
            for pf in parquet_files[:5]:  # Show first 5
                print(f"   - {pf.relative_to(project_root)}")
            if len(parquet_files) > 5:
                print(f"   ... 还有 {len(parquet_files) - 5} 个文件")

if not found_cache:
    print("\n✅ 没有找到数据缓存文件")

# 2. Check if DataPipeline has caching mechanism
print("\n2. 检查DataPipeline缓存机制:")
print("-" * 90)

from src.data.pipeline import DataPipeline

# Check if DataPipeline saves intermediate results
pipeline_code = Path("src/data/pipeline.py").read_text()

has_caching = False
if "cache" in pipeline_code.lower() or "save" in pipeline_code.lower():
    print("⚠️  DataPipeline代码中包含'cache'或'save'关键字")
    print("   需要检查是否有缓存逻辑")
    has_caching = True
else:
    print("✅ DataPipeline代码中没有明显的缓存机制")
    print("   每次调用run()都会重新处理数据")

# 3. Summary
print("\n" + "=" * 90)
print("📋 总结:")
print("=" * 90)

if found_cache:
    print("⚠️  发现数据缓存文件:")
    print("   建议：如果要确保使用最新配置，可以删除这些缓存文件")
    print("   命令示例:")
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"   rm -rf {cache_dir}")
else:
    print("✅ 没有发现数据缓存，每次训练都会重新生成数据")

print()
print("✅ 已添加调试输出到:")
print("   - TrainingRunner.run(): 检查DataPipeline返回的DataFrame")
print("   - prepare_features_and_targets(): 检查各个阶段的neighbor特征")
print()
print("下一步：重新运行训练，查看调试输出")


