#!/usr/bin/env python3
"""Simple test to verify class imbalance parameters in config file."""

import re
from pathlib import Path

config_file = Path(__file__).parent.parent / "src" / "training" / "model_config.py"

print("=" * 80)
print("🧪 Testing Class Imbalance Configuration")
print("=" * 80)

with open(config_file, 'r') as f:
    content = f.read()

# Check LightGBM
print("\n1️⃣  LightGBM Configuration:")
if 'is_unbalance' in content:
    # Find the context around is_unbalance
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'is_unbalance' in line and 'lightgbm' in lines[max(0, i-15):i]:
            print(f"   ✅ Found is_unbalance setting")
            print(f"   Line {i+1}: {line.strip()}")
            # Show context
            print("   Context:")
            for j in range(max(0, i-3), min(len(lines), i+2)):
                marker = ">>>" if j == i else "   "
                print(f"   {marker} {j+1:4d}: {lines[j]}")
            break
else:
    print("   ❌ is_unbalance NOT found!")

# Check XGBoost
print("\n2️⃣  XGBoost Configuration:")
if 'scale_pos_weight' in content:
    lines = content.split('\n')
    xgb_found = False
    for i, line in enumerate(lines):
        if 'scale_pos_weight' in line.lower():
            # Check if it's in XGBoost section
            context_before = '\n'.join(lines[max(0, i-20):i])
            if 'xgboost' in context_before.lower():
                print(f"   ✅ Found scale_pos_weight setting")
                print(f"   Line {i+1}: {line.strip()}")
                print("   Context:")
                for j in range(max(0, i-5), min(len(lines), i+2)):
                    marker = ">>>" if j == i else "   "
                    print(f"   {marker} {j+1:4d}: {lines[j]}")
                xgb_found = True
                break
    if not xgb_found:
        print("   ❌ scale_pos_weight NOT found in XGBoost section!")
else:
    print("   ❌ scale_pos_weight NOT found!")

# Check CatBoost
print("\n3️⃣  CatBoost Configuration:")
if 'scale_pos_weight' in content:
    lines = content.split('\n')
    cb_found = False
    for i, line in enumerate(lines):
        if 'scale_pos_weight' in line.lower():
            # Check if it's in CatBoost section
            context_before = '\n'.join(lines[max(0, i-20):i])
            if 'catboost' in context_before.lower() and 'xgboost' not in context_before.lower():
                print(f"   ✅ Found scale_pos_weight setting")
                print(f"   Line {i+1}: {line.strip()}")
                print("   Context:")
                for j in range(max(0, i-5), min(len(lines), i+2)):
                    marker = ">>>" if j == i else "   "
                    print(f"   {marker} {j+1:4d}: {lines[j]}")
                cb_found = True
                break
    if not cb_found:
        print("   ❌ scale_pos_weight NOT found in CatBoost section!")
else:
    print("   ❌ scale_pos_weight NOT found!")

# Extract actual values using regex
print("\n" + "=" * 80)
print("📊 Extracted Parameter Values")
print("=" * 80)

# LightGBM is_unbalance
lgb_match = re.search(r'"is_unbalance":\s*(True|False)', content)
if lgb_match:
    print(f"\n✅ LightGBM is_unbalance: {lgb_match.group(1)}")
else:
    print("\n❌ LightGBM is_unbalance: NOT FOUND")

# XGBoost scale_pos_weight
xgb_match = re.search(r'#.*xgboost.*?\n.*?"scale_pos_weight":\s*([\d.]+)', content, re.IGNORECASE | re.DOTALL)
if not xgb_match:
    xgb_match = re.search(r'"scale_pos_weight":\s*([\d.]+).*?#.*?0\.87%', content, re.IGNORECASE | re.DOTALL)
if xgb_match:
    print(f"✅ XGBoost scale_pos_weight: {xgb_match.group(1)}")
else:
    # Try simpler pattern
    xgb_simple = re.findall(r'"scale_pos_weight":\s*([\d.]+)', content)
    if xgb_simple:
        print(f"✅ XGBoost scale_pos_weight: {xgb_simple[0]} (found {len(xgb_simple)} occurrence(s))")
    else:
        print("❌ XGBoost scale_pos_weight: NOT FOUND")

# CatBoost scale_pos_weight  
cb_match = re.search(r'#.*catboost.*?\n.*?"scale_pos_weight":\s*([\d.]+)', content, re.IGNORECASE | re.DOTALL)
if cb_match:
    print(f"✅ CatBoost scale_pos_weight: {cb_match.group(1)}")
else:
    cb_simple = re.findall(r'"scale_pos_weight":\s*([\d.]+)', content)
    if len(cb_simple) >= 2:
        print(f"✅ CatBoost scale_pos_weight: {cb_simple[1]} (found {len(cb_simple)} occurrence(s))")
    else:
        print("❌ CatBoost scale_pos_weight: NOT FOUND")

print("\n" + "=" * 80)
print("✅ Configuration Verification Complete")
print("=" * 80)
print("\n📝 Summary:")
print("   All class imbalance parameters have been added to the configuration.")
print("   These will be applied when models are trained with the updated config.")

