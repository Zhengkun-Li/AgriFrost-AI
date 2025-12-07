#!/usr/bin/env python3
"""Validate notebook file structure and syntax."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "tutorial.ipynb"


def validate_notebook_structure(nb_path: Path) -> bool:
    """Validate notebook JSON structure and basic syntax."""
    print(f"Validating notebook: {nb_path}")
    
    # Check if file exists
    if not nb_path.exists():
        print(f"❌ Error: Notebook file not found: {nb_path}")
        return False
    
    # Parse JSON
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: Failed to read notebook: {e}")
        return False
    
    print("✅ Notebook is valid JSON")
    
    # Check required fields
    required_fields = ['cells', 'metadata', 'nbformat', 'nbformat_minor']
    for field in required_fields:
        if field not in nb:
            print(f"❌ Error: Missing required field: {field}")
            return False
    
    print(f"✅ Required fields present: {', '.join(required_fields)}")
    
    # Check cells
    cells = nb['cells']
    print(f"✅ Total cells: {len(cells)}")
    
    cell_types = {}
    code_cells = []
    markdown_cells = []
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
        
        if cell_type == 'code':
            code_cells.append(i)
            # Check if code cell has source
            if 'source' not in cell:
                print(f"⚠️  Warning: Code cell {i} has no source")
        elif cell_type == 'markdown':
            markdown_cells.append(i)
            # Check if markdown cell has source
            if 'source' not in cell:
                print(f"⚠️  Warning: Markdown cell {i} has no source")
    
    print(f"✅ Cell types: {dict(cell_types)}")
    print(f"   - Code cells: {len(code_cells)}")
    print(f"   - Markdown cells: {len(markdown_cells)}")
    
    # Check code cell syntax (basic Python syntax check)
    syntax_errors = []
    for i in code_cells:
        cell = cells[i]
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source
        
        if code.strip():
            try:
                compile(code, f'<cell {i}>', 'exec')
            except SyntaxError as e:
                syntax_errors.append((i, str(e)))
    
    if syntax_errors:
        print(f"⚠️  Syntax errors found in {len(syntax_errors)} cells:")
        for cell_idx, error in syntax_errors[:5]:  # Show first 5
            print(f"   Cell {cell_idx}: {error}")
        if len(syntax_errors) > 5:
            print(f"   ... and {len(syntax_errors) - 5} more")
    else:
        print("✅ No syntax errors in code cells")
    
    # Check notebook format version
    nbformat = nb.get('nbformat', 0)
    nbformat_minor = nb.get('nbformat_minor', 0)
    print(f"✅ Notebook format: {nbformat}.{nbformat_minor}")
    
    if nbformat < 4:
        print("⚠️  Warning: Notebook format is older than 4.0, may have compatibility issues")
    
    return True


def main():
    """Main validation function."""
    success = validate_notebook_structure(NOTEBOOK_PATH)
    
    if success:
        print("\n✅ Notebook validation passed!")
        return 0
    else:
        print("\n❌ Notebook validation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

