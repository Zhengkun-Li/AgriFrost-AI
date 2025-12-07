# Scripts Directory

⚠️ **This directory has been simplified. All main functionality should use the new CLI commands: `python -m src.cli ...`**

## New Project Structure

This project has fully migrated to a unified CLI interface. The `scripts/` directory now only contains:

1. **Tool Scripts** (`tools/`) - Independent tool scripts, such as fetching metadata
2. **Test Scripts** (`test/`) - Project test scripts
3. **Shell Scripts** - Training-related shell scripts (if still in use)

## Using the New CLI

All main functionality has been migrated to `src/cli`:

```bash
# Training
python -m src.cli train single --model-name lightgbm --matrix-cell B --track top175_features --horizon-h 12
python -m src.cli train matrix --config config/pipeline/matrix_experiments.yaml

# Evaluation
python -m src.cli evaluate model --model-dir experiments/model_dir --config config/evaluation.yaml
python -m src.cli evaluate compare --model-dirs dir1 dir2 --output-dir comparison/
python -m src.cli evaluate matrix --experiments-dir experiments/ --output-dir matrix/

# Inference
python -m src.cli inference predict --model-dir experiments/model --input data.csv --output pred.csv

# Analysis
python -m src.cli analysis full --data-path data.csv --output-dir analysis/
python -m src.cli analysis compare-sets --feature-sets '[{"name": "raw", "path": "data/raw.csv"}]' --output-dir comparison/
```

## Get Help

```bash
# View all commands
python -m src.cli --help

# View help for specific commands
python -m src.cli train --help
python -m src.cli evaluate --help
python -m src.cli inference --help
python -m src.cli analysis --help
```

## Directory Structure

```
scripts/
├── README.md              # This file
├── MIGRATION.md           # Detailed migration guide (if upgrading from old version)
├── tools/                 # Independent tool scripts
│   ├── fetch_station_metadata.py
│   ├── generate_station_map.py
│   ├── run_full_pipeline.py
│   └── select_features.py
└── test/                  # Test scripts
    └── test_graph_builder.py
```

## Why Migrate?

1. **Unified Interface**: All commands use the same pattern `python -m src.cli <command> <subcommand>`
2. **Better Help**: Use `--help` to view detailed usage
3. **Type Safety**: Complete type checking and validation
4. **Extensibility**: Easy to add new commands and features
5. **Production Ready**: Comprehensive error handling and resource management
6. **Code Reuse**: Eliminate duplicate code, unified implementation

## Old Scripts Completely Removed

All old script wrappers have been completely removed:

- ❌ `scripts/train/` - Deleted (use `python -m src.cli train ...`)
- ❌ `scripts/evaluate/` - Deleted (use `python -m src.cli evaluate ...`)
- ❌ `scripts/inference/` - Deleted (use `python -m src.cli inference ...`)
- ❌ `scripts/analysis/` - Deleted (use `python -m src.cli analysis ...`)

All functionality has been integrated into `src/cli`. **This is a completely new project structure, no backward compatibility needed.**

## Tool Scripts

Scripts in the `tools/` directory are independent tools that may need to be run separately:

### `fetch_station_metadata.py`

Fetch CIMIS station metadata (including latitude, longitude, names, etc.).

**Usage:**
```bash
python scripts/tools/fetch_station_metadata.py
```

**Output:**
- `data/external/cimis_station_metadata.json`
- `data/external/cimis_station_metadata.csv`

### `generate_station_map.py`

Generate an interactive station distribution map showing all 18 CIMIS station locations.

**Usage:**
```bash
# First fetch station metadata (if needed)
python scripts/tools/fetch_station_metadata.py

# Generate map
python scripts/tools/generate_station_map.py
```

**Requirements:**
- Station metadata must be available (run `fetch_station_metadata.py` first)
- Optional: Mapbox token in `config/settings.json` for better map tiles

**Output:**
- `docs/figures/station_distribution_map.html` - Interactive HTML map

### `run_full_pipeline.py`

Run complete pipeline (specific use case)

### `select_features.py`

Feature selection tool

---

**Note**: These scripts can be used as needed, they do not depend on the new CLI structure.
