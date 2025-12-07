# Deep Learning Models Quick Test Guide

## Overview

`test_deep_models_quick.py` is a quick test script to verify that deep learning models (LSTM, GRU, TCN) are working correctly.

## Features

- ✅ **Memory friendly**: Uses only 5000 samples
- ✅ **Quick verification**: Each model trains for 5 epochs (~2-5 minutes)
- ✅ **Complete testing**: Tests both classification and regression tasks
- ✅ **Non-interfering**: Does not interfere with running Random Forest training

## Usage

```bash
# Navigate to project directory
cd /home/zhengkun-li/frost-risk-forecast-challenge

# Run test
python scripts/test/test_deep_models_quick.py
```

## Test Content

The script tests the following models:
1. **LSTM** - Long Short-Term Memory network
2. **GRU** - Gated Recurrent Unit network
3. **TCN** - Temporal Convolutional Network

Each model is tested for:
- Classification task (frost prediction)
- Regression task (temperature prediction)

## Configuration Parameters

You can modify the following parameters in the script:

```python
models_to_test = ['lstm', 'gru', 'tcn']  # List of models to test
horizon = 3                               # Forecast time window (hours)
max_samples = 5000                        # Number of samples to use
```

## Expected Output

When tests succeed, it will display:
```
✅ LSTM test PASSED!
✅ GRU test PASSED!
✅ TCN test PASSED!

TEST SUMMARY
============
LSTM: ✅ PASSED
GRU: ✅ PASSED
TCN: ✅ PASSED

Total: 3/3 models passed
```

## Notes

1. **Memory usage**: Tests use small amounts of data, but deep learning models still require some memory. If system memory is tight, you can reduce `max_samples`.

2. **GPU support**: If the system has a GPU, PyTorch will automatically use it. You can set `CUDA_VISIBLE_DEVICES` in the script to control which GPU to use.

3. **Training time**: Each model takes ~2-5 minutes, depending on hardware configuration.

4. **Dependencies**: Ensure PyTorch is installed:
   ```bash
   pip install torch
   ```

## Troubleshooting

**Issue**: ImportError: PyTorch is required
**Solution**: Install PyTorch: `pip install torch`

**Issue**: CUDA out of memory
**Solution**: Reduce `max_samples` or `batch_size` parameter

**Issue**: Test fails but error message is unclear
**Solution**: Check log output, usually there will be detailed error messages

## Next Steps

After tests pass, you can use the complete training workflow to train deep learning models:

```bash
python -m src.cli train single \
    --model-name lstm \
    --matrix-cell A \
    --track raw \
    --horizon-h 3 \
    --output-dir experiments/lstm/raw/A/full_training/horizon_3h
```
