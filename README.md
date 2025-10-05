# Transliteration Model Training Pipeline

Complete modular pipeline for training and evaluating 9 transliteration models.

## Models

1. **RNN** - Vanilla RNN encoder-decoder
2. **LSTM** - Vanilla LSTM encoder-decoder
3. **GRU** - Vanilla GRU encoder-decoder
4. **RNN + Attention** - RNN with Bahdanau attention
5. **LSTM + Attention** - LSTM with Bahdanau attention
6. **GRU + Attention** - GRU with Bahdanau attention
7. **BiLSTM + Attention** - Bidirectional LSTM with attention
8. **Transformer** - Standard Transformer architecture
9. **ByT5** - Pre-trained byte-level T5

## File Structure

```
project/
├── models/
│   ├── __init__.py
│   ├── base_seq2seq.py       # Base encoder/decoder components
│   ├── vanilla_models.py     # RNN/LSTM/GRU models
│   └── attention_models.py   # Models with attention
├── config.py                 # All configurations
├── trainer.py                # Generic training loop
├── evaluator.py              # Evaluation and metrics
├── train_all.py              # Main orchestrator script
├── data_inspection.py        # Your existing data loader
├── attention.py              # Your existing Transformer
├── train_biLSTM.py          # Your existing BiLSTM
├── train_byt5.py            # Your existing ByT5
└── results/
    ├── checkpoints/          # Saved models
    ├── predictions_all.csv   # All predictions
    └── metrics_summary.csv   # Summary statistics
```

## Installation

```bash
# Required packages
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy matplotlib seaborn tqdm
```

## Usage

### Train All Models

```bash
python train_all.py
```

### Train Specific Models

```bash
# Train only basic models
python train_all.py --models rnn,lstm,gru

# Train only attention models
python train_all.py --models rnn_attention,lstm_attention,gru_attention

# Train advanced models
python train_all.py --models bilstm_attention,transformer,byt5
```

### Evaluate Only (Skip Training)

```bash
python train_all.py --eval-only
```

This will evaluate all models that have saved checkpoints.

## Output Files

### predictions_all.csv

Contains detailed predictions for every test sample from all models:

| model | source | target | prediction | correct | edit_distance | source_len | inference_ms |
|-------|--------|--------|------------|---------|---------------|------------|--------------|
| rnn | namaste | नमस्ते | नमस्ते | True | 0 | 7 | 2.3 |
| lstm | namaste | नमस्ते | नमस्ते | True | 0 | 7 | 2.8 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### metrics_summary.csv

Aggregated metrics for each model:

| model | accuracy | avg_edit_distance | total_params | avg_inference_ms |
|-------|----------|-------------------|--------------|------------------|
| transformer | 0.8945 | 0.23 | 12.5M | 5.2 |
| lstm_attention | 0.8623 | 0.34 | 3.2M | 3.1 |
| ... | ... | ... | ... | ... |

## Configuration

Edit `config.py` to modify:

- **Data paths**: Train/val/test dataset locations
- **Hyperparameters**: Hidden dim, layers, dropout, etc.
- **Training params**: Batch size, learning rate, epochs
- **Model-specific settings**: Transformer heads, ByT5 model name

## Features

- **Automatic checkpoint management**: Skips already-trained models
- **Consistent evaluation**: All models tested on same test set
- **Detailed metrics**: Accuracy, edit distance, inference time
- **Error analysis**: Per-sample predictions for debugging
- **Resume capability**: Can stop and resume training anytime

## Training Progress

Each model shows:
- Training/validation loss per epoch
- Validation accuracy
- Sample predictions every 5 epochs
- Best model checkpointing

## Memory Management

Models are trained sequentially to avoid GPU memory issues. If you have limited memory:

1. Train in batches: `python train_all.py --models rnn,lstm,gru`
2. Then: `python train_all.py --models rnn_attention,lstm_attention,gru_attention`
3. Finally: `python train_all.py --models bilstm_attention,transformer,byt5`

## Analysis

Use the CSV files for downstream analysis:

```python
import pandas as pd

# Load results
predictions = pd.read_csv('results/predictions_all.csv')
summary = pd.read_csv('results/metrics_summary.csv')

# Compare models
print(summary.sort_values('accuracy', ascending=False))

# Error analysis
errors = predictions[~predictions['correct']]
print(errors.groupby('model')['edit_distance'].mean())

# Length-based accuracy
predictions.groupby(['model', 'source_len'])['correct'].mean()
```

## Extending

To add a new model:

1. Create model class in `models/` directory
2. Add config to `MODEL_CONFIGS` in `config.py`
3. Add training function in `train_all.py`
4. Add loading logic in `evaluator.py`

## Troubleshooting

**CUDA out of memory**: Reduce batch size in `config.py`

**Module not found**: Ensure all dependencies are installed

**Checkpoint not found**: Model hasn't been trained yet

**ByT5 download issues**: Check internet connection, model downloads ~300MB