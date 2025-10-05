"""
Configuration for all models
"""

import torch

# Data paths
DATA_CONFIG = {
    'train_path': '/home/swapnil/Desktop/Placement26/ADAS/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv',
    'val_path': '/home/swapnil/Desktop/Placement26/ADAS/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv',
    'test_path': '/home/swapnil/Desktop/Placement26/ADAS/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv',
}

# Common hyperparameters
COMMON_CONFIG = {
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 30,
    'teacher_forcing_ratio': 0.5,
    'gradient_clip': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Model-specific configurations
MODEL_CONFIGS = {
    'rnn': {
        **COMMON_CONFIG,
        'cell_type': 'RNN',
        'save_dir': 'results/checkpoints/rnn',
    },
    'lstm': {
        **COMMON_CONFIG,
        'cell_type': 'LSTM',
        'save_dir': 'results/checkpoints/lstm',
    },
    'gru': {
        **COMMON_CONFIG,
        'cell_type': 'GRU',
        'save_dir': 'results/checkpoints/gru',
    },
    'rnn_attention': {
        **COMMON_CONFIG,
        'cell_type': 'RNN',
        'use_attention': True,
        'save_dir': 'results/checkpoints/rnn_attention',
    },
    'lstm_attention': {
        **COMMON_CONFIG,
        'cell_type': 'LSTM',
        'use_attention': True,
        'save_dir': 'results/checkpoints/lstm_attention',
    },
    'gru_attention': {
        **COMMON_CONFIG,
        'cell_type': 'GRU',
        'use_attention': True,
        'save_dir': 'results/checkpoints/gru_attention',
    },
    'bilstm_attention': {
        **COMMON_CONFIG,
        'hidden_dim': 768,
        'dropout': 0.5,
        'bidirectional': True,
        'save_dir': 'results/checkpoints/bilstm_attention',
    },
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'max_seq_length': 100,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'num_epochs': 50,
        'warmup_steps': 4000,
        'gradient_clip': 1.0,
        'device': COMMON_CONFIG['device'],
        'save_dir': 'results/checkpoints/transformer',
    },
    'byt5': {
        'model_name': 'google/byt5-small',
        'max_length': 50,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'num_epochs': 10,
        'warmup_steps': 500,
        'device': COMMON_CONFIG['device'],
        'save_dir': 'results/checkpoints/byt5',
    },
}

# Evaluation config
EVAL_CONFIG = {
    'batch_size': 64,
    'beam_size': 1,  # For beam search (if applicable)
    'results_dir': 'results',
}