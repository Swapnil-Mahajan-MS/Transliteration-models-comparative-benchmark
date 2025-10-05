"""
Main script to train and evaluate all models
"""

import torch
from torch.utils.data import DataLoader
import os
import argparse
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_inspection import TransliterationDataset, collate_fn
from config import DATA_CONFIG, MODEL_CONFIGS, EVAL_CONFIG
from trainer import train_model
from evaluator import evaluate_all_models


def train_vanilla_models(models_to_train, train_loader, val_loader, 
                        source_vocab, target_vocab, device):
    """Train vanilla RNN/LSTM/GRU models"""
    from models.vanilla_models import build_vanilla_model
    
    for model_name in models_to_train:
        if model_name not in ['rnn', 'lstm', 'gru']:
            continue
        
        config = MODEL_CONFIGS[model_name]
        
        # Check if already trained
        checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
        if os.path.exists(checkpoint_path):
            print(f"\n⚠ {model_name} already trained, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Build model
        model = build_vanilla_model(
            len(source_vocab), len(target_vocab), config, device
        )
        
        # Train
        best_acc, history = train_model(
            model, train_loader, val_loader, source_vocab, target_vocab,
            config, device, model_type='seq2seq'
        )
        
        print(f"\n✓ {model_name} training complete. Best accuracy: {best_acc:.4f}")


def train_attention_models(models_to_train, train_loader, val_loader,
                           source_vocab, target_vocab, device):
    """Train RNN/LSTM/GRU with attention"""
    from models.attention_models import build_attention_model
    
    for model_name in models_to_train:
        if model_name not in ['rnn_attention', 'lstm_attention', 'gru_attention']:
            continue
        
        config = MODEL_CONFIGS[model_name]
        
        # Check if already trained
        checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
        if os.path.exists(checkpoint_path):
            print(f"\n⚠ {model_name} already trained, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Build model
        model = build_attention_model(
            len(source_vocab), len(target_vocab), config, device
        )
        
        # Train
        best_acc, history = train_model(
            model, train_loader, val_loader, source_vocab, target_vocab,
            config, device, model_type='seq2seq'
        )
        
        print(f"\n✓ {model_name} training complete. Best accuracy: {best_acc:.4f}")


def train_bilstm_attention(train_loader, val_loader, source_vocab, target_vocab, device):
    """Train BiLSTM with attention"""
    from train_biLSTM import BiLSTMEncoder, AttentionDecoder, BiLSTMSeq2Seq
    
    config = MODEL_CONFIGS['bilstm_attention']
    
    # Check if already trained
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"\n⚠ bilstm_attention already trained, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"TRAINING: BILSTM ATTENTION")
    print(f"{'='*60}")
    
    # Build model
    encoder_hidden = config['hidden_dim'] * 2
    encoder = BiLSTMEncoder(
        len(source_vocab), config['embedding_dim'], config['hidden_dim'],
        config['num_layers'], config['dropout']
    ).to(device)
    
    decoder = AttentionDecoder(
        len(target_vocab), config['embedding_dim'], config['hidden_dim'],
        config['num_layers'], encoder_hidden, config['dropout']
    ).to(device)
    
    model = BiLSTMSeq2Seq(encoder, decoder, device).to(device)
    
    # Train
    best_acc, history = train_model(
        model, train_loader, val_loader, source_vocab, target_vocab,
        config, device, model_type='seq2seq'
    )
    
    print(f"\n✓ bilstm_attention training complete. Best accuracy: {best_acc:.4f}")


def train_transformer(train_loader, val_loader, source_vocab, target_vocab, device):
    """Train Transformer model"""
    from attention import TransformerTransliteration, TransformerLRScheduler
    
    config = MODEL_CONFIGS['transformer']
    
    # Check if already trained
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        print(f"\n⚠ transformer already trained, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"TRAINING: TRANSFORMER")
    print(f"{'='*60}")
    
    # Build model
    model = TransformerTransliteration(
        len(source_vocab), len(target_vocab),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Custom optimizer and scheduler for Transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, config['d_model'], config['warmup_steps'])
    
    # Train
    best_acc, history = train_model(
        model, train_loader, val_loader, source_vocab, target_vocab,
        config, device, model_type='transformer', scheduler=scheduler
    )
    
    print(f"\n✓ transformer training complete. Best accuracy: {best_acc:.4f}")


def train_byt5(source_vocab, target_vocab, device):
    """Train ByT5 model"""
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import torch.nn as nn
    
    config = MODEL_CONFIGS['byt5']
    
    # Check if already trained
    checkpoint_path = os.path.join(config['save_dir'], 'best_byt5')
    if os.path.exists(checkpoint_path):
        print(f"\n⚠ byt5 already trained, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"TRAINING: BYT5")
    print(f"{'='*60}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Custom dataset for ByT5
    class ByT5Dataset(Dataset):
        def __init__(self, data_path, tokenizer, max_length=50):
            self.data = []
            self.tokenizer = tokenizer
            self.max_length = max_length
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        target, source = parts[0], parts[1]
                        if len(source) <= max_length and len(target) <= max_length:
                            self.data.append((source, target))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            source, target = self.data[idx]
            input_text = f"transliterate: {source}"
            
            inputs = self.tokenizer(input_text, max_length=self.max_length,
                                   padding='max_length', truncation=True,
                                   return_tensors='pt')
            labels = self.tokenizer(target, max_length=self.max_length,
                                   padding='max_length', truncation=True,
                                   return_tensors='pt')
            
            labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': labels['input_ids'].squeeze()
            }
    
    # Load model and tokenizer
    print("Loading pre-trained ByT5...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = T5ForConditionalGeneration.from_pretrained(
        config['model_name'], use_safetensors=True
    ).to(device)
    
    # Load data
    train_dataset = ByT5Dataset(DATA_CONFIG['train_path'], tokenizer, config['max_length'])
    val_dataset = ByT5Dataset(DATA_CONFIG['val_path'], tokenizer, config['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config['save_dir'], 'best_byt5')
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")
    
    print(f"\n✓ byt5 training complete. Best val loss: {best_val_loss:.4f}")


def main(args):
    """Main function"""
    print("="*60)
    print("TRANSLITERATION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create results directory
    os.makedirs('results/checkpoints', exist_ok=True)
    
    # Determine which models to train
    if args.models == 'all':
        models_to_train = list(MODEL_CONFIGS.keys())
    else:
        models_to_train = args.models.split(',')
    
    print(f"\nModels to train: {', '.join(models_to_train)}")
    
    # Load data (needed for all models except byt5 uses its own)
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    train_dataset = TransliterationDataset(DATA_CONFIG['train_path'])
    source_vocab = train_dataset.source_vocab
    target_vocab = train_dataset.target_vocab
    
    val_dataset = TransliterationDataset(
        DATA_CONFIG['val_path'], 
        source_vocab=source_vocab, 
        target_vocab=target_vocab
    )
    
    test_dataset = TransliterationDataset(
        DATA_CONFIG['test_path'],
        source_vocab=source_vocab,
        target_vocab=target_vocab
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
    )
    
    print(f"✓ Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    
    if not args.eval_only:
        print(f"\n{'='*60}")
        print("TRAINING PHASE")
        print(f"{'='*60}")
        
        # Train vanilla models (RNN, LSTM, GRU)
        train_vanilla_models(models_to_train, train_loader, val_loader,
                           source_vocab, target_vocab, device)
        
        # Train attention models (RNN+Attn, LSTM+Attn, GRU+Attn)
        train_attention_models(models_to_train, train_loader, val_loader,
                              source_vocab, target_vocab, device)
        
        # Train BiLSTM with attention
        if 'bilstm_attention' in models_to_train:
            train_bilstm_attention(train_loader, val_loader, source_vocab, 
                                  target_vocab, device)
        
        # Train Transformer
        if 'transformer' in models_to_train:
            train_transformer(train_loader, val_loader, source_vocab,
                            target_vocab, device)
        
        # Train ByT5
        if 'byt5' in models_to_train:
            train_byt5(source_vocab, target_vocab, device)
        
        print(f"\n{'='*60}")
        print("ALL TRAINING COMPLETE")
        print(f"{'='*60}")
    
    # ========================================================================
    # EVALUATION PHASE
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("EVALUATION PHASE")
    print(f"{'='*60}")
    
    # Filter MODEL_CONFIGS to only include models we want to evaluate
    eval_configs = {k: v for k, v in MODEL_CONFIGS.items() if k in models_to_train}
    
    # Evaluate all models
    all_predictions, summary = evaluate_all_models(
        eval_configs, test_dataset, source_vocab, target_vocab,
        device, 'results'
    )
    
    if summary is not None:
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print("\nGenerated files:")
        print("  results/predictions_all.csv")
        print("  results/metrics_summary.csv")
        print("\nTop 3 models by accuracy:")
        top3 = summary.nlargest(3, 'accuracy')
        for idx, row in top3.iterrows():
            print(f"  {idx+1}. {row['model']}: {row['accuracy']:.4f}")
    else:
        print("\n✗ No models were evaluated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate transliteration models')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to train, or "all" (default: all)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip training and only evaluate existing models')
    
    args = parser.parse_args()
    
    main(args)