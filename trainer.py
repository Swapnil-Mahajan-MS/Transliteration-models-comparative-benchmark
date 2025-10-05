"""
Generic trainer for all models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time


def train_epoch(model, dataloader, optimizer, criterion, device, config, model_type='seq2seq'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        source_lengths = batch['source_lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'transformer':
            output = model(source, target)
        else:
            output = model(source, target, source_lengths, config.get('teacher_forcing_ratio', 0.5))
        
        # Calculate loss
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, config, model_type='seq2seq'):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_lengths = batch['source_lengths']
            
            if model_type == 'transformer':
                output = model(source, target)
            else:
                output = model(source, target, source_lengths, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def calculate_accuracy(model, dataloader, source_vocab, target_vocab, device, model_type='seq2seq'):
    """Calculate word-level accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating accuracy"):
            sources = batch['source_text']
            targets = batch['target_text']
            
            predictions = model.predict(sources, source_vocab, target_vocab, device=device)
            
            for pred, target in zip(predictions, targets):
                if pred == target:
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0


def train_model(model, train_loader, val_loader, source_vocab, target_vocab, 
                config, device, model_type='seq2seq', scheduler=None):
    """
    Complete training loop
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"{'='*60}")
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config, model_type)
        val_loss = evaluate(model, val_loader, criterion, device, config, model_type)
        val_acc = calculate_accuracy(model, val_loader, source_vocab, target_vocab, device, model_type)
        
        if scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'source_vocab': source_vocab,
                'target_vocab': target_vocab,
                'config': config,
                'val_acc': val_acc,
                'history': history
            }
            
            save_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
        
        if (epoch + 1) % 5 == 0:
            print(f"\nSample predictions:")
            samples = ['namaste', 'dhanyavaad', 'kripaya', 'shubh']
            preds = model.predict(samples, source_vocab, target_vocab, device=device)
            for src, pred in zip(samples, preds):
                print(f"  '{src}' → '{pred}'")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training time: {training_time/60:.2f} minutes")
    
    return best_val_acc, history