"""
Evaluation module for all models
"""

import torch
import pandas as pd
from tqdm import tqdm
import time
import os


def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def evaluate_single_model(model, dataset, source_vocab, target_vocab, 
                          device, model_name, model_type='seq2seq'):
    """
    Evaluate a single model on the test set
    
    Args:
        model: Trained model
        dataset: Test dataset
        source_vocab: Source vocabulary
        target_vocab: Target vocabulary
        device: torch device
        model_name: Name of the model
        model_type: Type of model ('seq2seq', 'transformer', etc.)
    
    Returns:
        results_df: DataFrame with predictions and metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    predictions = []
    
    total_time = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Testing {model_name}"):
            source = dataset.data[i][0]
            target = dataset.data[i][1]
            
            # Time prediction
            start_time = time.time()
            pred = model.predict([source], source_vocab, target_vocab, device=device)[0]
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            total_time += inference_time
            
            predictions.append({
                'model': model_name,
                'source': source,
                'target': target,
                'prediction': pred,
                'correct': pred == target,
                'source_len': len(source),
                'target_len': len(target),
                'pred_len': len(pred),
                'edit_distance': levenshtein_distance(pred, target) if pred != target else 0,
                'inference_ms': inference_time
            })
    
    df = pd.DataFrame(predictions)
    
    # Calculate metrics
    accuracy = df['correct'].mean()
    avg_edit_dist = df['edit_distance'].mean()
    avg_inference_time = df['inference_ms'].mean()
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg Edit Distance: {avg_edit_dist:.2f}")
    print(f"  Avg Inference Time: {avg_inference_time:.2f} ms")
    
    return df


def evaluate_all_models(model_configs, test_dataset, source_vocab, target_vocab, 
                        device, results_dir):
    """
    Evaluate all trained models
    
    Args:
        model_configs: Dict of model configurations
        test_dataset: Test dataset
        source_vocab: Source vocabulary
        target_vocab: Target vocabulary
        device: torch device
        results_dir: Directory to save results
    
    Returns:
        all_predictions: DataFrame with all predictions
        summary: DataFrame with summary statistics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING ALL MODELS")
    print(f"{'='*60}")
    
    all_results = []
    summary_data = []
    
    for model_name, config in model_configs.items():
        checkpoint_path = os.path.join(config['save_dir'], 'best_model.pth')
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠ Checkpoint not found for {model_name}, skipping...")
            continue
        
        try:
            # Load model
            print(f"\nLoading {model_name}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Rebuild model based on type
            if model_name in ['rnn', 'lstm', 'gru']:
                from models.vanilla_models import build_vanilla_model
                model = build_vanilla_model(
                    len(source_vocab), len(target_vocab), config, device
                )
                model_type = 'seq2seq'
            elif model_name in ['rnn_attention', 'lstm_attention', 'gru_attention']:
                from models.attention_models import build_attention_model
                model = build_attention_model(
                    len(source_vocab), len(target_vocab), config, device
                )
                model_type = 'seq2seq'
            elif model_name == 'bilstm_attention':
                from train_biLSTM import BiLSTMEncoder, AttentionDecoder, BiLSTMSeq2Seq
                encoder_hidden = config['hidden_dim'] * 2
                encoder = BiLSTMEncoder(
                    len(source_vocab), config['embedding_dim'], 
                    config['hidden_dim'], config['num_layers'], 
                    config['dropout']
                ).to(device)
                decoder = AttentionDecoder(
                    len(target_vocab), config['embedding_dim'], 
                    config['hidden_dim'], config['num_layers'], 
                    encoder_hidden, config['dropout']
                ).to(device)
                model = BiLSTMSeq2Seq(encoder, decoder, device).to(device)
                model_type = 'seq2seq'
            elif model_name == 'transformer':
                from attention import TransformerTransliteration
                model = TransformerTransliteration(
                    len(source_vocab), len(target_vocab),
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_encoder_layers=config['num_encoder_layers'],
                    num_decoder_layers=config['num_decoder_layers'],
                    dim_feedforward=config['dim_feedforward'],
                    dropout=config['dropout']
                ).to(device)
                model_type = 'transformer'
            elif model_name == 'byt5':
                from transformers import T5ForConditionalGeneration, AutoTokenizer
                
                class ByT5Wrapper:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer
                    
                    def eval(self):
                        self.model.eval()
                    
                    def predict(self, sources, source_vocab, target_vocab, device='cuda'):
                        results = []
                        for source in sources:
                            input_text = f"transliterate: {source}"
                            inputs = self.tokenizer(input_text, return_tensors='pt').to(device)
                            outputs = self.model.generate(**inputs, max_length=50, num_beams=1)
                            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                            results.append(pred)
                        return results
                
                base_model = T5ForConditionalGeneration.from_pretrained(
                    config['save_dir'], use_safetensors=True
                ).to(device)
                tokenizer = AutoTokenizer.from_pretrained(config['save_dir'])
                model = ByT5Wrapper(base_model, tokenizer)
                model_type = 'byt5'
            else:
                print(f"Unknown model type: {model_name}")
                continue
            
            # Load weights (except for ByT5 which loads directly)
            if model_name != 'byt5':
                model.load_state_dict(checkpoint['model_state_dict'])
            
            model.eval()
            
            # Count parameters
            if model_name != 'byt5':
                total_params = sum(p.numel() for p in model.parameters())
            else:
                total_params = sum(p.numel() for p in model.model.parameters())
            
            # Evaluate
            results_df = evaluate_single_model(
                model, test_dataset, source_vocab, target_vocab, 
                device, model_name, model_type
            )
            
            all_results.append(results_df)
            
            # Summary statistics
            summary_data.append({
                'model': model_name,
                'accuracy': results_df['correct'].mean(),
                'avg_edit_distance': results_df['edit_distance'].mean(),
                'total_params': total_params,
                'avg_inference_ms': results_df['inference_ms'].mean(),
            })
            
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        all_predictions = pd.concat(all_results, ignore_index=True)
        summary = pd.DataFrame(summary_data)
        
        # Save results
        predictions_path = os.path.join(results_dir, 'predictions_all.csv')
        summary_path = os.path.join(results_dir, 'metrics_summary.csv')
        
        all_predictions.to_csv(predictions_path, index=False)
        summary.to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("RESULTS SAVED")
        print(f"{'='*60}")
        print(f"  Predictions: {predictions_path}")
        print(f"  Summary: {summary_path}")
        
        # Display summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(summary.to_string(index=False))
        
        return all_predictions, summary
    else:
        print("\n✗ No models evaluated successfully")
        return None, None