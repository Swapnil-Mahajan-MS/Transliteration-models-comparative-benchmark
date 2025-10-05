"""
Step 1: Data Loading and Exploration
Let's build the data loader step by step with proper debugging
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd

# ============================================================================
# STEP 1: CHECK IF DATA FILE EXISTS AND INSPECT IT
# ============================================================================

def inspect_data_file(file_path):
    """Inspect the data file to understand its format"""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {file_path}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File does not exist!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {file_path}")
        return False
    
    print(f"✓ File exists!")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes ({file_size/1024:.2f} KB)")
    
    # Read first few lines
    print(f"\nFirst 10 lines:")
    print("-" * 60)
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"{i+1}: {line.strip()}")
    
    # Count total lines
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"\nTotal lines: {total_lines}")
    
    # Analyze format
    print(f"\nAnalyzing format...")
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if '\t' in first_line:
            parts = first_line.split('\t')
            print(f"✓ Tab-separated format detected")
            print(f"Number of columns: {len(parts)}")
            for i, part in enumerate(parts):
                print(f"  Column {i}: '{part}' (length: {len(part)})")
        else:
            print(f"⚠ WARNING: No tab found in first line")
            print(f"Line content: '{first_line}'")
    
    return True


# ============================================================================
# STEP 2: BUILD VOCABULARY CLASS
# ============================================================================

class Vocabulary:
    """Vocabulary class for character-level encoding"""
    
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.char_counts = Counter()
        
    def build_vocab(self, texts, min_freq=1):
        """Build vocabulary from list of texts"""
        print(f"\nBuilding vocabulary from {len(texts)} texts...")
        
        # Count characters
        for text in texts:
            self.char_counts.update(text)
        
        print(f"Unique characters found: {len(self.char_counts)}")
        print(f"Top 20 most common characters:")
        for char, count in self.char_counts.most_common(20):
            print(f"  '{char}': {count}")
        
        # Add characters to vocab
        idx = len(self.char2idx)
        for char, count in self.char_counts.items():
            if count >= min_freq and char not in self.char2idx:
                self.char2idx[char] = idx
                self.idx2char[idx] = char
                idx += 1
        
        print(f"Final vocabulary size: {len(self.char2idx)}")
        
    def encode(self, text):
        """Convert text to indices"""
        indices = [self.char2idx['<SOS>']]
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx['<UNK>']))
        indices.append(self.char2idx['<EOS>'])
        return indices
    
    def decode(self, indices):
        """Convert indices to text"""
        chars = []
        for idx in indices:
            if idx == self.char2idx['<EOS>']:
                break
            if idx not in [self.char2idx['<PAD>'], self.char2idx['<SOS>']]:
                chars.append(self.idx2char.get(idx, '<UNK>'))
        return ''.join(chars)
    
    def __len__(self):
        return len(self.char2idx)


# ============================================================================
# STEP 3: BUILD DATASET CLASS WITH DEBUGGING
# ============================================================================

class TransliterationDataset(Dataset):
    """Dataset class for transliteration pairs with detailed debugging"""
    
    def __init__(self, data_path, source_vocab=None, target_vocab=None, max_len=50):
        """
        Args:
            data_path: Path to TSV file
            source_vocab: Vocabulary for source language (Latin)
            target_vocab: Vocabulary for target language (Devanagari)
            max_len: Maximum sequence length
        """
        print(f"\n{'='*60}")
        print(f"INITIALIZING DATASET: {data_path}")
        print(f"{'='*60}")
        
        self.data = []
        self.max_len = max_len
        self.skipped = {'too_long': 0, 'malformed': 0}
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Read data
        print(f"Reading data from: {data_path}")
        line_count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Split by tab
                parts = line.split('\t')
                
                # Handle both 2-column and 3-column formats
                if len(parts) == 3:
                    # Format: Devanagari \t Latin \t Quality
                    target, source, quality = parts
                elif len(parts) == 2:
                    # Format: Devanagari \t Latin
                    target, source = parts
                else:
                    self.skipped['malformed'] += 1
                    if line_num <= 5:  # Show first few errors
                        print(f"⚠ Line {line_num}: Expected 2 or 3 columns, got {len(parts)}")
                        print(f"   Content: '{line}'")
                    continue
                
                # Check length
                if len(source) > max_len or len(target) > max_len:
                    self.skipped['too_long'] += 1
                    continue
                
                self.data.append((source, target))
        
        print(f"\n✓ Successfully loaded {len(self.data)} pairs")
        print(f"  Total lines read: {line_count}")
        print(f"  Skipped (too long): {self.skipped['too_long']}")
        print(f"  Skipped (malformed): {self.skipped['malformed']}")
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data loaded from {data_path}!")
        
        # Show sample data
        print(f"\nSample pairs:")
        for i in range(min(5, len(self.data))):
            src, tgt = self.data[i]
            print(f"  {i+1}. '{src}' → '{tgt}'")
        
        # Build vocabularies if not provided
        if source_vocab is None:
            print(f"\nBuilding source vocabulary...")
            self.source_vocab = Vocabulary()
            self.source_vocab.build_vocab([pair[0] for pair in self.data])
        else:
            self.source_vocab = source_vocab
            print(f"\nUsing provided source vocabulary (size: {len(source_vocab)})")
            
        if target_vocab is None:
            print(f"\nBuilding target vocabulary...")
            self.target_vocab = Vocabulary()
            self.target_vocab.build_vocab([pair[1] for pair in self.data])
        else:
            self.target_vocab = target_vocab
            print(f"\nUsing provided target vocabulary (size: {len(target_vocab)})")
        
        # Show encoding example
        if len(self.data) > 0:
            src, tgt = self.data[0]
            src_encoded = self.source_vocab.encode(src)
            tgt_encoded = self.target_vocab.encode(tgt)
            print(f"\nEncoding example:")
            print(f"  Source: '{src}' → {src_encoded}")
            print(f"  Target: '{tgt}' → {tgt_encoded}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source, target = self.data[idx]
        
        # Convert to indices
        source_indices = self.source_vocab.encode(source)
        target_indices = self.target_vocab.encode(target)
        
        return {
            'source': torch.tensor(source_indices, dtype=torch.long),
            'target': torch.tensor(target_indices, dtype=torch.long),
            'source_text': source,
            'target_text': target
        }


# ============================================================================
# STEP 4: COLLATE FUNCTION
# ============================================================================

def collate_fn(batch):
    """Custom collate function for batching"""
    sources = [item['source'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Get lengths
    source_lengths = torch.tensor([len(s) for s in sources])
    target_lengths = torch.tensor([len(t) for t in targets])
    
    # Pad sequences
    sources_padded = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return {
        'source': sources_padded,
        'target': targets_padded,
        'source_lengths': source_lengths,
        'target_lengths': target_lengths,
        'source_text': [item['source_text'] for item in batch],
        'target_text': [item['target_text'] for item in batch]
    }


# ============================================================================
# STEP 5: TEST THE DATA LOADER
# ============================================================================

def test_data_loader(train_path, val_path=None, test_path=None):
    """Test the data loader with actual files"""
    
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60)
    
    # Step 1: Inspect the file
    if not inspect_data_file(train_path):
        print("\n❌ Cannot proceed - file issues detected")
        return
    
    # Step 2: Load training data
    try:
        print(f"\n{'='*60}")
        print("LOADING TRAINING DATA")
        print(f"{'='*60}")
        train_dataset = TransliterationDataset(train_path)
        print(f"✓ Training dataset created successfully!")
        print(f"  Size: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ Error creating training dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Load validation data (if provided)
    if val_path:
        try:
            print(f"\n{'='*60}")
            print("LOADING VALIDATION DATA")
            print(f"{'='*60}")
            val_dataset = TransliterationDataset(
                val_path,
                source_vocab=train_dataset.source_vocab,
                target_vocab=train_dataset.target_vocab
            )
            print(f"✓ Validation dataset created successfully!")
            print(f"  Size: {len(val_dataset)}")
        except Exception as e:
            print(f"❌ Error creating validation dataset: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 4: Create DataLoader
    try:
        print(f"\n{'='*60}")
        print("CREATING DATALOADER")
        print(f"{'='*60}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn
        )
        print(f"✓ DataLoader created successfully!")
        print(f"  Number of batches: {len(train_loader)}")
    except Exception as e:
        print(f"❌ Error creating DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test iterating through one batch
    try:
        print(f"\n{'='*60}")
        print("TESTING BATCH ITERATION")
        print(f"{'='*60}")
        batch = next(iter(train_loader))
        
        print(f"✓ Successfully retrieved one batch!")
        print(f"\nBatch contents:")
        print(f"  source shape: {batch['source'].shape}")
        print(f"  target shape: {batch['target'].shape}")
        print(f"  source_lengths: {batch['source_lengths'][:5]}...")
        print(f"  target_lengths: {batch['target_lengths'][:5]}...")
        
        print(f"\nFirst example in batch:")
        print(f"  Source text: '{batch['source_text'][0]}'")
        print(f"  Target text: '{batch['target_text'][0]}'")
        print(f"  Source indices: {batch['source'][0].tolist()}")
        print(f"  Target indices: {batch['target'][0].tolist()}")
        
        # Test decoding
        decoded_source = train_dataset.source_vocab.decode(batch['source'][0].tolist())
        decoded_target = train_dataset.target_vocab.decode(batch['target'][0].tolist())
        print(f"  Decoded source: '{decoded_source}'")
        print(f"  Decoded target: '{decoded_target}'")
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS PASSED!")
        print(f"{'='*60}")
        
        return train_dataset, val_dataset if val_path else None
        
    except Exception as e:
        print(f"❌ Error iterating through batch: {e}")
        import traceback
        traceback.print_exc()
        return


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

if __name__ == '__main__':
    
    # =======================================================================
    # EDIT THESE PATHS TO MATCH YOUR DATA LOCATION
    # =======================================================================
    
    TRAIN_PATH = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
    VAL_PATH = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
    TEST_PATH = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'
    
    # =======================================================================
    
    print("="*60)
    print("DATA LOADER TEST SCRIPT")
    print("="*60)
    print(f"\nCurrent directory: {os.getcwd()}")
    print(f"\nPaths to test:")
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Val:   {VAL_PATH}")
    print(f"  Test:  {TEST_PATH}")
    
    # Test with your data
    test_data_loader(TRAIN_PATH, VAL_PATH, TEST_PATH)
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("\nTo use this data loader in your main code:")
    print("""
from data_loader import TransliterationDataset, collate_fn

# Load data
train_dataset = TransliterationDataset('path/to/train.tsv')
val_dataset = TransliterationDataset(
    'path/to/val.tsv',
    source_vocab=train_dataset.source_vocab,
    target_vocab=train_dataset.target_vocab
)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, 
                         shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64,
                       shuffle=False, collate_fn=collate_fn)
    """)