import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesConversationDataset(Dataset):
    """Dataset class for sales conversations"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256, include_context: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_context = include_context
        
        # Preprocess context features
        self.context_encoders = self._create_context_encoders()
        
    def _create_context_encoders(self):
        """Create label encoders for categorical context features"""
        encoders = {}
        
        # Extract all context features
        context_features = ['company_size', 'industry', 'contact_role', 'lead_source', 'urgency', 'budget_authority']
        
        for feature in context_features:
            values = [item['customer_context'][feature] for item in self.data]
            encoder = LabelEncoder()
            encoder.fit(values)
            encoders[feature] = encoder
            
        return encoders
    
    def _encode_context(self, context: Dict) -> np.ndarray:
        """Encode context features as numerical values"""
        features = []
        
        # Categorical features
        categorical_features = ['company_size', 'industry', 'contact_role', 'lead_source', 'urgency', 'budget_authority']
        for feature in categorical_features:
            encoded_value = self.context_encoders[feature].transform([context[feature]])[0]
            features.append(encoded_value)
        
        # Numerical features
        features.append(context['previous_interactions'])
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare text input
        text = item['transcript']
        
        # Add context to text if specified
        if self.include_context:
            context_text = self._context_to_text(item['customer_context'])
            text = f"{context_text} [SEP] {text}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare context features
        context_features = self._encode_context(item['customer_context'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['conversion_label'], dtype=torch.long),
            'context_features': torch.tensor(context_features, dtype=torch.float32),
            'text': text,
            'original_data': item
        }
    
    def _context_to_text(self, context: Dict) -> str:
        """Convert context to natural language text"""
        context_parts = [
            f"Company: {context['company_size']} {context['industry']} company",
            f"Contact: {context['contact_role']}",
            f"Lead source: {context['lead_source']}",
            f"Urgency: {context['urgency']}",
            f"Authority: {context['budget_authority']}"
        ]
        return " | ".join(context_parts)

class SalesDataProcessor:
    """Main data processing class"""
    
    def __init__(self, data_path: str = "sales_conversations_dataset.json"):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self) -> List[Dict]:
        """Load raw data from JSON file"""
        logger.info(f"Loading data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        logger.info(f"Loaded {len(self.raw_data)} conversations")
        return self.raw_data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize case (keep as-is for now to preserve proper nouns)
        text = text.strip()
        
        return text
    
    def preprocess_data(self) -> List[Dict]:
        """Preprocess and clean the data"""
        logger.info("Preprocessing data...")
        
        if self.raw_data is None:
            self.load_data()
        
        processed_data = []
        
        for item in self.raw_data:
            # Clean transcript
            cleaned_transcript = self.clean_text(item['transcript'])
            
            # Create processed item
            processed_item = {
                'id': item['id'],
                'transcript': cleaned_transcript,
                'conversion_label': item['conversion_label'],
                'outcome': item['outcome'],
                'customer_context': item['customer_context'],
                'conversation_length': len(cleaned_transcript.split()),
                'original_length': item['conversation_length']
            }
            
            processed_data.append(processed_item)
        
        self.processed_data = processed_data
        logger.info(f"Preprocessed {len(processed_data)} conversations")
        
        return processed_data
    
    def create_splits(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # Extract labels for stratified split
        labels = [item['conversion_label'] for item in self.processed_data]
        
        # First split: train+val vs test
        train_val_data, test_data = train_test_split(
            self.processed_data,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        train_val_labels = [item['conversion_label'] for item in train_val_data]
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
        
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        logger.info(f"Train: {len(train_data)} examples")
        logger.info(f"Validation: {len(val_data)} examples")
        logger.info(f"Test: {len(test_data)} examples")
        
        # Check balance
        train_positive = sum(1 for item in train_data if item['conversion_label'] == 1)
        val_positive = sum(1 for item in val_data if item['conversion_label'] == 1)
        test_positive = sum(1 for item in test_data if item['conversion_label'] == 1)
        
        logger.info(f"Train positive rate: {train_positive/len(train_data)*100:.1f}%")
        logger.info(f"Validation positive rate: {val_positive/len(val_data)*100:.1f}%")
        logger.info(f"Test positive rate: {test_positive/len(test_data)*100:.1f}%")
        
        return train_data, val_data, test_data
    
    def save_splits(self, output_dir: str = "."):
        """Save train/val/test splits to files"""
        if self.train_data is None:
            raise ValueError("No splits created. Call create_splits() first.")
        
        # Save as JSON
        with open(f"{output_dir}/train_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.train_data, f, indent=2)
        
        with open(f"{output_dir}/val_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.val_data, f, indent=2)
        
        with open(f"{output_dir}/test_data.json", 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, indent=2)
        
        logger.info(f"Splits saved to {output_dir}/")
    
    def get_data_loaders(self, tokenizer, batch_size: int = 16, max_length: int = 256, include_context: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training"""
        if self.train_data is None:
            self.create_splits()
        
        # Create datasets
        train_dataset = SalesConversationDataset(self.train_data, tokenizer, max_length, include_context)
        val_dataset = SalesConversationDataset(self.val_data, tokenizer, max_length, include_context)
        test_dataset = SalesConversationDataset(self.test_data, tokenizer, max_length, include_context)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def analyze_data(self) -> Dict:
        """Analyze the processed data"""
        if self.processed_data is None:
            self.preprocess_data()
        
        analysis = {
            'total_examples': len(self.processed_data),
            'positive_examples': sum(1 for item in self.processed_data if item['conversion_label'] == 1),
            'negative_examples': sum(1 for item in self.processed_data if item['conversion_label'] == 0),
            'avg_length': np.mean([item['conversation_length'] for item in self.processed_data]),
            'min_length': min(item['conversation_length'] for item in self.processed_data),
            'max_length': max(item['conversation_length'] for item in self.processed_data),
            'industries': len(set(item['customer_context']['industry'] for item in self.processed_data)),
            'company_sizes': len(set(item['customer_context']['company_size'] for item in self.processed_data)),
            'contact_roles': len(set(item['customer_context']['contact_role'] for item in self.processed_data))
        }
        
        analysis['positive_rate'] = analysis['positive_examples'] / analysis['total_examples'] * 100
        
        return analysis

def create_contrastive_pairs(data: List[Dict], num_pairs: int = 1000) -> List[Dict]:
    """Create positive and negative pairs for contrastive learning"""
    logger.info(f"Creating {num_pairs} contrastive pairs...")
    
    # Separate successful and failed conversations
    successful = [item for item in data if item['conversion_label'] == 1]
    failed = [item for item in data if item['conversion_label'] == 0]
    
    pairs = []
    
    for i in range(num_pairs):
        # Create positive pair (same label)
        if np.random.random() < 0.5:
            # Both successful
            if len(successful) >= 2:
                anchor, positive = np.random.choice(successful, 2, replace=False)
                negative = np.random.choice(failed, 1)[0]
            else:
                continue
        else:
            # Both failed
            if len(failed) >= 2:
                anchor, positive = np.random.choice(failed, 2, replace=False)
                negative = np.random.choice(successful, 1)[0]
            else:
                continue
        
        pairs.append({
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        })
    
    logger.info(f"Created {len(pairs)} contrastive pairs")
    return pairs

def extract_sales_features(transcript: str, context: Dict) -> Dict:
    """Extract sales-specific features from transcript and context"""
    features = {}
    
    # Text-based features
    text_lower = transcript.lower()
    
    # Buying signals
    buying_signals = ['price', 'cost', 'budget', 'timeline', 'implementation', 'demo', 'trial', 'contract', 'purchase']
    features['buying_signals'] = sum(1 for signal in buying_signals if signal in text_lower)
    
    # Objection patterns
    objections = ['expensive', 'costly', 'think about', 'not sure', 'not ready', 'concerns', 'hesitant']
    features['objections'] = sum(1 for obj in objections if obj in text_lower)
    
    # Engagement indicators
    features['question_count'] = transcript.count('?')
    features['word_count'] = len(transcript.split())
    
    # Positive/negative sentiment words
    positive_words = ['excited', 'interested', 'great', 'perfect', 'excellent', 'ready', 'forward']
    negative_words = ['difficult', 'challenging', 'problem', 'issue', 'concern', 'worried']
    
    features['positive_words'] = sum(1 for word in positive_words if word in text_lower)
    features['negative_words'] = sum(1 for word in negative_words if word in text_lower)
    
    # Context-based features
    features['urgency_score'] = {'low': 1, 'medium': 2, 'high': 3}[context['urgency']]
    features['authority_score'] = {'end_user': 1, 'influencer': 2, 'decision_maker': 3}[context['budget_authority']]
    features['company_size_score'] = {'startup': 1, 'small business': 2, 'mid-market': 3, 'enterprise': 4}[context['company_size']]
    features['previous_interactions'] = context['previous_interactions']
    
    return features

def main():
    """Main function to demonstrate data pipeline"""
    logger.info("=== SALES CONVERSATION DATA PIPELINE ===")
    
    # Initialize processor
    processor = SalesDataProcessor()
    
    # Load and preprocess data
    processor.load_data()
    processor.preprocess_data()
    
    # Analyze data
    analysis = processor.analyze_data()
    
    logger.info("=== DATA ANALYSIS ===")
    for key, value in analysis.items():
        logger.info(f"{key}: {value}")
    
    # Create splits
    train_data, val_data, test_data = processor.create_splits()
    
    # Save splits
    processor.save_splits()
    
    # Create contrastive pairs for contrastive learning
    contrastive_pairs = create_contrastive_pairs(train_data, num_pairs=500)
    
    # Save contrastive pairs
    with open('contrastive_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(contrastive_pairs, f, indent=2)
    
    logger.info("=== PIPELINE COMPLETE ===")
    logger.info("Files created:")
    logger.info("- train_data.json")
    logger.info("- val_data.json") 
    logger.info("- test_data.json")
    logger.info("- contrastive_pairs.json")
    
    return processor

if __name__ == "__main__":
    processor = main() 