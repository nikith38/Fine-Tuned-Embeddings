import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContrastiveSalesDataset(Dataset):
    """Dataset for contrastive learning with sales conversations"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256, 
                 num_negatives: int = 1, include_context: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.include_context = include_context
        
        # Separate by labels for efficient sampling
        self.successful_examples = [item for item in data if item['conversion_label'] == 1]
        self.failed_examples = [item for item in data if item['conversion_label'] == 0]
        
        logger.info(f"Dataset: {len(self.successful_examples)} successful, {len(self.failed_examples)} failed")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = anchor['conversion_label']
        
        # Find positive example (same label)
        if anchor_label == 1 and len(self.successful_examples) > 1:
            positive_candidates = [ex for ex in self.successful_examples if ex['id'] != anchor['id']]
            positive = random.choice(positive_candidates) if positive_candidates else anchor
        elif anchor_label == 0 and len(self.failed_examples) > 1:
            positive_candidates = [ex for ex in self.failed_examples if ex['id'] != anchor['id']]
            positive = random.choice(positive_candidates) if positive_candidates else anchor
        else:
            positive = anchor  # Fallback
        
        # Find negative example (different label)
        if anchor_label == 1 and self.failed_examples:
            negative = random.choice(self.failed_examples)
        elif anchor_label == 0 and self.successful_examples:
            negative = random.choice(self.successful_examples)
        else:
            # Fallback: use a random different example
            negative_candidates = [ex for ex in self.data if ex['id'] != anchor['id']]
            negative = random.choice(negative_candidates) if negative_candidates else anchor
        
        # Prepare texts
        anchor_text = self._prepare_text(anchor)
        positive_text = self._prepare_text(positive)
        negative_text = self._prepare_text(negative)
        
        # Tokenize
        anchor_tokens = self._tokenize(anchor_text)
        positive_tokens = self._tokenize(positive_text)
        negative_tokens = self._tokenize(negative_text)
        
        return {
            'anchor_input_ids': anchor_tokens['input_ids'].flatten(),
            'anchor_attention_mask': anchor_tokens['attention_mask'].flatten(),
            'positive_input_ids': positive_tokens['input_ids'].flatten(),
            'positive_attention_mask': positive_tokens['attention_mask'].flatten(),
            'negative_input_ids': negative_tokens['input_ids'].flatten(),
            'negative_attention_mask': negative_tokens['attention_mask'].flatten(),
            'anchor_label': torch.tensor(anchor_label, dtype=torch.long),
            'positive_label': torch.tensor(positive['conversion_label'], dtype=torch.long),
            'negative_label': torch.tensor(negative['conversion_label'], dtype=torch.long)
        }
    
    def _prepare_text(self, item: Dict) -> str:
        """Prepare text with optional context"""
        text = item['transcript']
        
        if self.include_context:
            context = item['customer_context']
            context_text = f"Company: {context['company_size']} {context['industry']} | " \
                          f"Contact: {context['contact_role']} | " \
                          f"Urgency: {context['urgency']} | " \
                          f"Conversation: {text}"
            return context_text
        
        return text
    
    def _tokenize(self, text: str) -> Dict:
        """Tokenize text"""
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

class ContrastiveSalesModel(nn.Module):
    """Contrastive learning model for sales conversations"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', 
                 embedding_dim: int = 128, temperature: float = 0.1,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        else:
            # Unfreeze only last few layers for efficiency
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze last 2 layers
            for param in self.encoder.transformer.layer[-2:].parameters():
                param.requires_grad = True
            
            logger.info("Last 2 encoder layers unfrozen for fine-tuning")
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        )
        
        # Classification head (for evaluation)
        self.classifier = nn.Linear(embedding_dim, 2)
        
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """Forward pass"""
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token
        pooled_output = encoder_outputs.last_hidden_state[:, 0]
        
        # Project to embedding space
        embeddings = self.projection(pooled_output)
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_embeddings:
            return embeddings
        
        # Classification logits
        logits = self.classifier(embeddings)
        return logits, embeddings
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get normalized embeddings"""
        return self.forward(input_ids, attention_mask, return_embeddings=True)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for triplet learning"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute contrastive loss
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        """
        # Compute similarities
        pos_similarity = F.cosine_similarity(anchor, positive, dim=1)
        neg_similarity = F.cosine_similarity(anchor, negative, dim=1)
        
        # Contrastive loss: maximize positive similarity, minimize negative similarity
        loss = torch.relu(self.margin - pos_similarity + neg_similarity)
        
        return loss.mean()

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negative):
        """
        Compute InfoNCE loss
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        """
        # Compute similarities
        pos_similarity = F.cosine_similarity(anchor, positive, dim=1) / self.temperature
        neg_similarity = F.cosine_similarity(anchor, negative, dim=1) / self.temperature
        
        # InfoNCE loss
        logits = torch.stack([pos_similarity, neg_similarity], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

class ContrastiveTrainer:
    """Trainer for contrastive learning"""
    
    def __init__(self, model: ContrastiveSalesModel, device: str = 'auto'):
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.training_time = 0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, learning_rate: float = 1e-5,
              loss_type: str = 'contrastive') -> Dict:
        """Train the contrastive model"""
        
        logger.info(f"Starting contrastive training for {epochs} epochs...")
        start_time = time.time()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup loss function
        if loss_type == 'contrastive':
            contrastive_criterion = ContrastiveLoss(temperature=self.model.temperature)
        elif loss_type == 'infonce':
            contrastive_criterion = InfoNCELoss(temperature=self.model.temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        classification_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # Move to device
                anchor_ids = batch['anchor_input_ids'].to(self.device)
                anchor_mask = batch['anchor_attention_mask'].to(self.device)
                positive_ids = batch['positive_input_ids'].to(self.device)
                positive_mask = batch['positive_attention_mask'].to(self.device)
                negative_ids = batch['negative_input_ids'].to(self.device)
                negative_mask = batch['negative_attention_mask'].to(self.device)
                
                # Get embeddings
                anchor_embeddings = self.model.get_embeddings(anchor_ids, anchor_mask)
                positive_embeddings = self.model.get_embeddings(positive_ids, positive_mask)
                negative_embeddings = self.model.get_embeddings(negative_ids, negative_mask)
                
                # Compute contrastive loss
                contrastive_loss = contrastive_criterion(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )
                
                # Optional: Add classification loss
                anchor_labels = batch['anchor_label'].to(self.device)
                anchor_logits, _ = self.model(anchor_ids, anchor_mask)
                classification_loss = classification_criterion(anchor_logits, anchor_labels)
                
                # Combined loss
                total_loss = contrastive_loss + 0.1 * classification_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
                progress_bar.set_postfix({
                    'cont_loss': f'{contrastive_loss.item():.4f}',
                    'cls_loss': f'{classification_loss.item():.4f}',
                    'total_loss': f'{total_loss.item():.4f}'
                })
            
            # Validation phase
            val_loss, val_accuracy = self._validate(val_loader, contrastive_criterion, classification_criterion)
            
            # Save metrics
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'training_time': self.training_time
        }
    
    def _validate(self, val_loader: DataLoader, contrastive_criterion, classification_criterion):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                anchor_ids = batch['anchor_input_ids'].to(self.device)
                anchor_mask = batch['anchor_attention_mask'].to(self.device)
                positive_ids = batch['positive_input_ids'].to(self.device)
                positive_mask = batch['positive_attention_mask'].to(self.device)
                negative_ids = batch['negative_input_ids'].to(self.device)
                negative_mask = batch['negative_attention_mask'].to(self.device)
                anchor_labels = batch['anchor_label'].to(self.device)
                
                # Get embeddings and logits
                anchor_embeddings = self.model.get_embeddings(anchor_ids, anchor_mask)
                positive_embeddings = self.model.get_embeddings(positive_ids, positive_mask)
                negative_embeddings = self.model.get_embeddings(negative_ids, negative_mask)
                
                anchor_logits, _ = self.model(anchor_ids, anchor_mask)
                
                # Compute losses
                contrastive_loss = contrastive_criterion(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )
                classification_loss = classification_criterion(anchor_logits, anchor_labels)
                total_loss += (contrastive_loss + 0.1 * classification_loss).item()
                
                # Get predictions
                _, predicted = torch.max(anchor_logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(anchor_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def evaluate_embeddings(self, test_loader: DataLoader) -> Dict:
        """Evaluate the quality of learned embeddings"""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating embeddings"):
                anchor_ids = batch['anchor_input_ids'].to(self.device)
                anchor_mask = batch['anchor_attention_mask'].to(self.device)
                anchor_labels = batch['anchor_label'].to(self.device)
                
                # Get embeddings and predictions
                embeddings = self.model.get_embeddings(anchor_ids, anchor_mask)
                logits, _ = self.model(anchor_ids, anchor_mask)
                
                _, predicted = torch.max(logits, 1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.extend(anchor_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Combine embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        # Calculate embedding quality metrics
        successful_embeddings = embeddings[np.array(all_labels) == 1]
        failed_embeddings = embeddings[np.array(all_labels) == 0]
        
        # Intra-class similarity (higher is better)
        if len(successful_embeddings) > 1:
            successful_sim = np.mean(cosine_similarity(successful_embeddings))
        else:
            successful_sim = 0.0
        
        if len(failed_embeddings) > 1:
            failed_sim = np.mean(cosine_similarity(failed_embeddings))
        else:
            failed_sim = 0.0
        
        # Inter-class similarity (lower is better)
        if len(successful_embeddings) > 0 and len(failed_embeddings) > 0:
            inter_class_sim = np.mean(cosine_similarity(successful_embeddings, failed_embeddings))
        else:
            inter_class_sim = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'embeddings': embeddings,
            'labels': all_labels,
            'successful_intra_similarity': successful_sim,
            'failed_intra_similarity': failed_sim,
            'inter_class_similarity': inter_class_sim,
            'embedding_quality_score': (successful_sim + failed_sim) / 2 - inter_class_sim
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_name': self.model.model_name,
                'embedding_dim': self.model.embedding_dim,
                'temperature': self.model.temperature
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'training_time': self.training_time
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")

def train_contrastive_model(train_data_path: str = 'train_data.json',
                           val_data_path: str = 'val_data.json',
                           test_data_path: str = 'test_data.json',
                           model_name: str = 'distilbert-base-uncased',
                           batch_size: int = 16,
                           epochs: int = 10,
                           learning_rate: float = 1e-5) -> Tuple[ContrastiveTrainer, Dict]:
    """Train a contrastive learning model"""
    
    logger.info("=== TRAINING CONTRASTIVE LEARNING MODEL ===")
    
    # Load data
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = ContrastiveSalesDataset(train_data, tokenizer)
    val_dataset = ContrastiveSalesDataset(val_data, tokenizer)
    test_dataset = ContrastiveSalesDataset(test_data, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ContrastiveSalesModel(model_name=model_name, embedding_dim=128, temperature=0.1)
    trainer = ContrastiveTrainer(model)
    
    # Train model
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        loss_type='contrastive'
    )
    
    # Evaluate embeddings
    logger.info("Evaluating learned embeddings...")
    embedding_results = trainer.evaluate_embeddings(test_loader)
    
    logger.info(f"Embedding Results:")
    logger.info(f"  Accuracy: {embedding_results['accuracy']:.4f}")
    logger.info(f"  F1 Score: {embedding_results['f1_score']:.4f}")
    logger.info(f"  Embedding Quality Score: {embedding_results['embedding_quality_score']:.4f}")
    logger.info(f"  Successful Intra-class Similarity: {embedding_results['successful_intra_similarity']:.4f}")
    logger.info(f"  Failed Intra-class Similarity: {embedding_results['failed_intra_similarity']:.4f}")
    logger.info(f"  Inter-class Similarity: {embedding_results['inter_class_similarity']:.4f}")
    
    # Save model
    trainer.save_model('contrastive_sales_model.pth')
    
    results = {
        'training_results': training_results,
        'embedding_results': embedding_results,
        'model_info': {
            'model_name': model_name,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        }
    }
    
    return trainer, results

def main():
    """Main function"""
    # Load and prepare data
    from data_pipeline import SalesDataProcessor
    
    processor = SalesDataProcessor()
    processor.load_data()
    processor.preprocess_data()
    processor.create_splits()
    processor.save_splits()
    
    # Train contrastive model
    trainer, results = train_contrastive_model()
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main() 