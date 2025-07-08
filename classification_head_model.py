import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from data_pipeline import SalesDataProcessor, SalesConversationDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesClassificationModel(nn.Module):
    """Sales conversion prediction model with frozen encoder and trainable classification head"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_classes: int = 2, 
                 freeze_encoder: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        else:
            logger.info("Encoder weights will be fine-tuned")
        
        # Get hidden size
        self.hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """Forward pass"""
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token representation
        pooled_output = encoder_outputs.last_hidden_state[:, 0]  # CLS token
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        if return_embeddings:
            return logits, pooled_output
        else:
            return logits
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get embeddings without classification"""
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            return encoder_outputs.last_hidden_state[:, 0]  # CLS token

class SalesClassificationTrainer:
    """Trainer for sales classification model"""
    
    def __init__(self, model: SalesClassificationModel, device: str = 'auto'):
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
              epochs: int = 5, learning_rate: float = 2e-5, 
              warmup_steps: int = 100, weight_decay: float = 0.01) -> Dict:
        """Train the model"""
        
        logger.info(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
            
            # Validation phase
            val_loss, val_accuracy, val_metrics = self.evaluate(val_loader, criterion)
            
            # Save metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
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
    
    def evaluate(self, data_loader: DataLoader, criterion: nn.Module = None) -> Tuple[float, float, Dict]:
        """Evaluate the model"""
        self.model.eval()
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, [probs[1] for probs in all_probabilities])
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        return avg_loss, accuracy, metrics
    
    def predict(self, data_loader: DataLoader) -> Tuple[List[int], List[float]]:
        """Make predictions on data"""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
        
        return all_predictions, all_probabilities
    
    def get_embeddings(self, data_loader: DataLoader) -> np.ndarray:
        """Extract embeddings from the model"""
        self.model.eval()
        
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                embeddings = self.model.get_embeddings(input_ids, attention_mask)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model.model_name,
            'num_classes': self.model.num_classes,
            'freeze_encoder': self.model.freeze_encoder,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'training_time': self.training_time
            }
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            self.train_losses = history['train_losses']
            self.val_losses = history['val_losses']
            self.val_accuracies = history['val_accuracies']
            self.training_time = history['training_time']
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

def train_classification_model(train_data_path: str = 'train_data.json',
                             val_data_path: str = 'val_data.json',
                             test_data_path: str = 'test_data.json',
                             model_name: str = 'distilbert-base-uncased',
                             batch_size: int = 16,
                             epochs: int = 5,
                             learning_rate: float = 2e-5,
                             max_length: int = 256) -> Tuple[SalesClassificationTrainer, Dict]:
    """Train a classification model"""
    
    logger.info("=== TRAINING CLASSIFICATION HEAD MODEL ===")
    
    # Load data
    logger.info("Loading data...")
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_data_path, 'r') as f:
        val_data = json.load(f)
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create data loaders
    train_dataset = SalesConversationDataset(train_data, tokenizer, max_length)
    val_dataset = SalesConversationDataset(val_data, tokenizer, max_length)
    test_dataset = SalesConversationDataset(test_data, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SalesClassificationModel(model_name=model_name, freeze_encoder=True)
    trainer = SalesClassificationTrainer(model)
    
    # Train model
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy, test_metrics = trainer.evaluate(test_loader)
    
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"  AUC: {test_metrics['auc']:.4f}")
    
    # Save model
    trainer.save_model('classification_head_model.pth')
    
    # Plot training history
    trainer.plot_training_history('training_history.png')
    
    results = {
        'training_results': training_results,
        'test_metrics': test_metrics,
        'model_info': {
            'model_name': model_name,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'max_length': max_length
        }
    }
    
    return trainer, results

def compare_frozen_vs_unfrozen(train_data_path: str = 'train_data.json',
                              val_data_path: str = 'val_data.json',
                              test_data_path: str = 'test_data.json') -> Dict:
    """Compare frozen encoder vs unfrozen encoder"""
    
    logger.info("=== COMPARING FROZEN VS UNFROZEN ENCODER ===")
    
    results = {}
    
    for freeze_encoder in [True, False]:
        mode = "frozen" if freeze_encoder else "unfrozen"
        logger.info(f"\nTraining with {mode} encoder...")
        
        # Load data
        with open(train_data_path, 'r') as f:
            train_data = json.load(f)
        
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Setup
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        train_dataset = SalesConversationDataset(train_data, tokenizer, 256)
        val_dataset = SalesConversationDataset(val_data, tokenizer, 256)
        test_dataset = SalesConversationDataset(test_data, tokenizer, 256)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Train model
        model = SalesClassificationModel(freeze_encoder=freeze_encoder)
        trainer = SalesClassificationTrainer(model)
        
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,  # Fewer epochs for comparison
            learning_rate=2e-5 if freeze_encoder else 1e-5  # Lower LR for unfrozen
        )
        
        # Evaluate
        test_loss, test_accuracy, test_metrics = trainer.evaluate(test_loader)
        
        results[mode] = {
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_score'],
            'training_time': training_results['training_time'],
            'best_val_accuracy': training_results['best_val_accuracy']
        }
        
        logger.info(f"{mode.title()} Results:")
        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Test F1: {test_metrics['f1_score']:.4f}")
        logger.info(f"  Training Time: {training_results['training_time']:.2f}s")
    
    # Print comparison
    print("\n=== COMPARISON RESULTS ===")
    print(f"{'Mode':<10} {'Accuracy':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 45)
    for mode, metrics in results.items():
        print(f"{mode:<10} {metrics['test_accuracy']:<10.4f} {metrics['test_f1']:<10.4f} {metrics['training_time']:<10.1f}")
    
    return results

def main():
    """Main function"""
    # First, create data splits
    processor = SalesDataProcessor()
    processor.load_data()
    processor.preprocess_data()
    processor.create_splits()
    processor.save_splits()
    
    # Train classification model
    trainer, results = train_classification_model()
    
    # Compare frozen vs unfrozen
    comparison_results = compare_frozen_vs_unfrozen()
    
    return trainer, results, comparison_results

if __name__ == "__main__":
    trainer, results, comparison_results = main() 