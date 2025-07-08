import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import pickle
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FewShotSalesPredictor:
    """Few-shot learning predictor for sales conversations"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.5):
        """
        Initialize the few-shot predictor
        
        Args:
            model_name: Name of the sentence transformer model
            similarity_threshold: Threshold for similarity-based prediction
        """
        logger.info(f"Loading pre-trained model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        
        # Storage for training examples
        self.successful_examples = []
        self.failed_examples = []
        self.all_examples = []
        
        # Performance tracking
        self.training_time = 0
        self.prediction_times = []
        
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        return self.embedding_model.encode([text])[0]
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embedding vectors"""
        return self.embedding_model.encode(texts)
    
    def train(self, train_data: List[Dict], include_context: bool = True) -> None:
        """
        Train the few-shot model by storing examples
        
        Args:
            train_data: List of training examples
            include_context: Whether to include context in text
        """
        logger.info(f"Training few-shot model with {len(train_data)} examples...")
        start_time = time.time()
        
        # Prepare texts for batch encoding
        texts = []
        labels = []
        
        for item in train_data:
            if include_context:
                text = self._add_context_to_text(item['transcript'], item['customer_context'])
            else:
                text = item['transcript']
            
            texts.append(text)
            labels.append(item['conversion_label'])
        
        # Batch encode all texts
        logger.info("Encoding all training examples...")
        embeddings = self._encode_batch(texts)
        
        # Store examples with embeddings
        for i, (text, label, embedding) in enumerate(zip(texts, labels, embeddings)):
            example = {
                'text': text,
                'embedding': embedding,
                'original_data': train_data[i],
                'label': label
            }
            
            self.all_examples.append(example)
            
            if label == 1:
                self.successful_examples.append(example)
            else:
                self.failed_examples.append(example)
        
        self.training_time = time.time() - start_time
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        logger.info(f"Stored {len(self.successful_examples)} successful examples")
        logger.info(f"Stored {len(self.failed_examples)} failed examples")
    
    def _add_context_to_text(self, transcript: str, context: Dict) -> str:
        """Add context information to transcript"""
        context_text = f"Company: {context['company_size']} {context['industry']} | " \
                      f"Contact: {context['contact_role']} | " \
                      f"Urgency: {context['urgency']} | " \
                      f"Authority: {context['budget_authority']} | " \
                      f"Conversation: {transcript}"
        return context_text
    
    def predict_single(self, text: str, context: Optional[Dict] = None, method: str = 'similarity') -> Tuple[float, Dict]:
        """
        Predict conversion probability for a single example
        
        Args:
            text: Input transcript
            context: Customer context (optional)
            method: Prediction method ('similarity', 'knn', 'weighted')
            
        Returns:
            Tuple of (probability, details)
        """
        start_time = time.time()
        
        # Prepare input text
        if context:
            input_text = self._add_context_to_text(text, context)
        else:
            input_text = text
        
        # Encode input
        input_embedding = self._encode_text(input_text)
        
        # Calculate similarities
        if method == 'similarity':
            probability, details = self._predict_similarity(input_embedding)
        elif method == 'knn':
            probability, details = self._predict_knn(input_embedding, k=5)
        elif method == 'weighted':
            probability, details = self._predict_weighted(input_embedding)
        else:
            raise ValueError(f"Unknown prediction method: {method}")
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        details['prediction_time'] = prediction_time
        details['input_text'] = input_text
        
        return probability, details
    
    def _predict_similarity(self, input_embedding: np.ndarray) -> Tuple[float, Dict]:
        """Predict using maximum similarity approach"""
        if not self.successful_examples or not self.failed_examples:
            return 0.5, {'method': 'similarity', 'reason': 'insufficient_examples'}
        
        # Calculate similarities to successful examples
        success_similarities = []
        for example in self.successful_examples:
            similarity = cosine_similarity([input_embedding], [example['embedding']])[0][0]
            success_similarities.append(similarity)
        
        # Calculate similarities to failed examples
        fail_similarities = []
        for example in self.failed_examples:
            similarity = cosine_similarity([input_embedding], [example['embedding']])[0][0]
            fail_similarities.append(similarity)
        
        max_success_sim = max(success_similarities)
        max_fail_sim = max(fail_similarities)
        
        # Calculate probability
        if max_success_sim + max_fail_sim == 0:
            probability = 0.5
        else:
            probability = max_success_sim / (max_success_sim + max_fail_sim)
        
        details = {
            'method': 'similarity',
            'max_success_similarity': max_success_sim,
            'max_fail_similarity': max_fail_sim,
            'avg_success_similarity': np.mean(success_similarities),
            'avg_fail_similarity': np.mean(fail_similarities)
        }
        
        return probability, details
    
    def _predict_knn(self, input_embedding: np.ndarray, k: int = 5) -> Tuple[float, Dict]:
        """Predict using k-nearest neighbors"""
        if len(self.all_examples) < k:
            k = len(self.all_examples)
        
        # Calculate similarities to all examples
        similarities = []
        for example in self.all_examples:
            similarity = cosine_similarity([input_embedding], [example['embedding']])[0][0]
            similarities.append((similarity, example['label']))
        
        # Get top k similar examples
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]
        
        # Calculate probability as proportion of successful examples in top k
        successful_count = sum(1 for _, label in top_k if label == 1)
        probability = successful_count / k
        
        details = {
            'method': 'knn',
            'k': k,
            'top_k_similarities': [sim for sim, _ in top_k],
            'top_k_labels': [label for _, label in top_k],
            'successful_in_top_k': successful_count
        }
        
        return probability, details
    
    def _predict_weighted(self, input_embedding: np.ndarray) -> Tuple[float, Dict]:
        """Predict using weighted similarity approach"""
        if not self.all_examples:
            return 0.5, {'method': 'weighted', 'reason': 'no_examples'}
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        similarities = []
        for example in self.all_examples:
            similarity = cosine_similarity([input_embedding], [example['embedding']])[0][0]
            weight = max(0, similarity)  # Only positive similarities
            
            total_weight += weight
            weighted_score += weight * example['label']
            similarities.append(similarity)
        
        if total_weight == 0:
            probability = 0.5
        else:
            probability = weighted_score / total_weight
        
        details = {
            'method': 'weighted',
            'total_weight': total_weight,
            'weighted_score': weighted_score,
            'avg_similarity': np.mean(similarities),
            'max_similarity': max(similarities)
        }
        
        return probability, details
    
    def predict_batch(self, texts: List[str], contexts: Optional[List[Dict]] = None, method: str = 'similarity') -> List[Tuple[float, Dict]]:
        """Predict conversion probabilities for batch of examples"""
        logger.info(f"Predicting {len(texts)} examples using {method} method...")
        
        results = []
        for i, text in enumerate(tqdm(texts, desc="Predicting")):
            context = contexts[i] if contexts else None
            probability, details = self.predict_single(text, context, method)
            results.append((probability, details))
        
        return results
    
    def evaluate(self, test_data: List[Dict], method: str = 'similarity', threshold: float = 0.5) -> Dict:
        """Evaluate model performance on test data"""
        logger.info(f"Evaluating on {len(test_data)} test examples...")
        
        # Prepare test inputs
        texts = [item['transcript'] for item in test_data]
        contexts = [item['customer_context'] for item in test_data]
        true_labels = [item['conversion_label'] for item in test_data]
        
        # Get predictions
        predictions = self.predict_batch(texts, contexts, method)
        probabilities = [prob for prob, _ in predictions]
        predicted_labels = [1 if prob >= threshold else 0 for prob in probabilities]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # Calculate average prediction time
        avg_prediction_time = np.mean(self.prediction_times) if self.prediction_times else 0
        
        results = {
            'method': method,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_test_examples': len(test_data),
            'training_time': self.training_time,
            'avg_prediction_time': avg_prediction_time,
            'probabilities': probabilities,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels
        }
        
        logger.info(f"Results: Accuracy={accuracy:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        model_data = {
            'successful_examples': self.successful_examples,
            'failed_examples': self.failed_examples,
            'all_examples': self.all_examples,
            'similarity_threshold': self.similarity_threshold,
            'training_time': self.training_time,
            'model_name': self.embedding_model.get_sentence_embedding_dimension()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.successful_examples = model_data['successful_examples']
        self.failed_examples = model_data['failed_examples']
        self.all_examples = model_data['all_examples']
        self.similarity_threshold = model_data['similarity_threshold']
        self.training_time = model_data['training_time']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_most_similar_examples(self, text: str, context: Optional[Dict] = None, top_k: int = 5) -> List[Dict]:
        """Get most similar training examples for interpretability"""
        if context:
            input_text = self._add_context_to_text(text, context)
        else:
            input_text = text
        
        input_embedding = self._encode_text(input_text)
        
        # Calculate similarities
        similarities = []
        for example in self.all_examples:
            similarity = cosine_similarity([input_embedding], [example['embedding']])[0][0]
            similarities.append((similarity, example))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k
        return [
            {
                'similarity': sim,
                'text': example['text'],
                'label': example['label'],
                'outcome': 'successful' if example['label'] == 1 else 'failed'
            }
            for sim, example in similarities[:top_k]
        ]

def compare_prediction_methods(model: FewShotSalesPredictor, test_data: List[Dict]) -> Dict:
    """Compare different prediction methods"""
    logger.info("Comparing prediction methods...")
    
    methods = ['similarity', 'knn', 'weighted']
    results = {}
    
    for method in methods:
        logger.info(f"Evaluating {method} method...")
        result = model.evaluate(test_data, method=method)
        results[method] = result
    
    # Print comparison
    print("\n=== METHOD COMPARISON ===")
    print(f"{'Method':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(ms)':<10}")
    print("-" * 70)
    
    for method, result in results.items():
        print(f"{method:<12} {result['accuracy']:<10.3f} {result['precision']:<10.3f} "
              f"{result['recall']:<10.3f} {result['f1_score']:<10.3f} "
              f"{result['avg_prediction_time']*1000:<10.1f}")
    
    return results

def main():
    """Main function to demonstrate few-shot learning"""
    logger.info("=== FEW-SHOT SALES CONVERSION PREDICTION ===")
    
    # Load data
    logger.info("Loading training data...")
    with open('train_data.json', 'r') as f:
        train_data = json.load(f)
    
    with open('test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Initialize model
    model = FewShotSalesPredictor(model_name='all-MiniLM-L6-v2')
    
    # Train model
    model.train(train_data, include_context=True)
    
    # Evaluate with different methods
    results = compare_prediction_methods(model, test_data)
    
    # Save model
    model.save_model('few_shot_sales_model.pkl')
    
    # Test single prediction
    logger.info("\n=== SINGLE PREDICTION EXAMPLE ===")
    test_example = test_data[0]
    probability, details = model.predict_single(
        test_example['transcript'], 
        test_example['customer_context'],
        method='similarity'
    )
    
    print(f"Input: {test_example['transcript'][:100]}...")
    print(f"True label: {test_example['conversion_label']}")
    print(f"Predicted probability: {probability:.3f}")
    print(f"Prediction details: {details}")
    
    # Show similar examples
    similar_examples = model.get_most_similar_examples(
        test_example['transcript'],
        test_example['customer_context'],
        top_k=3
    )
    
    print(f"\nMost similar training examples:")
    for i, example in enumerate(similar_examples):
        print(f"{i+1}. Similarity: {example['similarity']:.3f} | "
              f"Label: {example['outcome']} | "
              f"Text: {example['text'][:80]}...")
    
    return model, results

if __name__ == "__main__":
    model, results = main() 