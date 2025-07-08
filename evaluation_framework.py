import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer
import time
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our models
from few_shot_model import FewShotSalesPredictor
from classification_head_model import SalesClassificationModel, SalesClassificationTrainer
from contrastive_learning_model import ContrastiveSalesModel, ContrastiveTrainer
from data_pipeline import SalesDataProcessor, SalesConversationDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesModelEvaluator:
    """Comprehensive evaluation framework for sales conversion prediction models"""
    
    def __init__(self, test_data_path: str = 'test_data.json'):
        self.test_data_path = test_data_path
        self.test_data = None
        self.results = {}
        self.load_test_data()
    
    def load_test_data(self):
        """Load test data"""
        with open(self.test_data_path, 'r') as f:
            self.test_data = json.load(f)
        logger.info(f"Loaded {len(self.test_data)} test examples")
    
    def evaluate_few_shot_model(self, model_path: str = None, 
                               train_data_path: str = 'train_data.json') -> Dict:
        """Evaluate few-shot learning model"""
        logger.info("=== EVALUATING FEW-SHOT MODEL ===")
        
        # Load training data for few-shot model
        with open(train_data_path, 'r') as f:
            train_data = json.load(f)
        
        # Initialize and train model
        start_time = time.time()
        model = FewShotSalesPredictor(model_name='all-MiniLM-L6-v2')
        
        if model_path:
            model.load_model(model_path)
        else:
            model.train(train_data, include_context=True)
        
        training_time = time.time() - start_time
        
        # Evaluate different methods
        methods = ['similarity', 'knn', 'weighted']
        method_results = {}
        
        for method in methods:
            logger.info(f"Evaluating few-shot with {method} method...")
            
            # Prepare test inputs
            texts = [item['transcript'] for item in self.test_data]
            contexts = [item['customer_context'] for item in self.test_data]
            true_labels = [item['conversion_label'] for item in self.test_data]
            
            # Get predictions
            start_pred_time = time.time()
            predictions = model.predict_batch(texts, contexts, method)
            prediction_time = time.time() - start_pred_time
            
            probabilities = [prob for prob, _ in predictions]
            predicted_labels = [1 if prob >= 0.5 else 0 for prob in probabilities]
            
            # Calculate metrics
            metrics = self._calculate_metrics(true_labels, predicted_labels, probabilities)
            metrics['training_time'] = training_time
            metrics['prediction_time'] = prediction_time
            metrics['avg_prediction_time'] = prediction_time / len(self.test_data)
            
            method_results[method] = metrics
        
        # Use best method for final results
        best_method = max(method_results.keys(), key=lambda x: method_results[x]['f1_score'])
        best_results = method_results[best_method]
        best_results['best_method'] = best_method
        best_results['all_methods'] = method_results
        
        self.results['few_shot'] = best_results
        logger.info(f"Best few-shot method: {best_method} (F1: {best_results['f1_score']:.4f})")
        
        return best_results
    
    def evaluate_classification_model(self, model_path: str = 'classification_head_model.pth') -> Dict:
        """Evaluate classification head model"""
        logger.info("=== EVALUATING CLASSIFICATION HEAD MODEL ===")
        
        try:
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SalesClassificationModel()
            trainer = SalesClassificationTrainer(model, device=device)
            trainer.load_model(model_path)
            
            # Prepare test data
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            test_dataset = SalesConversationDataset(self.test_data, tokenizer, max_length=256)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Evaluate
            start_time = time.time()
            test_loss, test_accuracy, test_metrics = trainer.evaluate(test_loader)
            evaluation_time = time.time() - start_time
            
            # Add timing information
            test_metrics['evaluation_time'] = evaluation_time
            test_metrics['avg_prediction_time'] = evaluation_time / len(self.test_data)
            test_metrics['training_time'] = getattr(trainer, 'training_time', 0)
            
            self.results['classification_head'] = test_metrics
            logger.info(f"Classification model F1: {test_metrics['f1_score']:.4f}")
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {'error': str(e)}
    
    def evaluate_contrastive_model(self, model_path: str = 'contrastive_sales_model.pth') -> Dict:
        """Evaluate contrastive learning model"""
        logger.info("=== EVALUATING CONTRASTIVE LEARNING MODEL ===")
        
        try:
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = ContrastiveSalesModel()
            trainer = ContrastiveTrainer(model, device=device)
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Prepare test data
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            from contrastive_learning_model import ContrastiveSalesDataset
            test_dataset = ContrastiveSalesDataset(self.test_data, tokenizer)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Evaluate
            start_time = time.time()
            results = trainer.evaluate_embeddings(test_loader)
            evaluation_time = time.time() - start_time
            
            # Add timing information
            results['evaluation_time'] = evaluation_time
            results['avg_prediction_time'] = evaluation_time / len(self.test_data)
            results['training_time'] = checkpoint.get('training_history', {}).get('training_time', 0)
            
            self.results['contrastive'] = results
            logger.info(f"Contrastive model F1: {results['f1_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating contrastive model: {e}")
            return {'error': str(e)}
    
    def _calculate_metrics(self, true_labels: List[int], predicted_labels: List[int], 
                          probabilities: List[float]) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels, zero_division=0),
            'recall': recall_score(true_labels, predicted_labels, zero_division=0),
            'f1_score': f1_score(true_labels, predicted_labels, zero_division=0),
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'probabilities': probabilities
        }
        
        # Calculate AUC if possible
        try:
            metrics['auc'] = roc_auc_score(true_labels, probabilities)
        except:
            metrics['auc'] = 0.0
        
        return metrics
    
    def compare_all_models(self) -> Dict:
        """Compare all models and generate comprehensive report"""
        logger.info("=== COMPARING ALL MODELS ===")
        
        # Evaluate all models
        few_shot_results = self.evaluate_few_shot_model()
        classification_results = self.evaluate_classification_model()
        contrastive_results = self.evaluate_contrastive_model()
        
        # Create comparison dataframe
        comparison_data = []
        
        models = [
            ('Few-Shot', few_shot_results),
            ('Classification Head', classification_results),
            ('Contrastive Learning', contrastive_results)
        ]
        
        for model_name, results in models:
            if 'error' not in results:
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results.get('accuracy', 0),
                    'Precision': results.get('precision', 0),
                    'Recall': results.get('recall', 0),
                    'F1 Score': results.get('f1_score', 0),
                    'AUC': results.get('auc', 0),
                    'Training Time (s)': results.get('training_time', 0),
                    'Avg Prediction Time (ms)': results.get('avg_prediction_time', 0) * 1000
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison table
        print("\n=== MODEL COMPARISON RESULTS ===")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        if not comparison_df.empty:
            best_model_idx = comparison_df['F1 Score'].idxmax()
            best_model = comparison_df.iloc[best_model_idx]['Model']
            best_f1 = comparison_df.iloc[best_model_idx]['F1 Score']
            
            print(f"\n🏆 Best Model: {best_model} (F1 Score: {best_f1:.4f})")
        
        return {
            'comparison_table': comparison_df,
            'individual_results': self.results,
            'best_model': best_model if not comparison_df.empty else None
        }
    
    def plot_model_comparison(self, save_path: str = 'model_comparison.png'):
        """Create visualization comparing all models"""
        if not self.results:
            logger.warning("No results to plot. Run comparison first.")
            return
        
        # Prepare data for plotting
        models = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        data = {metric: [] for metric in metrics}
        
        for model_name, results in self.results.items():
            if 'error' not in results:
                models.append(model_name.replace('_', ' ').title())
                for metric in metrics:
                    data[metric].append(results.get(metric, 0))
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(models, data[metric], alpha=0.7)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, data[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(models) > 2:
                ax.tick_params(axis='x', rotation=45)
        
        # Plot training time comparison
        ax = axes[5]
        training_times = [self.results[model].get('training_time', 0) for model in self.results.keys() if 'error' not in self.results[model]]
        model_names = [name.replace('_', ' ').title() for name in self.results.keys() if 'error' not in self.results[name]]
        
        bars = ax.bar(model_names, training_times, alpha=0.7, color='orange')
        ax.set_title('Training Time')
        ax.set_ylabel('Seconds')
        
        for bar, value in zip(bars, training_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01,
                   f'{value:.1f}s', ha='center', va='bottom')
        
        if len(model_names) > 2:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
        plt.show()
    
    def plot_confusion_matrices(self, save_path: str = 'confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        models_with_predictions = []
        
        for model_name, results in self.results.items():
            if 'error' not in results and 'true_labels' in results and 'predicted_labels' in results:
                models_with_predictions.append((model_name, results))
        
        if not models_with_predictions:
            logger.warning("No models with predictions to plot confusion matrices.")
            return
        
        n_models = len(models_with_predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(models_with_predictions):
            cm = confusion_matrix(results['true_labels'], results['predicted_labels'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Failed', 'Successful'],
                       yticklabels=['Failed', 'Successful'])
            
            axes[i].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")
        plt.show()
    
    def plot_roc_curves(self, save_path: str = 'roc_curves.png'):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            if 'error' not in results and 'true_labels' in results and 'probabilities' in results:
                try:
                    fpr, tpr, _ = roc_curve(results['true_labels'], results['probabilities'])
                    auc = results.get('auc', 0)
                    plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f})')
                except:
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
        plt.show()
    
    def analyze_prediction_errors(self) -> Dict:
        """Analyze prediction errors across models"""
        logger.info("=== ANALYZING PREDICTION ERRORS ===")
        
        error_analysis = {}
        
        for model_name, results in self.results.items():
            if 'error' not in results and 'true_labels' in results and 'predicted_labels' in results:
                true_labels = np.array(results['true_labels'])
                predicted_labels = np.array(results['predicted_labels'])
                
                # Find misclassified examples
                misclassified = true_labels != predicted_labels
                misclassified_indices = np.where(misclassified)[0]
                
                # Analyze types of errors
                false_positives = np.where((true_labels == 0) & (predicted_labels == 1))[0]
                false_negatives = np.where((true_labels == 1) & (predicted_labels == 0))[0]
                
                error_analysis[model_name] = {
                    'total_errors': len(misclassified_indices),
                    'error_rate': len(misclassified_indices) / len(true_labels),
                    'false_positives': len(false_positives),
                    'false_negatives': len(false_negatives),
                    'misclassified_examples': [
                        {
                            'index': int(idx),
                            'true_label': int(true_labels[idx]),
                            'predicted_label': int(predicted_labels[idx]),
                            'transcript': self.test_data[idx]['transcript'][:100] + '...',
                            'industry': self.test_data[idx]['customer_context']['industry'],
                            'company_size': self.test_data[idx]['customer_context']['company_size']
                        }
                        for idx in misclassified_indices[:5]  # Show first 5 errors
                    ]
                }
        
        # Print error analysis
        for model_name, analysis in error_analysis.items():
            print(f"\n=== {model_name.replace('_', ' ').title()} Error Analysis ===")
            print(f"Total errors: {analysis['total_errors']}")
            print(f"Error rate: {analysis['error_rate']:.3f}")
            print(f"False positives: {analysis['false_positives']}")
            print(f"False negatives: {analysis['false_negatives']}")
            
            if analysis['misclassified_examples']:
                print("\nSample misclassified examples:")
                for i, example in enumerate(analysis['misclassified_examples']):
                    print(f"{i+1}. True: {example['true_label']}, Pred: {example['predicted_label']}")
                    print(f"   Industry: {example['industry']}, Size: {example['company_size']}")
                    print(f"   Text: {example['transcript']}")
        
        return error_analysis
    
    def generate_final_report(self, save_path: str = 'sales_model_evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        logger.info("=== GENERATING FINAL REPORT ===")
        
        # Run all evaluations
        comparison_results = self.compare_all_models()
        error_analysis = self.analyze_prediction_errors()
        
        # Create plots
        self.plot_model_comparison()
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        
        # Compile final report
        report = {
            'evaluation_summary': {
                'test_data_size': len(self.test_data),
                'models_evaluated': list(self.results.keys()),
                'best_model': comparison_results.get('best_model'),
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'model_comparison': comparison_results['comparison_table'].to_dict('records'),
            'detailed_results': self.results,
            'error_analysis': error_analysis,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {save_path}")
        return report
    
    def _generate_recommendations(self) -> Dict:
        """Generate recommendations based on evaluation results"""
        recommendations = {
            'best_overall_model': None,
            'best_for_speed': None,
            'best_for_accuracy': None,
            'deployment_recommendations': [],
            'improvement_suggestions': []
        }
        
        if not self.results:
            return recommendations
        
        # Find best models for different criteria
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if valid_results:
            # Best overall (F1 score)
            best_f1_model = max(valid_results.keys(), 
                               key=lambda x: valid_results[x].get('f1_score', 0))
            recommendations['best_overall_model'] = best_f1_model
            
            # Best for speed (lowest prediction time)
            best_speed_model = min(valid_results.keys(),
                                  key=lambda x: valid_results[x].get('avg_prediction_time', float('inf')))
            recommendations['best_for_speed'] = best_speed_model
            
            # Best for accuracy
            best_accuracy_model = max(valid_results.keys(),
                                     key=lambda x: valid_results[x].get('accuracy', 0))
            recommendations['best_for_accuracy'] = best_accuracy_model
            
            # Deployment recommendations
            recommendations['deployment_recommendations'] = [
                f"For production deployment, consider {best_overall_model} for best overall performance",
                f"For real-time applications, consider {best_speed_model} for fastest predictions",
                f"For maximum accuracy, consider {best_accuracy_model}"
            ]
            
            # Improvement suggestions
            recommendations['improvement_suggestions'] = [
                "Consider ensemble methods combining multiple approaches",
                "Experiment with different hyperparameters",
                "Collect more training data for better performance",
                "Add domain-specific features to improve predictions"
            ]
        
        return recommendations

def main():
    """Main evaluation function"""
    logger.info("=== SALES CONVERSION PREDICTION - MODEL EVALUATION ===")
    
    # Initialize evaluator
    evaluator = SalesModelEvaluator()
    
    # Generate comprehensive report
    report = evaluator.generate_final_report()
    
    print("\n🎯 EVALUATION COMPLETE!")
    print(f"📊 Report saved as: sales_model_evaluation_report.json")
    print(f"📈 Visualizations created: model_comparison.png, confusion_matrices.png, roc_curves.png")
    
    return evaluator, report

if __name__ == "__main__":
    evaluator, report = main() 