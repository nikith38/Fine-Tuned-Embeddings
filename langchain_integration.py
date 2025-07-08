import json
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Tuple
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import logging
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Import our models
from few_shot_model import FewShotSalesPredictor
from classification_head_model import SalesClassificationModel, SalesClassificationTrainer
from contrastive_learning_model import ContrastiveSalesModel, ContrastiveTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesEmbeddings(Embeddings):
    """Custom LangChain embeddings class for sales conversations"""
    
    def __init__(self, model_type: str = 'sentence_transformer', 
                 model_path: str = None, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize sales embeddings
        
        Args:
            model_type: Type of model ('sentence_transformer', 'classification_head', 'contrastive')
            model_path: Path to saved model (for fine-tuned models)
            model_name: Name of base model
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model"""
        if self.model_type == 'sentence_transformer':
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            
        elif self.model_type == 'classification_head':
            if not self.model_path:
                raise ValueError("model_path required for classification_head model")
            
            self.model = SalesClassificationModel(model_name=self.model_name)
            self.trainer = SalesClassificationTrainer(self.model, device=self.device)
            self.trainer.load_model(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded classification head model from {self.model_path}")
            
        elif self.model_type == 'contrastive':
            if not self.model_path:
                raise ValueError("model_path required for contrastive model")
            
            self.model = ContrastiveSalesModel(model_name=self.model_name)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded contrastive model from {self.model_path}")
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self.model_type == 'sentence_transformer':
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
            
        elif self.model_type in ['classification_head', 'contrastive']:
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    if self.model_type == 'classification_head':
                        embedding = self.model.get_embeddings(
                            inputs['input_ids'], inputs['attention_mask']
                        )
                    else:  # contrastive
                        embedding = self.model.get_embeddings(
                            inputs['input_ids'], inputs['attention_mask']
                        )
                    
                    embeddings.append(embedding.cpu().numpy().flatten())
            
            return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

class SalesConversionPredictor(LLM):
    """LangChain LLM wrapper for sales conversion prediction"""
    
    def __init__(self, model_type: str = 'few_shot', model_path: str = None,
                 train_data_path: str = 'train_data.json'):
        """
        Initialize sales conversion predictor
        
        Args:
            model_type: Type of model ('few_shot', 'classification_head', 'contrastive')
            model_path: Path to saved model
            train_data_path: Path to training data (for few-shot model)
        """
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load the prediction model"""
        if self.model_type == 'few_shot':
            # Load training data
            with open(self.train_data_path, 'r') as f:
                train_data = json.load(f)
            
            self.model = FewShotSalesPredictor()
            
            if self.model_path:
                self.model.load_model(self.model_path)
            else:
                self.model.train(train_data, include_context=True)
            
            logger.info("Loaded few-shot sales predictor")
            
        elif self.model_type == 'classification_head':
            if not self.model_path:
                raise ValueError("model_path required for classification_head model")
            
            self.model = SalesClassificationModel()
            self.trainer = SalesClassificationTrainer(self.model, device=self.device)
            self.trainer.load_model(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            logger.info("Loaded classification head model")
            
        elif self.model_type == 'contrastive':
            if not self.model_path:
                raise ValueError("model_path required for contrastive model")
            
            self.model = ContrastiveSalesModel()
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            logger.info("Loaded contrastive model")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForChainRun] = None) -> str:
        """Make a prediction call"""
        # Parse the prompt to extract transcript and context
        transcript, context = self._parse_prompt(prompt)
        
        # Get prediction
        probability = self._predict(transcript, context)
        
        # Format response
        prediction = "successful" if probability >= 0.5 else "failed"
        confidence = max(probability, 1 - probability)
        
        response = f"Conversion Prediction: {prediction}\n"
        response += f"Probability: {probability:.3f}\n"
        response += f"Confidence: {confidence:.3f}"
        
        return response
    
    def _parse_prompt(self, prompt: str) -> Tuple[str, Optional[Dict]]:
        """Parse prompt to extract transcript and context"""
        # Simple parsing - in practice, you might want more sophisticated parsing
        lines = prompt.strip().split('\n')
        
        transcript = ""
        context = None
        
        for line in lines:
            if line.startswith("Transcript:"):
                transcript = line.replace("Transcript:", "").strip()
            elif line.startswith("Context:"):
                try:
                    context_str = line.replace("Context:", "").strip()
                    context = json.loads(context_str)
                except:
                    context = None
        
        if not transcript:
            transcript = prompt  # Use entire prompt as transcript if no explicit format
        
        return transcript, context
    
    def _predict(self, transcript: str, context: Optional[Dict] = None) -> float:
        """Make prediction using the loaded model"""
        if self.model_type == 'few_shot':
            probability, _ = self.model.predict_single(transcript, context, method='similarity')
            return probability
            
        elif self.model_type in ['classification_head', 'contrastive']:
            # Prepare input
            if context:
                text = f"Company: {context.get('company_size', '')} {context.get('industry', '')} | " \
                       f"Contact: {context.get('contact_role', '')} | " \
                       f"Conversation: {transcript}"
            else:
                text = transcript
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                if self.model_type == 'classification_head':
                    logits = self.model(inputs['input_ids'], inputs['attention_mask'])
                    probabilities = torch.softmax(logits, dim=1)
                    return probabilities[0][1].item()  # Probability of class 1
                else:  # contrastive
                    logits, _ = self.model(inputs['input_ids'], inputs['attention_mask'])
                    probabilities = torch.softmax(logits, dim=1)
                    return probabilities[0][1].item()
    
    @property
    def _llm_type(self) -> str:
        return "sales_conversion_predictor"

class SalesAnalysisChain(Chain):
    """LangChain chain for comprehensive sales conversation analysis"""
    
    input_keys: List[str] = ["transcript", "context"]
    output_keys: List[str] = ["prediction", "analysis", "recommendations"]
    
    def __init__(self, predictor: SalesConversionPredictor, 
                 embeddings: SalesEmbeddings, vectorstore: FAISS = None):
        super().__init__()
        self.predictor = predictor
        self.embeddings = embeddings
        self.vectorstore = vectorstore
    
    def _call(self, inputs: Dict[str, Any], 
              run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """Execute the sales analysis chain"""
        transcript = inputs["transcript"]
        context = inputs.get("context", {})
        
        # Get conversion prediction
        prediction_prompt = f"Transcript: {transcript}\nContext: {json.dumps(context)}"
        prediction_result = self.predictor(prediction_prompt)
        
        # Parse prediction
        lines = prediction_result.split('\n')
        prediction = lines[0].split(': ')[1] if len(lines) > 0 else "unknown"
        probability = float(lines[1].split(': ')[1]) if len(lines) > 1 else 0.0
        confidence = float(lines[2].split(': ')[1]) if len(lines) > 2 else 0.0
        
        # Generate analysis
        analysis = self._analyze_conversation(transcript, context, probability)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(transcript, context, prediction, probability)
        
        # Find similar conversations if vectorstore is available
        similar_conversations = []
        if self.vectorstore:
            similar_docs = self.vectorstore.similarity_search(transcript, k=3)
            similar_conversations = [doc.page_content for doc in similar_docs]
        
        return {
            "prediction": {
                "outcome": prediction,
                "probability": probability,
                "confidence": confidence
            },
            "analysis": analysis,
            "recommendations": recommendations,
            "similar_conversations": similar_conversations
        }
    
    def _analyze_conversation(self, transcript: str, context: Dict, probability: float) -> Dict:
        """Analyze the conversation for insights"""
        analysis = {
            "length": len(transcript.split()),
            "sentiment": "positive" if probability > 0.6 else "negative" if probability < 0.4 else "neutral",
            "key_indicators": [],
            "risk_factors": []
        }
        
        text_lower = transcript.lower()
        
        # Identify key indicators
        positive_indicators = ['price', 'budget', 'timeline', 'demo', 'contract', 'implementation', 'excited', 'interested']
        negative_indicators = ['expensive', 'think about', 'not sure', 'concerns', 'hesitant', 'not ready']
        
        for indicator in positive_indicators:
            if indicator in text_lower:
                analysis["key_indicators"].append(f"Mentioned {indicator}")
        
        for indicator in negative_indicators:
            if indicator in text_lower:
                analysis["risk_factors"].append(f"Expressed {indicator}")
        
        # Context analysis
        if context:
            if context.get('urgency') == 'high':
                analysis["key_indicators"].append("High urgency customer")
            if context.get('budget_authority') == 'decision_maker':
                analysis["key_indicators"].append("Speaking with decision maker")
            if context.get('company_size') == 'enterprise':
                analysis["key_indicators"].append("Enterprise customer")
        
        return analysis
    
    def _generate_recommendations(self, transcript: str, context: Dict, 
                                prediction: str, probability: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction == "successful":
            recommendations.extend([
                "🎯 High conversion probability - prioritize this lead",
                "📞 Schedule follow-up call within 24 hours",
                "📋 Prepare detailed proposal or demo",
                "🤝 Involve technical team if needed"
            ])
        else:
            recommendations.extend([
                "⚠️ Low conversion probability - needs nurturing",
                "📚 Provide additional educational content",
                "🔄 Address specific concerns mentioned",
                "⏰ Schedule follow-up in 1-2 weeks"
            ])
        
        # Context-specific recommendations
        if context:
            if context.get('urgency') == 'high':
                recommendations.append("⚡ Customer has high urgency - fast response critical")
            
            if context.get('budget_authority') != 'decision_maker':
                recommendations.append("👥 Identify and engage decision makers")
            
            if context.get('company_size') == 'enterprise':
                recommendations.append("🏢 Prepare enterprise-level proposal and pricing")
        
        return recommendations

class SalesVectorStore:
    """Wrapper for creating and managing sales conversation vector store"""
    
    def __init__(self, embeddings: SalesEmbeddings):
        self.embeddings = embeddings
        self.vectorstore = None
    
    def create_from_conversations(self, conversations: List[Dict], 
                                save_path: str = "sales_vectorstore") -> FAISS:
        """Create vector store from sales conversations"""
        logger.info(f"Creating vector store from {len(conversations)} conversations...")
        
        # Prepare documents
        documents = []
        for conv in conversations:
            # Create document content
            content = conv['transcript']
            
            # Add metadata
            metadata = {
                'id': conv['id'],
                'conversion_label': conv['conversion_label'],
                'outcome': conv['outcome'],
                'industry': conv['customer_context']['industry'],
                'company_size': conv['customer_context']['company_size'],
                'contact_role': conv['customer_context']['contact_role'],
                'urgency': conv['customer_context']['urgency']
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save vector store
        self.vectorstore.save_local(save_path)
        logger.info(f"Vector store saved to {save_path}")
        
        return self.vectorstore
    
    def load_vectorstore(self, path: str = "sales_vectorstore") -> FAISS:
        """Load existing vector store"""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        logger.info(f"Vector store loaded from {path}")
        return self.vectorstore
    
    def search_similar_conversations(self, query: str, k: int = 5, 
                                   filter_by: Dict = None) -> List[Dict]:
        """Search for similar conversations"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': None  # FAISS doesn't return scores by default
            }
            
            # Apply filters if specified
            if filter_by:
                matches = all(doc.metadata.get(key) == value for key, value in filter_by.items())
                if matches:
                    results.append(result)
            else:
                results.append(result)
        
        return results

def create_sales_pipeline(model_type: str = 'few_shot', 
                         model_path: str = None,
                         train_data_path: str = 'train_data.json') -> Tuple[SalesAnalysisChain, SalesVectorStore]:
    """Create complete sales analysis pipeline"""
    logger.info(f"Creating sales pipeline with {model_type} model...")
    
    # Initialize embeddings
    if model_type == 'few_shot':
        embeddings = SalesEmbeddings(model_type='sentence_transformer')
    else:
        embeddings = SalesEmbeddings(model_type=model_type, model_path=model_path)
    
    # Initialize predictor
    predictor = SalesConversionPredictor(
        model_type=model_type,
        model_path=model_path,
        train_data_path=train_data_path
    )
    
    # Create vector store
    vector_store = SalesVectorStore(embeddings)
    
    # Load training data for vector store
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    vectorstore = vector_store.create_from_conversations(train_data)
    
    # Create analysis chain
    analysis_chain = SalesAnalysisChain(
        predictor=predictor,
        embeddings=embeddings,
        vectorstore=vectorstore
    )
    
    logger.info("Sales pipeline created successfully!")
    
    return analysis_chain, vector_store

def demo_sales_pipeline():
    """Demonstrate the sales pipeline"""
    logger.info("=== SALES PIPELINE DEMO ===")
    
    # Create pipeline
    chain, vector_store = create_sales_pipeline(model_type='few_shot')
    
    # Demo conversation
    demo_transcript = """
    Customer was very excited about our CRM software. They asked detailed questions about 
    pricing and implementation timeline. They mentioned they have a budget of $50,000 
    and want to move forward quickly. They asked about training and support options.
    """
    
    demo_context = {
        "company_size": "mid-market",
        "industry": "technology",
        "contact_role": "VP of Sales",
        "urgency": "high",
        "budget_authority": "decision_maker"
    }
    
    # Run analysis
    result = chain({
        "transcript": demo_transcript,
        "context": demo_context
    })
    
    # Print results
    print("\n=== SALES CONVERSATION ANALYSIS ===")
    print(f"Transcript: {demo_transcript[:100]}...")
    print(f"\nPrediction: {result['prediction']}")
    print(f"\nAnalysis: {result['analysis']}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")
    
    if result['similar_conversations']:
        print(f"\nSimilar conversations found:")
        for i, conv in enumerate(result['similar_conversations'][:2]):
            print(f"  {i+1}. {conv[:80]}...")
    
    return chain, vector_store, result

def main():
    """Main function to demonstrate LangChain integration"""
    # Run demo
    chain, vector_store, result = demo_sales_pipeline()
    
    print("\n🚀 LangChain integration complete!")
    print("✅ Sales embeddings implemented")
    print("✅ Conversion predictor integrated")
    print("✅ Analysis chain created")
    print("✅ Vector store for similarity search")
    
    return chain, vector_store

if __name__ == "__main__":
    chain, vector_store = main() 