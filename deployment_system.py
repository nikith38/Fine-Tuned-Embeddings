import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import our models and integrations
from langchain_integration import create_sales_pipeline, SalesAnalysisChain, SalesVectorStore
from evaluation_framework import SalesModelEvaluator
from few_shot_model import FewShotSalesPredictor
from classification_head_model import SalesClassificationModel, SalesClassificationTrainer
from contrastive_learning_model import ContrastiveSalesModel, ContrastiveTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sales_prediction_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class CustomerContext(BaseModel):
    """Customer context information"""
    company_size: str = Field(..., description="Company size (startup, small business, mid-market, enterprise)")
    industry: str = Field(..., description="Industry sector")
    contact_role: str = Field(..., description="Contact person's role")
    urgency: str = Field(..., description="Urgency level (low, medium, high)")
    budget_authority: str = Field(..., description="Budget authority (end_user, influencer, decision_maker)")
    previous_interactions: int = Field(default=1, description="Number of previous interactions")
    lead_source: str = Field(default="unknown", description="Lead source")

class PredictionRequest(BaseModel):
    """Request model for prediction"""
    transcript: str = Field(..., description="Sales conversation transcript")
    context: Optional[CustomerContext] = Field(None, description="Customer context information")
    model_type: Optional[str] = Field("auto", description="Model type to use (auto, few_shot, classification_head, contrastive)")
    include_analysis: bool = Field(True, description="Include detailed analysis")
    include_similar: bool = Field(True, description="Include similar conversations")

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    similar_conversations: Optional[List[str]] = None
    model_used: str
    processing_time: float
    confidence_score: float

class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_prediction_time: float
    total_predictions: int
    last_updated: str

@dataclass
class PredictionLog:
    """Log entry for predictions"""
    timestamp: str
    transcript: str
    context: Optional[Dict]
    prediction: Dict
    model_used: str
    processing_time: float
    confidence_score: float

class SalesModelManager:
    """Manager for loading and switching between different models"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.performance_metrics = {}
        self.prediction_logs = []
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        logger.info("Loading sales prediction models...")
        
        try:
            # Load few-shot model
            self.models['few_shot'] = {
                'chain': None,
                'vector_store': None,
                'loaded': False
            }
            
            # Try to load fine-tuned models if they exist
            model_paths = {
                'classification_head': 'classification_head_model.pth',
                'contrastive': 'contrastive_sales_model.pth'
            }
            
            for model_type, path in model_paths.items():
                if Path(path).exists():
                    self.models[model_type] = {
                        'chain': None,
                        'vector_store': None,
                        'path': path,
                        'loaded': False
                    }
                    logger.info(f"Found {model_type} model at {path}")
                else:
                    logger.warning(f"Model file not found: {path}")
            
            # Load default model
            self.load_model('few_shot')
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_model(self, model_type: str):
        """Load a specific model"""
        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} not available")
        
        try:
            logger.info(f"Loading {model_type} model...")
            
            if model_type == 'few_shot':
                chain, vector_store = create_sales_pipeline(
                    model_type='few_shot',
                    train_data_path='train_data.json'
                )
            else:
                model_path = self.models[model_type].get('path')
                chain, vector_store = create_sales_pipeline(
                    model_type=model_type,
                    model_path=model_path,
                    train_data_path='train_data.json'
                )
            
            self.models[model_type]['chain'] = chain
            self.models[model_type]['vector_store'] = vector_store
            self.models[model_type]['loaded'] = True
            self.current_model = model_type
            
            logger.info(f"Successfully loaded {model_type} model")
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {e}")
            raise
    
    def get_best_model(self) -> str:
        """Get the best performing model based on metrics"""
        if not self.performance_metrics:
            return 'few_shot'  # Default fallback
        
        best_model = max(
            self.performance_metrics.keys(),
            key=lambda x: self.performance_metrics[x].get('f1_score', 0)
        )
        return best_model
    
    def predict(self, transcript: str, context: Optional[Dict] = None, 
                model_type: str = 'auto') -> Tuple[Dict, str, float]:
        """Make prediction using specified or best model"""
        
        # Select model
        if model_type == 'auto':
            model_type = self.get_best_model()
        
        if model_type not in self.models or not self.models[model_type]['loaded']:
            # Fallback to few-shot if requested model not available
            model_type = 'few_shot'
            if not self.models[model_type]['loaded']:
                self.load_model(model_type)
        
        # Make prediction
        start_time = time.time()
        
        chain = self.models[model_type]['chain']
        result = chain({
            'transcript': transcript,
            'context': context or {}
        })
        
        processing_time = time.time() - start_time
        confidence_score = result['prediction']['confidence']
        
        # Log prediction
        self.log_prediction(transcript, context, result, model_type, processing_time, confidence_score)
        
        return result, model_type, processing_time
    
    def log_prediction(self, transcript: str, context: Optional[Dict], 
                      result: Dict, model_used: str, processing_time: float, 
                      confidence_score: float):
        """Log prediction for monitoring"""
        log_entry = PredictionLog(
            timestamp=datetime.now().isoformat(),
            transcript=transcript[:100] + "..." if len(transcript) > 100 else transcript,
            context=context,
            prediction=result['prediction'],
            model_used=model_used,
            processing_time=processing_time,
            confidence_score=confidence_score
        )
        
        self.prediction_logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(self.prediction_logs) > 1000:
            self.prediction_logs = self.prediction_logs[-1000:]
    
    def get_performance_metrics(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models"""
        metrics = {}
        
        for model_type in self.models.keys():
            if model_type in self.performance_metrics:
                metrics[model_type] = ModelPerformance(
                    model_name=model_type,
                    **self.performance_metrics[model_type],
                    last_updated=datetime.now().isoformat()
                )
        
        return metrics
    
    def update_performance_metrics(self, model_type: str, metrics: Dict):
        """Update performance metrics for a model"""
        self.performance_metrics[model_type] = metrics
        logger.info(f"Updated performance metrics for {model_type}")

# Global model manager
model_manager = SalesModelManager()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Sales Conversion Prediction API")
    yield
    # Shutdown
    logger.info("Shutting down Sales Conversion Prediction API")

app = FastAPI(
    title="Sales Conversion Prediction API",
    description="AI-powered sales conversation analysis and conversion prediction",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sales Conversion Prediction API",
        "version": "1.0.0",
        "status": "active",
        "available_models": list(model_manager.models.keys()),
        "current_model": model_manager.current_model
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_conversion(request: PredictionRequest):
    """Predict sales conversion probability"""
    try:
        logger.info(f"Prediction request received for transcript: {request.transcript[:50]}...")
        
        # Convert context to dict if provided
        context = None
        if request.context:
            context = request.context.dict()
        
        # Make prediction
        result, model_used, processing_time = model_manager.predict(
            transcript=request.transcript,
            context=context,
            model_type=request.model_type
        )
        
        # Prepare response
        response = PredictionResponse(
            prediction=result['prediction'],
            analysis=result.get('analysis') if request.include_analysis else None,
            recommendations=result.get('recommendations') if request.include_analysis else None,
            similar_conversations=result.get('similar_conversations') if request.include_similar else None,
            model_used=model_used,
            processing_time=processing_time,
            confidence_score=result['prediction']['confidence']
        )
        
        logger.info(f"Prediction completed: {result['prediction']['outcome']} "
                   f"(probability: {result['prediction']['probability']:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available models and their status"""
    models_info = {}
    
    for model_type, info in model_manager.models.items():
        models_info[model_type] = {
            "loaded": info['loaded'],
            "current": model_type == model_manager.current_model,
            "path": info.get('path', 'N/A')
        }
    
    return {
        "models": models_info,
        "current_model": model_manager.current_model
    }

@app.post("/models/{model_type}/load")
async def load_model(model_type: str, background_tasks: BackgroundTasks):
    """Load a specific model"""
    if model_type not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
    
    try:
        background_tasks.add_task(model_manager.load_model, model_type)
        return {"message": f"Loading {model_type} model in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
async def get_performance():
    """Get performance metrics for all models"""
    return model_manager.get_performance_metrics()

@app.get("/logs")
async def get_prediction_logs(limit: int = 100):
    """Get recent prediction logs"""
    logs = model_manager.prediction_logs[-limit:]
    return {
        "logs": [asdict(log) for log in logs],
        "total_predictions": len(model_manager.prediction_logs)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": sum(1 for info in model_manager.models.values() if info['loaded']),
        "total_models": len(model_manager.models),
        "total_predictions": len(model_manager.prediction_logs)
    }

@app.post("/evaluate")
async def evaluate_models(background_tasks: BackgroundTasks):
    """Trigger model evaluation"""
    try:
        background_tasks.add_task(run_model_evaluation)
        return {"message": "Model evaluation started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_model_evaluation():
    """Run comprehensive model evaluation"""
    try:
        logger.info("Starting model evaluation...")
        
        evaluator = SalesModelEvaluator()
        results = evaluator.compare_all_models()
        
        # Update performance metrics
        for model_name, metrics in results['individual_results'].items():
            if 'error' not in metrics:
                model_manager.update_performance_metrics(model_name, metrics)
        
        logger.info("Model evaluation completed")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")

class SalesAPIClient:
    """Client for interacting with the Sales Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, transcript: str, context: Optional[Dict] = None, 
                model_type: str = "auto") -> Dict:
        """Make a prediction request"""
        import requests
        
        payload = {
            "transcript": transcript,
            "context": context,
            "model_type": model_type,
            "include_analysis": True,
            "include_similar": True
        }
        
        response = requests.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_models(self) -> Dict:
        """Get available models"""
        import requests
        
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        
        return response.json()
    
    def get_performance(self) -> Dict:
        """Get performance metrics"""
        import requests
        
        response = requests.get(f"{self.base_url}/performance")
        response.raise_for_status()
        
        return response.json()

def create_demo_script():
    """Create a demo script for testing the API"""
    demo_script = '''
import requests
import json

# Demo sales conversations
demo_conversations = [
    {
        "transcript": "Customer was very excited about our CRM software. They asked detailed questions about pricing and implementation timeline. They mentioned they have a budget of $50,000 and want to move forward quickly.",
        "context": {
            "company_size": "mid-market",
            "industry": "technology",
            "contact_role": "VP of Sales",
            "urgency": "high",
            "budget_authority": "decision_maker",
            "previous_interactions": 2,
            "lead_source": "website"
        }
    },
    {
        "transcript": "Customer seemed hesitant throughout the call. They raised concerns about cost and said they need to think about it more. They mentioned budget constraints.",
        "context": {
            "company_size": "startup",
            "industry": "healthcare",
            "contact_role": "IT Director",
            "urgency": "low",
            "budget_authority": "influencer",
            "previous_interactions": 1,
            "lead_source": "cold_call"
        }
    }
]

# Test predictions
base_url = "http://localhost:8000"

for i, demo in enumerate(demo_conversations):
    print(f"\\n=== Demo Conversation {i+1} ===")
    
    response = requests.post(f"{base_url}/predict", json={
        "transcript": demo["transcript"],
        "context": demo["context"],
        "model_type": "auto",
        "include_analysis": True,
        "include_similar": True
    })
    
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']['outcome']}")
        print(f"Probability: {result['prediction']['probability']:.3f}")
        print(f"Model used: {result['model_used']}")
        print(f"Processing time: {result['processing_time']:.3f}s")
        
        if result['recommendations']:
            print("Recommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Check API health
health_response = requests.get(f"{base_url}/health")
print(f"\\n=== API Health ===")
print(health_response.json())
'''
    
    with open('api_demo.py', 'w') as f:
        f.write(demo_script)
    
    logger.info("Demo script created: api_demo.py")

def main():
    """Main function to start the deployment system"""
    logger.info("=== SALES CONVERSION PREDICTION - DEPLOYMENT SYSTEM ===")
    
    # Create demo script
    create_demo_script()
    
    # Create requirements file
    requirements = '''
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
torch>=1.13.0
transformers>=4.21.0
sentence-transformers>=2.2.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
langchain>=0.0.350
faiss-cpu>=1.7.0
requests>=2.28.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    logger.info("Requirements file created: requirements.txt")
    
    # Print deployment instructions
    print("\n🚀 DEPLOYMENT SYSTEM READY!")
    print("\n📋 To start the API server:")
    print("   pip install -r requirements.txt")
    print("   python -m uvicorn deployment_system:app --host 0.0.0.0 --port 8000")
    print("\n🧪 To test the API:")
    print("   python api_demo.py")
    print("\n📊 API Documentation:")
    print("   http://localhost:8000/docs")
    print("\n💡 Key Features:")
    print("   ✅ Multiple model support (few-shot, classification head, contrastive)")
    print("   ✅ Automatic model selection")
    print("   ✅ Performance monitoring")
    print("   ✅ Prediction logging")
    print("   ✅ Similar conversation search")
    print("   ✅ Comprehensive analysis and recommendations")
    print("   ✅ RESTful API with FastAPI")
    print("   ✅ Background model evaluation")
    
    return app

if __name__ == "__main__":
    app = main()
    # Run with: uvicorn deployment_system:app --host 0.0.0.0 --port 8000 --reload 