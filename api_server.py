#!/usr/bin/env python3
"""
D&D LoRA API Server
===================

FastAPI server for real-time comparison between original and LoRA fine-tuned models.
Optimized for production deployment with proper error handling.

Usage:
    python api_server.py
    python api_server.py --model distilgpt2 --port 8000
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Dict, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dnd_lora_core import DnDModelComparator

# Global model comparator
comparator: Optional[DnDModelComparator] = None

# API Models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 150  # Increased for more complete responses
    temperature: float = 0.8  # Slightly higher for more creativity

class GenerateResponse(BaseModel):
    prompt: str
    original_response: str
    lora_response: str
    original_dnd_terms: int
    lora_dnd_terms: int
    improvement: int
    generation_time: float

class HealthResponse(BaseModel):
    status: str
    model_name: str
    lora_available: bool

# Initialize FastAPI app
app = FastAPI(
    title="D&D LoRA Model API",
    description="Compare original and D&D LoRA fine-tuned language models",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>D&D LoRA Model API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 30px; border-radius: 10px; text-align: center; }
            .section { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }
            .endpoint { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; 
                       border-left: 4px solid #667eea; }
            .example { background: #e9ecef; padding: 10px; border-radius: 5px; margin: 10px 0; }
            code { background: #f1f3f4; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🐉 D&D LoRA Model API</h1>
            <p>Compare original and D&D knowledge-enhanced language models</p>
        </div>

        <div class="section">
            <h2>🚀 API Endpoints</h2>
            
            <div class="endpoint">
                <h3>POST /generate</h3>
                <p>Compare responses from original and LoRA models</p>
                <div class="example">
                    <strong>Request:</strong><br>
                    <code>{"prompt": "What is a Fireball spell?", "max_length": 150}</code>
                </div>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Check API and model status</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /docs</h3>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
        </div>

        <div class="section">
            <h2>🎯 Quick Test</h2>
            <p>Try these D&D questions:</p>
            <ul>
                <li>"What is a Fireball spell in D&D?"</li>
                <li>"What are the racial traits of an Elf?"</li>
                <li>"What is a Beholder?"</li>
                <li>"What damage does a Longsword deal?"</li>
            </ul>
        </div>

        <div class="section">
            <h2>📚 Documentation</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
            <p>Visit <a href="/redoc">/redoc</a> for alternative documentation</p>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if comparator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return HealthResponse(
        status="healthy",
        model_name=comparator.model_name,
        lora_available=comparator.lora_model is not None
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_comparison(request: GenerateRequest):
    """Generate and compare responses from both models"""
    if comparator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        start_time = time.time()
        
        # Generate comparison
        result = comparator.compare_responses(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        generation_time = time.time() - start_time
        
        return GenerateResponse(
            prompt=result["prompt"],
            original_response=result["original_response"],
            lora_response=result["lora_response"],
            original_dnd_terms=result["original_dnd_terms"],
            lora_dnd_terms=result["lora_dnd_terms"],
            improvement=result["improvement"],
            generation_time=generation_time
        )
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global comparator
    
    logging.info("🚀 Starting D&D LoRA API Server...")
    
    try:
        # Get model configuration from startup args
        model_name = getattr(app.state, 'model_name', 'distilgpt2')
        lora_path = getattr(app.state, 'lora_path', 'models/dnd-lora')
        
        logging.info(f"Loading models: {model_name}")
        logging.info(f"LoRA path: {lora_path}")
        
        comparator = DnDModelComparator(
            model_name=model_name,
            lora_path=lora_path
        )
        
        logging.info("✅ Models loaded successfully!")
        
    except Exception as e:
        logging.error(f"❌ Failed to load models: {e}")
        comparator = None

def main():
    parser = argparse.ArgumentParser(description="D&D LoRA API Server")
    parser.add_argument("--model", default="distilgpt2",
                       help="Base model name")
    parser.add_argument("--lora-path", default="models/dnd-lora",
                       help="Path to LoRA adapter")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Store model config in app state for startup event
    app.state.model_name = args.model
    app.state.lora_path = args.lora_path
    
    # Check if LoRA model exists
    lora_path = Path(args.lora_path)
    if not lora_path.exists():
        logging.warning(f"⚠️  LoRA model not found: {lora_path}")
        logging.info("Server will start but LoRA comparisons will fail")
        logging.info("Train a model first: python train_dnd_lora.py")
    
    logging.info(f"🌐 Starting server on http://{args.host}:{args.port}")
    logging.info("📚 API docs available at: /docs")
    
    # Start server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
