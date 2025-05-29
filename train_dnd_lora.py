#!/usr/bin/env python3
"""
D&D LoRA Training Script
========================

Train a LoRA adapter to inject D&D 5e SRD knowledge into a language model.

Usage:
    python train_dnd_lora.py --model distilgpt2 --data data/dnd_srd_qa.jsonl
    python train_dnd_lora.py --model EleutherAI/pythia-1.4b --epochs 2
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dnd_lora_core import DnDLoRATrainer, DnDDataProcessor

def main():
    parser = argparse.ArgumentParser(description="Train D&D LoRA model")
    parser.add_argument("--model", default="distilgpt2", 
                       help="Base model name (default: distilgpt2)")
    parser.add_argument("--data", default="data/dnd_srd_qa.jsonl",
                       help="Training data path")
    parser.add_argument("--output", default="models/dnd-lora",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 Starting D&D LoRA Training")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Initialize trainer
        trainer = DnDLoRATrainer(
            model_name=args.model,
            output_dir=args.output
        )
        
        # Setup LoRA
        trainer.setup_lora(rank=args.lora_rank, alpha=args.lora_alpha)
        
        # Prepare dataset
        if not Path(args.data).exists():
            logger.error(f"Data file not found: {args.data}")
            logger.info("Please run: python prepare_dnd_data.py first")
            return 1
            
        dataset = trainer.prepare_dataset(args.data)
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        
        # Train model
        trainer.train(
            dataset=dataset,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Model saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
