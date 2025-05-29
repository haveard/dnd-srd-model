#!/usr/bin/env python3
"""
D&D SRD Data Preparation Script
===============================

Converts D&D 5e SRD JSON data into training-ready Q&A format.
Reads from local ./data/raw directory by default.

Usage:
    python prepare_dnd_data.py
    python prepare_dnd_data.py --raw-data-path custom/raw/path --output data/custom_qa.jsonl
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dnd_lora_core import DnDDataProcessor

def main():
    parser = argparse.ArgumentParser(description="Prepare D&D training data")
    parser.add_argument("--raw-data-path", default=None,
                       help="Path to raw data directory (defaults to ./data/raw)")
    parser.add_argument("--output", default="data/dnd_srd_qa.jsonl",
                       help="Output JSONL file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("📚 Preparing D&D SRD Training Data")
    
    try:
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load SRD data from local raw directory (or specified path)
        if args.raw_data_path:
            raw_data_path = Path(args.raw_data_path)
            if not raw_data_path.exists():
                logger.error(f"Raw data path not found: {raw_data_path}")
                return 1
            logger.info(f"Loading SRD data from: {raw_data_path}")
            srd_data = DnDDataProcessor.load_srd_data(str(raw_data_path))
        else:
            logger.info("Loading SRD data from local ./data/raw directory")
            srd_data = DnDDataProcessor.load_srd_data()  # Use default path
        
        if not srd_data:
            logger.error("No SRD data loaded!")
            return 1
        
        # Create Q&A pairs
        logger.info("Creating question-answer pairs...")
        qa_pairs = DnDDataProcessor.create_qa_pairs(srd_data)
        
        if not qa_pairs:
            logger.error("No Q&A pairs created!")
            return 1
        
        # Save dataset
        DnDDataProcessor.save_dataset(qa_pairs, str(output_path))
        
        logger.info("✅ Data preparation completed successfully!")
        logger.info(f"Created {len(qa_pairs)} training examples")
        logger.info(f"Dataset saved to: {output_path}")
        
        # Show sample
        logger.info("\nSample Q&A pair:")
        sample = qa_pairs[0]
        logger.info(f"Q: {sample['prompt']}")
        logger.info(f"A: {sample['completion'][:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
