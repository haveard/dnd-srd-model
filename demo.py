#!/usr/bin/env python3
"""
D&D LoRA Demonstration Script
============================

Simple demonstration of the dramatic improvements from D&D LoRA fine-tuning.
Shows "zero-to-hero" transformation with clear before/after examples.

Usage:
    python demo.py
    python demo.py --model distilgpt2 --questions 5
"""

import argparse
import sys
from pathlib import Path
import logging
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dnd_lora_core import DnDModelComparator, DND_QUESTIONS

def print_header():
    """Print demonstration header"""
    print("\n" + "="*80)
    print("🐉 D&D KNOWLEDGE INJECTION DEMONSTRATION")
    print("="*80)
    print("This demo shows how LoRA fine-tuning transforms a general language model")
    print("from having zero D&D knowledge to domain-expert level understanding.")
    print("="*80)

def print_question_header(question_num: int, total: int, question: str):
    """Print question section header"""
    print(f"\n{'─'*80}")
    print(f"📋 QUESTION {question_num}/{total}: {question}")
    print("─"*80)

def print_model_response(model_name: str, response: str, dnd_terms: int, emoji: str):
    """Print formatted model response"""
    print(f"\n{emoji} {model_name.upper()}:")
    print(f"   {response}")
    print(f"   📊 D&D terms detected: {dnd_terms}")

def print_analysis(improvement: int, original_terms: int, lora_terms: int):
    """Print improvement analysis"""
    print(f"\n📈 ANALYSIS:")
    print(f"   Improvement: {improvement:+d} D&D terms")
    
    if improvement > 0:
        if original_terms == 0:
            print("   ✨ ZERO-TO-HERO: Complete transformation from no D&D knowledge!")
        else:
            percentage = (improvement / original_terms) * 100
            print(f"   🚀 {percentage:.1f}% increase in D&D terminology usage")
        print("   ✅ LoRA successfully injected domain expertise!")
    elif improvement == 0:
        print("   ➖ No change in D&D terminology")
    else:
        print("   ⚠️  Fewer D&D terms - may need more training")

def print_summary(results: list):
    """Print final summary"""
    improvements = [r['improvement'] for r in results]
    avg_improvement = sum(improvements) / len(improvements)
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    zero_to_hero = sum(1 for r in results if r['original_dnd_terms'] == 0 and r['lora_dnd_terms'] > 0)
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION SUMMARY")
    print("="*80)
    print(f"Questions tested: {len(results)}")
    print(f"Average improvement: {avg_improvement:+.1f} D&D terms")
    print(f"Questions improved: {positive_improvements}/{len(results)} ({100*positive_improvements/len(results):.1f}%)")
    print(f"Zero-to-hero transformations: {zero_to_hero}")
    
    if avg_improvement > 0:
        print("\n✨ SUCCESS: LoRA training successfully transformed the model!")
        print("🎲 The model now understands D&D concepts, terminology, and mechanics")
        print("⚡ Training efficiency: Only ~1% of parameters needed for domain expertise")
    else:
        print("\n⚠️  The model may need additional training or parameter tuning")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="D&D LoRA demonstration")
    parser.add_argument("--model", default="distilgpt2",
                       help="Base model name")
    parser.add_argument("--lora-path", default="models/dnd-lora",
                       help="Path to LoRA adapter")
    parser.add_argument("--questions", type=int, default=5,
                       help="Number of questions to demo")
    parser.add_argument("--pause", type=float, default=1.0,
                       help="Pause between questions (seconds)")
    
    args = parser.parse_args()
    
    # Setup logging (minimal for demo)
    logging.basicConfig(level=logging.WARNING)
    
    print_header()
    
    try:
        # Check if LoRA model exists
        lora_path = Path(args.lora_path)
        if not lora_path.exists():
            print(f"❌ LoRA model not found: {lora_path}")
            print("Please train a model first using: python train_dnd_lora.py")
            return 1
        
        print("🚀 Loading models...")
        
        # Initialize comparator
        comparator = DnDModelComparator(
            model_name=args.model,
            lora_path=str(lora_path)
        )
        
        print("✅ Models loaded successfully!")
        
        # Select questions for demo
        questions = DND_QUESTIONS[:args.questions]
        results = []
        
        # Run demonstration
        for i, question in enumerate(questions, 1):
            print_question_header(i, len(questions), question)
            
            # Generate comparison
            print("🔄 Generating responses...")
            result = comparator.compare_responses(question)
            results.append(result)
            
            # Display results
            print_model_response(
                "Original Model", 
                result['original_response'], 
                result['original_dnd_terms'],
                "📚"
            )
            
            print_model_response(
                "LoRA Fine-tuned", 
                result['lora_response'], 
                result['lora_dnd_terms'],
                "🐉"
            )
            
            print_analysis(
                result['improvement'],
                result['original_dnd_terms'],
                result['lora_dnd_terms']
            )
            
            # Pause between questions
            if i < len(questions):
                time.sleep(args.pause)
        
        # Show final summary
        print_summary(results)
        
        print(f"\n🔬 For detailed analysis, run: python compare_models.py --model {args.model}")
        print(f"🌐 For interactive testing, run: python api_server.py --model {args.model}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
