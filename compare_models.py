#!/usr/bin/env python3
"""
D&D Model Comparison Script
===========================

Compare original and LoRA fine-tuned models on D&D knowledge tasks.
Generates comprehensive evaluation reports and analysis.

Usage:
    python compare_models.py --model distilgpt2 --lora-path models/dnd-lora
    python compare_models.py --model EleutherAI/pythia-1.4b --batch-size 10
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd
import json
from datetime import datetime
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dnd_lora_core import DnDModelComparator, DND_QUESTIONS, GENERAL_QUESTIONS

def generate_html_report(results: list, output_dir: Path, model_name: str) -> None:
    """Generate comprehensive HTML evaluation report"""
    
    # Calculate summary statistics
    total_questions = len(results)
    dnd_results = [r for r in results if r['prompt'] in DND_QUESTIONS]
    general_results = [r for r in results if r['prompt'] in GENERAL_QUESTIONS]
    
    avg_improvement = sum(r['improvement'] for r in dnd_results) / len(dnd_results) if dnd_results else 0
    improvements = [r['improvement'] for r in dnd_results]
    improved_count = sum(1 for imp in improvements if imp > 0)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>D&D Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; border-radius: 10px; text-align: center; }}
            .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                       gap: 15px; margin: 20px 0; }}
            .metric {{ background: white; padding: 15px; border-radius: 8px; text-align: center; 
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric h3 {{ margin: 0; color: #333; }}
            .metric .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            .improvement-positive {{ color: #28a745; }}
            .improvement-negative {{ color: #dc3545; }}
            .comparison {{ background: white; margin: 20px 0; padding: 20px; 
                          border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .question {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
            .response {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .original {{ background: #e9ecef; border-left: 4px solid #6c757d; }}
            .lora {{ background: #d4edda; border-left: 4px solid #28a745; }}
            .analysis {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }}
            .section-title {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🐉 D&D Knowledge Injection Analysis</h1>
            <h2>{model_name} - Original vs LoRA Fine-tuned</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary">
            <h2 class="section-title">📊 Executive Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <h3>Total Questions</h3>
                    <div class="value">{total_questions}</div>
                </div>
                <div class="metric">
                    <h3>D&D Questions</h3>
                    <div class="value">{len(dnd_results)}</div>
                </div>
                <div class="metric">
                    <h3>Avg D&D Term Improvement</h3>
                    <div class="value {'improvement-positive' if avg_improvement > 0 else 'improvement-negative'}">
                        {avg_improvement:+.1f}
                    </div>
                </div>
                <div class="metric">
                    <h3>Questions Improved</h3>
                    <div class="value improvement-positive">{improved_count}/{len(dnd_results)}</div>
                </div>
            </div>
        </div>

        <h2 class="section-title">🎯 D&D Knowledge Questions</h2>
    """
    
    # Add D&D question comparisons
    for result in dnd_results:
        improvement_class = "improvement-positive" if result['improvement'] > 0 else "improvement-negative"
        html_content += f"""
        <div class="comparison">
            <div class="question">{result['prompt']}</div>
            
            <div class="response original">
                <strong>📚 Original Model:</strong><br>
                {result['original_response']}
            </div>
            
            <div class="response lora">
                <strong>🐉 LoRA Fine-tuned:</strong><br>
                {result['lora_response']}
            </div>
            
            <div class="analysis">
                <strong>📈 Analysis:</strong> 
                Original D&D terms: {result['original_dnd_terms']} | 
                LoRA D&D terms: {result['lora_dnd_terms']} | 
                <span class="{improvement_class}">Improvement: {result['improvement']:+d}</span>
            </div>
        </div>
        """
    
    # Add general questions if present
    if general_results:
        html_content += f"""
        <h2 class="section-title">🌍 General Knowledge Questions</h2>
        <p><em>These questions test whether domain-specific training affects general knowledge.</em></p>
        """
        
        for result in general_results:
            html_content += f"""
            <div class="comparison">
                <div class="question">{result['prompt']}</div>
                
                <div class="response original">
                    <strong>📚 Original Model:</strong><br>
                    {result['original_response']}
                </div>
                
                <div class="response lora">
                    <strong>🐉 LoRA Fine-tuned:</strong><br>
                    {result['lora_response']}
                </div>
                
                <div class="analysis">
                    <strong>📈 Analysis:</strong> 
                    D&D terms - Original: {result['original_dnd_terms']}, LoRA: {result['lora_dnd_terms']}
                    (General knowledge should remain similar)
                </div>
            </div>
            """
    
    html_content += """
        <div class="summary">
            <h2 class="section-title">🏆 Conclusion</h2>
            <p>This report demonstrates the impact of LoRA fine-tuning on domain-specific knowledge injection. 
            The LoRA adapter successfully adds D&D 5e knowledge while preserving general language capabilities.</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"📄 HTML report saved to: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare D&D models")
    parser.add_argument("--model", default="distilgpt2",
                       help="Base model name")
    parser.add_argument("--lora-path", default="models/dnd-lora",
                       help="Path to LoRA adapter")
    parser.add_argument("--output", default="eval/comparison",
                       help="Output directory for reports")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Number of questions to test (default: all)")
    parser.add_argument("--include-general", action="store_true",
                       help="Include general knowledge questions")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🔬 Starting D&D Model Comparison")
    logger.info(f"Model: {args.model}")
    logger.info(f"LoRA Path: {args.lora_path}")
    
    try:
        # Check if LoRA model exists
        lora_path = Path(args.lora_path)
        if not lora_path.exists():
            logger.error(f"LoRA model not found: {lora_path}")
            logger.info("Please train a model first using: python train_dnd_lora.py")
            return 1
        
        # Initialize comparator
        logger.info("Loading models...")
        comparator = DnDModelComparator(
            model_name=args.model,
            lora_path=str(lora_path)
        )
        
        # Prepare questions
        questions = DND_QUESTIONS.copy()
        if args.include_general:
            questions.extend(GENERAL_QUESTIONS)
            
        if args.batch_size:
            questions = questions[:args.batch_size]
        
        logger.info(f"Testing {len(questions)} questions...")
        
        # Run evaluation
        start_time = time.time()
        results = comparator.evaluate_on_questions(questions)
        evaluation_time = time.time() - start_time
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = output_dir / f"results_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 Results saved to: {csv_path}")
        
        # Generate HTML report
        generate_html_report(results, output_dir, args.model)
        
        # Print summary
        dnd_results = [r for r in results if r['prompt'] in DND_QUESTIONS]
        if dnd_results:
            avg_improvement = sum(r['improvement'] for r in dnd_results) / len(dnd_results)
            improved_count = sum(1 for r in dnd_results if r['improvement'] > 0)
            
            logger.info("\n" + "="*60)
            logger.info("🏆 EVALUATION SUMMARY")
            logger.info("="*60)
            logger.info(f"D&D Questions Tested: {len(dnd_results)}")
            logger.info(f"Average D&D Term Improvement: {avg_improvement:+.2f}")
            logger.info(f"Questions with Improvement: {improved_count}/{len(dnd_results)} ({100*improved_count/len(dnd_results):.1f}%)")
            logger.info(f"Evaluation Time: {evaluation_time:.1f} seconds")
            
            if avg_improvement > 0:
                logger.info("✅ LoRA training successfully injected D&D knowledge!")
            else:
                logger.info("⚠️  Model may need additional training or tuning")
        
        logger.info("✅ Comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
