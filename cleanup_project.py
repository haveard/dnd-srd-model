#!/usr/bin/env python3
"""
Project Cleanup Script
======================

Organizes the D&D LoRA project by moving legacy development scripts
to an archive directory and cleaning up the main project structure.

Usage:
    python cleanup_project.py
    python cleanup_project.py --dry-run  # Preview changes only
"""

import argparse
import shutil
from pathlib import Path
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_archive_structure(base_path: Path, dry_run: bool = False):
    """Create archive directory structure"""
    archive_path = base_path / "scripts_archive"
    
    if not dry_run:
        archive_path.mkdir(exist_ok=True)
        (archive_path / "training").mkdir(exist_ok=True)
        (archive_path / "comparison").mkdir(exist_ok=True)
        (archive_path / "demos").mkdir(exist_ok=True)
        (archive_path / "api").mkdir(exist_ok=True)
        (archive_path / "testing").mkdir(exist_ok=True)
        (archive_path / "utilities").mkdir(exist_ok=True)
    
    logging.info(f"{'Would create' if dry_run else 'Created'} archive structure: {archive_path}")
    return archive_path

def get_script_mappings():
    """Define which scripts go where in the archive"""
    return {
        # Training scripts
        "training": [
            "train_distilgpt2_dnd.py",
            "train_lora_pythia.py", 
            "train_lora_pythia_fixed.py",
            "quick_test_training.py",
            "training_monitor.py",
            "training_dashboard.py"
        ],
        # Comparison scripts
        "comparison": [
            "compare_distilgpt2_dnd.py",
            "compare_original_vs_lora.py",
            "compare_lora_vs_gpt4_eval.py"
        ],
        # Demo scripts
        "demos": [
            "demo_comparison.py",
            "demo_dramatic_improvement.py",
            "live_demo.py"
        ],
        # API scripts
        "api": [
            "api_server.py",
            "inference_server_lora.py",
            "test_api_server.py"
        ],
        # Testing scripts
        "testing": [
            "quick_model_test.py",
            "test_training_format.py",
            "test_smaller_models.py",
            "test_more_small_models.py"
        ],
        # Utility scripts
        "utilities": [
            "prepare_dataset.py",
            "pipeline_status.py"
        ]
    }

def move_scripts(scripts_dir: Path, archive_dir: Path, dry_run: bool = False):
    """Move legacy scripts to archive"""
    mappings = get_script_mappings()
    moved_count = 0
    
    for category, script_list in mappings.items():
        category_dir = archive_dir / category
        
        for script_name in script_list:
            source_path = scripts_dir / script_name
            dest_path = category_dir / script_name
            
            if source_path.exists():
                if not dry_run:
                    shutil.move(str(source_path), str(dest_path))
                logging.info(f"{'Would move' if dry_run else 'Moved'} {script_name} -> {category}/")
                moved_count += 1
            else:
                logging.warning(f"Script not found: {script_name}")
    
    return moved_count

def create_archive_readme(archive_dir: Path, dry_run: bool = False):
    """Create README for the archive directory"""
    readme_content = """# Legacy Scripts Archive

This directory contains the original development scripts used during the D&D LoRA project development. These scripts were used for experimentation, testing, and iterative development before the functionality was consolidated into the main refactored scripts.

## Directory Structure

- **training/**: Original training scripts and experiments
- **comparison/**: Various model comparison approaches
- **demos/**: Different demonstration scripts
- **api/**: API server iterations and tests  
- **testing/**: Quick tests and model experiments
- **utilities/**: Data preparation and utility scripts

## Main Project Scripts

The consolidated, production-ready scripts are in the parent directory:

- `train_dnd_lora.py` - Unified training script
- `compare_models.py` - Comprehensive model comparison
- `demo.py` - Simple demonstration
- `api_server.py` - Production API server
- `prepare_dnd_data.py` - Data preparation

## Using Legacy Scripts

These scripts are preserved for reference and may contain useful experimental code. However, they are not maintained and may have dependencies or compatibility issues.

For active development, use the main project scripts instead.
"""
    
    readme_path = archive_dir / "README.md"
    if not dry_run:
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    logging.info(f"{'Would create' if dry_run else 'Created'} archive README")

def main():
    parser = argparse.ArgumentParser(description="Clean up D&D LoRA project structure")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview changes without actually moving files")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Get project paths
    project_root = Path(__file__).parent
    scripts_dir = project_root / "scripts"
    
    if not scripts_dir.exists():
        logging.error(f"Scripts directory not found: {scripts_dir}")
        return 1
    
    logging.info("🧹 Starting D&D LoRA Project Cleanup")
    logging.info(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'EXECUTE (will move files)'}")
    
    try:
        # Create archive structure
        archive_dir = create_archive_structure(project_root, args.dry_run)
        
        # Move legacy scripts
        moved_count = move_scripts(scripts_dir, archive_dir, args.dry_run)
        
        # Create archive documentation
        create_archive_readme(archive_dir, args.dry_run)
        
        # Summary
        logging.info("\n" + "="*60)
        logging.info("🎉 CLEANUP SUMMARY")
        logging.info("="*60)
        logging.info(f"Scripts processed: {moved_count}")
        logging.info(f"Archive location: {archive_dir}")
        
        if args.dry_run:
            logging.info("\n⚠️  This was a DRY RUN - no files were actually moved")
            logging.info("Run without --dry-run to execute the cleanup")
        else:
            logging.info("\n✅ Project cleanup completed successfully!")
            logging.info("Legacy scripts archived and main project structure cleaned")
        
        logging.info("\n📁 Current main scripts:")
        main_scripts = [
            "train_dnd_lora.py",
            "compare_models.py", 
            "demo.py",
            "api_server.py",
            "prepare_dnd_data.py"
        ]
        
        for script in main_scripts:
            script_path = project_root / script
            status = "✅" if script_path.exists() else "❌"
            logging.info(f"  {status} {script}")
        
    except Exception as e:
        logging.error(f"❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
