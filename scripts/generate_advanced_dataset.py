#!/usr/bin/env python3
"""
Master Script: Generate Advanced Training Dataset

Orchestrates the complete pipeline for generating high-quality typo correction
training data using all the advanced components.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

from source_text_diversifier import SourceTextDiversifier
from advanced_training_data_generator import AdvancedTrainingDataGenerator
from quality_validator import QualityValidator
from training_data_analyzer import TrainingDataAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Generate advanced typo correction training dataset")
    parser.add_argument('--target-size', type=int, default=100000,
                       help='Target number of training examples')
    parser.add_argument('--source-sentences', type=int, default=50000,
                       help='Number of source sentences to collect')
    parser.add_argument('--output-dir', default='data/advanced_dataset',
                       help='Output directory')
    parser.add_argument('--skip-source-collection', action='store_true',
                       help='Skip source sentence collection (use existing file)')
    parser.add_argument('--source-file', default='data/diverse_source_sentences.json',
                       help='Existing source sentences file')
    parser.add_argument('--validate-quality', action='store_true',
                       help='Run quality validation')
    parser.add_argument('--create-analysis', action='store_true',
                       help='Create comprehensive analysis and plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 ADVANCED TRAINING DATA GENERATION PIPELINE")
    print(f"🎯 Target: {args.target_size:,} training examples")
    print(f"📁 Output: {output_dir}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Collect diverse source sentences
    source_sentences_file = output_dir / "source_sentences.json"
    
    if not args.skip_source_collection:
        print(f"\n🔹 Step 1: Collecting {args.source_sentences:,} diverse source sentences")
        print("-" * 50)
        
        diversifier = SourceTextDiversifier()
        source_sentences = diversifier.collect_diverse_sentences(args.source_sentences)
        diversifier.save_collected_sentences(source_sentences, str(source_sentences_file))
        
    else:
        print(f"\n🔹 Step 1: Loading existing source sentences from {args.source_file}")
        print("-" * 50)
        
        if Path(args.source_file).exists():
            with open(args.source_file, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            source_sentences = source_data.get("sentences_by_domain", {})
            
            # Save to output directory for consistency
            with open(source_sentences_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": source_data.get("metadata", {}),
                    "sentences_by_domain": source_sentences
                }, f, indent=2, ensure_ascii=False)
            
            total_sentences = sum(len(sentences) for sentences in source_sentences.values())
            print(f"✅ Loaded {total_sentences:,} source sentences from {len(source_sentences)} domains")
        else:
            print(f"❌ Source file not found: {args.source_file}")
            return
    
    # Step 2: Generate advanced training dataset
    print(f"\n🔹 Step 2: Generating {args.target_size:,} advanced training examples")
    print("-" * 50)
    
    generator = AdvancedTrainingDataGenerator()
    examples = generator.generate_training_dataset(source_sentences, args.target_size)
    
    # Save training dataset
    training_file = output_dir / "training_dataset.jsonl"
    stats = generator.save_training_dataset(examples, str(training_file))
    
    print(f"\n📊 Generation Statistics:")
    print(f"  Examples Generated: {len(examples):,}")
    print(f"  Success Rate: {len(examples)/args.target_size:.1%}")
    print(f"  Generation Methods: {len(stats['generation_stats'])}")
    print(f"  Complexity Distribution:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"    {complexity:8}: {count:6,} ({count/len(examples):5.1%})")
    
    # Step 3: Quality validation (if requested)
    validation_results = None
    if args.validate_quality:
        print(f"\n🔹 Step 3: Quality Validation")
        print("-" * 50)
        
        validator = QualityValidator()
        validation_results = validator.validate_dataset_quality(examples)
        
        # Save validation report
        validation_file = output_dir / "validation_report.json"
        validator.save_validation_report(validation_results, str(validation_file))
        
        print(f"\n📋 Validation Results:")
        print(f"  Total Issues: {validation_results['validation_stats']['total_issues']:,}")
        print(f"  Failed Examples: {len(validation_results['failed_examples']):,}")
        print(f"  Warning Examples: {len(validation_results['warning_examples']):,}")
        print(f"  Clean Examples: {len(examples) - len(validation_results['failed_examples']) - len(validation_results['warning_examples']):,}")
        
        if validation_results['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(validation_results['recommendations'], 1):
                print(f"  {i}. {rec}")
    
    # Step 4: Comprehensive analysis (if requested)
    analysis_results = None
    if args.create_analysis:
        print(f"\n🔹 Step 4: Comprehensive Analysis")
        print("-" * 50)
        
        analyzer = TrainingDataAnalyzer()
        analyzer.examples = [ex.__dict__ if hasattr(ex, '__dict__') else ex for ex in examples]
        analysis_results = analyzer.generate_analysis_report()
        
        # Save analysis report
        analysis_file = output_dir / "analysis_report.json"
        analyzer.save_analysis_report(str(analysis_file))
        
        # Create visualizations
        plots_dir = output_dir / "plots"
        analyzer.create_visualizations(str(plots_dir))
        
        # Print summary
        analyzer.print_summary_report()
    
    # Step 5: Generate final summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"📁 Output Directory: {output_dir}")
    print(f"📊 Training Dataset: {training_file}")
    print(f"📈 Examples Generated: {len(examples):,}")
    
    # Create summary report
    summary = {
        "pipeline_config": {
            "target_size": args.target_size,
            "source_sentences": args.source_sentences,
            "seed": args.seed,
            "generation_time_minutes": total_time / 60
        },
        "results": {
            "examples_generated": len(examples),
            "success_rate": len(examples) / args.target_size,
            "output_files": {
                "training_dataset": str(training_file),
                "source_sentences": str(source_sentences_file)
            }
        },
        "statistics": stats
    }
    
    if validation_results:
        summary["validation"] = {
            "total_issues": validation_results['validation_stats']['total_issues'],
            "failed_examples": len(validation_results['failed_examples']),
            "quality_score": 1 - (len(validation_results['failed_examples']) / len(examples))
        }
        summary["results"]["output_files"]["validation_report"] = str(output_dir / "validation_report.json")
    
    if analysis_results:
        summary["analysis"] = {
            "balance_scores": analysis_results["quality_metrics"]["balance_scores"],
            "quality_flags": analysis_results["quality_metrics"]["quality_flags"]
        }
        summary["results"]["output_files"]["analysis_report"] = str(output_dir / "analysis_report.json")
        summary["results"]["output_files"]["plots_directory"] = str(output_dir / "plots")
    
    # Save summary
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📋 Summary Report: {summary_file}")
    
    # Final recommendations
    print(f"\n🎯 NEXT STEPS:")
    if validation_results and len(validation_results['failed_examples']) > len(examples) * 0.05:
        print(f"  ⚠️  Consider regenerating to reduce failed examples ({len(validation_results['failed_examples'])} failures)")
    else:
        print(f"  ✅ Dataset quality looks good for training!")
    
    print(f"  🚂 Ready to train: python scripts/train_byt5_nocallback.py --train-file {training_file}")
    print(f"  📊 Analyze results: python scripts/evaluate_all_checkpoints.py --model-path models/[your-model]")
    
    print(f"\n📁 All files saved to: {output_dir}/")
    print(f"   • training_dataset.jsonl - Main training file")
    print(f"   • source_sentences.json - Source sentence collection")
    print(f"   • generation_summary.json - Complete pipeline summary")
    
    if args.validate_quality:
        print(f"   • validation_report.json - Quality validation results")
    
    if args.create_analysis:
        print(f"   • analysis_report.json - Comprehensive dataset analysis")
        print(f"   • plots/ - Visualization charts")


if __name__ == "__main__":
    main()