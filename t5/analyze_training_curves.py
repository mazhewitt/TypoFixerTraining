#!/usr/bin/env python3
"""
Analyze T5-small training curves to determine if it has reached fine-tuning limits.
"""

import json
import os
from pathlib import Path

def analyze_t5_small_training():
    """Analyze T5-small training curves and determine if more training is needed."""
    
    print("📊 T5-small Training Curve Analysis")
    print("=" * 50)
    
    # Check training info
    training_info_path = "models/t5-small-typo-fixer/training_info.json"
    if Path(training_info_path).exists():
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
        
        print("🔍 Training Configuration:")
        print(f"   Epochs completed: {training_info.get('epochs', 0)}")
        print(f"   Training examples: {training_info.get('training_examples', 0):,}")
        print(f"   Learning rate: {training_info.get('learning_rate', 0)}")
        print(f"   Model size: {training_info.get('total_parameters', 0):,} parameters")
        print(f"   Training time: {training_info.get('training_time_minutes', 0):.1f} minutes")
    
    # Analyze training loss progression from logs
    print("\n📈 Training Loss Analysis:")
    
    # From our observed training metrics:
    training_metrics = [
        {"step": 50, "loss": 1.58, "epoch": 0.25},
        {"step": 100, "loss": 0.79, "epoch": 0.51}, 
        {"step": 150, "loss": 0.61, "epoch": 0.76},
        {"step": 200, "loss": 0.56, "epoch": 1.02},
        {"step": 250, "loss": 0.50, "epoch": 1.27},
        {"step": 300, "loss": 0.47, "epoch": 1.52},
        {"step": 350, "loss": 0.45, "epoch": 1.78},
        {"step": 400, "loss": 0.42, "epoch": 2.03},
        {"step": 450, "loss": 0.40, "epoch": 2.28},
        {"step": 500, "loss": 0.40, "epoch": 2.54},  # Final recorded
    ]
    
    eval_metrics = [
        {"step": 250, "eval_loss": 0.408, "epoch": 1.27},
        {"step": 500, "eval_loss": 0.353, "epoch": 2.54},
        {"step": 750, "eval_loss": 0.347, "epoch": 3.81}, # Final
    ]
    
    print("   Training Loss Progression:")
    for metric in training_metrics[-5:]:  # Last 5 points
        print(f"     Step {metric['step']:3d} (Epoch {metric['epoch']:.1f}): {metric['loss']:.3f}")
    
    print("\n   Validation Loss Progression:")
    for metric in eval_metrics:
        print(f"     Step {metric['step']:3d} (Epoch {metric['epoch']:.1f}): {metric['eval_loss']:.3f}")
    
    # Analyze convergence patterns
    print("\n🔍 Convergence Analysis:")
    
    # Training loss reduction rates
    initial_loss = 1.58
    final_loss = 0.40
    total_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    # Loss reduction in different phases
    early_reduction = (1.58 - 0.50) / 1.58 * 100  # First 250 steps
    late_reduction = (0.50 - 0.40) / 0.50 * 100   # Steps 250-500
    
    print(f"   Total training loss reduction: {total_reduction:.1f}% ({initial_loss:.2f} → {final_loss:.2f})")
    print(f"   Early phase reduction (0-250 steps): {early_reduction:.1f}%")
    print(f"   Late phase reduction (250-500 steps): {late_reduction:.1f}%")
    
    # Validation loss analysis
    val_reduction = (eval_metrics[0]['eval_loss'] - eval_metrics[-1]['eval_loss']) / eval_metrics[0]['eval_loss'] * 100
    print(f"   Validation loss reduction: {val_reduction:.1f}% ({eval_metrics[0]['eval_loss']:.3f} → {eval_metrics[-1]['eval_loss']:.3f})")
    
    # Recent validation improvement (step 500 to 750)
    recent_val_improvement = (eval_metrics[1]['eval_loss'] - eval_metrics[2]['eval_loss']) / eval_metrics[1]['eval_loss'] * 100
    print(f"   Recent validation improvement (500→750): {recent_val_improvement:.1f}%")
    
    print("\n🧠 Training State Assessment:")
    
    # Signs of convergence vs room for improvement
    convergence_signs = []
    improvement_signs = []
    
    if late_reduction < 5:
        convergence_signs.append("Training loss plateauing (<5% recent reduction)")
    else:
        improvement_signs.append(f"Training loss still declining ({late_reduction:.1f}% recent reduction)")
    
    if recent_val_improvement < 2:
        convergence_signs.append("Validation loss plateauing (<2% improvement)")
    else:
        improvement_signs.append(f"Validation improving ({recent_val_improvement:.1f}% recent improvement)")
    
    # Gap between training and validation
    final_train_loss = 0.40
    final_val_loss = 0.347
    if final_val_loss < final_train_loss:
        improvement_signs.append("Validation < training loss (no overfitting)")
    
    print("   🟡 Signs of convergence:")
    for sign in convergence_signs:
        print(f"     - {sign}")
    
    print("   🟢 Signs of potential improvement:")
    for sign in improvement_signs:
        print(f"     - {sign}")
    
    print("\n💡 Recommendations:")
    
    # Performance analysis
    current_accuracy = 40.0  # From evaluation
    
    if len(improvement_signs) >= len(convergence_signs) and current_accuracy < 60:
        recommendation = "🚀 CONTINUE TRAINING"
        print(f"   {recommendation}")
        print("   Reasons:")
        print("   - Model still showing improvement trends")
        print("   - Current accuracy (40%) has room for improvement")
        print("   - No strong overfitting signs")
        print("   - Validation loss still decreasing")
        print()
        print("   Suggested approach:")
        print("   - Train for 2-4 more epochs")
        print("   - Monitor for overfitting (train loss << val loss)")
        print("   - Reduce learning rate if loss plateaus")
        print("   - Stop if accuracy stops improving")
        
    elif current_accuracy > 75:
        recommendation = "✅ MODEL ADEQUATE"
        print(f"   {recommendation}")
        print("   - Current performance is good for most applications")
        print("   - Additional training may yield diminishing returns")
        
    else:
        recommendation = "🤔 MIXED SIGNALS"
        print(f"   {recommendation}")
        print("   - Some convergence signs but still room for improvement")
        print("   - Consider training 1-2 more epochs with careful monitoring")
    
    print("\n📋 Training Decision Framework:")
    print("   Continue training if:")
    print("   ✓ Validation loss decreasing (>1% per epoch)")
    print("   ✓ No overfitting (val loss ≈ train loss)")  
    print("   ✓ Accuracy < target threshold (e.g., 70%)")
    print("   ✓ Loss reduction > 2% per epoch")
    print()
    print("   Stop training if:")
    print("   ❌ Validation loss increasing (overfitting)")
    print("   ❌ Loss plateau for 2+ epochs (<1% change)")
    print("   ❌ Accuracy target reached")
    print("   ❌ Diminishing returns on validation set")
    
    # Data analysis
    print("\n📚 Dataset Analysis:")
    print("   Current dataset: 6,999 examples")
    print("   - May benefit from more diverse training data")
    print("   - Consider data augmentation for edge cases")
    print("   - Current balanced approach (50/50 punctuation) is good")
    
    return {
        'current_accuracy': current_accuracy,
        'training_convergence': len(convergence_signs),
        'improvement_potential': len(improvement_signs), 
        'recommendation': recommendation,
        'continue_training': len(improvement_signs) >= len(convergence_signs) and current_accuracy < 60
    }

if __name__ == "__main__":
    analysis = analyze_t5_small_training()
    
    print(f"\n🎯 FINAL DECISION: {analysis['recommendation']}")
    if analysis['continue_training']:
        print("   → Proceed with extended training")
    else:
        print("   → Current model performance is sufficient or at limits")