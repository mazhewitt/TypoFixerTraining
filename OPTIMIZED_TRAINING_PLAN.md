# Optimized Training Plan for 90%+ Accuracy

Based on systematic analysis and targeted improvements to reach 90%+ token accuracy with minimal compute.

## 🎯 **Quick Setup on New Machine**

### **Step 1: Environment Setup**
```bash
# Clone repository
git clone https://github.com/mazhewitt/TypoFixerTraining.git
cd TypoFixerTraining

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-colab.txt  # Optimized for cloud

# Verify GPU  
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# RTX 5090 Fix: Install PyTorch nightly for sm_120 support
# If you get "no kernel image available" error, run:
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Alternative: Force CPU training (slower but works):
# export CUDA_VISIBLE_DEVICES=""
```

### **Step 2: Generate Improved Training Data**
```bash
# Generate 100K examples with systematic improvements
python src/data_generation_v2.py \
  --num_examples 100000 \
  --output data/improved_train_100k.jsonl \
  --curriculum \
  --identity_rate 0.15

# Expected improvements:
# ✅ 25% corruption rate (vs 15% baseline)
# ✅ Character-level noise (adjacent swaps)  
# ✅ Missing/extra space corruptions
# ✅ 15% identity examples (prevents overcorrection)
# ✅ Enhanced keyboard layouts
```

### **Step 3: Optimized Training**
```bash
# Train with improved hyperparameters
python src/train.py \
  --train_file data/improved_train_100k.jsonl \
  --output_dir models/optimized_typo_fixer \
  --num_train_epochs 4 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --weight_decay 0.03 \
  --max_seq_len 96 \
  --save_steps 4000 \
  --eval_steps 2000 \
  --logging_steps 500

# Key optimizations:
# ✅ Increased weight decay: 0.01 → 0.03 (reduces overfitting)
# ✅ Longer sequences: 64 → 96 (more context for multi-word typos)  
# ✅ 4 epochs (sweet spot for this data size)
# ✅ Identity examples train model to "leave correct words alone"
```

### **Step 4: Advanced Validation**
```bash
# Test with improved constrained decoding
python src/validate_mlm.py --model_dir models/optimized_typo_fixer

# Expected improvements:
# ✅ Beam search (top-k=5) vs greedy decoding
# ✅ Plausibility filtering (prevents "beutiful" → "gautiful")
# ✅ 60% similarity threshold for corrections
# ✅ Keyboard-aware correction validation
```

---

## 📊 **Expected Performance Progression**

| Stage | Token Accuracy | Key Improvements |
|-------|---------------|------------------|
| **Baseline (untrained)** | 71% | Raw DistilBERT |
| **V1 (original training)** | 72% | Basic MLM on 100K examples |
| **V2 (improved data)** | 78-80% | Higher corruption + identity examples |
| **V3 (constrained decoding)** | 84-86% | Beam search + plausibility filtering |
| **V4 (curriculum + tweaks)** | **88-90%** | Full optimization stack |

---

## 🔧 **Advanced Optimizations (If Needed)**

### **Curriculum Learning** (Extra 3-4% gain)
```bash
# Generate curriculum dataset
python src/data_generation_v2.py \
  --num_examples 150000 \
  --output data/curriculum_150k.jsonl \
  --curriculum \
  --identity_rate 0.15

# Train in stages:
# Stage 1: Easy corruptions (25K examples, 1 epoch)
# Stage 2: Mixed corruptions (75K examples, 2 epochs)  
# Stage 3: Hard corruptions (50K examples, 1 epoch)
```

### **Hyperparameter Fine-tuning**
```bash
# If performance is still below 88%:
python src/train.py \
  --train_file data/improved_train_100k.jsonl \
  --output_dir models/fine_tuned_typo_fixer \
  --num_train_epochs 6 \
  --per_device_train_batch_size 64 \
  --learning_rate 1.5e-5 \
  --weight_decay 0.05 \
  --max_seq_len 128 \
  --gradient_clip 0.5

# Additional tweaks:
# ✅ Lower learning rate for stability
# ✅ Higher weight decay for regularization  
# ✅ Gradient clipping for training stability
```

### **Bidirectional Repair** (Extra 1-2% gain)
```python
# Run model left→right, then right→left, keep best per-token probability
# Implementation in validate_mlm.py can be extended for this
```

---

## 🎯 **Success Criteria & Benchmarks**

### **Target Metrics:**
- **Token Accuracy: ≥88%** (vs 72% baseline)
- **Exact Matches: ≥60%** (vs 10% baseline)
- **Partial Improvements: ≥80%** (vs 20% baseline)
- **Overcorrection Rate: <5%** (key improvement from constrained decoding)

### **Test Cases:**
```
1. "Thi sis a test sentenc with typos" → "This is a test sentence with typos" 
2. "The quikc brown fox jumps over teh lazy dog" → "The quick brown fox jumps over the lazy dog"
3. "I went too the stor to buy som milk" → "I went to the store to buy some milk"
4. "Ther are many mistaks in this sentance" → "There are many mistakes in this sentence"
5. "Its a beutiful day outsid today" → "It's a beautiful day outside today"
```

**Expected Results with Optimizations:**
- Cases 1-4: **90-95% token accuracy**
- Case 5: **85% accuracy** (apostrophes are harder)
- **No more hallucinations** like "gautiful"

---

## ⏱️ **Time & Cost Estimates**

### **RTX 5090 (Your GPU):**
- **Data generation:** 10 minutes (100K examples)
- **Training:** 15-20 minutes (4 epochs, 100K examples) 
- **Validation:** 1 minute
- **Total time:** ~30 minutes
- **Cost:** ~$10-15 on cloud GPU

### **RTX 5070/5080:**
- **Data generation:** 15 minutes (100K examples)
- **Training:** 20-25 minutes (4 epochs, 100K examples)
- **Validation:** 2 minutes
- **Total time:** ~45 minutes
- **Cost:** ~$15-20 on cloud GPU

### **Comparison vs Original:**
- **Original:** 72% accuracy in 4 minutes
- **Optimized RTX 5090:** 88-90% accuracy in 30 minutes
- **ROI:** 16-18% accuracy gain for 7x time investment

---

## 🚀 **Deployment Pipeline**

```bash
# 1. Train optimized model
python src/train.py [optimized parameters]

# 2. Validate performance  
python src/validate_mlm.py --model_dir models/optimized_typo_fixer

# 3. Upload to Hugging Face Hub
python src/upload_to_hub.py \
  --model_path models/optimized_typo_fixer \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-optimized

# 4. Download for ANE conversion
python src/download_from_hub.py \
  --hub_model_name YOUR_USERNAME/distilbert-typo-fixer-optimized \
  --local_path models/production_model

# 5. Convert to ANE format
python src/apple_ane_conversion.py \
  --input_model models/production_model \
  --coreml_output models/production_model_ANE.mlpackage

# 6. Benchmark ANE performance
python src/ane_vs_cpu_benchmark.py
```

---

## 🎉 **Expected Final Results**

With this optimized approach, you should achieve:

- **88-90% token accuracy** (vs 72% baseline)
- **60-70% exact match rate** (vs 10% baseline)
- **<2% hallucination rate** (vs ~15% baseline)
- **1.3x+ ANE speedup** (maintained from original)
- **Production-ready model** in ~45 minutes

**This systematic approach addresses all the key failure modes identified in the original training while maintaining ANE compatibility!** 🎯