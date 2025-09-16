#!/bin/bash
"""
Production Dataset Generation Script

Generates a comprehensive 100K training dataset using the advanced pipeline
with real datasets, quality validation, and comprehensive analysis.
"""

echo "🚀 GENERATING PRODUCTION-READY TRAINING DATASET"
echo "================================================"

# Set parameters
TARGET_SIZE=100000
SOURCE_SENTENCES=50000
OUTPUT_DIR="data/production_advanced_dataset"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "📊 Configuration:"
echo "  Target Examples: ${TARGET_SIZE}"
echo "  Source Sentences: ${SOURCE_SENTENCES}" 
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Timestamp: ${TIMESTAMP}"
echo ""

# Create timestamped backup directory name
BACKUP_DIR="${OUTPUT_DIR}_${TIMESTAMP}"

echo "🔄 Running advanced dataset generation pipeline..."
python scripts/generate_advanced_dataset.py \
    --target-size ${TARGET_SIZE} \
    --source-sentences ${SOURCE_SENTENCES} \
    --validate-quality \
    --create-analysis \
    --output-dir ${OUTPUT_DIR} \
    --seed 42

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dataset generation completed successfully!"
    
    # Create backup with timestamp
    if [ -d "${OUTPUT_DIR}" ]; then
        cp -r "${OUTPUT_DIR}" "${BACKUP_DIR}"
        echo "💾 Backup created: ${BACKUP_DIR}"
    fi
    
    # Show final statistics
    echo ""
    echo "📈 FINAL DATASET STATISTICS:"
    echo "=============================="
    
    # Count actual examples generated
    if [ -f "${OUTPUT_DIR}/training_dataset.jsonl" ]; then
        ACTUAL_COUNT=$(wc -l < "${OUTPUT_DIR}/training_dataset.jsonl")
        echo "  Examples Generated: ${ACTUAL_COUNT}"
        echo "  Success Rate: $(python3 -c "print(f'{${ACTUAL_COUNT}/${TARGET_SIZE}:.1%}')")"
    fi
    
    # Show file sizes
    echo "  File Sizes:"
    ls -lh "${OUTPUT_DIR}"/*.json* 2>/dev/null | awk '{print "    " $9 ": " $5}'
    
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "=============="
    echo "1. Review quality validation report:"
    echo "   cat ${OUTPUT_DIR}/validation_report.json"
    echo ""
    echo "2. Train ByT5 model:"
    echo "   python scripts/train_byt5_nocallback.py \\"
    echo "     --train-file ${OUTPUT_DIR}/training_dataset.jsonl \\"
    echo "     --output-dir models/byt5-advanced-typo-fixer-${TIMESTAMP} \\"
    echo "     --num-epochs 3 \\"
    echo "     --batch-size 8 \\"
    echo "     --learning-rate 3e-5"
    echo ""
    echo "3. Evaluate systematically:"
    echo "   python scripts/evaluate_all_checkpoints.py \\"
    echo "     --model-path models/byt5-advanced-typo-fixer-${TIMESTAMP}"
    echo ""
    echo "4. Upload best model:"
    echo "   python scripts/upload_and_replace_model.py \\"
    echo "     --model-path models/byt5-advanced-typo-fixer-${TIMESTAMP}/checkpoint-[BEST] \\"
    echo "     --hub-model-id mazhewitt/byt5-advanced-typo-fixer"
    echo ""
    echo "📁 All files saved to: ${OUTPUT_DIR}/"
    echo "💾 Backup available at: ${BACKUP_DIR}/"
    
else
    echo "❌ Dataset generation failed!"
    exit 1
fi