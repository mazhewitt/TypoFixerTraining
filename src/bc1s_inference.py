#!/usr/bin/env python3
"""
BC1S inference interface for Apple ANE DistilBERT typo correction.
Uses Apple's Core ML model directly with BC1S reshaping in Python.
"""

import argparse
import logging
import numpy as np
import coremltools as ct
from transformers import DistilBertTokenizer
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BC1SInferenceEngine:
    """
    Inference engine using Apple's ANE Core ML model with BC1S reshaping.
    Handles all tensor format conversions in Python for maximum compatibility.
    """
    
    def __init__(self, coreml_model_path: str, tokenizer_path: str, max_sequence_length: int = 128):
        """
        Initialize the BC1S inference engine.
        
        Args:
            coreml_model_path: Path to Apple's ANE Core ML model
            tokenizer_path: Path to tokenizer
            max_sequence_length: Maximum sequence length
        """
        self.max_seq_len = max_sequence_length
        
        # Load Apple's ANE Core ML model
        logger.info(f"Loading Apple ANE Core ML model from {coreml_model_path}")
        self.coreml_model = self._load_apple_coreml_model(coreml_model_path)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        
        logger.info("✅ BC1S inference engine ready!")
    
    def _load_apple_coreml_model(self, model_path: str):
        """Load Apple's ANE-optimized Core ML model."""
        try:
            # Load Apple's official ANE model
            model = ct.models.MLModel(model_path)
            logger.info(f"Loaded Apple ANE Core ML model")
            logger.info(f"Input description: {model.input_description}")
            logger.info(f"Output description: {model.output_description}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Apple ANE model: {e}")
            raise
    
    def _prepare_inputs_for_ane(self, input_text: str):
        """
        Prepare inputs for Apple ANE model using BC1S reshaping approach.
        
        Args:
            input_text: Text with potential typos
            
        Returns:
            Tuple of (input_ids, attention_mask) ready for ANE inference
        """
        # Tokenize input using DistilBERT tokenizer
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True
        )
        
        # Extract tokenized inputs
        input_ids = inputs['input_ids'].astype(np.int32)
        attention_mask = inputs['attention_mask'].astype(np.int32)
        
        logger.debug(f"Tokenized input_ids shape: {input_ids.shape}")
        logger.debug(f"Tokenized attention_mask shape: {attention_mask.shape}")
        
        return input_ids, attention_mask
    
    def _reshape_for_bc1s_inference(self, logits):
        """
        Reshape ANE outputs from BC1S format to standard format.
        
        Args:
            logits: Raw logits from Apple ANE model
            
        Returns:
            Reshaped logits in standard [batch, seq_len, vocab_size] format
        """
        logger.debug(f"Raw ANE output shape: {logits.shape}")
        
        # Apple's ANE model outputs in BC1S format
        # We need to handle various possible output shapes
        if len(logits.shape) == 4:
            # BC1S format: [batch, vocab_size, 1, seq_len] -> [batch, seq_len, vocab_size]
            logits = logits.squeeze(2).transpose(1, 2)
        elif len(logits.shape) == 3:
            # Already in [batch, seq_len, vocab_size] format
            pass
        elif len(logits.shape) == 2:
            # [batch, vocab_size] format - single token prediction
            logits = np.expand_dims(logits, axis=1)  # Add seq_len dimension
        
        logger.debug(f"Reshaped logits shape: {logits.shape}")
        return logits
    
    def predict_corrections(self, input_text: str, return_probabilities: bool = False):
        """
        Predict typo corrections for input text using Apple ANE acceleration.
        
        Args:
            input_text: Text with potential typos
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary containing corrected text and optionally probabilities
        """
        logger.info(f"Predicting corrections for: '{input_text}'")
        
        # Prepare inputs for ANE
        input_ids, attention_mask = self._prepare_inputs_for_ane(input_text)
        
        # Run ANE inference
        try:
            logger.debug("Running Apple ANE inference...")
            start_time = time.time()
            
            prediction = self.coreml_model.predict({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            
            inference_time = time.time() - start_time
            logger.debug(f"ANE inference completed in {inference_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"ANE inference failed: {e}")
            raise
        
        # Extract and reshape logits
        logits_key = list(prediction.keys())[0]  # Get first output key
        raw_logits = prediction[logits_key]
        logits = self._reshape_for_bc1s_inference(raw_logits)
        
        # Get predicted tokens
        predicted_ids = np.argmax(logits, axis=-1)
        
        # Decode predicted text
        if len(predicted_ids.shape) > 1:
            predicted_ids = predicted_ids[0]  # Remove batch dimension
        
        corrected_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        
        result = {
            'input_text': input_text,
            'corrected_text': corrected_text.strip(),
            'inference_time_ms': inference_time * 1000
        }
        
        if return_probabilities:
            # Get top-k probabilities for analysis
            probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            result['probabilities'] = probabilities
        
        logger.info(f"Corrected: '{input_text}' → '{result['corrected_text']}'")
        return result
    
    def batch_predict(self, texts: list, show_progress: bool = True):
        """
        Predict corrections for multiple texts using Apple ANE.
        
        Args:
            texts: List of texts to correct
            show_progress: Whether to show progress
            
        Returns:
            List of correction results
        """
        logger.info(f"Running batch prediction on {len(texts)} texts...")
        
        results = []
        total_time = 0
        
        for i, text in enumerate(texts):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing {i+1}/{len(texts)}...")
            
            result = self.predict_corrections(text)
            results.append(result)
            total_time += result['inference_time_ms']
        
        avg_time = total_time / len(texts)
        logger.info(f"Batch completed: {len(texts)} texts in {total_time:.1f}ms")
        logger.info(f"Average ANE inference time: {avg_time:.2f}ms per text")
        logger.info(f"ANE throughput: {1000/avg_time:.1f} corrections/second")
        
        return results
    
    def benchmark_ane_performance(self, num_runs: int = 50, text: str = None):
        """
        Benchmark Apple ANE performance.
        
        Args:
            num_runs: Number of benchmark runs
            text: Test text (uses default if None)
            
        Returns:
            Performance statistics
        """
        if text is None:
            text = "This is a test sentence with some potential typos to correct using Apple Neural Engine"
        
        logger.info(f"Benchmarking Apple ANE with {num_runs} runs...")
        
        # Warmup runs
        logger.info("Warming up Apple Neural Engine...")
        for _ in range(5):
            self.predict_corrections(text)
        
        # Benchmark runs
        times = []
        logger.info("Running benchmark...")
        
        for i in range(num_runs):
            start_time = time.time()
            result = self.predict_corrections(text)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        throughput = 1000 / avg_time
        
        stats = {
            'num_runs': num_runs,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'throughput_per_sec': throughput,
            'test_text': text
        }
        
        logger.info("🚀 Apple ANE Benchmark Results:")
        logger.info(f"   Average time: {avg_time:.2f}ms")
        logger.info(f"   Min time: {min_time:.2f}ms") 
        logger.info(f"   Max time: {max_time:.2f}ms")
        logger.info(f"   Std deviation: {std_time:.2f}ms")
        logger.info(f"   Throughput: {throughput:.1f} corrections/second")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description="BC1S inference with Apple ANE Core ML")
    parser.add_argument('--coreml_model', type=str, 
                       default='/Users/mazdahewitt/projects/train-typo-fixer/apple-ane-distilbert/DistilBERT_fp16.mlpackage',
                       help='Path to Apple ANE Core ML model')
    parser.add_argument('--tokenizer', type=str,
                       default='/Users/mazdahewitt/projects/train-typo-fixer/apple-ane-distilbert',
                       help='Path to tokenizer')
    parser.add_argument('--text', type=str,
                       help='Text to correct (interactive mode if not provided)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    parser.add_argument('--num_runs', type=int, default=50,
                       help='Number of benchmark runs')
    parser.add_argument('--max_seq_len', type=int, default=128,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    try:
        # Initialize BC1S inference engine
        logger.info("🍎 Initializing Apple ANE BC1S inference engine...")
        engine = BC1SInferenceEngine(
            coreml_model_path=args.coreml_model,
            tokenizer_path=args.tokenizer,
            max_sequence_length=args.max_seq_len
        )
        
        if args.benchmark:
            # Run benchmark
            engine.benchmark_ane_performance(num_runs=args.num_runs, text=args.text)
        elif args.text:
            # Single prediction
            result = engine.predict_corrections(args.text, return_probabilities=True)
            print(f"Input: {result['input_text']}")
            print(f"Corrected: {result['corrected_text']}")
            print(f"Time: {result['inference_time_ms']:.2f}ms")
        else:
            # Interactive mode
            logger.info("🎯 Interactive typo correction mode (type 'quit' to exit)")
            
            test_examples = [
                "Thi sis a test sentenc with typos",
                "The quikc brown fox jumps over teh lazy dog",
                "I went too the stor to buy som milk",
                "Ther are many mistaks in this sentance",
                "Its a beutiful day outsid today"
            ]
            
            logger.info("Example test cases:")
            for i, example in enumerate(test_examples, 1):
                result = engine.predict_corrections(example)
                print(f"{i}. '{example}' → '{result['corrected_text']}' ({result['inference_time_ms']:.1f}ms)")
            
            print("\nEnter your own text to correct:")
            while True:
                try:
                    user_text = input("\n> ").strip()
                    if user_text.lower() in ['quit', 'exit', 'q']:
                        break
                    if user_text:
                        result = engine.predict_corrections(user_text)
                        print(f"Corrected: {result['corrected_text']} ({result['inference_time_ms']:.1f}ms)")
                except KeyboardInterrupt:
                    break
        
        logger.info("✅ Apple ANE BC1S inference completed!")
        
    except Exception as e:
        logger.error(f"❌ BC1S inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()