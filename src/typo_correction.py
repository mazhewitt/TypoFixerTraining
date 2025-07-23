#!/usr/bin/env python3
"""
BERT-based typo correction library using multi-pass iterative improvement.
Uses masked language modeling to identify and correct suspicious tokens.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from transformers import (
    DistilBertForMaskedLM, 
    DistilBertTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)
# Use Levenshtein distance implementation without external dependency
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
import logging

logger = logging.getLogger(__name__)

class TypoCorrector:
    """BERT-based typo correction using iterative masked language modeling."""
    
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        low_prob_threshold: float = -5.0,
        edit_penalty_lambda: float = 2.0,
        top_k: int = 10,
        max_passes: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize typo corrector.
        
        Args:
            model: Pre-trained masked language model (e.g., DistilBertForMaskedLM)
            tokenizer: Corresponding tokenizer
            low_prob_threshold: Log probability threshold for suspicious tokens
            edit_penalty_lambda: Weight for edit distance penalty
            top_k: Number of top predictions to consider for each mask
            max_passes: Maximum correction passes
            device: Device to run model on (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.low_prob_threshold = low_prob_threshold
        self.edit_penalty_lambda = edit_penalty_lambda
        self.top_k = top_k
        self.max_passes = max_passes
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"TypoCorrector initialized on {self.device}")
        logger.info(f"Thresholds: prob={low_prob_threshold}, edit_penalty={edit_penalty_lambda}")
    
    def get_token_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for each token in the sequence."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            
            # Convert to log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get log prob of actual tokens
            token_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=input_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, seq_len]
            
            return token_log_probs
    
    def identify_suspects(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        token_log_probs: torch.Tensor
    ) -> List[List[int]]:
        """Identify suspicious token spans for correction."""
        suspects = []
        seq_len = input_ids.shape[1]
        
        # Skip [CLS] and [SEP] tokens
        for i in range(1, seq_len - 1):
            token_id = input_ids[0, i].item()
            log_prob = token_log_probs[0, i].item()
            
            # Check if token is suspicious
            is_suspicious = (
                log_prob < self.low_prob_threshold or 
                token_id == self.tokenizer.unk_token_id
            )
            
            if is_suspicious:
                # Add single token suspect
                suspects.append([i])
                
                # Check if next token is also suspicious (for 2-token spans)
                if i + 1 < seq_len - 1:
                    next_log_prob = token_log_probs[0, i + 1].item()
                    if next_log_prob < self.low_prob_threshold:
                        suspects.append([i, i + 1])
        
        return suspects
    
    def get_mask_predictions(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        mask_positions: List[int]
    ) -> List[Tuple[List[int], float]]:
        """Get top-k predictions for masked positions."""
        # Create masked input
        masked_input_ids = input_ids.clone()
        for pos in mask_positions:
            masked_input_ids[0, pos] = self.tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = self.model(input_ids=masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Get predictions for each masked position
            predictions = []
            
            if len(mask_positions) == 1:
                # Single token prediction
                pos = mask_positions[0]
                top_logits, top_indices = torch.topk(logits[pos], self.top_k)
                top_log_probs = torch.log_softmax(logits[pos], dim=-1)[top_indices]
                
                for idx, log_prob in zip(top_indices, top_log_probs):
                    predictions.append(([idx.item()], log_prob.item()))
            
            else:
                # Multi-token prediction using Cartesian product of top-k for each position
                predictions = self._get_cartesian_predictions(
                    input_ids, attention_mask, mask_positions, logits
                )
        
        return predictions
    
    def _get_cartesian_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, 
        mask_positions: List[int],
        masked_logits: torch.Tensor
    ) -> List[Tuple[List[int], float]]:
        """Get Cartesian product of top-k predictions for multi-token spans and re-rank."""
        from itertools import product
        
        # Get top-k candidates for each position
        position_candidates = []
        for pos in mask_positions:
            top_indices = torch.topk(masked_logits[pos], min(self.top_k, 5))[1]  # Limit to 5 for efficiency
            position_candidates.append([(idx.item(), pos) for idx in top_indices])
        
        # Generate Cartesian product of all combinations
        all_combinations = list(product(*position_candidates))
        
        # Limit combinations to prevent exponential explosion
        max_combinations = min(50, len(all_combinations))  # Cap at 50 combinations
        combinations_to_evaluate = all_combinations[:max_combinations]
        
        # Re-rank combinations by evaluating them in full context
        scored_predictions = []
        
        for combination in combinations_to_evaluate:
            # Extract token IDs and positions
            token_ids = [token_id for token_id, pos in combination]
            
            # Create candidate sequence with this combination
            candidate_input_ids = input_ids.clone()
            for i, (token_id, pos) in enumerate(combination):
                candidate_input_ids[0, pos] = token_id
            
            # Score this combination using the model
            with torch.no_grad():
                outputs = self.model(input_ids=candidate_input_ids, attention_mask=attention_mask)
                candidate_logits = outputs.logits[0]  # [seq_len, vocab_size]
                
                # Calculate log probability for each token in the span
                total_log_prob = 0.0
                for i, pos in enumerate(mask_positions):
                    token_id = token_ids[i]
                    token_log_prob = torch.log_softmax(candidate_logits[pos], dim=-1)[token_id]
                    total_log_prob += token_log_prob.item()
                
                # Average log probability across span
                avg_log_prob = total_log_prob / len(mask_positions)
                
                scored_predictions.append((token_ids, avg_log_prob))
        
        # Sort by score (higher is better)
        scored_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top predictions
        return scored_predictions[:self.top_k]
    
    def calculate_edit_distance(self, original_tokens: List[int], candidate_tokens: List[int]) -> int:
        """Calculate edit distance between token sequences."""
        # Convert to strings for edit distance calculation
        original_text = self.tokenizer.decode(original_tokens, skip_special_tokens=True)
        candidate_text = self.tokenizer.decode(candidate_tokens, skip_special_tokens=True)
        
        return levenshtein_distance(original_text, candidate_text)
    
    def correct_typos(self, sentence: str) -> Tuple[str, Dict[str, Any]]:
        """
        Correct typos in a sentence using iterative masked language modeling.
        
        Args:
            sentence: Input sentence with potential typos
            
        Returns:
            Tuple of (corrected_sentence, correction_stats)
        """
        # Tokenize input
        inputs = self.tokenizer(
            sentence, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        original_tokens = input_ids.clone()
        corrections_made = []
        
        for pass_num in range(self.max_passes):
            logger.debug(f"Correction pass {pass_num + 1}/{self.max_passes}")
            
            # Get token probabilities
            token_log_probs = self.get_token_log_probs(input_ids, attention_mask)
            
            # Identify suspicious tokens/spans
            suspects = self.identify_suspects(input_ids, attention_mask, token_log_probs)
            
            if not suspects:
                logger.debug("No suspicious tokens found, stopping")
                break
            
            updated = False
            
            # Try to correct each suspect span
            for span in suspects:
                original_span_tokens = [input_ids[0, i].item() for i in span]
                
                # Get predictions for masked span
                predictions = self.get_mask_predictions(input_ids, attention_mask, span)
                
                best_score = float('-inf')
                best_replacement = original_span_tokens
                
                # Evaluate each prediction
                for candidate_tokens, lm_score in predictions:
                    # Calculate edit penalty
                    edit_dist = self.calculate_edit_distance(original_span_tokens, candidate_tokens)
                    edit_penalty = self.edit_penalty_lambda * edit_dist
                    
                    # Total score = language model score - edit penalty
                    total_score = lm_score - edit_penalty
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_replacement = candidate_tokens
                
                # Apply correction if improvement found
                if best_replacement != original_span_tokens:
                    logger.debug(f"Correcting span {span}: {original_span_tokens} -> {best_replacement}")
                    
                    # Update input_ids
                    for i, token_id in enumerate(best_replacement):
                        if i < len(span):  # Handle different length replacements
                            input_ids[0, span[i]] = token_id
                    
                    corrections_made.append({
                        'pass': pass_num + 1,
                        'span': span,
                        'original': self.tokenizer.decode(original_span_tokens, skip_special_tokens=True),
                        'corrected': self.tokenizer.decode(best_replacement, skip_special_tokens=True),
                        'score': best_score
                    })
                    
                    updated = True
                    break  # Only correct one span per pass to avoid conflicts
            
            if not updated:
                logger.debug("No improvements found, stopping")
                break
        
        # Decode final result
        corrected_sentence = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Prepare stats
        stats = {
            'original': sentence,
            'corrected': corrected_sentence,
            'passes_used': pass_num + 1,
            'corrections_made': corrections_made,
            'total_corrections': len(corrections_made)
        }
        
        return corrected_sentence, stats

def load_pretrained_corrector(
    model_name: str = "distilbert-base-uncased",
    **kwargs
) -> TypoCorrector:
    """Load a pre-trained typo corrector."""
    logger.info(f"Loading pre-trained corrector: {model_name}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForMaskedLM.from_pretrained(model_name)
    
    return TypoCorrector(model, tokenizer, **kwargs)

def correct_sentence(sentence: str, model_name: str = "distilbert-base-uncased") -> str:
    """Simple utility function to correct a sentence."""
    corrector = load_pretrained_corrector(model_name)
    corrected, _ = corrector.correct_typos(sentence)
    return corrected