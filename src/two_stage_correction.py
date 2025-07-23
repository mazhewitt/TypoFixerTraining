#!/usr/bin/env python3
"""
Two-stage typo correction: 
1. Detection model identifies misspelled tokens
2. Correction model fixes the identified tokens
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM, 
    DistilBertTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel
)
import logging

logger = logging.getLogger(__name__)

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

class TypoDetector:
    """Model that identifies which tokens are likely misspelled."""
    
    def __init__(self, detection_model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize typo detector.
        
        Args:
            detection_model_path: Path to fine-tuned detection model (if None, uses heuristics)
            device: Device to run on
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        if detection_model_path:
            # Load fine-tuned detection model
            self.tokenizer = DistilBertTokenizer.from_pretrained(detection_model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(detection_model_path)
            self.model.to(self.device)
            self.model.eval()
            self.use_model = True
            logger.info(f"Loaded detection model from {detection_model_path}")
        else:
            # Use base model with heuristics
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
            self.model.to(self.device)
            self.model.eval()
            self.use_model = False
            logger.info("Using heuristic-based typo detection")
    
    def detect_typos(self, text: str, threshold: float = -4.0) -> List[Tuple[int, str, float]]:
        """
        Detect typos in text.
        
        Returns:
            List of (token_position, token_text, confidence_score) for suspected typos
        """
        if self.use_model:
            return self._detect_with_model(text)
        else:
            return self._detect_with_heuristics(text, threshold)
    
    def _detect_with_heuristics(self, text: str, threshold: float) -> List[Tuple[int, str, float]]:
        """Detect typos using token probability heuristics."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        suspected_typos = []
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Get log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=-1, index=input_ids[0].unsqueeze(-1)).squeeze(-1)
            
            # Check each token
            for pos in range(1, input_ids.shape[1] - 1):  # Skip CLS and SEP
                if attention_mask[0, pos] == 0:  # Skip padding
                    break
                
                token_id = input_ids[0, pos].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                log_prob = token_log_probs[pos].item()
                
                # Skip very short tokens and subword pieces
                if len(token_text) <= 1 or token_text.startswith('##'):
                    continue
                
                # Check if probability is suspiciously low
                if log_prob < threshold:
                    suspected_typos.append((pos, token_text, -log_prob))  # Convert to positive confidence
        
        return suspected_typos
    
    def _detect_with_model(self, text: str) -> List[Tuple[int, str, float]]:
        """Detect typos using fine-tuned detection model."""
        # TODO: Implement when we have a trained detection model
        # For now, fall back to heuristics
        return self._detect_with_heuristics(text, -4.0)

class TypoCorrector:
    """Model that corrects identified misspelled tokens."""
    
    def __init__(self, correction_model_path: str, device: Optional[str] = None):
        """
        Initialize typo corrector.
        
        Args:
            correction_model_path: Path to fine-tuned correction model
            device: Device to run on
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(correction_model_path)
        self.model = DistilBertForMaskedLM.from_pretrained(correction_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded correction model from {correction_model_path}")
    
    def correct_token(self, text: str, token_position: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Correct a specific token at the given position.
        
        Args:
            text: Original text
            token_position: Position of token to correct
            top_k: Number of correction candidates to return
            
        Returns:
            List of (corrected_token, confidence_score) candidates
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Create masked version
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, token_position] = self.tokenizer.mask_token_id
        
        candidates = []
        
        with torch.no_grad():
            outputs = self.model(input_ids=masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, token_position]  # Logits for the masked position
            
            # Get top-k candidates
            top_logits, top_indices = torch.topk(logits, top_k)
            top_probs = torch.softmax(top_logits, dim=-1)
            
            for idx, prob in zip(top_indices, top_probs):
                token = self.tokenizer.decode([idx.item()], skip_special_tokens=True).strip()
                if len(token) > 0:  # Skip empty tokens
                    candidates.append((token, prob.item()))
        
        return candidates

class TwoStageTypoCorrector:
    """Complete two-stage typo correction pipeline."""
    
    def __init__(
        self, 
        correction_model_path: str,
        detection_model_path: Optional[str] = None,
        detection_threshold: float = -3.5,
        edit_penalty_lambda: float = 1.0,
        max_corrections: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize two-stage corrector.
        
        Args:
            correction_model_path: Path to correction model
            detection_model_path: Path to detection model (None for heuristics)
            detection_threshold: Threshold for typo detection
            edit_penalty_lambda: Penalty for edit distance
            max_corrections: Maximum corrections per sentence
            device: Device to run on
        """
        self.detector = TypoDetector(detection_model_path, device)
        self.corrector = TypoCorrector(correction_model_path, device)
        
        self.detection_threshold = detection_threshold
        self.edit_penalty_lambda = edit_penalty_lambda
        self.max_corrections = max_corrections
        
        logger.info("TwoStageTypoCorrector initialized")
    
    def correct_text(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Correct typos in text using two-stage approach.
        
        Returns:
            Tuple of (corrected_text, correction_stats)
        """
        # Stage 1: Detect typos
        suspected_typos = self.detector.detect_typos(text, self.detection_threshold)
        
        if not suspected_typos:
            return text, {
                'original': text,
                'corrected': text,
                'typos_detected': 0,
                'corrections_made': [],
                'total_corrections': 0
            }
        
        # Sort by confidence (highest first) and limit
        suspected_typos.sort(key=lambda x: x[2], reverse=True)
        suspected_typos = suspected_typos[:self.max_corrections]
        
        # Stage 2: Correct identified typos
        corrected_text = text
        corrections_made = []
        
        # Process in reverse order to maintain token positions
        for pos, original_token, confidence in reversed(suspected_typos):
            # Get correction candidates
            candidates = self.corrector.correct_token(corrected_text, pos)
            
            if not candidates:
                continue
            
            # Find best candidate considering edit distance
            best_score = float('-inf')
            best_candidate = original_token
            
            for candidate_token, lm_prob in candidates:
                # Skip if same as original
                if candidate_token.lower() == original_token.lower():
                    continue
                
                # Calculate edit distance penalty
                edit_dist = levenshtein_distance(original_token.lower(), candidate_token.lower())
                edit_penalty = self.edit_penalty_lambda * edit_dist
                
                # Combined score
                total_score = np.log(lm_prob) - edit_penalty
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate_token
            
            # Apply correction if we found a better candidate
            if best_candidate != original_token:
                # Replace token in text (simple word-level replacement)
                words = corrected_text.split()
                if pos - 1 < len(words):  # Adjust for CLS token
                    words[pos - 1] = best_candidate
                    corrected_text = ' '.join(words)
                    
                    corrections_made.append({
                        'position': pos,
                        'original': original_token,
                        'corrected': best_candidate,
                        'confidence': confidence,
                        'score': best_score
                    })
        
        # Prepare stats
        stats = {
            'original': text,
            'corrected': corrected_text,
            'typos_detected': len(suspected_typos),
            'corrections_made': corrections_made,
            'total_corrections': len(corrections_made)
        }
        
        return corrected_text, stats

def load_two_stage_corrector(
    correction_model_path: str,
    detection_model_path: Optional[str] = None,
    **kwargs
) -> TwoStageTypoCorrector:
    """Load a two-stage typo corrector."""
    return TwoStageTypoCorrector(
        correction_model_path=correction_model_path,
        detection_model_path=detection_model_path,
        **kwargs
    )