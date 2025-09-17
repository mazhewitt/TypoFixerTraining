#!/usr/bin/env python3
"""
Source Text Diversifier for Multi-Domain Sentence Collection

Collects high-quality sentences from diverse domains to create a rich foundation
for typo correction training data.
"""

import random
import re
import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import json
from tqdm import tqdm

# Avoid multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import datasets
from datasets import load_dataset


class SourceTextDiversifier:
    """Multi-domain sentence collector with quality filtering"""
    
    def __init__(self):
        self.setup_domain_patterns()
        self.collected_sentences = defaultdict(list)
        self.sentence_fingerprints = set()
    
    def setup_domain_patterns(self):
        """Define patterns and keywords for different domains"""
        
        self.domain_configs = {
            "conversational": {
                "description": "Informal, everyday conversation",
                "keywords": ["think", "feel", "really", "pretty", "kind of", "sort of", "maybe", "probably"],
                "avoid_patterns": [r"\d{4}", r"[A-Z]{3,}", r"@", r"http", r"www"],
                "min_words": 5,
                "max_words": 18,
                "target_ratio": 0.25
            },
            
            "professional": {
                "description": "Business and professional communication",
                "keywords": ["project", "meeting", "deadline", "team", "client", "strategy", "process", "report"],
                "avoid_patterns": [r"\$\d+", r"%", r"CEO|CFO|CTO", r"Q[1-4]"],
                "min_words": 6,
                "max_words": 20,
                "target_ratio": 0.20
            },
            
            "educational": {
                "description": "Learning and academic content",
                "keywords": ["study", "learn", "research", "understand", "explain", "theory", "method", "analysis"],
                "avoid_patterns": [r"Figure \d+", r"Table \d+", r"Chapter \d+", r"pp\. \d+"],
                "min_words": 7,
                "max_words": 22,
                "target_ratio": 0.20
            },
            
            "creative": {
                "description": "Stories, descriptions, and creative writing",
                "keywords": ["suddenly", "quietly", "beautiful", "mysterious", "ancient", "journey", "discover", "imagine"],
                "avoid_patterns": [r"Chapter \d+", r"Part \d+", r"ISBN", r"copyright"],
                "min_words": 6,
                "max_words": 25,
                "target_ratio": 0.15
            },
            
            "instructional": {
                "description": "How-to guides and instructions",
                "keywords": ["first", "next", "then", "finally", "step", "process", "method", "ensure", "make sure"],
                "avoid_patterns": [r"Step \d+", r"\d+\.", r"a\)", r"b\)", r"i\."],
                "min_words": 5,
                "max_words": 20,
                "target_ratio": 0.10
            },
            
            "general": {
                "description": "General purpose sentences",
                "keywords": [],  # No specific keywords
                "avoid_patterns": [r"===", r"\[\[", r"\]\]", r"{{", r"}}", r"Category:"],
                "min_words": 5,
                "max_words": 18,
                "target_ratio": 0.10
            }
        }
    
    def create_sentence_fingerprint(self, sentence: str) -> str:
        """Create a normalized fingerprint to detect near-duplicates"""
        # Remove punctuation, lowercase, sort words
        words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
        return ' '.join(sorted(set(words)))
    
    def classify_sentence_domain(self, sentence: str) -> Optional[str]:
        """Classify a sentence into the most appropriate domain"""
        sentence_lower = sentence.lower()
        word_count = len(sentence.split())
        
        # Score each domain
        domain_scores = {}
        
        for domain, config in self.domain_configs.items():
            score = 0
            
            # Word count check
            if config["min_words"] <= word_count <= config["max_words"]:
                score += 10
            
            # Keyword matching
            if config["keywords"]:
                keyword_matches = sum(1 for keyword in config["keywords"] if keyword in sentence_lower)
                score += keyword_matches * 5
            
            # Avoid pattern penalty
            for pattern in config["avoid_patterns"]:
                if re.search(pattern, sentence):
                    score -= 20
            
            # Domain-specific bonuses
            if domain == "conversational":
                if any(word in sentence_lower for word in ["i ", "you ", "we ", "they "]):
                    score += 3
                if sentence.count("'") >= 1:  # Contractions
                    score += 2
            
            elif domain == "professional":
                if any(word in sentence_lower for word in ["will", "should", "must", "need to"]):
                    score += 3
                if re.search(r'\b(company|business|organization|department)\b', sentence_lower):
                    score += 3
            
            elif domain == "educational":
                if any(word in sentence_lower for word in ["according to", "research shows", "studies indicate"]):
                    score += 5
                if re.search(r'\b(study|research|data|evidence|findings)\b', sentence_lower):
                    score += 3
            
            elif domain == "creative":
                if any(word in sentence_lower for word in ["once", "long ago", "in the", "there was"]):
                    score += 3
                if len([w for w in sentence.split() if len(w) > 6]) >= 2:  # Longer words
                    score += 2
            
            elif domain == "instructional":
                if any(word in sentence_lower for word in ["first", "second", "next", "then", "finally"]):
                    score += 5
                if sentence_lower.startswith(("make sure", "be sure", "remember to", "don't forget")):
                    score += 3
            
            domain_scores[domain] = max(0, score)
        
        # Return domain with highest score, or None if no good matches
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain if domain_scores[best_domain] > 5 else "general"
        
        return "general"
    
    def is_high_quality_sentence(self, sentence: str, domain: str = None) -> bool:
        """Apply comprehensive quality filters"""
        
        # Basic length and structure
        if not (20 <= len(sentence) <= 200):
            return False
        
        words = sentence.split()
        if not (4 <= len(words) <= 30):
            return False
        
        # Must end with proper punctuation
        if not sentence.rstrip().endswith(('.', '!', '?')):
            return False
        
        # Character diversity (avoid repeated characters)
        if len(set(sentence.lower())) < len(sentence) * 0.3:
            return False
        
        # Avoid problematic patterns
        problematic_patterns = [
            r'^[A-Z\s]+$',  # ALL CAPS
            r'\d{4,}',  # Long numbers
            r'http|www|@',  # URLs/emails
            r'===|---|\*\*\*',  # Wiki markup
            r'[<>{}[\]\\]',  # Markup characters
            r'\([^)]{30,}\)',  # Long parenthetical
            r'"[^"]{50,}"',  # Long quoted text
            r'\.{3,}|!{2,}|\?{2,}',  # Multiple punctuation
            r'\b[A-Z]{3,}\b.*\b[A-Z]{3,}\b',  # Multiple acronyms
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, sentence):
                return False
        
        # Domain-specific quality checks
        if domain:
            config = self.domain_configs.get(domain, {})
            
            # Additional avoid patterns for domain
            for pattern in config.get("avoid_patterns", []):
                if re.search(pattern, sentence):
                    return False
        
        # Check for reasonable word variety
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        if len(unique_words) < len(words) * 0.7:  # Too many repeated words
            return False
        
        # Avoid sentences that are mostly numbers or special characters
        alpha_chars = sum(1 for c in sentence if c.isalpha())
        if alpha_chars < len(sentence) * 0.6:
            return False
        
        return True
    
    def collect_from_dataset(self, dataset_name: str, max_sentences: int = 15000) -> List[Tuple[str, str]]:
        """Collect sentences from a specific dataset"""
        
        print(f"📚 Collecting from {dataset_name}...")
        
        try:
            # Load dataset based on name
            if dataset_name == "wikitext":
                dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
                text_key = 'text'
            elif dataset_name == "bookcorpus":
                try:
                    dataset = load_dataset('bookcorpus', split='train[:8000]', streaming=False)
                    text_key = 'text'
                except:
                    print("⚠️  BookCorpus not available, using WikiText")
                    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
                    text_key = 'text'
            elif dataset_name == "openwebtext":
                try:
                    dataset = load_dataset('openwebtext', split='train[:5000]', streaming=False)
                    text_key = 'text'
                except:
                    print("⚠️  OpenWebText not available, using WikiText")
                    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
                    text_key = 'text'
            else:
                print(f"⚠️  Unknown dataset {dataset_name}, using WikiText")
                dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
                text_key = 'text'
            
            collected = []
            processed_count = 0
            
            for example in tqdm(dataset, desc=f"Processing {dataset_name}", disable=True):
                if len(collected) >= max_sentences or processed_count > 20000:
                    break
                
                processed_count += 1
                text = example[text_key].strip()
                
                if not text or len(text) < 30:
                    continue
                
                # Split into sentences
                sentences = re.split(r'[.!?]+\s+', text)
                
                for raw_sentence in sentences[:10]:  # Limit per document
                    if len(collected) >= max_sentences:
                        break
                    
                    sentence = raw_sentence.strip()
                    if not sentence:
                        continue
                    
                    # Ensure proper ending
                    if not sentence.endswith(('.', '!', '?')):
                        sentence += '.'
                    
                    # Apply quality filter
                    if not self.is_high_quality_sentence(sentence):
                        continue
                    
                    # Check for near-duplicates
                    fingerprint = self.create_sentence_fingerprint(sentence)
                    if fingerprint in self.sentence_fingerprints:
                        continue
                    
                    # Classify domain
                    domain = self.classify_sentence_domain(sentence)
                    if not domain:
                        continue
                    
                    # Store sentence with domain
                    collected.append((sentence, domain))
                    self.sentence_fingerprints.add(fingerprint)
            
            print(f"✅ Collected {len(collected)} sentences from {dataset_name}")
            return collected
        
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            raise
    
    def get_fallback_sentences(self, source: str) -> List[Tuple[str, str]]:
        """High-quality fallback sentences when datasets fail"""
        
        fallback_by_domain = {
            "conversational": [
                "I think this is probably the best approach we can take.",
                "You know, I really believe that's going to work well.",
                "Maybe we should try a different method this time.",
                "I feel like this project is going really smoothly.",
                "That sounds pretty interesting, but I'm not completely sure.",
                "I kind of expected this might happen eventually.",
                "Do you really think we can finish this by Friday?",
                "It seems like everyone is pretty happy with the results."
            ],
            
            "professional": [
                "The project deadline has been extended until next Friday.",
                "We need to schedule a meeting to discuss the client requirements.",
                "The team successfully completed the quarterly strategy review.",
                "Please submit your report before the end of the business day.",
                "The new process will improve efficiency across all departments.",
                "Our client expressed satisfaction with the delivered solution.",
                "The budget allocation for this quarter needs immediate attention.",
                "We should prioritize the most critical issues first."
            ],
            
            "educational": [
                "Research shows that regular practice improves learning outcomes significantly.",
                "Students should understand the fundamental principles before advancing further.",
                "The study demonstrates clear evidence of improved performance metrics.",
                "According to recent analysis, this method proves highly effective.",
                "Learning requires consistent effort and systematic approach to mastery.",
                "The theory explains why certain patterns emerge in natural systems.",
                "Scientific research continues to reveal fascinating discoveries about nature.",
                "Understanding these concepts will help students solve complex problems."
            ],
            
            "creative": [
                "The ancient library contained thousands of mysterious manuscripts.",
                "She discovered a beautiful garden hidden behind the old wall.",
                "The journey through the mountain valley proved more challenging than expected.",
                "Sunlight filtered through the leaves, creating dancing shadows below.",
                "The old musician played haunting melodies that echoed through the night.",
                "In the distance, a lone figure walked slowly across the bridge.",
                "The story began on a quiet morning in a small village.",
                "Magic seemed to flow through every corner of the enchanted forest."
            ],
            
            "instructional": [
                "First, make sure you have all the necessary materials ready.",
                "Next, carefully follow each step in the correct sequence.",
                "Then, verify that everything is properly aligned before proceeding.",
                "Finally, test the system to ensure it functions as expected.",
                "Remember to save your work frequently during the process.",
                "Be sure to double-check all measurements before making any cuts.",
                "Always wear appropriate safety equipment when handling tools.",
                "Don't forget to clean up your workspace when finished."
            ],
            
            "general": [
                "The weather forecast predicts sunny skies for the weekend.",
                "Technology continues to transform how we communicate daily.",
                "Regular exercise contributes to better physical and mental health.",
                "The library offers many resources for students and researchers.",
                "Environmental protection requires cooperation from all community members.",
                "Public transportation helps reduce traffic congestion in urban areas.",
                "The museum displays artifacts from many different historical periods.",
                "Online learning platforms provide access to education worldwide."
            ]
        }
        
        sentences_with_domains = []
        for domain, sentences in fallback_by_domain.items():
            for sentence in sentences:
                sentences_with_domains.append((sentence, domain))
        
        # Repeat and shuffle to get more variety
        sentences_with_domains = sentences_with_domains * 10
        random.shuffle(sentences_with_domains)
        
        print(f"✅ Generated {len(sentences_with_domains)} fallback sentences for {source}")
        return sentences_with_domains
    
    def collect_diverse_sentences(self, total_target: int = 50000) -> Dict[str, List[str]]:
        """Collect diverse sentences from multiple sources"""
        
        print(f"🎯 Collecting {total_target:,} diverse sentences across multiple domains")
        print("=" * 60)
        
        # Data sources with allocation (updated for availability)
        sources = [
            ("wikitext", 0.6),      # 60% - Educational/general content (increased since others may fail)
            ("cc_news", 0.2),       # 20% - News content (more available)
            ("xsum", 0.2),          # 20% - Summary/article content
        ]
        
        all_sentences_with_domains = []
        
        # Collect from each source
        for source_name, allocation in sources:
            target_for_source = int(total_target * allocation)
            sentences = self.collect_from_dataset(source_name, target_for_source)
            all_sentences_with_domains.extend(sentences)
            print()
        
        # Organize by domain
        sentences_by_domain = defaultdict(list)
        for sentence, domain in all_sentences_with_domains:
            sentences_by_domain[domain].append(sentence)
        
        # Balance according to target ratios
        balanced_sentences = {}
        total_collected = len(all_sentences_with_domains)
        
        print(f"🔄 Balancing {total_collected:,} sentences across domains...")
        
        for domain, config in self.domain_configs.items():
            target_count = int(total_target * config["target_ratio"])
            available = sentences_by_domain[domain]
            
            if len(available) >= target_count:
                # Randomly sample if we have too many
                selected = random.sample(available, target_count)
            else:
                # Use all available + duplicates if needed
                selected = available.copy()
                while len(selected) < target_count and available:
                    selected.extend(random.choices(available, k=min(len(available), target_count - len(selected))))
            
            balanced_sentences[domain] = selected
            print(f"  {domain:15}: {len(selected):5,} sentences ({config['target_ratio']:5.1%} target)")
        
        return balanced_sentences
    
    def save_collected_sentences(self, sentences_by_domain: Dict[str, List[str]], 
                                output_file: str = "data/diverse_source_sentences.json"):
        """Save collected sentences to file"""
        
        os.makedirs("data", exist_ok=True)
        
        # Prepare data with metadata
        output_data = {
            "metadata": {
                "total_sentences": sum(len(sentences) for sentences in sentences_by_domain.values()),
                "domains": list(sentences_by_domain.keys()),
                "domain_counts": {domain: len(sentences) for domain, sentences in sentences_by_domain.items()},
                "collection_method": "multi_source_diversified",
                "quality_filtered": True
            },
            "sentences_by_domain": sentences_by_domain
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        total = output_data["metadata"]["total_sentences"]
        print(f"💾 Saved {total:,} diverse sentences to {output_file}")
        
        # Show sample from each domain
        print(f"\n📝 Sample sentences by domain:")
        for domain, sentences in sentences_by_domain.items():
            if sentences:
                sample = random.choice(sentences)
                print(f"  {domain:15}: '{sample[:60]}{'...' if len(sample) > 60 else ''}'")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect diverse source sentences for training data")
    parser.add_argument('--target-sentences', type=int, default=50000,
                       help='Target number of sentences to collect')
    parser.add_argument('--output-file', default='data/diverse_source_sentences.json',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create diversifier and collect sentences
    diversifier = SourceTextDiversifier()
    sentences_by_domain = diversifier.collect_diverse_sentences(args.target_sentences)
    
    # Save results
    diversifier.save_collected_sentences(sentences_by_domain, args.output_file)
    
    print(f"\n✅ Source text collection complete!")
    print(f"📁 Output: {args.output_file}")


if __name__ == "__main__":
    main()