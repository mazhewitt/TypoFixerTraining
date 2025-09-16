#!/usr/bin/env python3
"""
Advanced Error Pattern Library for Realistic Typo Generation

This library implements sophisticated error patterns based on real human typing mistakes,
cognitive errors, and mobile device input patterns.
"""

import random
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ErrorType(Enum):
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    CAPITALIZATION = "capitalization"
    SPACING = "spacing"
    PHONETIC = "phonetic"
    KEYBOARD = "keyboard"

@dataclass
class ErrorPattern:
    name: str
    error_type: ErrorType
    probability: float
    description: str

class AdvancedErrorPatterns:
    """Advanced error pattern library with human-like mistakes"""
    
    def __init__(self):
        self.setup_error_patterns()
    
    def setup_error_patterns(self):
        """Initialize all error pattern data"""
        
        # Keyboard layout for adjacent key errors
        self.keyboard_layout = {
            'q': ['w', 'a', 's'], 'w': ['q', 'e', 'a', 's', 'd'], 'e': ['w', 'r', 's', 'd', 'f'],
            'r': ['e', 't', 'd', 'f', 'g'], 't': ['r', 'y', 'f', 'g', 'h'], 'y': ['t', 'u', 'g', 'h', 'j'],
            'u': ['y', 'i', 'h', 'j', 'k'], 'i': ['u', 'o', 'j', 'k', 'l'], 'o': ['i', 'p', 'k', 'l'],
            'p': ['o', 'l'], 'a': ['q', 'w', 's', 'z'], 's': ['w', 'e', 'a', 'd', 'z', 'x'],
            'd': ['e', 'r', 's', 'f', 'x', 'c'], 'f': ['r', 't', 'd', 'g', 'c', 'v'], 
            'g': ['t', 'y', 'f', 'h', 'v', 'b'], 'h': ['y', 'u', 'g', 'j', 'b', 'n'],
            'j': ['u', 'i', 'h', 'k', 'n', 'm'], 'k': ['i', 'o', 'j', 'l', 'm'], 
            'l': ['o', 'p', 'k'], 'z': ['a', 's', 'x'], 'x': ['s', 'd', 'z', 'c'],
            'c': ['d', 'f', 'x', 'v'], 'v': ['f', 'g', 'c', 'b'], 'b': ['g', 'h', 'v', 'n'],
            'n': ['h', 'j', 'b', 'm'], 'm': ['j', 'k', 'n']
        }
        
        # Common homophones and sound-based confusions
        self.homophones = {
            'there': ['their', 'they\'re'], 'their': ['there', 'they\'re'], 'they\'re': ['there', 'their'],
            'to': ['too', 'two'], 'too': ['to', 'two'], 'two': ['to', 'too'],
            'your': ['you\'re'], 'you\'re': ['your'], 'its': ['it\'s'], 'it\'s': ['its'],
            'then': ['than'], 'than': ['then'], 'affect': ['effect'], 'effect': ['affect'],
            'accept': ['except'], 'except': ['accept'], 'lose': ['loose'], 'loose': ['lose'],
            'choose': ['chose'], 'chose': ['choose'], 'break': ['brake'], 'brake': ['break'],
            'piece': ['peace'], 'peace': ['piece'], 'hear': ['here'], 'here': ['hear'],
            'wear': ['where'], 'where': ['wear'], 'weather': ['whether'], 'whether': ['weather']
        }
        
        # Common misspellings with multiple variants
        self.common_misspellings = {
            'definitely': ['definately', 'definatly', 'defiantly'], 
            'separate': ['seperate', 'seprate', 'separat'], 
            'necessary': ['neccessary', 'necesary', 'neccesary'],
            'receive': ['recieve', 'recive', 'receve'], 
            'believe': ['beleive', 'belive', 'beleave'],
            'achieve': ['acheive', 'achive', 'acheve'], 
            'occurred': ['occured', 'occurrd', 'ocurred'],
            'beginning': ['begining', 'beggining', 'begininng'], 
            'embarrass': ['embarass', 'embarras', 'embaress'],
            'government': ['goverment', 'govenment', 'govermnent'], 
            'restaurant': ['restarant', 'resturant', 'restaurent'],
            'business': ['buisness', 'bussiness', 'busines'], 
            'tomorrow': ['tomarow', 'tomorow', 'tommorow'],
            'language': ['langauge', 'languag', 'lanuage'], 
            'experience': ['experiance', 'experince', 'expereince'],
            'different': ['diferent', 'diffrent', 'diferrent'], 
            'important': ['importent', 'importnt', 'imporant'],
            'because': ['becuase', 'beacuse', 'becaus'], 
            'beautiful': ['beatiful', 'beutiful', 'beautifull'],
            'friend': ['freind', 'frend', 'freend'], 
            'sentence': ['sentnce', 'sentance', 'sentense'],
            'remember': ['remeber', 'remmeber', 'remembr'], 
            'exercise': ['excercise', 'exersize', 'exercize']
        }
        
        # Contractions and apostrophe patterns
        self.contraction_errors = {
            'don\'t': ['dont', 'do not'], 'can\'t': ['cant', 'cannot'], 'won\'t': ['wont', 'will not'],
            'isn\'t': ['isnt', 'is not'], 'aren\'t': ['arent', 'are not'], 'wasn\'t': ['wasnt', 'was not'],
            'weren\'t': ['werent', 'were not'], 'doesn\'t': ['doesnt', 'does not'], 'didn\'t': ['didnt', 'did not'],
            'shouldn\'t': ['shouldnt', 'should not'], 'wouldn\'t': ['wouldnt', 'would not'], 'couldn\'t': ['couldnt', 'could not'],
            'I\'m': ['Im', 'I am'], 'you\'re': ['youre', 'you are'], 'we\'re': ['were', 'we are'],
            'they\'re': ['theyre', 'they are'], 'I\'ve': ['Ive', 'I have'], 'you\'ve': ['youve', 'you have'],
            'we\'ve': ['weve', 'we have'], 'they\'ve': ['theyve', 'they have'], 'I\'ll': ['Ill', 'I will'],
            'you\'ll': ['youll', 'you will'], 'we\'ll': ['well', 'we will'], 'they\'ll': ['theyll', 'they will']
        }
        
        # Double letter patterns (common mistakes)
        self.double_letter_words = {
            'accommodate': 'accomodate', 'occurring': 'occuring', 'beginning': 'begining',
            'committee': 'commitee', 'possess': 'posess', 'address': 'adress',
            'success': 'sucess', 'access': 'acess', 'goddess': 'godess',
            'process': 'proces', 'necessary': 'necesary', 'embarrass': 'embaras'
        }
        
        # Compound word errors
        self.compound_word_errors = {
            'anyway': 'any way', 'everyone': 'every one', 'someone': 'some one',
            'anything': 'any thing', 'everything': 'every thing', 'something': 'some thing',
            'everybody': 'every body', 'somebody': 'some body', 'anybody': 'any body',
            'nobody': 'no body', 'everywhere': 'every where', 'somewhere': 'some where',
            'anywhere': 'any where', 'nowhere': 'no where', 'into': 'in to',
            'onto': 'on to', 'cannot': 'can not', 'forever': 'for ever'
        }

    def apply_keyboard_error(self, word: str) -> str:
        """Apply keyboard adjacency errors (fat finger mistakes)"""
        if len(word) < 2:
            return word
        
        pos = random.randint(0, len(word) - 1)
        char = word[pos].lower()
        
        if char in self.keyboard_layout and random.random() < 0.4:
            replacement = random.choice(self.keyboard_layout[char])
            return word[:pos] + (replacement.upper() if word[pos].isupper() else replacement) + word[pos + 1:]
        
        return word

    def apply_character_operation_error(self, word: str) -> str:
        """Apply character insertion, deletion, or transposition"""
        if len(word) < 3:
            return word
        
        operation = random.choice(['insert', 'delete', 'transpose', 'substitute'])
        
        if operation == 'insert':
            # Insert a character (often repeated)
            pos = random.randint(0, len(word))
            if pos > 0:
                char_to_insert = word[pos-1] if random.random() < 0.7 else random.choice('aeiou')
                return word[:pos] + char_to_insert + word[pos:]
        
        elif operation == 'delete':
            # Delete a character
            pos = random.randint(1, len(word) - 2)
            return word[:pos] + word[pos + 1:]
        
        elif operation == 'transpose':
            # Swap adjacent characters
            pos = random.randint(0, len(word) - 2)
            chars = list(word)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        
        elif operation == 'substitute':
            # Substitute with similar character
            pos = random.randint(0, len(word) - 1)
            char = word[pos].lower()
            similar_chars = {
                'a': ['e', 'o'], 'e': ['a', 'i'], 'i': ['e', 'o'], 'o': ['a', 'u'], 'u': ['o', 'i'],
                'b': ['p', 'v'], 'p': ['b'], 'v': ['b', 'f'], 'f': ['v'], 'g': ['c'], 'c': ['g'],
                'd': ['t'], 't': ['d'], 'n': ['m'], 'm': ['n'], 'l': ['r'], 'r': ['l']
            }
            if char in similar_chars:
                replacement = random.choice(similar_chars[char])
                return word[:pos] + (replacement.upper() if word[pos].isupper() else replacement) + word[pos + 1:]
        
        return word

    def apply_phonetic_error(self, word: str) -> str:
        """Apply phonetic/sound-based errors"""
        word_lower = word.lower()
        
        # Check for common misspellings first
        if word_lower in self.common_misspellings and random.random() < 0.6:
            typo = random.choice(self.common_misspellings[word_lower])
            return self.preserve_case(word, typo)
        
        # Check for homophones
        if word_lower in self.homophones and random.random() < 0.3:
            replacement = random.choice(self.homophones[word_lower])
            return self.preserve_case(word, replacement)
        
        # Apply phonetic substitutions
        phonetic_rules = [
            ('ph', 'f'), ('ough', 'uff'), ('augh', 'af'), ('eigh', 'ay'),
            ('tion', 'sion'), ('cious', 'sious'), ('ence', 'ance'), ('ance', 'ence')
        ]
        
        for pattern, replacement in phonetic_rules:
            if pattern in word_lower and random.random() < 0.2:
                new_word = word_lower.replace(pattern, replacement, 1)
                return self.preserve_case(word, new_word)
        
        return word

    def apply_contraction_error(self, word: str) -> str:
        """Apply contraction and apostrophe errors"""
        word_lower = word.lower()
        
        if word_lower in self.contraction_errors and random.random() < 0.5:
            error_variant = random.choice(self.contraction_errors[word_lower])
            return self.preserve_case(word, error_variant)
        
        return word

    def apply_compound_word_error(self, word: str) -> str:
        """Apply compound word spacing errors"""
        word_lower = word.lower()
        
        if word_lower in self.compound_word_errors and random.random() < 0.4:
            error_variant = self.compound_word_errors[word_lower]
            return self.preserve_case(word, error_variant)
        
        return word

    def apply_double_letter_error(self, word: str) -> str:
        """Apply double letter mistakes"""
        word_lower = word.lower()
        
        if word_lower in self.double_letter_words and random.random() < 0.4:
            error_variant = self.double_letter_words[word_lower]
            return self.preserve_case(word, error_variant)
        
        # General double letter errors
        if len(word) > 4:
            for i in range(len(word) - 1):
                if word[i].lower() == word[i + 1].lower():
                    # Remove one of the double letters sometimes
                    if random.random() < 0.3:
                        return word[:i] + word[i + 1:]
        
        return word

    def preserve_case(self, original: str, modified: str) -> str:
        """Preserve the capitalization pattern of the original word"""
        if not original or not modified:
            return modified
        
        result = []
        for i, char in enumerate(modified):
            if i < len(original):
                if original[i].isupper():
                    result.append(char.upper())
                else:
                    result.append(char.lower())
            else:
                result.append(char.lower())
        
        return ''.join(result)

    def apply_contextual_errors(self, words: List[str]) -> List[str]:
        """Apply errors that depend on context between words"""
        result = words.copy()
        
        for i in range(len(result)):
            word = result[i].lower()
            
            # Context-dependent homophones
            if i > 0 and word == 'then' and result[i-1].lower() in ['better', 'more', 'less', 'rather']:
                if random.random() < 0.4:
                    result[i] = self.preserve_case(result[i], 'than')
            
            elif word == 'than' and i > 0 and result[i-1].lower() not in ['better', 'more', 'less', 'rather']:
                if random.random() < 0.3:
                    result[i] = self.preserve_case(result[i], 'then')
        
        return result

    def corrupt_word(self, word: str, error_types: List[ErrorType] = None, intensity: float = 0.15) -> str:
        """
        Apply realistic corruption to a word based on specified error types
        
        Args:
            word: The word to corrupt
            error_types: List of error types to apply (None = random selection)
            intensity: Probability of applying corruption (0.0-1.0)
        
        Returns:
            Corrupted word
        """
        if len(word) <= 2 or random.random() > intensity:
            return word
        
        if error_types is None:
            error_types = [ErrorType.SPELLING, ErrorType.PHONETIC, ErrorType.KEYBOARD]
        
        # Select error type based on word characteristics and probabilities
        error_weights = {
            ErrorType.SPELLING: 0.3,
            ErrorType.PHONETIC: 0.25,
            ErrorType.KEYBOARD: 0.2,
            ErrorType.PUNCTUATION: 0.1,
            ErrorType.GRAMMAR: 0.05,
            ErrorType.SPACING: 0.1
        }
        
        # Filter error types to those we have weights for and are in error_types
        available_errors = [et for et in error_types if et in error_weights]
        if not available_errors:
            available_errors = [ErrorType.SPELLING]  # Default fallback
        
        weights = [error_weights[et] for et in available_errors]
        
        selected_error = random.choices(
            available_errors,
            weights=weights,
            k=1
        )[0]
        
        # Apply the selected error type
        if selected_error == ErrorType.KEYBOARD:
            return self.apply_keyboard_error(word)
        elif selected_error == ErrorType.PHONETIC:
            return self.apply_phonetic_error(word)
        elif selected_error == ErrorType.SPELLING:
            corrupted = self.apply_character_operation_error(word)
            if corrupted == word:  # If no change, try double letter
                corrupted = self.apply_double_letter_error(word)
            return corrupted
        elif selected_error == ErrorType.PUNCTUATION:
            return self.apply_contraction_error(word)
        elif selected_error == ErrorType.SPACING:
            return self.apply_compound_word_error(word)
        
        return word

    def corrupt_sentence(self, sentence: str, complexity: str = "medium", 
                        num_errors: Optional[int] = None) -> str:
        """
        Apply realistic corruption to an entire sentence
        
        Args:
            sentence: The sentence to corrupt
            complexity: "simple", "medium", or "complex"
            num_errors: Specific number of errors (None = auto-determine)
        
        Returns:
            Corrupted sentence
        """
        words = sentence.strip().split()
        if not words:
            return sentence
        
        # Determine number of errors based on complexity
        if num_errors is None:
            if complexity == "simple":
                num_errors = max(1, len(words) // 8)  # ~12% of words
            elif complexity == "medium":
                num_errors = max(1, len(words) // 5)   # ~20% of words  
            else:  # complex
                num_errors = max(2, len(words) // 3)   # ~33% of words
        
        # Select words to corrupt (avoid first and last words often)
        corruptible_indices = list(range(len(words)))
        if len(words) > 3:
            # Reduce probability for first and last words
            weights = [0.3] + [1.0] * (len(words) - 2) + [0.3]
        else:
            weights = [1.0] * len(words)
        
        indices_to_corrupt = random.choices(
            corruptible_indices, 
            weights=weights, 
            k=min(num_errors, len(words))
        )
        
        # Apply contextual errors first
        words = self.apply_contextual_errors(words)
        
        # Apply individual word corruptions
        for idx in set(indices_to_corrupt):  # Remove duplicates
            # Determine error intensity based on word position and length
            intensity = 0.8 if len(words[idx]) > 4 else 0.6
            
            # Apply corruption
            original_word = words[idx]
            corrupted_word = self.corrupt_word(original_word, intensity=intensity)
            
            # Ensure we actually made a change
            if corrupted_word == original_word and len(original_word) > 3:
                # Force a simple character operation if no change occurred
                corrupted_word = self.apply_character_operation_error(original_word)
            
            words[idx] = corrupted_word
        
        return ' '.join(words)

    def get_error_statistics(self) -> Dict:
        """Return statistics about available error patterns"""
        return {
            "keyboard_patterns": len(self.keyboard_layout),
            "homophones": len(self.homophones),
            "common_misspellings": len(self.common_misspellings),
            "contraction_errors": len(self.contraction_errors),
            "compound_word_errors": len(self.compound_word_errors),
            "double_letter_words": len(self.double_letter_words)
        }