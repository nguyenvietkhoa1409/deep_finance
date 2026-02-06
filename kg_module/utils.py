"""
Utility functions for Knowledge Graph module
"""

import re
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .config import (
    ENTITY_ALIASES, MAGNITUDE_PATTERNS, 
    POLARITY_KEYWORDS, CERTAINTY_KEYWORDS, RELATION_TYPES
)


def link_entity(mention: str, tickers: List[str]) -> str:
    """
    Link entity mention to canonical ticker.
    
    Args:
        mention: Entity string from news (e.g., "Amazon", "AWS")
        tickers: List of valid tickers to consider
    
    Returns:
        Canonical ticker or 'OTHER'
    
    Examples:
        >>> link_entity("Amazon Web Services", ["AMZN", "MSFT"])
        'AMZN'
        >>> link_entity("Google", ["AMZN", "MSFT"])
        'OTHER'
    """
    mention_lower = mention.lower().strip()
    
    # Check each ticker's aliases
    for ticker in tickers:
        if ticker not in ENTITY_ALIASES:
            continue
        
        aliases = ENTITY_ALIASES[ticker]
        for alias in aliases:
            if alias.lower() in mention_lower or mention_lower in alias.lower():
                return ticker
    
    return 'OTHER'


def extract_magnitude(text: str) -> Optional[float]:
    """
    Extract numerical magnitude from text using regex patterns.
    
    Args:
        text: News text
    
    Returns:
        Magnitude as float (e.g., 0.23 for "23%") or None
    
    Examples:
        >>> extract_magnitude("Revenue increased 23%")
        0.23
        >>> extract_magnitude("Spent $1.5 billion")
        1.5
    """
    text_lower = text.lower()
    
    for pattern in MAGNITUDE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            # Extract number (handle different group positions)
            for group in match.groups():
                try:
                    num = float(group)
                    
                    # Convert percentage to decimal
                    if '%' in text[match.start():match.end()]:
                        num = num / 100.0
                    
                    # Convert billion/million to base
                    if 'billion' in text[match.start():match.end()] or 'b' in text[match.start():match.end()].lower():
                        # Keep as billions for consistency
                        pass
                    elif 'million' in text[match.start():match.end()] or 'm' in text[match.start():match.end()].lower():
                        num = num / 1000.0  # Convert to billions
                    
                    return num
                except ValueError:
                    continue
    
    return None


def determine_polarity(text: str) -> int:
    """
    Determine sentiment polarity from text.
    
    Args:
        text: News text
    
    Returns:
        1 (positive), 0 (neutral), or -1 (negative)
    
    Examples:
        >>> determine_polarity("Revenue surged 23%")
        1
        >>> determine_polarity("Stock plunged on weak results")
        -1
    """
    text_lower = text.lower()
    
    pos_count = sum(1 for word in POLARITY_KEYWORDS['positive'] if word in text_lower)
    neg_count = sum(1 for word in POLARITY_KEYWORDS['negative'] if word in text_lower)
    
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return -1
    else:
        return 0


def determine_certainty(text: str) -> float:
    """
    Determine certainty level from text.
    
    Args:
        text: News text
    
    Returns:
        Certainty score in [0.3, 0.5, 0.8, 1.0]
    
    Examples:
        >>> determine_certainty("Amazon reported Q1 earnings")
        1.0
        >>> determine_certainty("Microsoft may acquire startup")
        0.5
    """
    text_lower = text.lower()
    
    # Check each certainty level (highest first)
    for score in sorted(CERTAINTY_KEYWORDS.keys(), reverse=True):
        keywords = CERTAINTY_KEYWORDS[score]
        if any(keyword in text_lower for keyword in keywords):
            return score
    
    # Default to confirmed if no marker found
    return 1.0


def classify_relation(text: str) -> Optional[str]:
    """
    Classify relation type from text using keyword matching.
    
    Args:
        text: News text
    
    Returns:
        Relation type string or None
    
    Examples:
        >>> classify_relation("Amazon Q1 revenue increased 23%")
        'revenue_change'
        >>> classify_relation("Microsoft acquires Activision")
        'acquisition'
    """
    text_lower = text.lower()
    
    # Score each relation type
    scores = {}
    for rel_type, rel_config in RELATION_TYPES.items():
        keywords = rel_config['keywords']
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[rel_type] = score
    
    if not scores:
        return None
    
    # Return relation with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def compute_temporal_decay(
    event_date: datetime,
    current_date: datetime,
    relation_type: str
) -> float:
    """
    Compute temporal decay weight for an event.
    
    Args:
        event_date: When event occurred
        current_date: Current date
        relation_type: Type of event
    
    Returns:
        Decay weight in [0, 1]
    
    Examples:
        >>> from datetime import datetime
        >>> event = datetime(2024, 1, 1)
        >>> current = datetime(2024, 1, 3)
        >>> compute_temporal_decay(event, current, 'revenue_change')
        0.5  # 2 days elapsed, half-life=2
    """
    days_elapsed = (current_date - event_date).days
    
    if days_elapsed < 0:
        return 1.0  # Future event (shouldn't happen, but handle gracefully)
    
    # Get half-life for this relation type
    half_life = RELATION_TYPES.get(relation_type, {}).get('half_life_days', 5)
    
    # Exponential decay: weight = 0.5^(days/half_life)
    decay = 0.5 ** (days_elapsed / half_life)
    
    return max(decay, 0.01)  # Minimum weight to prevent complete disappearance


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize feature vectors (zero mean, unit variance).
    
    Args:
        features: Array of shape (N, D)
    
    Returns:
        Normalized array of same shape
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def validate_triple(triple: Dict) -> bool:
    """
    Validate that a triple has required fields and valid values.
    
    Args:
        triple: Dictionary with triple data
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['subject', 'relation', 'object', 'polarity', 'certainty']
    
    # Check required fields exist
    if not all(field in triple for field in required_fields):
        return False
    
    # Check relation is valid
    if triple['relation'] not in RELATION_TYPES and triple['relation'] is not None:
        return False
    
    # Check polarity in valid range
    if triple['polarity'] not in [-1, 0, 1]:
        return False
    
    # Check certainty in valid range
    if not (0.0 <= triple['certainty'] <= 1.0):
        return False
    
    return True