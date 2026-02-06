"""
Stage 1: Triple Extraction
Rule-based + LLM fallback for extracting structured events from news
"""

import re
import json
from typing import Dict, List, Optional
from datetime import datetime

from .config import (
    RELATION_TYPES, USE_LLM_FALLBACK, LLM_MODEL, 
    LLM_MAX_RETRIES, LLM_TIMEOUT, LLM_PROMPT_TEMPLATE
)
from .utils import (
    link_entity, extract_magnitude, determine_polarity,
    determine_certainty, classify_relation, validate_triple
)


class TripleExtractor:
    """
    Extract structured triples from news headlines.
    
    Strategy:
        1. Fast path: Rule-based pattern matching (85% of cases)
        2. Slow path: LLM fallback for complex cases (15% of cases)
    
    Output Format:
        {
            'subject': 'AMZN',
            'relation': 'revenue_change',
            'object': 'Q1 revenue increased 23% to $127.4B',
            'magnitude': 0.23,
            'polarity': 1,
            'certainty': 1.0,
            'date': datetime(...),
            'source': 'Bloomberg'
        }
    """
    
    def __init__(self, use_llm: bool = USE_LLM_FALLBACK):
        """
        Initialize extractor.
        
        Args:
            use_llm: Whether to use LLM fallback for complex cases
        """
        self.use_llm = use_llm
        self.llm_client = None
        
        if self.use_llm:
            try:
                import openai
                self.llm_client = openai
            except ImportError:
                print("⚠️  OpenAI not installed. LLM fallback disabled.")
                self.use_llm = False
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'rule_based_success': 0,
            'llm_fallback_used': 0,
            'failed_extractions': 0
        }
    
    def extract(
        self, 
        headline: str, 
        ticker: str, 
        date: datetime,
        source: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Extract triple from a single news headline.
        
        Args:
            headline: News headline text
            ticker: Associated ticker symbol
            date: Publication date
            source: News source (optional)
        
        Returns:
            Triple dict or None if extraction failed
        
        Examples:
            >>> extractor = TripleExtractor()
            >>> triple = extractor.extract(
            ...     "Amazon Q1 revenue surged 23% to $127.4B",
            ...     "AMZN",
            ...     datetime(2024, 4, 30)
            ... )
            >>> triple['relation']
            'revenue_change'
            >>> triple['magnitude']
            0.23
        """
        self.stats['total_processed'] += 1
        
        # Try rule-based extraction first (fast path)
        triple = self._extract_rule_based(headline, ticker, date, source)
        
        if triple and validate_triple(triple):
            self.stats['rule_based_success'] += 1
            return triple
        
        # Fallback to LLM (slow path)
        if self.use_llm:
            try:
                triple = self._extract_llm(headline, ticker, date, source)
                if triple and validate_triple(triple):
                    self.stats['llm_fallback_used'] += 1
                    return triple
            except Exception as e:
                print(f"⚠️  LLM extraction failed: {e}")
        
        # Extraction failed
        self.stats['failed_extractions'] += 1
        return None
    
    def extract_batch(
        self,
        news_records: List[Dict]
    ) -> List[Dict]:
        """
        Extract triples from a batch of news records.
        
        Args:
            news_records: List of dicts with keys:
                - 'title' or 'headline': str
                - 'equity' or 'ticker': str
                - 'date': datetime or str
                - 'source': str (optional)
        
        Returns:
            List of extracted triples
        
        Examples:
            >>> records = [
            ...     {'title': 'AMZN revenue up', 'equity': 'AMZN', 'date': '2024-01-01'},
            ...     {'title': 'TSLA new factory', 'equity': 'TSLA', 'date': '2024-01-02'}
            ... ]
            >>> triples = extractor.extract_batch(records)
            >>> len(triples)
            2
        """
        triples = []
        
        for record in news_records:
            # Parse record
            headline = record.get('title') or record.get('headline', '')
            ticker = record.get('equity') or record.get('ticker', '')
            date = record.get('date')
            source = record.get('source')
            
            if not headline or not ticker:
                continue
            
            # Convert date if needed
            if isinstance(date, str):
                try:
                    date = datetime.fromisoformat(date.replace('Z', ''))
                except:
                    date = datetime.now()
            
            # Extract
            triple = self.extract(headline, ticker, date, source)
            if triple:
                triples.append(triple)
        
        return triples
    
    def _extract_rule_based(
        self,
        headline: str,
        ticker: str,
        date: datetime,
        source: Optional[str]
    ) -> Optional[Dict]:
        """
        Rule-based extraction using pattern matching.
        
        Returns:
            Triple dict or None
        """
        # Classify relation type
        relation = classify_relation(headline)
        if not relation:
            return None
        
        # Extract components
        magnitude = extract_magnitude(headline)
        polarity = determine_polarity(headline)
        certainty = determine_certainty(headline)
        
        # Build triple
        triple = {
            'subject': ticker,
            'relation': relation,
            'object': headline[:100],  # Truncate to 100 chars
            'magnitude': magnitude,
            'polarity': polarity,
            'certainty': certainty,
            'date': date,
            'source': source
        }
        
        return triple
    
    def _extract_llm(
        self,
        headline: str,
        ticker: str,
        date: datetime,
        source: Optional[str]
    ) -> Optional[Dict]:
        """
        LLM-based extraction for complex cases.
        
        Returns:
            Triple dict or None
        """
        if not self.llm_client:
            return None
        
        # Format prompt
        prompt = LLM_PROMPT_TEMPLATE.format(
            headline=headline,
            ticker=ticker
        )
        
        # Call LLM with retries
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = self.llm_client.ChatCompletion.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=LLM_TIMEOUT,
                    temperature=0.0  # Deterministic
                )
                
                # Parse response
                content = response.choices[0].message.content.strip()
                
                # Clean markdown if present
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                
                result = json.loads(content)
                
                # Validate and build triple
                if result.get('relation') in RELATION_TYPES or result.get('relation') is None:
                    triple = {
                        'subject': ticker,
                        'relation': result.get('relation'),
                        'object': result.get('object', headline[:100]),
                        'magnitude': result.get('magnitude'),
                        'polarity': result.get('polarity', 0),
                        'certainty': result.get('certainty', 0.8),
                        'date': date,
                        'source': source
                    }
                    return triple
                
            except Exception as e:
                if attempt == LLM_MAX_RETRIES - 1:
                    print(f"⚠️  LLM extraction failed after {LLM_MAX_RETRIES} attempts: {e}")
                continue
        
        return None
    
    def get_stats(self) -> Dict:
        """
        Get extraction statistics.
        
        Returns:
            Dict with statistics
        """
        stats = self.stats.copy()
        if stats['total_processed'] > 0:
            stats['success_rate'] = (
                (stats['rule_based_success'] + stats['llm_fallback_used']) /
                stats['total_processed']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def print_stats(self):
        """Print extraction statistics."""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("📊 TRIPLE EXTRACTION STATISTICS")
        print("="*50)
        print(f"Total processed:      {stats['total_processed']:,}")
        print(f"Rule-based success:   {stats['rule_based_success']:,} ({stats['rule_based_success']/max(stats['total_processed'],1)*100:.1f}%)")
        print(f"LLM fallback used:    {stats['llm_fallback_used']:,} ({stats['llm_fallback_used']/max(stats['total_processed'],1)*100:.1f}%)")
        print(f"Failed extractions:   {stats['failed_extractions']:,} ({stats['failed_extractions']/max(stats['total_processed'],1)*100:.1f}%)")
        print(f"Overall success rate: {stats['success_rate']*100:.1f}%")
        print("="*50 + "\n")