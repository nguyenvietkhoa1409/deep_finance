"""
Knowledge Graph Module for Stock Prediction
Simplified implementation for local machine (4-20 tickers)
"""

from .simple_kg import SimpleKnowledgeGraph
from .triple_extractor import TripleExtractor
from .graph_builder import StarGraphBuilder

__all__ = ['SimpleKnowledgeGraph', 'TripleExtractor', 'StarGraphBuilder']
__version__ = '1.0.0-simplified'