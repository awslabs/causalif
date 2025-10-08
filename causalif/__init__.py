# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Causalif: Language-Augmented Causal Reasoning with JAX and RAG
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import (
    AssociationResponse,
    AssociationType, 
    CausalDirection,
    KnowledgeBase
)

from .engine import CausalifEngine
from .prompts import CausalifPrompts
from .visualization import visualize_causalif_results
from .tools import (
    causalif_tool,
    set_causalif_engine,
    extract_factors_from_query
)

__all__ = [
    'AssociationResponse',
    'AssociationType',
    'CausalDirection', 
    'KnowledgeBase',
    'CausalifEngine',
    'CausalifPrompts',
    'causalif_tool',
    'set_causalif_engine',
    'extract_factors_from_query',
    'visualize_causalif_results'
]