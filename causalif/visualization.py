# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization utilities for Causalif"""

import math
from typing import Union, Dict
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

def visualize_causalif_results(causalif_result: Dict) -> go.Figure:
    """Create visualization from Causalif results with degree-based coloring"""
    if not causalif_result['success']:
        print("Cannot visualize failed Causalif analysis")
        return None

    # Import here to avoid circular imports
    from .engine import CausalifEngine
    
    # Create a simple engine for visualization
    max_degrees = causalif_result.get('max_degrees_used', 5)
    max_parallel_queries = causalif_result.get('max_parallel_queries_used', 50)
    viz_engine = CausalifEngine(
        model=None, 
        dataframe=pd.DataFrame({'ftg': [1, 2, 3], 'week': [30, 31, 32]}), 
        max_degrees=max_degrees, 
        max_parallel_queries=max_parallel_queries
    )

    # Reconstruct graph from results
    causal_graph = nx.DiGraph()
    causal_graph.add_nodes_from(causalif_result['causal_graph']['nodes'])
    causal_graph.add_edges_from(causalif_result['causal_graph']['edges'])

    # Visualize with target factor for degree coloring
    target_factor = causalif_result.get('target_factor')
    rag_status = "with RAG" if causalif_result.get('rag_support_enabled') else "no RAG"
    return viz_engine.visualize_graph(
        causal_graph, 
        f"Causalif Results ({rag_status}, Max {max_degrees} degrees, {max_parallel_queries} parallel)", 
        target_factor
    )