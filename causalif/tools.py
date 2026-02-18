# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain tools for Causalif"""

import re
from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx
from langchain_core.tools import tool
from langchain_aws import ChatBedrock

from .engine import CausalifEngine

# Global Causalif engine instance
_global_causalif_engine = None

def set_causalif_engine(model, retriever_tool=None, dataframe=None, factors=None, domains=None, max_degrees: int = 5, max_parallel_queries: int = 50):
    """Set the global Causalif engine instance with complete RAG support"""
    global _global_causalif_engine
    _global_causalif_engine = CausalifEngine(
        model=model,
        retriever_tool=retriever_tool,
        dataframe=dataframe,
        factors=factors,
        domains=domains,
        k_documents=3,
        max_degrees=max_degrees,
        max_parallel_queries=max_parallel_queries
    )
    print(f"Causalif engine configured with max_degrees={max_degrees}, max_parallel_queries={max_parallel_queries}")
    print(f"RAG retriever available: {retriever_tool is not None}")

def extract_factors_from_query(query: str, available_columns: List[str]) -> Tuple[str, List[str]]:
    """Extract target factor and related factors from query"""
    
    patterns = [
        r"why (?:is|are) ([\w_]+) (?:so )?(?:low|high|poor|bad|good)",
        r"what (?:causes|affects|influences) ([\w_]+)",
        r"([\w_]+) (?:is|are) (?:too )?(?:low|high)",
        r"analyze (?:the )?(?:causes (?:of|for) )?([\w_]+)",
        r"dependencies (?:of|for) ([\w_]+)",
        r"factors (?:affecting|influencing) ([\w_]+)"
    ]

    query_lower = query.lower()
    target_factor = None

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            target_factor = match.group(1)
            for col in available_columns:
                if target_factor in col.lower() or col.lower() in target_factor:
                    target_factor = col
                    break
            break

    if not target_factor:
        for col in available_columns:
            if col.lower() in query_lower:
                target_factor = col
                break

    if not target_factor:
        metric_terms = ['intake', 'volume', 'attainment', 'new_intake', 'buffer', 'performance', 'efficiency', 'total', 'ftg']
        for term in metric_terms:
            for col in available_columns:
                if term in col.lower():
                    target_factor = col
                    break
            if target_factor:
                break

    if not target_factor:
        target_factor = available_columns[0] if available_columns else "ftg"

    related_factors = [col for col in available_columns if col != target_factor][:12]
    return target_factor, related_factors

@tool
def causalif_tool(query: str) -> Dict:
    """
    Causalif (Language-Augmented Causal Reasoning) analysis with parallel LLM queries and complete RAG support.
    
    This tool implements the complete Causalif algorithm with parallelization and full knowledge base support:
    1. Background Knowledge Base (BG) processing using LLM background knowledge
    2. Document Knowledge Base (DOC) processing using RAG retrieval
    3. Parallel Edge Existence Verification using batched LLM queries
    4. Parallel Causal Orientation using batched LLM queries  
    5. Degree-limited analysis to focus on relationships within max_degrees of separation
    6. Interactive visualization showing degree-based coloring and filtering

    Args:
        query (str): A natural language query asking about why a factor is high/low or 
                    requesting causal analysis

    Returns:
        Dict: Same structure as original Causalif tool but with faster execution via parallelization
              and complete RAG support
        
    Note: Use set_causalif_engine() to configure the engine with retriever_tool before using this tool.
    """
    
    try:
        global _global_causalif_engine
        
        if _global_causalif_engine is None:
            # Create default engine if none configured
            print("Warning: No Causalif engine configured. Creating default engine...")
            _global_causalif_engine = CausalifEngine(
                model=None,
                retriever_tool=None,
                dataframe=pd.DataFrame({'ftg': [1, 2, 3], 'week': [30, 31, 32], 'country': ['AT', 'DE', 'FR']}),
                k_documents=3,
                max_degrees=3,
                max_parallel_queries=50
            )
        
        # Use the configured engine
        causalif_engine = _global_causalif_engine
        max_degrees = causalif_engine.max_degrees
        max_parallel_queries = causalif_engine.max_parallel_queries
        
        # Get available columns from dataframe
        if causalif_engine.dataframe is not None:
            available_columns = list(causalif_engine.dataframe.columns)
        elif causalif_engine.factors is not None:
            available_columns = causalif_engine.factors
        else:
            available_columns = ['life', 'peace', 'sleep', 'good_food']
        
        # Extract factors from query
        target_factor, related_factors = extract_factors_from_query(query, available_columns)
        
        print(f"Target factor: {target_factor}")
        print(f"Related factors: {related_factors}")
        print(f"Maximum degrees of separation: {max_degrees}")
        print(f"Maximum parallel queries: {max_parallel_queries}")
        print(f"RAG retriever available: {causalif_engine.retriever_tool is not None}")
        
        # Select subset of factors for analysis
        analysis_factors = [target_factor] + related_factors[:8]
        
        if causalif_engine.domains is not None:
            domains = causalif_engine.domains
        else:
            domains = ['life Style', 'Well Being', 'Food', 'Health']
        
        print(f"Running Causalif analysis with complete RAG support on factors: {analysis_factors}")
        
        # Run complete Causalif algorithm with parallelization and full RAG support
        skeleton_graph, causal_graph = causalif_engine.run_complete_causalif(analysis_factors, domains, target_factor)
        
        # Analyze degrees of separation
        degrees_analysis = causalif_engine.analyze_degrees_of_separation(causal_graph, target_factor)
        
        # Analyze causal relationships focusing on target factor
        causal_relationships = []
        target_influences = []
        target_effects = []
        
        # Get statistical evidence for all relationships
        for edge in causal_graph.edges():
            factor_a, factor_b = edge[0], edge[1]
            correlation_evidence = causalif_engine.get_correlation_evidence(factor_a, factor_b)
            
            # Calculate degree of separation for this relationship
            path_a = causalif_engine.get_relationship_path(causal_graph, target_factor, factor_a)
            path_b = causalif_engine.get_relationship_path(causal_graph, target_factor, factor_b)
            degree_a = len(path_a) - 1 if path_a else float('inf')
            degree_b = len(path_b) - 1 if path_b else float('inf')
            min_degree = min(degree_a, degree_b)
            
            relationship = {
                'cause': factor_a,
                'effect': factor_b,
                'evidence': correlation_evidence,
                'relationship_type': 'causal',
                'discovered_by': 'Causalif_algorithm_with_RAG',
                'degree_from_target': min_degree,
                'path_to_target': path_a if degree_a <= degree_b else path_b
            }
            causal_relationships.append(relationship)
            
            # Track influences on target factor
            if factor_b == target_factor:
                target_influences.append({
                    'influencing_factor': factor_a,
                    'evidence': correlation_evidence,
                    'relationship': relationship,
                    'degree': degree_a
                })
            
            # Track effects of target factor
            if factor_a == target_factor:
                target_effects.append({
                    'affected_factor': factor_b,
                    'evidence': correlation_evidence,
                    'relationship': relationship,
                    'degree': degree_b
                })
        
        # Sort influences by degree (closer relationships first)
        target_influences.sort(key=lambda x: x.get('degree', float('inf')))
        target_effects.sort(key=lambda x: x.get('degree', float('inf')))
        
        # Network summary statistics with degree information
        network_summary = {
            'total_factors': len(causal_graph.nodes()),
            'total_causal_relationships': len(causal_graph.edges()),
            'factors_influencing_target': len(target_influences),
            'factors_affected_by_target': len(target_effects),
            'skeleton_edges': len(skeleton_graph.edges()),
            'causal_edges': len(causal_graph.edges()),
            'edge_removal_rate': 1 - (len(causal_graph.edges()) / max(1, len(skeleton_graph.edges()))),
            'max_degrees_analyzed': max_degrees,
            'max_parallel_queries_used': max_parallel_queries,
            'rag_retriever_used': causalif_engine.retriever_tool is not None,
            'actual_max_degree_found': degrees_analysis.get('max_degree_found', 0),
            'factors_by_degree': degrees_analysis.get('factors_by_degree', {})
        }
        
        # Generate summary
        summary_parts = [
            f"🚀 Causalif Causal Analysis Results with Complete RAG Support for: {target_factor}",
            f"   (Max {max_degrees} degrees, {max_parallel_queries} parallel queries, RAG: {'✅' if causalif_engine.retriever_tool else '❌'})",
            f"\n📊 Network Structure:",
            f"   • Total factors analyzed: {network_summary['total_factors']}",
            f"   • Causal relationships discovered: {network_summary['total_causal_relationships']}",
            f"   • Factors causing {target_factor}: {network_summary['factors_influencing_target']}",
            f"   • Factors affected by {target_factor}: {network_summary['factors_affected_by_target']}",
            f"   • Maximum degree found: {network_summary['actual_max_degree_found']}/{max_degrees}",
        ]
        
        summary = "\n".join(summary_parts)
        
        return {
            'target_factor': target_factor,
            'related_factors': related_factors,
            'skeleton_graph': {
                'nodes': list(skeleton_graph.nodes()),
                'edges': list(skeleton_graph.edges())
            },
            'causal_graph': {
                'nodes': list(causal_graph.nodes()),
                'edges': list(causal_graph.edges())
            },
            'degrees_analysis': degrees_analysis,
            'causal_relationships': causal_relationships,
            'strongest_causal_influences': target_influences,
            'network_summary': network_summary,
            'max_degrees_used': max_degrees,
            'max_parallel_queries_used': max_parallel_queries,
            'rag_support_enabled': causalif_engine.retriever_tool is not None,
            'causalif_enhanced': True,
            'summary': summary,
            'query': query,
            'success': True
        }
        
    except Exception as e:
        return {
            'target_factor': None,
            'related_factors': [],
            'skeleton_graph': {'nodes': [], 'edges': []},
            'causal_graph': {'nodes': [], 'edges': []},
            'degrees_analysis': {},
            'causal_relationships': [],
            'strongest_causal_influences': [],
            'network_summary': {},
            'max_degrees_used': 3,
            'max_parallel_queries_used': 50,
            'rag_support_enabled': False,
            'causalif_enhanced': True,
            'summary': f"Causalif Analysis Failed: {str(e)}",
            'query': query,
            'success': False
        }