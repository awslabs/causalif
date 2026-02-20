# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain tool wrappers and helper functions for CausalIF"""

from typing import List, Tuple, Dict
import re
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from langchain_core.tools import tool
import streamlit as st

from .engine import CausalIFEngine
from .visualization import visualize_causalif_results
from .prompts import format_causal_graph_for_llm, generate_llm_interpretation

# Global CausalIF engine instance
_global_causalif_engine = None


def set_causalif_engine(model, retriever_tool=None, dataframe=None, max_degrees: int = None, max_parallel_queries: int = 50,
                    excluded_target_columns: List[str] = None, excluded_related_columns: List[str] = None,
                    related_factors: List[str] = None, selected_dataframe_columns: List[str] = None,
                    enable_causal_estimate: bool = None):
    """
    Set the global CausalIF engine instance with Bayesian causal inference.
    
    Args:
        model: LLM model for CausalIF queries
        retriever_tool: RAG retriever for document knowledge
        dataframe: Observational data for Bayesian inference
        max_degrees: Maximum degrees of separation (None = no filtering, shows entire graph)
        max_parallel_queries: Maximum parallel LLM queries
        excluded_target_columns: List of column names to exclude from target factor selection
        excluded_related_columns: List of column names to exclude from related factors
        related_factors: List of factors to append with dataframe columns for CausalIF 1 analysis
        selected_dataframe_columns: List of specific column names to select from dataframe (None = use all columns)
                                    This will filter the dataframe AND be used for factor list
        enable_causal_estimate: If True, fit CPDs and enable causal inference methods (default: False)
    """
    global _global_causalif_engine
    
    # Filter the dataframe if selected_dataframe_columns is provided
    filtered_dataframe = dataframe
    dataframe_columns_to_use = []
    
    if dataframe is not None and hasattr(dataframe, 'columns'):
        if selected_dataframe_columns is not None:
            # Filter to only the specified columns that exist in the dataframe
            valid_columns = [col for col in selected_dataframe_columns if col in dataframe.columns]
            
            if valid_columns:
                # Create filtered dataframe with only selected columns
                filtered_dataframe = dataframe[valid_columns].copy()
                dataframe_columns_to_use = valid_columns
                print(f"✓ Filtered dataframe to {len(valid_columns)} selected columns")
            else:
                print(f"⚠️  Warning: None of the selected columns exist in the dataframe")
                filtered_dataframe = dataframe
                dataframe_columns_to_use = list(dataframe.columns)
            
            # Warn about columns that don't exist
            missing_cols = [col for col in selected_dataframe_columns if col not in dataframe.columns]
            if missing_cols:
                print(f"⚠️  Warning: The following selected columns are not in the dataframe: {missing_cols}")
        else:
            # Use all dataframe columns
            dataframe_columns_to_use = list(dataframe.columns)
    
    # Start with provided related_factors (can include factors not in dataframe)
    combined_related_factors = related_factors.copy() if related_factors else []
    
    # Append selected/filtered dataframe columns to the list
    for col in dataframe_columns_to_use:
        if col not in combined_related_factors:
            combined_related_factors.append(col)
    
    # Remove duplicates while preserving order
    combined_related_factors = list(dict.fromkeys(combined_related_factors))
    
    _global_causalif_engine = CausalIFEngine(
        model=model,
        retriever_tool=retriever_tool,
        dataframe=filtered_dataframe,  # Use filtered dataframe
        k_documents=3,
        max_degrees=max_degrees,
        max_parallel_queries=max_parallel_queries,
        excluded_target_columns=excluded_target_columns,
        excluded_related_columns=excluded_related_columns,
        related_factors=combined_related_factors,
        selected_dataframe_columns=selected_dataframe_columns,
        enable_causal_estimate=enable_causal_estimate  # NEW
    )
    print(f"CausalIF engine configured with Bayesian causal inference")
    print(f"max_degrees={max_degrees if max_degrees is not None else 'None (no filtering)'}, max_parallel_queries={max_parallel_queries}")
    print(f"Causal inference: {'ENABLED ✓' if enable_causal_estimate else 'DISABLED (use enable_causal_estimate=True to enable)'}")
    print(f"RAG retriever available: {retriever_tool is not None}")
    print(f"Dataframe available: {filtered_dataframe is not None}")
    if filtered_dataframe is not None:
        print(f"Dataframe shape: {filtered_dataframe.shape} ({len(filtered_dataframe)} rows, {len(filtered_dataframe.columns)} columns)")
    if excluded_target_columns:
        print(f"Excluded target columns: {excluded_target_columns}")
    if excluded_related_columns:
        print(f"Excluded related columns: {excluded_related_columns}")
    if selected_dataframe_columns:
        print(f"Selected dataframe columns: {len(dataframe_columns_to_use)} of {len(selected_dataframe_columns)} requested")
        print(f"  Columns: {dataframe_columns_to_use}")
    if combined_related_factors:
        print(f"Related factors for CausalIF 1 analysis: {len(combined_related_factors)} factors")
        print(f"  (includes {len(related_factors) if related_factors else 0} provided + {len(dataframe_columns_to_use)} from dataframe)")



def extract_factors_from_query(query: str, available_columns: List[str], 
                              excluded_target_columns: List[str] = None,
                              excluded_related_columns: List[str] = None) -> Tuple[str, List[str]]:
    """
    Extract target factor and related factors from query.
    
    Args:
        query: User query string
        available_columns: List of available column names from dataframe
        excluded_target_columns: Columns to exclude from target factor selection
        excluded_related_columns: Columns to exclude from related factors
        
    Returns:
        Tuple of (target_factor, related_factors)
    """
    
    # Default exclusions if not provided
    if excluded_target_columns is None:
        # Columns to exclude from target factor selection (dimensional/grouping columns)
        excluded_target_columns = ['week', 'country', 'destination_country', 'station', 'node', 
                                   'dow', 'date', 'sub_wk', 'rep_wk', 'plan_type']
    
    if excluded_related_columns is None:
        # Columns to exclude from related factors (can participate as causal factors but not as target)
        # Excludes temporal and geographic dimensions
        excluded_related_columns = ['country', 'destination_country', 'station', 'node', 
                                    'date', 'week', 'dow', 'sub_wk', 'rep_wk', 'plan_type']
    
    # Convert to sets for faster lookup
    EXCLUDED_COLUMNS = set(col.lower() for col in excluded_target_columns)
    EXCLUDED_RELATED_FACTORS = set(col.lower() for col in excluded_related_columns)
    
    # Old patterns (commented out)
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
    context = None
    
    # STEP 1: Try exact word-by-word matching first (highest priority)
    # Look for exact column names in the query
    for col in available_columns:
        if col.lower() not in EXCLUDED_COLUMNS:
            # Check if the exact column name appears as a word in the query
            # Use word boundaries to avoid partial matches
            import re as re_module
            pattern = r'\b' + re_module.escape(col.lower()) + r'\b'
            if re_module.search(pattern, query_lower):
                target_factor = col
                print(f"Exact match found: '{col}' in query")
                break
    
    # If exact match found, skip pattern matching
    if target_factor:
        related_factors = [col for col in available_columns 
                          if col != target_factor and col.lower() not in EXCLUDED_RELATED_FACTORS][:12]
        return target_factor, related_factors

    # STEP 2: Pattern matching (if no exact match)
    import re
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            # Extract metric and context from capture groups
            groups = match.groups()
            # Filter out None values from groups
            non_none_groups = [g for g in groups if g is not None]
            
            if len(non_none_groups) >= 2:
                # Pattern with context (e.g., "intake in pva")
                target_factor = non_none_groups[0]
                context = non_none_groups[1]
            elif len(non_none_groups) == 1:
                # Pattern without context (e.g., "ftg")
                target_factor = non_none_groups[0]
                context = None
            break
    
    # If target_factor was extracted but doesn't exist in available_columns,
    # try to find a matching column based on metric and context
    if target_factor and target_factor not in available_columns:
        # Try to find columns containing the target_factor and context
        # Exclude dimensional/grouping columns
        matching_cols = [col for col in available_columns 
                        if target_factor in col.lower() and col.lower() not in EXCLUDED_COLUMNS]
        
        if context:
            # Filter by context (pva or uvp)
            context_cols = [col for col in matching_cols if context in col.lower()]
            if context_cols:
                matching_cols = context_cols
                print(f"Filtered columns by context '{context}': {context_cols}")
        
        if matching_cols:
            # Prefer columns with "baseline" or "actuals" for metrics
            baseline_cols = [col for col in matching_cols if 'baseline' in col.lower()]
            actual_cols = [col for col in matching_cols if 'actual' in col.lower()]
            
            if baseline_cols:
                target_factor = baseline_cols[0]
                context_str = f" in {context}" if context else ""
                print(f"Mapped '{non_none_groups[0]}{context_str}' → '{target_factor}' (baseline metric)")
            elif actual_cols:
                target_factor = actual_cols[0]
                context_str = f" in {context}" if context else ""
                print(f"Mapped '{non_none_groups[0]}{context_str}' → '{target_factor}' (actuals metric)")
            else:
                target_factor = matching_cols[0]
                context_str = f" in {context}" if context else ""
                print(f"Mapped '{non_none_groups[0]}{context_str}' → '{target_factor}'")
        else:
            # No matching column found, reset target_factor
            context_str = f" in {context}" if context else ""
            print(f"Warning: Extracted '{target_factor}{context_str}' not found in dataframe columns")
            target_factor = None
        
    if not target_factor:
        # Exclude dimensional columns from fallback search
        for col in available_columns:
            if col.lower() in query_lower and col.lower() not in EXCLUDED_COLUMNS:
                target_factor = col
                break
            
    if not target_factor:
        metric_terms = ['intake', 'volume', 'attainment', 'buffer', 'new_intake', 'performance', 'efficiency', 'total', 'ftg']
        for term in metric_terms:
            for col in available_columns:
                if term in col.lower() and col.lower() not in EXCLUDED_COLUMNS:
                    target_factor = col
                    break
            if target_factor:
                break
            
    if not target_factor:
        # Last resort: pick first non-excluded column
        non_excluded_cols = [col for col in available_columns if col.lower() not in EXCLUDED_COLUMNS]
        target_factor = non_excluded_cols[0] if non_excluded_cols else (available_columns[0] if available_columns else "ftg")

    related_factors = [col for col in available_columns 
                      if col != target_factor and col.lower() not in EXCLUDED_RELATED_FACTORS][:12]
    return target_factor, related_factors




def causalif(query: str) -> Dict:
    """
    CausalIF (Language-Augmented Causal Reasoning) analysis with Bayesian causal inference.
    
    This tool implements the CausalIF algorithm with Bayesian structure learning for causal orientation:
    1. Background Knowledge Base (BG) processing using LLM background knowledge
    2. Document Knowledge Base (DOC) processing using RAG retrieval
    3. Edge Existence Verification using batched LLM queries
    4. Bayesian Causal Orientation using pgmpy structure learning
    5. Degree-limited analysis to focus on relationships within max_degrees of separation
    6. Interactive visualization showing degree-based coloring and filtering

    Args:
        query (str): A natural language query asking about why a factor is high/low or 
                    requesting causal analysis

    Returns:
        Dict: Analysis results with causal graph and insights
        
    Note: Use set_causalif_engine() to configure the engine before using this tool.
    """
    
    try:
        global _global_causalif_engine
        
        if _global_causalif_engine is None:
            print("Warning: No CausalIF engine configured. Creating default engine...")
            _global_causalif_engine = CausalIFEngine(
                model=None,
                retriever_tool=None,
                dataframe=pd.DataFrame({'ftg': [1, 2, 3], 'week': [30, 31, 32], 'country': ['AT', 'DE', 'FR']}),
                k_documents=3,
                max_degrees=3,
                max_parallel_queries=50
            )
        
        causalif_engine = _global_causalif_engine
        max_degrees = causalif_engine.max_degrees
        max_parallel_queries = causalif_engine.max_parallel_queries
        
        # Check if dataframe is available - required for causal analysis
        if causalif_engine.dataframe is None:
            raise ValueError(
                "Dataframe is not available. CausalIF causal analysis requires observational data.\n"
                "Please configure the CausalIF engine with a dataframe using:\n"
                "  set_causalif_engine(model=..., dataframe=your_dataframe, ...)\n"
                "The dataframe should contain the factors you want to analyze."
            )
        
        # Get available columns from dataframe
        if hasattr(causalif_engine.dataframe, "columns"):
            available_columns = list(causalif_engine.dataframe.columns)
        else:
            available_columns = list(causalif_engine.dataframe)

        # Create combined list: related_factors + dataframe columns
        # This allows target factor to be chosen from the full list (including related_factors not in dataframe)
        combined_factors = []
        if causalif_engine.related_factors:
            combined_factors.extend(causalif_engine.related_factors)
        # Add dataframe columns that aren't already in the list
        for col in available_columns:
            if col not in combined_factors:
                combined_factors.append(col)
        
        # Remove duplicates while preserving order
        combined_factors = list(dict.fromkeys(combined_factors))

        # Extract target factor from query using the combined list
        target_factor, _ = extract_factors_from_query(
            query, 
            combined_factors,  # Use combined list instead of just available_columns
            excluded_target_columns=causalif_engine.excluded_target_columns,
            excluded_related_columns=causalif_engine.excluded_related_columns
        )
        
        # For CausalIF 1 analysis, use all factors from combined list except target
        analysis_factors_all = [f for f in combined_factors if f != target_factor]
        
        # Remove duplicates
        analysis_factors_all = list(dict.fromkeys(analysis_factors_all))
        
        print(f"Target factor: {target_factor}")
        print(f"Analysis factors for CausalIF 1: {len(analysis_factors_all)} factors")
        print(f"  Combined list includes: {len(causalif_engine.related_factors) if causalif_engine.related_factors else 0} related_factors + {len(available_columns)} dataframe columns")
        print(f"  Factors: {analysis_factors_all}")
        print(f"Maximum degrees of separation: {max_degrees}")
        print(f"Maximum parallel queries: {max_parallel_queries}")
        print(f"RAG retriever available: {causalif_engine.retriever_tool is not None}")
        print(f"Bayesian causal inference: ENABLED")
        
        # Use combined factors for CausalIF 1 analysis
        analysis_factors = [target_factor] + analysis_factors_all
        domains = ['supply_chain', 'logistics', 'operations', 'performance_metrics']
        
        print(f"\nRunning CausalIF analysis on {len(analysis_factors)} total factors")
        
        skeleton_graph, causal_graph = causalif_engine.run_complete_causalif(analysis_factors, domains, target_factor)
        
        degrees_analysis = causalif_engine.analyze_degrees_of_separation(causal_graph, target_factor)
        
        causal_relationships = []
        target_influences = []
        target_effects = []
        
        for edge in causal_graph.edges():
            factor_a, factor_b = edge[0], edge[1]
            
            # Get edge strength if available
            edge_strength = causal_graph[factor_a][factor_b].get('strength', None) if causal_graph.has_edge(factor_a, factor_b) else None
            
            path_a = causalif_engine.get_relationship_path(causal_graph, target_factor, factor_a)
            path_b = causalif_engine.get_relationship_path(causal_graph, target_factor, factor_b)
            degree_a = len(path_a) - 1 if path_a else float('inf')
            degree_b = len(path_b) - 1 if path_b else float('inf')
            min_degree = min(degree_a, degree_b)
            
            # Original CausalIF: Evidence comes from LLM reasoning and Bayesian inference, not correlation
            evidence_description = f"Causal relationship discovered by CausalIF algorithm"
            if edge_strength is not None:
                evidence_description += f" (strength: {edge_strength:.3f})"
            
            relationship = {
                'cause': factor_a,
                'effect': factor_b,
                'evidence': evidence_description,
                'causal_strength': edge_strength,  # Added: Bayesian edge strength
                'relationship_type': 'causal',
                'discovered_by': 'CausalIF_with_Bayesian_inference',
                'degree_from_target': min_degree,
                'path_to_target': path_a if degree_a <= degree_b else path_b
            }
            causal_relationships.append(relationship)
            
            if factor_b == target_factor:
                target_influences.append({
                    'influencing_factor': factor_a,
                    'evidence': evidence_description,
                    'causal_strength': edge_strength, 
                    'relationship': relationship,
                    'degree': degree_a
                })
            
            if factor_a == target_factor:
                target_effects.append({
                    'affected_factor': factor_b,
                    'evidence': evidence_description,
                    'causal_strength': edge_strength, 
                    'relationship': relationship,
                    'degree': degree_b
                })
        
        target_influences.sort(key=lambda x: x.get('degree', float('inf')))
        target_effects.sort(key=lambda x: x.get('degree', float('inf')))
        
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
            'bayesian_inference_used': True,
            'actual_max_degree_found': degrees_analysis.get('max_degree_found', 0),
            'factors_by_degree': degrees_analysis.get('factors_by_degree', {})
        }
        
        causalif_insights = []
        causalif_insights.append(f"✓ Bayesian Framework: PRIOR (skeleton) → POSTERIOR (directed graph)")
        causalif_insights.append(f"✓ PRIOR: {len(skeleton_graph.edges())} associations from LLM + RAG")
        causalif_insights.append(f"✓ POSTERIOR: {len(causal_graph.edges())} causal directions from Bayesian inference")
        causalif_insights.append(f"✓ Analyzed {len(analysis_factors)} factors with {max_parallel_queries} parallel queries")
        
        if causalif_engine.retriever_tool:
            causalif_insights.append(f"✓ RAG retrieval enabled for domain knowledge in PRIOR")
        
        if causalif_engine.dataframe is not None:
            causalif_insights.append(f"✓ Observational data ({len(causalif_engine.dataframe)} samples) used for POSTERIOR")
        
        if target_influences:
            causalif_insights.append(f"✓ Found {len(target_influences)} factors causally influencing {target_factor}")
        
        recommendations = []
        if target_influences:
            top_influence = target_influences[0]
            recommendations.append(f"Primary driver: {top_influence['influencing_factor']} → {target_factor}")
        
        algorithm_details = {
            'method': 'Bayesian CausalIF (Prior → Posterior)',
            'prior_method': 'LLM + RAG for edge existence (CausalIF 1)',
            'posterior_method': 'Bayesian structure learning with BDeu score (CausalIF 2))',
            'orientation_algorithm': 'Hill Climbing constrained by prior skeleton',
            'bayesian_score': 'BDeu (Bayesian Dirichlet equivalent uniform)',
            'prior_constraint': 'Skeleton graph from CausalIF 1',
            'max_degrees': max_degrees,
            'parallel_queries': max_parallel_queries
        }
        
        summary = f"✅ Bayesian CausalIF Causal Analysis Complete\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"🎯 Target Factor: {target_factor}\n"
        summary += f"📊 PRIOR (Associations): {len(skeleton_graph.edges())} edges\n"
        summary += f"🔗 POSTERIOR (Causal): {len(causal_graph.edges())} directed edges\n"
        summary += f"⚡ Influencing Factors: {len(target_influences)}\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Add top causal influences with strengths
        if target_influences:
            summary += f"\n🔍 BAYESIAN CAUSAL ANALYSIS RESULTS:\n"
            summary += f"The following factors were identified as causal drivers\n"
            summary += f"through Bayesian structure learning:\n\n"
            
            # Sort by causal strength (descending)
            sorted_influences = sorted(
                [inf for inf in target_influences if inf.get('causal_strength') is not None],
                key=lambda x: x.get('causal_strength', 0),
                reverse=True
            )
            
            if sorted_influences:
                for i, inf in enumerate(sorted_influences, 1):
                    strength = inf.get('causal_strength', 0)
                    strength_bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                    strength_label = "STRONG" if strength > 0.7 else "MODERATE" if strength > 0.4 else "WEAK" if strength > 0.2 else "VERY WEAK"
                    summary += f"{i}. {inf['influencing_factor']} → {target_factor}\n"
                    summary += f"   Causal Strength: |{strength_bar}| {strength:.3f} ({strength_label})\n"
                    summary += f"   {inf.get('evidence', 'No statistical evidence available')}\n\n"
            else:
                summary += "⚠️ No causal influences with quantified strengths were found.\n"
                summary += "This may indicate:\n"
                summary += "- The Bayesian method rejected weak associations\n"
                summary += "- Data quality issues\n"
                summary += "- The target factor may be independent of analyzed factors\n\n"
        else:
            summary += f"\n⚠️ No direct causal influences found for {target_factor}\n"
            summary += f"The Bayesian analysis did not identify any factors that\n"
            if max_degrees is not None:
                summary += f"causally influence {target_factor} within {max_degrees} degree(s).\n\n"
            else:
                summary += f"causally influence {target_factor}.\n\n"
        
        # Add interpretation guidance
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"📌 INTERPRETATION GUIDE:\n"
        summary += f"• Use ONLY the factors listed above in your analysis\n"
        summary += f"• Causal strength indicates the LLM-based prior confidence\n"
        summary += f"• Higher strength = stronger domain knowledge support\n"
        summary += f"• Bayesian method determines causal directions from data\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # DEBUG: Show results for different max_degrees (1-5)
        print(f"\n{'='*80}")
        print(f"DEGREE ANALYSIS: Showing results for max_degrees 1 through 5")
        print(f"{'='*80}")
        
        for test_degree in range(1, 6):
            print(f"\n--- Max Degrees = {test_degree} ---")
            
            # Filter the causal graph by this degree
            test_filtered_graph = causalif_engine.filter_graph_by_degrees(causal_graph, target_factor, test_degree)
            
            # Count edges and nodes
            num_nodes = len(test_filtered_graph.nodes())
            num_edges = len(test_filtered_graph.edges())
            
            # Find influences at this degree
            test_influences = []
            for u, v in test_filtered_graph.edges():
                if v == target_factor:
                    edge_data = test_filtered_graph[u][v]
                    test_influences.append({
                        'factor': u,
                        'strength': edge_data.get('strength', 0) or 0  # Use 'strength' key from edge data
                    })
            
            print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
            print(f"  Factors influencing {target_factor}: {len(test_influences)}")
            
            if test_influences:
                # Sort by strength
                test_influences.sort(key=lambda x: x.get('strength', 0) or 0, reverse=True)
                for inf in test_influences:
                    strength = inf.get('strength', 0) or 0
                    strength_bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                    print(f"    • {inf['factor']:30s} |{strength_bar}| {strength:.3f}")
            else:
                print(f"    (No direct influences found)")
        
        print(f"{'='*80}\n")
        
        visualization_data = {
            'skeleton': {
                'nodes': list(skeleton_graph.nodes()),
                'edges': list(skeleton_graph.edges())
            },
            'causal': {
                'nodes': list(causal_graph.nodes()),
                'edges': [(u, v, causal_graph[u][v]) for u, v in causal_graph.edges()]  # Include edge attributes
            }
        }
        
        # Generate LLM interpretation of the causal graph
        llm_interpretation = ""
        if causalif_engine.model is not None:
            try:
                llm_interpretation = generate_llm_interpretation(
                    causalif_result={
                        'success': True,
                        'target_factor': target_factor,
                        'causal_relationships': causal_relationships,
                        'strongest_causal_influences': target_influences,
                        'network_summary': network_summary
                    },
                    original_query=query,
                    model=causalif_engine.model
                )
            except Exception as e:
                print(f"Warning: Could not generate LLM interpretation: {e}")
                llm_interpretation = "LLM interpretation unavailable."
        else:
            llm_interpretation = "LLM model not configured. Use set_causalif_engine() to enable interpretation."
        
        return {
            'target_factor': target_factor,
            'related_factors': analysis_factors_all,  # Fixed: use analysis_factors_all instead of undefined related_factors
            'skeleton_graph': visualization_data['skeleton'],
            'causal_graph': visualization_data['causal'],
            'degrees_analysis': degrees_analysis,
            'causal_relationships': causal_relationships,
            'strongest_causal_influences': target_influences,
            'network_summary': network_summary,
            'visualization_data': visualization_data,
            'causalif_insights': causalif_insights,
            'recommendations': recommendations,
            'algorithm_details': algorithm_details,
            'max_degrees_used': max_degrees,
            'max_parallel_queries_used': max_parallel_queries,
            'rag_support_enabled': causalif_engine.retriever_tool is not None,
            'bayesian_inference_enabled': True,
            'summary': summary,
            'llm_interpretation': llm_interpretation,  
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
            'visualization_data': {},
            'causalif_insights': [f"Error in CausalIF analysis: {str(e)}"],
            'recommendations': [],
            'algorithm_details': {'error': str(e)},
            'max_degrees_used': 3,
            'max_parallel_queries_used': 50,
            'rag_support_enabled': False,
            'bayesian_inference_enabled': True,
            'summary': f"❌ CausalIF Analysis Failed: {str(e)}",
            'llm_interpretation': f"Analysis failed: {str(e)}",
            'query': query,
            'success': False
        }



@tool
def causalif_tool(query: str) -> Dict:
    """
    Executes the CausalIF (Causal Inference Framework) pipeline with Bayesian causal inference. 
    Initializes the CausalIF engine with a Bedrock model and retriever, runs the query, 
    and returns structured results including a summary.
    
    Dataframe Selection Logic:
    - If 'ftg' in query → merge df_ftg + df_pva
    - If 'uvp' in query → use df_uvp
    - Otherwise → use df_pva (default)

    Args:
        query (str): The natural language query to process.

    Returns:
        Dict: A dictionary containing the full results and a summary.
    """
    try:
        # Import dataframes from tool_create_df
        import tool_create_df as df_module
        
        # Intelligent dataframe selection based on query content
        query_lower = query.lower()
        
        # Check for 'ftg' first - merge df_ftg and df_pva
        if 'ftg' in query_lower:
            print(f"Query mentions FTG - merging df_ftg and df_pva")
            # Merge df_ftg and df_pva with all columns from both
            if df_module.df_ftg is not None and df_module.df_pva is not None:
                # Find common columns for merging (likely 'week', 'jurisdiction', 'country', etc.)
                common_cols = list(set(df_module.df_ftg.columns) & set(df_module.df_pva.columns))
                if common_cols:
                    print(f"Merging on common columns: {common_cols}")
                    dataframe_to_use = pd.merge(
                        df_module.df_ftg, 
                        df_module.df_pva, 
                        on=common_cols, 
                        how='outer',
                        suffixes=('_ftg', '_pva')
                    )
                    print(f"Merged dataframe shape: {dataframe_to_use.shape}")
                else:
                    # No common columns, concatenate side by side
                    print("No common columns found, concatenating dataframes")
                    dataframe_to_use = pd.concat([df_module.df_ftg, df_module.df_pva], axis=1)
            else:
                print("Warning: df_ftg or df_pva is None, falling back to df_pva")
                dataframe_to_use = df_module.df_pva
        
        # Check for 'uvp' mention
        elif 'uvp' in query_lower:
            print(f"Query mentions UVP - using df_uvp")
            dataframe_to_use = df_module.df_uvp if df_module.df_uvp is not None else df_module.df_pva
        
        # Default case: use df_pva
        else:
            print(f"Default case - using df_pva")
            dataframe_to_use = df_module.df_pva if df_module.df_pva is not None else df_module.df_uvp
        
        # Verify dataframe is not None
        if dataframe_to_use is None:
            print("Warning: Selected dataframe is None, POSTERIOR will not be computed")
        else:
            print(f"Using dataframe with shape: {dataframe_to_use.shape}")
            print(f"All columns ({len(dataframe_to_use.columns)}): {list(dataframe_to_use.columns)}")
        
        result = causalif(query)
        fig = visualize_causalif_results(result)
        if fig:
            # Store visualization in session state to persist across reruns
            if 'causalif_visualizations' not in st.session_state:
                st.session_state.causalif_visualizations = []
            
            # Add timestamp and query info to the visualization
            import datetime
            viz_data = {
                'figure': fig,
                'query': query,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'target_factor': result.get('target_factor', 'Unknown')
            }
            st.session_state.causalif_visualizations.append(viz_data)
            
            # Display all stored visualizations
            st.markdown("---")
            
            # Header with clear button
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown("### 📊 CausalIF Analysis History")
            with col2:
                if st.button("🗑️ Clear History", key="clear_causalif_history"):
                    st.session_state.causalif_visualizations = []
                    st.rerun()
            
            # Display visualizations
            if st.session_state.causalif_visualizations:
                for i, viz in enumerate(st.session_state.causalif_visualizations, 1):
                    with st.expander(f"Analysis {i}: {viz['target_factor']} ({viz['timestamp']})", expanded=(i == len(st.session_state.causalif_visualizations))):
                        st.markdown(f"**Query:** {viz['query']}")
                        st.plotly_chart(viz['figure'], use_container_width=True, key=f"causalif_viz_{i}_{viz['timestamp']}")
            else:
                st.info("No CausalIF analyses in history. Run a query to see visualizations here.")
            
            st.markdown("---")

        return {
            "summary": result['summary']
        }
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        print("Please configure with your actual model, retriever, and data using set_causalif_engine()")
        return {
            "summary": f"Error: {str(e)}"
        }
