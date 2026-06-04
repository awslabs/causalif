# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain tool wrappers and helper functions for CausalIF"""

import logging
import re
import threading
from typing import List, Dict, Optional
from langchain_core.tools import tool

from .engine import CausalIFEngine
from .prompts import generate_llm_interpretation

logger = logging.getLogger(__name__)

# Thread-safe global CausalIF engine instance
_engine_lock = threading.Lock()
_global_causalif_engine: Optional[CausalIFEngine] = None


def _get_engine() -> Optional[CausalIFEngine]:
    """Thread-safe accessor for the global CausalIF engine."""
    with _engine_lock:
        return _global_causalif_engine


def set_causalif_engine(model, retriever_tool=None, retriever=None, dataframe=None, max_token_limit: int = None,
                    max_degrees: int = None, max_parallel_queries: int = None,
                    excluded_target_columns: List[str] = None, excluded_related_columns: List[str] = None,
                    related_factors: List[str] = None, selected_dataframe_columns: List[str] = None,
                    enable_causal_estimate: bool = None, domains: List[str] = None,
                    bootstrap_iterations: int = None, bootstrap_threshold: float = None,
                    factor_descriptions: str = None):
    """
    Set the global CausalIF engine instance with Bayesian causal inference.
    
    Args:
        model: LLM model for CausalIF queries
        retriever_tool: RAG retriever tool (LangChain tool wrapper) for document knowledge
        retriever: Raw retriever instance (e.g. AmazonKnowledgeBasesRetriever) for metadata access.
                   When provided, enables tracking of unique source document counts per edge.
                   If only retriever_tool is provided, document counts will be unavailable.
        dataframe: Observational data for Bayesian inference
        max_token_limit: Maximum token count per document before summarization (default: 150000).
                         Documents exceeding this limit are summarized via LLM instead of truncated.
        max_degrees: Maximum degrees of separation (None = no filtering, shows entire graph)
        max_parallel_queries: Maximum parallel LLM queries
        excluded_target_columns: List of column names to exclude from target factor selection
        excluded_related_columns: List of column names to exclude from related factors
        related_factors: List of factors to append with dataframe columns for CausalIF 1 analysis
        selected_dataframe_columns: List of specific column names to select from dataframe (None = use all columns)
                                    This will filter the dataframe AND be used for factor list
        enable_causal_estimate: If True, fit CPDs and enable causal inference methods (default: False)
        domains: List of domain names for context (default: ['supply_chain', 'logistics', 'operations', 'performance_metrics'])
    """
    global _global_causalif_engine
    
    # Filter the dataframe if selected_dataframe_columns is provided
    filtered_dataframe = dataframe
    dataframe_columns_to_use = []
    
    if dataframe is not None and hasattr(dataframe, 'columns'):
        if selected_dataframe_columns is not None:
            valid_columns = [col for col in selected_dataframe_columns if col in dataframe.columns]
            
            if valid_columns:
                filtered_dataframe = dataframe[valid_columns].copy()
                dataframe_columns_to_use = valid_columns
                logger.info(f"✓ Filtered dataframe to {len(valid_columns)} selected columns")
            else:
                logger.warning("None of the selected columns exist in the dataframe")
                filtered_dataframe = dataframe
                dataframe_columns_to_use = list(dataframe.columns)
            
            missing_cols = [col for col in selected_dataframe_columns if col not in dataframe.columns]
            if missing_cols:
                logger.warning(f"The following selected columns are not in the dataframe: {missing_cols}")
        else:
            dataframe_columns_to_use = list(dataframe.columns)
    
    # Start with provided related_factors (can include factors not in dataframe)
    combined_related_factors = related_factors.copy() if related_factors else []
    
    # Append selected/filtered dataframe columns to the list
    for col in dataframe_columns_to_use:
        if col not in combined_related_factors:
            combined_related_factors.append(col)
    
    # Remove duplicates while preserving order
    combined_related_factors = list(dict.fromkeys(combined_related_factors))
    
    # Build kwargs, only passing non-None values so CausalIFEngine defaults are preserved
    engine_kwargs = dict(
        model=model,
        retriever_tool=retriever_tool,
        retriever=retriever,
        dataframe=filtered_dataframe,
        excluded_target_columns=excluded_target_columns,
        excluded_related_columns=excluded_related_columns,
        related_factors=combined_related_factors,
        selected_dataframe_columns=selected_dataframe_columns,
    )
    if max_token_limit is not None:
        engine_kwargs['max_token_limit'] = max_token_limit
    if max_degrees is not None:
        engine_kwargs['max_degrees'] = max_degrees
    if max_parallel_queries is not None:
        engine_kwargs['max_parallel_queries'] = max_parallel_queries
    if enable_causal_estimate is not None:
        engine_kwargs['enable_causal_estimate'] = enable_causal_estimate
    if domains is not None:
        engine_kwargs['domains'] = domains
    if bootstrap_iterations is not None:
        engine_kwargs['bootstrap_iterations'] = bootstrap_iterations
    if bootstrap_threshold is not None:
        engine_kwargs['bootstrap_threshold'] = bootstrap_threshold
    if factor_descriptions is not None:
        # If it's an S3 URI, fetch the content
        if factor_descriptions.startswith('s3://'):
            import boto3
            parts = factor_descriptions[5:].split('/', 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=bucket, Key=key)
            factor_descriptions = obj['Body'].read().decode('utf-8')
            logger.info(f"✓ Loaded factor descriptions from s3://{bucket}/{key}")
        engine_kwargs['factor_descriptions'] = factor_descriptions
    
    with _engine_lock:
        _global_causalif_engine = CausalIFEngine(**engine_kwargs)

    # Warn loudly if factor_descriptions not provided
    if factor_descriptions is None:
        logger.warning("")
        logger.warning("⚠️" * 20)
        logger.warning("⚠️  WARNING: factor_descriptions NOT PROVIDED")
        logger.warning("⚠️  Causal directions may be MISREPRESENTED because the LLM")
        logger.warning("⚠️  cannot understand abbreviated column names without definitions.")
        logger.warning("⚠️  Pass factor_descriptions='- col_name: description\\n...' to fix this.")
        logger.warning("⚠️" * 20)
        logger.warning("")

    logger.info("CausalIF engine configured with Bayesian causal inference")
    logger.info(f"max_token_limit={max_token_limit if max_token_limit is not None else '150000 (default)'}, "
                f"max_degrees={max_degrees if max_degrees is not None else 'None (no filtering)'}, "
                f"max_parallel_queries={max_parallel_queries}")
    logger.info(f"Causal inference: {'ENABLED ✓' if enable_causal_estimate else 'DISABLED (use enable_causal_estimate=True to enable)'}")
    logger.info(f"RAG retriever available: {retriever_tool is not None or retriever is not None}")
    logger.info(f"RAG raw retriever (metadata access): {retriever is not None}")
    logger.info(f"Dataframe available: {filtered_dataframe is not None}")
    if filtered_dataframe is not None:
        logger.info(f"Dataframe shape: {filtered_dataframe.shape} ({len(filtered_dataframe)} rows, {len(filtered_dataframe.columns)} columns)")
    if excluded_target_columns:
        logger.info(f"Excluded target columns: {excluded_target_columns}")
    if excluded_related_columns:
        logger.info(f"Excluded related columns: {excluded_related_columns}")
    if selected_dataframe_columns:
        logger.info(f"Selected dataframe columns: {len(dataframe_columns_to_use)} of {len(selected_dataframe_columns)} requested")
        logger.info(f"  Columns: {dataframe_columns_to_use}")
    if combined_related_factors:
        logger.info(f"Related factors for CausalIF 1 analysis: {len(combined_related_factors)} factors")
        logger.info(f"  (includes {len(related_factors) if related_factors else 0} provided + {len(dataframe_columns_to_use)} from dataframe)")
    if domains:
        logger.info(f"Domains: {domains}")



def extract_factors_from_query(query: str, available_columns: List[str], 
                              excluded_target_columns: List[str] = None) -> str:
    """
    Extract target factor from query by matching against available columns.
    
    Args:
        query: User query string
        available_columns: List of available column names from dataframe
        excluded_target_columns: Columns to exclude from target factor selection
        
    Returns:
        str: The extracted target factor
    """
    
    EXCLUDED_COLUMNS = set(col.lower() for col in excluded_target_columns) if excluded_target_columns else set()
    
    # Patterns with optional context support (e.g., "revenue in pva")
    patterns = [
         r"why (?:is|are) ([\w_]+) (?:so )?(?:low|high|poor|bad|good)(?: in ([\w_]+))?",
         r"what (?:causes|affects|influences) ([\w_]+)(?: in ([\w_]+))?",
         r"([\w_]+) (?:is|are) (?:too )?(?:low|high)(?: in ([\w_]+))?",
         r"analyze (?:the )?(?:causes (?:of|for) )?([\w_]+)(?: in ([\w_]+))?",
         r"dependencies (?:of|for) ([\w_]+)(?: in ([\w_]+))?",
         r"factors (?:affecting|influencing) ([\w_]+)(?: in ([\w_]+))?"
    ]

    query_lower = query.lower()
    target_factor = None
    context = None
    # Track the raw extracted token for logging in the fallback column-matching block
    extracted_token = None
    
    # STEP 1: Try exact word-by-word matching first (highest priority)
    for col in available_columns:
        if col.lower() not in EXCLUDED_COLUMNS:
            pattern = r'\b' + re.escape(col.lower()) + r'\b'
            if re.search(pattern, query_lower):
                target_factor = col
                logger.info(f"Exact match found: '{col}' in query")
                break
    
    if target_factor:
        return target_factor

    # STEP 2: Pattern matching (if no exact match)
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            non_none_groups = [g for g in groups if g is not None]
            
            if len(non_none_groups) >= 2:
                target_factor = non_none_groups[0]
                context = non_none_groups[1]
            elif len(non_none_groups) == 1:
                target_factor = non_none_groups[0]
                context = None
            # Save the raw extracted token for logging
            extracted_token = non_none_groups[0] if non_none_groups else None
            break
    
    # If target_factor was extracted but doesn't exist in available_columns,
    # try to find a matching column based on metric and context
    if target_factor and target_factor not in available_columns:
        matching_cols = [col for col in available_columns 
                        if target_factor in col.lower() and col.lower() not in EXCLUDED_COLUMNS]
        
        if context:
            context_cols = [col for col in matching_cols if context in col.lower()]
            if context_cols:
                matching_cols = context_cols
                logger.info(f"Filtered columns by context '{context}': {context_cols}")
        
        context_str = f" in {context}" if context else ""
        if matching_cols:
            baseline_cols = [col for col in matching_cols if 'baseline' in col.lower()]
            actual_cols = [col for col in matching_cols if 'actual' in col.lower()]
            
            if baseline_cols:
                target_factor = baseline_cols[0]
                logger.info(f"Mapped '{extracted_token}{context_str}' → '{target_factor}' (baseline metric)")
            elif actual_cols:
                target_factor = actual_cols[0]
                logger.info(f"Mapped '{extracted_token}{context_str}' → '{target_factor}' (actuals metric)")
            else:
                target_factor = matching_cols[0]
                logger.info(f"Mapped '{extracted_token}{context_str}' → '{target_factor}'")
        else:
            logger.warning(f"Extracted '{target_factor}{context_str}' not found in dataframe columns")
            target_factor = None
        
    if not target_factor:
        raise ValueError(
            f"Could not identify target factor from query: '{query}'. "
            f"Please specify a valid column name from the background data or supplied dataframe. "
            f"Available columns: {', '.join(available_columns[:10])}{'...' if len(available_columns) > 10 else ''}"
        )

    return target_factor


def parse_intervention_query(query: str, available_columns: List[str]) -> Optional[Dict]:
    """
    Detect if a query is an interventional (do-operator) question and extract
    the target variable and intervention variables.

    Supports multi-word column names (e.g. "shipping cost") via regex alternation
    built from the available columns list.

    Supported natural language patterns:
      - "what happens to Y if X is high"
      - "what happens to Y if we set X to high"
      - "what is the effect on Y if X is low"
      - "what would Y be if X is high and Z is low"
      - "if X is high, what happens to Y"
      - "how does Y change if X is high"
      - "what if X is high, what happens to Y"
      - "effect of setting X to high on Y"

    Args:
        query: Natural language query string.
        available_columns: List of known variable/column names.

    Returns:
        Dict with 'target' and 'do_vars' if interventional, None otherwise.
    """
    q_lower = query.strip().lower()

    intervention_keywords = [
        'what happens', 'what would', 'what if', 'how does', 'how would',
        'effect of setting', 'effect on', 'if we set', 'if we change',
        'impact on', 'impact of setting',
    ]
    if not any(kw in q_lower for kw in intervention_keywords):
        return None

    # Build a regex alternation of column names (longest first to avoid partial matches).
    # re.escape handles spaces and special characters in column names.
    sorted_cols = sorted(available_columns, key=len, reverse=True)
    col_pattern = '|'.join(re.escape(c) for c in sorted_cols)
    col_re = re.compile(col_pattern, re.IGNORECASE)

    value_words = r'(?:high|low|medium|very\s+high|very\s+low|\d+)'

    # --- Pattern group 1: "what happens to Y if X is V [and Z is W]" ---
    m = re.search(
        r'(?:what\s+happens\s+to|what\s+would|how\s+does|how\s+would)\s+'
        r'(.+?)\s+'
        r'(?:be\s+)?(?:if|when)\s+(.+)',
        q_lower
    )
    if m:
        target_raw = m.group(1).strip().rstrip('?')
        conditions_raw = m.group(2).strip().rstrip('?')
        return _build_intervention(target_raw, conditions_raw, col_re, value_words, available_columns)

    # --- Pattern group 2: "if X is V, what happens to Y" ---
    m = re.search(
        r'(?:what\s+if|if)\s+(.+?),?\s*'
        r'(?:what\s+happens\s+to|what\s+(?:is|would\s+be)\s+(?:the\s+)?(?:effect\s+on)?)\s*'
        r'(.+)',
        q_lower
    )
    if m:
        conditions_raw = m.group(1).strip().rstrip('?')
        target_raw = m.group(2).strip().rstrip('?')
        return _build_intervention(target_raw, conditions_raw, col_re, value_words, available_columns)

    # --- Pattern group 3: "effect of setting X to V on Y" ---
    m = re.search(
        r'(?:effect|impact)\s+of\s+setting\s+(.+?)\s+on\s+(.+)',
        q_lower
    )
    if m:
        conditions_raw = m.group(1).strip().rstrip('?')
        target_raw = m.group(2).strip().rstrip('?')
        return _build_intervention(target_raw, conditions_raw, col_re, value_words, available_columns)

    return None


def _build_intervention(
    target_raw: str,
    conditions_raw: str,
    col_re: re.Pattern,
    value_words: str,
    available_columns: List[str],
) -> Optional[Dict]:
    """Helper: resolve target and intervention variables from raw text fragments."""
    target = _match_column(target_raw, col_re, available_columns)
    if not target:
        return None

    do_vars = {}
    parts = re.split(r'\s+and\s+', conditions_raw)
    for part in parts:
        m = re.search(
            r'(.+?)\s+(?:is|to|=|equals?)\s+(' + value_words + r')',
            part.strip()
        )
        if m:
            col_raw = m.group(1).strip()
            val = m.group(2).strip()
            col = _match_column(col_raw, col_re, available_columns)
            if col:
                do_vars[col] = val

    if not do_vars:
        return None

    return {'target': target, 'do_vars': do_vars}


def _match_column(text: str, col_re: re.Pattern, available_columns: List[str]) -> Optional[str]:
    """Find the best matching column name in a text fragment."""
    m = col_re.search(text)
    if m:
        matched = m.group(0)
        for col in available_columns:
            if col.lower() == matched.lower():
                return col
    return None


def _resolve_intervention_values(engine: 'CausalIFEngine', do_vars: Dict[str, str]) -> Dict[str, str]:
    """
    Map descriptive intervention values ('high', 'low', 'medium') to the
    discretized bin labels ('0', '1', '2', ...) used by the fitted model.

    If the causal model has not been fitted yet, returns the raw values as-is
    with a warning — the caller is expected to validate model readiness before
    invoking do-operator queries.
    """
    if engine.causal_model is None:
        logger.warning(
            "Causal model is not fitted. Cannot resolve intervention values to bin labels. "
            "Run a causal analysis query first so the CausalIF pipeline builds the model."
        )
        return do_vars

    resolved = {}
    model_nodes = set(engine.causal_model.nodes())

    for var, val in do_vars.items():
        if var not in model_nodes:
            resolved[var] = val
            continue

        cpd = engine.causal_model.get_cpds(var)
        states = [str(s) for s in cpd.state_names[var]]

        if val in states:
            resolved[var] = val
            continue

        val_lower = val.lower().strip()
        n = len(states)

        if val_lower in ('low', 'very low'):
            resolved[var] = states[0]
        elif val_lower == 'medium':
            resolved[var] = states[n // 2]
        elif val_lower in ('high', 'very high'):
            resolved[var] = states[-1]
        else:
            try:
                idx = int(val_lower)
                if 0 <= idx < n:
                    resolved[var] = states[idx]
                else:
                    resolved[var] = val
            except ValueError:
                resolved[var] = val

    return resolved


def causalif_intervene(query: str) -> Dict:
    """
    Handle an interventional (do-operator) query.

    Requires that the CausalIF pipeline has already been run at least once
    (so the causal model is fitted) and that ``enable_causal_estimate=True``.

    Args:
        query: Natural language interventional question.

    Returns:
        Dict with do-operator results or an error message.
    """
    engine = _get_engine()

    if engine is None:
        return {
            'success': False,
            'error': 'No CausalIF engine configured. Call set_causalif_engine() first.',
        }

    if not engine.enable_causal_estimate:
        return {
            'success': False,
            'error': (
                'Causal inference is disabled. '
                'Set enable_causal_estimate=True in set_causalif_engine() to use interventional queries.'
            ),
        }

    if engine.causal_model is None or engine.causal_inference_engine is None:
        return {
            'success': False,
            'error': (
                'Causal model has not been fitted yet. '
                'Run a causal analysis query first (e.g. "what causes high shipping_cost") '
                'so that the full CausalIF pipeline builds the model.'
            ),
        }

    model_nodes = sorted(engine.causal_model.nodes())

    parsed = parse_intervention_query(query, model_nodes)
    if parsed is None:
        return {
            'success': False,
            'error': (
                f'Could not parse interventional query: "{query}". '
                f'Try a format like: "what happens to Y if X is high". '
                f'Available variables: {model_nodes}'
            ),
        }

    target = parsed['target']
    do_vars = _resolve_intervention_values(engine, parsed['do_vars'])

    logger.info(f"[Interventional Query] target={target}, do={do_vars}")

    try:
        result = engine.do(target=target, do_vars=do_vars)

        if 'error' in result:
            return {
                'success': False,
                'error': result['message'],
                'suggestion': result.get('suggestion', ''),
                'target': target,
                'do_vars': do_vars,
                'summary': f"⚠️ {result['message']}\n💡 {result.get('suggestion', '')}",
            }

        return _build_intervention_summary(result, target, do_vars)

    except Exception as e:
        return {
            'success': False,
            'error': f'do-operator failed: {str(e)}',
            'target': target,
            'do_vars': do_vars,
        }


def _build_intervention_summary(result: Dict, target: str, do_vars: Dict) -> Dict:
    """Build the human-readable summary dict for an interventional query result."""
    do_str = ', '.join(f'{k}={v}' for k, v in do_vars.items())
    summary = f"🔬 Interventional Analysis: P({target} | do({do_str}))\n"
    summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    display_dist = result.get('distribution_with_ranges', result['distribution'])
    raw_dist = result['distribution']
    sorted_keys = sorted(raw_dist.keys(), key=lambda k: -raw_dist[k])

    for state in sorted_keys:
        prob = raw_dist[state]
        bar = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
        marker = ' ◀ most likely' if state == result['most_likely_value'] else ''
        display_key = next((dk for dk in display_dist if dk == state or dk.startswith(f"bin {state} ")), state)
        summary += f"  {target}={display_key}: |{bar}| {prob:.4f}{marker}\n"

    most_likely_display = result.get('most_likely_display', result['most_likely_value'])
    summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    summary += f"  → Most likely outcome: {target} = {most_likely_display} (p={result['max_probability']:.4f})"

    return {
        'success': True,
        'query_type': 'intervention',
        'target': target,
        'do_vars': do_vars,
        'distribution': result['distribution'],
        'distribution_with_ranges': result.get('distribution_with_ranges', {}),
        'most_likely_value': result['most_likely_value'],
        'most_likely_display': most_likely_display,
        'max_probability': result['max_probability'],
        'summary': summary,
    }


# ---------------------------------------------------------------------------
# causalif() helper functions (split from the monolithic function)
# ---------------------------------------------------------------------------

def _validate_engine_and_dataframe(engine: CausalIFEngine) -> List[str]:
    """Validate engine state and return available columns. Raises on failure."""
    if engine.dataframe is None:
        raise ValueError(
            "Dataframe is not available. CausalIF causal analysis requires observational data.\n"
            "Please configure the CausalIF engine with a dataframe using:\n"
            "  set_causalif_engine(model=..., dataframe=your_dataframe, ...)\n"
            "The dataframe should contain the factors you want to analyze."
        )
    if hasattr(engine.dataframe, "columns"):
        return list(engine.dataframe.columns)
    return list(engine.dataframe)


def _build_combined_factors(engine: CausalIFEngine, available_columns: List[str]) -> List[str]:
    """Return the deduplicated factor list from the engine.

    engine.related_factors already contains related_factors + dataframe columns
    (merged in set_causalif_engine), so we just return it directly.
    """
    if engine.related_factors:
        return list(dict.fromkeys(engine.related_factors))
    return list(dict.fromkeys(available_columns))


def _extract_causal_relationships(engine: CausalIFEngine, causal_graph, target_factor: str):
    """Walk causal graph edges and build relationship / influence / effect lists."""
    causal_relationships = []
    target_influences = []
    target_effects = []

    for factor_a, factor_b in causal_graph.edges():
        edge_data = causal_graph[factor_a][factor_b] if causal_graph.has_edge(factor_a, factor_b) else {}
        llm_confidence = edge_data.get('prior_strength', None)

        path_a = engine.get_relationship_path(causal_graph, target_factor, factor_a)
        path_b = engine.get_relationship_path(causal_graph, target_factor, factor_b)
        degree_a = len(path_a) - 1 if path_a else float('inf')
        degree_b = len(path_b) - 1 if path_b else float('inf')
        min_degree = min(degree_a, degree_b)

        evidence_description = "Causal relationship discovered by CausalIF algorithm"
        if llm_confidence is not None:
            evidence_description += f" (LLM confidence: {llm_confidence:.3f})"

        relationship = {
            'cause': factor_a,
            'effect': factor_b,
            'evidence': evidence_description,
            'llm_confidence': llm_confidence,
            'relationship_type': 'causal',
            'discovered_by': 'CausalIF_with_Bayesian_inference',
            'degree_from_target': min_degree,
            'path_to_target': path_a if degree_a <= degree_b else path_b,
        }
        causal_relationships.append(relationship)

        if factor_b == target_factor:
            target_influences.append({
                'influencing_factor': factor_a,
                'evidence': evidence_description,
                'llm_confidence': llm_confidence,
                'relationship': relationship,
                'degree': degree_a,
            })

        if factor_a == target_factor:
            target_effects.append({
                'affected_factor': factor_b,
                'evidence': evidence_description,
                'llm_confidence': llm_confidence,
                'relationship': relationship,
                'degree': degree_b,
            })

    target_influences.sort(key=lambda x: x.get('degree', float('inf')))
    target_effects.sort(key=lambda x: x.get('degree', float('inf')))
    return causal_relationships, target_influences, target_effects


def _build_network_summary(engine: CausalIFEngine, skeleton_graph, causal_graph,
                           target_influences, target_effects, degrees_analysis,
                           max_degrees, max_parallel_queries) -> Dict:
    """Assemble the network_summary dict."""
    return {
        'total_factors': len(causal_graph.nodes()),
        'total_causal_relationships': len(causal_graph.edges()),
        'factors_influencing_target': len(target_influences),
        'factors_affected_by_target': len(target_effects),
        'skeleton_edges': len(skeleton_graph.edges()),
        'causal_edges': len(causal_graph.edges()),
        'edge_removal_rate': 1 - (len(causal_graph.edges()) / max(1, len(skeleton_graph.edges()))),
        'max_degrees_analyzed': max_degrees,
        'max_parallel_queries_used': max_parallel_queries,
        'rag_retriever_used': engine.retriever_tool is not None or engine.retriever is not None,
        'bayesian_inference_used': True,
        'actual_max_degree_found': degrees_analysis.get('max_degree_found', 0),
        'factors_by_degree': degrees_analysis.get('factors_by_degree', {}),
        'rag_document_stats': engine.rag_document_stats,
    }


def _build_insights(engine: CausalIFEngine, skeleton_graph, causal_graph,
                    analysis_factors, target_factor, target_influences,
                    max_parallel_queries) -> List[str]:
    """Build the causalif_insights list."""
    insights = [
        f"✓ Bayesian Framework: PRIOR (skeleton) → POSTERIOR (directed graph)",
        f"✓ PRIOR: {len(skeleton_graph.edges())} associations from LLM + RAG",
        f"✓ POSTERIOR: {len(causal_graph.edges())} causal directions from Bayesian inference",
        f"✓ Analyzed {len(analysis_factors)} factors with {max_parallel_queries} parallel queries",
    ]

    if engine.retriever_tool or engine.retriever:
        insights.append("✓ RAG retrieval enabled for domain knowledge in PRIOR")
        if engine.rag_document_stats:
            all_sources = set()
            for s in engine.rag_document_stats.values():
                all_sources.update(s.get('source_uris', []))
            if all_sources:
                insights.append(f"✓ RAG matched {len(all_sources)} unique source documents across all edge queries")

    if engine.dataframe is not None:
        insights.append(f"✓ Observational data ({len(engine.dataframe)} samples) used for POSTERIOR")

    if target_influences:
        insights.append(f"✓ Found {len(target_influences)} factors causally influencing {target_factor}")

    return insights


def _build_summary_text(engine: CausalIFEngine, skeleton_graph, causal_graph,
                        target_factor, target_influences, max_degrees) -> str:
    """Build the human-readable summary string."""
    summary = f"✅ Bayesian CausalIF Causal Analysis Complete\n"
    summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    summary += f"🎯 Target Factor: {target_factor}\n"
    summary += f"📊 PRIOR (Associations): {len(skeleton_graph.edges())} edges\n"
    summary += f"🔗 POSTERIOR (Causal): {len(causal_graph.edges())} directed edges\n"
    summary += f"⚡ Influencing Factors: {len(target_influences)}\n"

    # RAG document stats
    if engine.rag_document_stats:
        all_sources = set()
        total_chunks = 0
        for s in engine.rag_document_stats.values():
            all_sources.update(s.get('source_uris', []))
            total_chunks += s.get('chunks_retrieved', 0)
        if all_sources:
            summary += f"📄 RAG Documents Matched: {len(all_sources)} unique sources ({total_chunks} total chunks)\n"
        elif total_chunks > 0:
            summary += f"📄 RAG Chunks Retrieved: {total_chunks} (pass raw retriever for source count)\n"
    summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    # Top causal influences
    if target_influences:
        summary += f"\n🔍 BAYESIAN CAUSAL ANALYSIS RESULTS:\n"
        summary += f"The following factors were identified as causal drivers\n"
        summary += f"through Bayesian structure learning:\n\n"

        sorted_influences = sorted(
            [inf for inf in target_influences if inf.get('llm_confidence') is not None],
            key=lambda x: x.get('llm_confidence', 0),
            reverse=True,
        )

        if sorted_influences:
            for i, inf in enumerate(sorted_influences, 1):
                confidence = inf.get('llm_confidence', 0)
                max_possible = max(confidence, 3)
                norm_confidence = min(confidence / max_possible, 1.0) if max_possible > 0 else 0
                strength_bar = "█" * int(norm_confidence * 10) + "░" * (10 - int(norm_confidence * 10))
                confidence_label = "HIGH" if confidence >= 3 else "MODERATE" if confidence >= 2 else "LOW"
                summary += f"{i}. {inf['influencing_factor']} → {target_factor}\n"
                summary += f"   LLM Confidence: |{strength_bar}| {confidence:.1f} ({confidence_label}) — {int(confidence)} of {int(max_possible)} KBs agreed\n"
                summary += f"   {inf.get('evidence', 'No statistical evidence available')}\n\n"
        else:
            summary += "⚠️ No causal influences with quantified confidence were found.\n"
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

    summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    summary += f"📌 INTERPRETATION GUIDE:\n"
    summary += f"• Use ONLY the factors listed above in your analysis\n"
    summary += f"• LLM confidence indicates how many knowledge bases agreed the edge exists\n"
    summary += f"• Higher confidence = more knowledge sources support this relationship\n"
    summary += f"• Bayesian method determines causal directions from data\n"
    summary += f"• LLM confidence is NOT causal effect size — it is edge existence agreement\n"
    summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return summary


def _build_error_result(query: str, error: str) -> Dict:
    """Build the standard error result dict."""
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
        'causalif_insights': [f"Error in CausalIF analysis: {error}"],
        'recommendations': [],
        'algorithm_details': {'error': error},
        'max_degrees_used': 3,
        'max_parallel_queries_used': 50,
        'rag_support_enabled': False,
        'bayesian_inference_enabled': True,
        'summary': f"❌ CausalIF Analysis Failed: {error}",
        'llm_interpretation': f"Analysis failed: {error}",
        'query': query,
        'success': False,
    }


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
        causalif_engine = _get_engine()
        
        if causalif_engine is None:
            logger.warning("No CausalIF engine configured. Raising error")
            raise ValueError(
                "❌ CausalIF Analysis Failed: No CausalIF engine configured. "
                "Please call set_causalif_engine() first.\n")

        # Check if this is an interventional (do-operator) query first.
        # Only route to do-operator if the model is already fitted.
        # If not fitted, fall through to run a full causal discovery.
        if causalif_engine.enable_causal_estimate:
            if causalif_engine.causal_model is not None:
                model_nodes = sorted(causalif_engine.causal_model.nodes())
                parsed = parse_intervention_query(query, model_nodes)
                if parsed is not None:
                    logger.info("Detected interventional query — routing to do-operator")
                    return causalif_intervene(query)
            else:
                # Check if the query *looks* interventional but model isn't fitted
                # Use a lightweight keyword check to give a helpful error
                q_lower = query.lower()
                intervention_hints = ['what happens', 'what would', 'what if', 'how does',
                                      'effect of setting', 'if we set']
                if any(kw in q_lower for kw in intervention_hints):
                    return {
                        'success': False,
                        'error': (
                            'This looks like an interventional query, but the causal model has not been '
                            'fitted yet. Run a causal analysis query first (e.g. "what causes high X") '
                            'so that the full CausalIF pipeline builds the model, then retry your '
                            'interventional question.'
                        ),
                        'query': query,
                    }

        max_degrees = causalif_engine.max_degrees
        max_parallel_queries = causalif_engine.max_parallel_queries
        
        available_columns = _validate_engine_and_dataframe(causalif_engine)
        combined_factors = _build_combined_factors(causalif_engine, available_columns)

        target_factor = extract_factors_from_query(
            query, 
            combined_factors,
            excluded_target_columns=causalif_engine.excluded_target_columns,
        )
        
        analysis_factors_all = list(dict.fromkeys([f for f in combined_factors if f != target_factor]))
        
        logger.info(f"Target factor: {target_factor}")
        logger.info(f"Analysis factors for CausalIF 1: {len(analysis_factors_all)} factors")
        logger.info(f"  Combined list includes: {len(causalif_engine.related_factors) if causalif_engine.related_factors else 0} related_factors + {len(available_columns)} dataframe columns")
        logger.info(f"Maximum degrees of separation: {max_degrees}")
        logger.info(f"Maximum parallel queries: {max_parallel_queries}")
        logger.info(f"RAG retriever available: {causalif_engine.retriever_tool is not None or causalif_engine.retriever is not None}")
        
        analysis_factors = [target_factor] + analysis_factors_all
        domains = causalif_engine.domains
        
        logger.info(f"Running CausalIF analysis on {len(analysis_factors)} total factors")
        logger.info(f"Domains: {domains}")
        
        skeleton_graph, causal_graph = causalif_engine.run_complete_causalif(analysis_factors, domains, target_factor)
        
        # Get causal inference summary — reuse the one computed inside
        # run_complete_causalif when enable_causal_estimate is True to avoid
        # running estimate_causal_effects / estimate_downstream_effects twice.
        causal_inference_summary = None
        if causalif_engine.enable_causal_estimate and causalif_engine.causal_inference_engine:
            causal_inference_summary = causalif_engine.get_causal_summary_lightweight(target_factor, causal_graph)
        
        degrees_analysis = causalif_engine.analyze_degrees_of_separation(causal_graph, target_factor)
        
        causal_relationships, target_influences, target_effects = _extract_causal_relationships(
            causalif_engine, causal_graph, target_factor,
        )
        
        network_summary = _build_network_summary(
            causalif_engine, skeleton_graph, causal_graph,
            target_influences, target_effects, degrees_analysis,
            max_degrees, max_parallel_queries,
        )
        
        causalif_insights = _build_insights(
            causalif_engine, skeleton_graph, causal_graph,
            analysis_factors, target_factor, target_influences,
            max_parallel_queries,
        )
        
        recommendations = []
        if target_influences:
            top_influence = target_influences[0]
            recommendations.append(f"Primary driver: {top_influence['influencing_factor']} → {target_factor}")
        
        algorithm_details = {
            'method': 'Bayesian CausalIF (Prior → Posterior)',
            'prior_method': 'LLM + RAG for edge existence (CausalIF 1)',
            'posterior_method': 'Bayesian structure learning with BDeu score (CausalIF 2)',
            'orientation_algorithm': 'Hill Climbing constrained by prior skeleton',
            'bayesian_score': 'BDeu (Bayesian Dirichlet equivalent uniform)',
            'prior_constraint': 'Skeleton graph from CausalIF 1',
            'max_degrees': max_degrees,
            'parallel_queries': max_parallel_queries,
        }
        
        summary = _build_summary_text(
            causalif_engine, skeleton_graph, causal_graph,
            target_factor, target_influences, max_degrees,
        )
        
        visualization_data = {
            'skeleton': {
                'nodes': list(skeleton_graph.nodes()),
                'edges': list(skeleton_graph.edges()),
            },
            'causal': {
                'nodes': list(causal_graph.nodes()),
                'edges': [(u, v, causal_graph[u][v]) for u, v in causal_graph.edges()],
            },
        }
        
        return {
            'target_factor': target_factor,
            'related_factors': analysis_factors_all,
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
            'rag_support_enabled': causalif_engine.retriever_tool is not None or causalif_engine.retriever is not None,
            'bayesian_inference_enabled': True,
            'summary': summary,
            'llm_interpretation': '',
            'causal_inference_summary': causal_inference_summary,
            'query': query,
            'success': True,
        }
        
    except Exception as e:
        logger.exception(f"CausalIF Analysis Failed: {e}")
        return _build_error_result(query, str(e))

@tool
def causalif_tool(query: str) -> Dict:
    """
    Executes the CausalIF (Causal Inference Framework) pipeline with Bayesian causal inference.
    This is a LangChain tool wrapper around the causalif() function.

    Returns the same result dict as causalif(), with an additional LLM interpretation
    (only generated in agentic tool use, not in library/notebook use).

    Note: You must configure the CausalIF engine first using set_causalif_engine()
    with your model, retriever, and dataframe.

    Args:
        query (str): The natural language query to process.

    Returns:
        Dict: A dictionary containing summary, structured results, and graph data.
    """
    result = causalif(query)

    if result.get('success'):
        engine = _get_engine()
        if engine and engine.model is not None:
            try:
                result['llm_interpretation'] = generate_llm_interpretation(
                    causalif_result=result,
                    original_query=query,
                    model=engine.model,
                )
            except Exception as e:
                logger.warning(f"Could not generate LLM interpretation: {e}")
                result['llm_interpretation'] = "LLM interpretation unavailable."

    return result