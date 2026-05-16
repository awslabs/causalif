# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CausalIF Engine implementation"""

import concurrent.futures
import logging
import math
import random
import re
import time
from typing import Dict, List, Union, Tuple, Callable, Any
from collections import defaultdict, deque

from urllib3.exceptions import ProtocolError, ReadTimeoutError
from http.client import HTTPException

import pandas as pd
import networkx as nx
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.causal_discovery import HillClimbSearch, ExpertKnowledge
from pgmpy.structure_score import BaseStructureScore, BDeu
from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork

from .core import KnowledgeBase
from .prompts import CausalIFPrompts

logger = logging.getLogger(__name__)


def _run_hill_climb(data, scoring_method, max_iter, max_indegree,
                    start_dag=None, tabu_length=100, expert_knowledge=None,
                    show_progress=False):
    """Run HillClimbSearch using the pgmpy >= 1.1 API.

    Uses HillClimbSearch(...).fit(data) which returns self with causal_graph_ attribute.
    """
    hc = HillClimbSearch(
        scoring_method=scoring_method,
        start_dag=start_dag,
        tabu_length=tabu_length,
        max_indegree=max_indegree,
        expert_knowledge=expert_knowledge,
        return_type='dag',
        epsilon=1e-4,
        max_iter=max_iter,
        show_progress=show_progress,
    )
    hc.fit(data)
    return hc.causal_graph_

def parse_association_type_response(response_text: str) -> Dict:
    """Parse LLM response from Association Type Verifier (Step 3 of LACR 1).
    
    Extracts association type (direct/indirect/unknown) and intermediary factors.
    See paper Section 3.2.1: (D) Directly, (E) Indirectly, (C) Unknown.
    """
    result = {
        "type": "unknown",
        "intermediary_factors": [],
        "raw_response": response_text
    }
    
    response_upper = response_text.upper()
    
    # Extract association type: check (E) first to avoid false match with (D)
    if "(E)" in response_upper or "INDIRECTLY ASSOCIATED" in response_upper:
        result["type"] = "indirect"
    elif "(D)" in response_upper or "DIRECTLY ASSOCIATED" in response_upper:
        result["type"] = "direct"
    elif "(C)" in response_upper:
        result["type"] = "unknown"
    else:
        if "INDIRECT" in response_upper:
            result["type"] = "indirect"
        elif "DIRECT" in response_upper:
            result["type"] = "direct"
    
    # Extract intermediary factors if indirect
    if result["type"] == "indirect":
        lines = response_text.split('\n')
        for line in lines:
            if "intermediary factor" in line.lower():
                if ':' in line:
                    factors_text = line.split(':', 1)[1].strip()
                    if factors_text.lower() in ['none', 'n/a', '']:
                        break
                    factors_text = factors_text.replace('[', '').replace(']', '')
                    factors_text = factors_text.replace(' and ', ', ').replace(';', ',')
                    factors = [f.strip() for f in factors_text.split(',')]
                    factors = [f for f in factors if f and f.lower() not in ['none', 'n/a']]
                    result["intermediary_factors"] = factors
                    break
    
    return result


def parse_batch_association_response(response_text: str, expected_count: int) -> List[str]:
    """Parse a batch LLM response containing multiple edge association results.
    
    Sections delimited by ``--- Edge N ---``. Each classified as (A) associated,
    (B) independent, or (C) unknown. Tries Answer: line first, falls back to full section.
    Returns exactly ``expected_count`` items.
    """
    # Split by --- Edge N --- delimiters
    sections = re.split(r'---\s*Edge\s+\d+\s*---', response_text)
    if sections:
        sections = sections[1:]

    def _classify_text(text: str) -> str:
        upper = text.upper()
        if "(B)" in upper:
            return "independent"
        if "(A)" in upper:
            return "associated"
        if "(C)" in upper:
            return "unknown"
        if "INDEPENDENT" in upper:
            return "independent"
        if "ASSOCIATED" in upper:
            return "associated"
        return "unknown"

    results: List[str] = []
    for section in sections:
        answer_line = None
        for line in section.split('\n'):
            if line.strip().upper().startswith("ANSWER"):
                answer_line = line
                break
        results.append(_classify_text(answer_line) if answer_line else _classify_text(section))

    # Pad or truncate to exactly expected_count
    if len(results) < expected_count:
        results.extend(["unknown"] * (expected_count - len(results)))
    else:
        results = results[:expected_count]

    return results


def parse_batch_association_type_response(response_text: str, expected_count: int) -> List[Dict]:
    """Parse a batch LLM response for association TYPE queries (Phase 3).
    
    Sections delimited by ``--- Edge N ---``, each parsed via
    parse_association_type_response. Returns exactly ``expected_count`` items.
    """
    default_result: Dict = {"type": "unknown", "intermediary_factors": []}

    sections = re.split(r'---\s*Edge\s+\d+\s*---', response_text)
    if sections:
        sections = sections[1:]

    results: List[Dict] = []
    for section in sections:
        stripped = section.strip()
        if stripped:
            try:
                parsed = parse_association_type_response(stripped)
                results.append(parsed)
            except Exception:
                results.append(dict(default_result))
        else:
            results.append(dict(default_result))

    # Pad or truncate to exactly expected_count
    while len(results) < expected_count:
        results.append(dict(default_result))
    results = results[:expected_count]

    return results


def check_intermediary_factors_in_v(
    intermediary_factors: List[str],
    factors: List[str],
    factor_a: str,
    factor_b: str
) -> bool:
    r"""Check if any intermediary factor is in V\{vi,vj}.
    
    Step 4 of LACR 1: verifies intermediary factors are from the variable set V.
    External variables are ignored per the paper — if intermediaries are not in V,
    the association is effectively direct.
    """
    available_factors = set(factors) - {factor_a, factor_b}
    normalized_available = {f.strip().lower(): f for f in available_factors}

    for intermediary in intermediary_factors:
        if intermediary.strip().lower() in normalized_available:
            return True
    return False


def apply_rechecker_logic(association_type: str, intermediary_in_v: bool) -> str:
    r"""Apply Association Rechecker logic (Step 4 of LACR 1).
    
    - direct → directly_associated
    - indirect + intermediaries in V → indirectly_associated
    - indirect + intermediaries NOT in V → directly_associated (corrected)
    - unknown → unknown
    """
    if association_type == "direct":
        return "directly_associated"
    elif association_type == "indirect":
        if intermediary_in_v:
            return "indirectly_associated"
        else:
            return "directly_associated"
    else:
        return "unknown"


def map_classification_to_vote(final_classification: str, initial_association: str) -> int:
    """Map final classification to vote value (Step 5 of LACR 1).
    
    Vote scoring: direct → +1, indirect → -1, independent → -1, unknown → 0.
    Edge kept if S = Σ votes > 0, removed if S ≤ 0.
    """
    if initial_association == "independent":
        return -1
    if final_classification == "directly_associated":
        return 1
    elif final_classification == "indirectly_associated":
        return -1
    else:
        return 0



class PriorWeightedBDeu(BaseStructureScore):
    """Custom scoring: BDeu + CausalIF 1 structure priors.
    
    Score: log P(G|D) = Σ_v [ BDeu_local(v, pa(v)) + λ × Σ_{p ∈ pa(v)} log(prior_strength + 1) ]
    
    λ = α_structure / (n_nodes − 1) by default, where α_structure = ESS.
    This treats the LLM structure prior as equivalent to α_structure imaginary
    data points (rows) of structural evidence, distributed across the maximum
    possible parent edges per variable. As n_samples grows, data dominates
    the posterior; as n_samples is small, the prior has proportionally more
    influence — standard Bayesian updating (Heckerman et al. 1995).
    """
    
    def __init__(self, data, skeleton_graph, prior_weight='auto', equivalent_sample_size=10, quiet=False, **kwargs):
        super(PriorWeightedBDeu, self).__init__(data, **kwargs)
        
        self.bdeu = BDeu(data, equivalent_sample_size=equivalent_sample_size)
        self.skeleton = skeleton_graph
        
        n_nodes = len(skeleton_graph.nodes()) if len(skeleton_graph.nodes()) > 0 else 1
        
        if prior_weight == 'auto':
            # α_structure = ESS: the LLM prior is worth ESS imaginary data points.
            # Divide by max fan-in (n_nodes − 1) so the total prior budget per
            # variable stays bounded regardless of graph size.
            alpha_structure = equivalent_sample_size
            self.prior_weight = alpha_structure / max(1, n_nodes - 1)
        else:
            self.prior_weight = prior_weight
        
        self.prior_strengths = {}
        for u, v in skeleton_graph.edges():
            prior_strength = skeleton_graph[u][v].get('prior_strength', 0.5)
            self.prior_strengths[(u, v)] = prior_strength
            self.prior_strengths[(v, u)] = prior_strength
        
        if not quiet:
            logger.info(f"\n[Prior-Weighted Scoring] Initialized with {len(self.prior_strengths)//2} prior edges")
            logger.info(f"  Prior weight (λ): {self.prior_weight:.4f} {'(adaptive: ESS / (n_nodes-1))' if prior_weight == 'auto' else '(manual)'}")
            logger.info(f"  α_structure (ESS): {equivalent_sample_size}, n_nodes: {n_nodes}")
            logger.info(f"  Sample size: {len(data)}")
            logger.info(f"  Prior-to-data ratio: ~{equivalent_sample_size}/{len(data)} = {equivalent_sample_size/max(1,len(data)):.4f} per variable")
            if self.prior_strengths:
                s_min = min(self.prior_strengths.values())
                s_max = max(self.prior_strengths.values())
                logger.info(f"  Prior strength range: [{s_min:.3f}, {s_max:.3f}]")
                min_contrib = self.prior_weight * math.log(s_min + 1)
                max_contrib = self.prior_weight * math.log(s_max + 1)
                logger.info(f"  Effective prior contribution per edge: [{min_contrib:.4f}, {max_contrib:.4f}]")
                max_budget = (n_nodes - 1) * max_contrib
                logger.info(f"  Max prior budget per variable (all parents): {max_budget:.4f}")
    
    def score(self, model):
        """Overall model score = Σ local_score(v, parents(v)) over all nodes.
        
        Sums local_score() so the prior contributions are included consistently
        with Hill Climb's incremental scoring. Delegating to BDeu.score() would
        bypass the prior, causing score() and local_score() to disagree.
        """
        score_val = 0.0
        for node in model.nodes():
            parents = list(model.predecessors(node))
            score_val += self.local_score(node, parents)
        return score_val
    
    def local_score(self, variable, parents):
        """Local score = BDeu local + prior contribution for parent edges."""
        bdeu_local = self.bdeu.local_score(variable, parents)
        
        prior_local = 0.0
        for parent in parents:
            if (parent, variable) in self.prior_strengths:
                prior_local += self.prior_weight * np.log(self.prior_strengths[(parent, variable)] + 1)
        
        return bdeu_local + prior_local



class CausalIFEngine:
    """CausalIF implementation with Bayesian causal inference for orientation"""
    
    def __init__(self, model, retriever_tool=None, retriever=None, dataframe: pd.DataFrame = None, 
                 max_token_limit: int = 150000, max_degrees: int = None, max_parallel_queries: int = 25,
                 excluded_target_columns: List[str] = None, excluded_related_columns: List[str] = None,
                 related_factors: List[str] = None, selected_dataframe_columns: List[str] = None,
                 enable_causal_estimate: bool = False, domains: List[str] = None,
                 batch_size: int = 10, read_timeout: int = 120, bootstrap_iterations: int = 50,
                 bootstrap_threshold: float = 0.7):
        self.model = model
        self.retriever_tool = retriever_tool
        self.retriever = retriever
        self.dataframe = dataframe
        self.max_token_limit = max_token_limit
        self.max_degrees = max_degrees
        self.max_parallel_queries = max_parallel_queries
        self.excluded_target_columns = excluded_target_columns or []
        self.excluded_related_columns = excluded_related_columns or []
        self.related_factors = related_factors or []
        self.selected_dataframe_columns = selected_dataframe_columns
        self.prompts = CausalIFPrompts()
        self.enable_causal_estimate = enable_causal_estimate
        self.causal_model = None
        self.causal_inference_engine = None
        self.domains = domains or ['supply_chain', 'logistics', 'operations', 'performance_metrics']
        self.batch_size = max(batch_size, 1)
        self.rag_document_stats = {}
        self._cached_discretizers = {}
        self.bootstrap_iterations = max(bootstrap_iterations, 0)
        self.bootstrap_threshold = bootstrap_threshold

        self._apply_read_timeout(read_timeout)

    def _apply_read_timeout(self, read_timeout: int) -> None:
        """Patch the underlying Bedrock client's read timeout if accessible."""
        try:
            from botocore.config import Config

            client = getattr(self.model, 'client', None)
            if client is None:
                client = getattr(self.model, '_client', None)

            if client is not None and hasattr(client, '_endpoint'):
                current_timeout = getattr(client._endpoint, 'timeout', None)
                client._endpoint.timeout = max(read_timeout, 60)
                logger.info(f"[CausalIFEngine] Bedrock client read_timeout set to {max(read_timeout, 60)}s (was {current_timeout})")
            elif client is not None and hasattr(client, 'meta'):
                logger.info(f"[CausalIFEngine] ℹ️  Could not patch read_timeout directly. "
                      f"Pass read_timeout={read_timeout} in your boto3 Config.")
            else:
                logger.info(f"[CausalIFEngine] ℹ️  Model does not expose a Bedrock client. "
                      f"Set read_timeout={read_timeout} on your boto3 client config.")
        except Exception as e:
            logger.info(f"[CausalIFEngine] ⚠️  Could not set read_timeout: {e}.")

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _prepare_doc_content(self, content: str) -> str:
        """Return content as-is if within token limit, otherwise summarize via LLM."""
        estimated_tokens = self._estimate_tokens(content)
        if estimated_tokens <= self.max_token_limit:
            return content
        
        # Content exceeds token limit — summarize using LLM
        logger.info(f"  📝 Document content exceeds {self.max_token_limit} tokens (~{estimated_tokens} est.), summarizing...")
        try:
            summary_prompt = (
                f"Summarize the following document concisely, preserving all key facts, "
                f"relationships, causal claims, and statistical evidence. "
                f"Keep the summary under {self.max_token_limit // 2} tokens.\n\n"
                f"Document:\n{content}"
            )
            response = self.model.invoke(summary_prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"  ✓ Summarized from ~{estimated_tokens} to ~{self._estimate_tokens(summary)} tokens")
            return summary
        except Exception as e:
            # Fallback: truncate to approximate char limit if summarization fails
            max_chars = self.max_token_limit * 4
            logger.warning(f"  ⚠️ Summarization failed ({e}), truncating to {max_chars} chars")
            return content[:max_chars]

    def _discretize_dataframe(self, df: pd.DataFrame, label: str = "") -> pd.DataFrame:
        """Discretize numeric columns using KBinsDiscretizer with Sturges' adaptive bins.
        
        Uses quantile bins for skewed columns (|skew| > 2), kmeans otherwise.
        Fitted discretizers are cached for deterministic bins across CausalIF 2 & 3.
        NaN values are preserved (not passed to sklearn) and filled downstream.
        """
        DISCRETE_THRESHOLD = 7
        MAX_BINS = 5
        MIN_BINS = 2
        SKEW_THRESHOLD = 2.0

        context = f" for {label}" if label else ""
        logger.info(f"\n  Discretizing numeric columns{context} (KBinsDiscretizer, Sturges' adaptive bins, max {MAX_BINS}):")

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                n_unique = df[col].nunique()
                if n_unique <= DISCRETE_THRESHOLD:
                    df[col] = df[col].astype(str)
                    logger.info(f"    ✓ '{col}' kept as-is ({n_unique} unique, already discrete)")
                elif col in self._cached_discretizers:
                    # Reuse previously fitted discretizer for deterministic bins
                    kbd = self._cached_discretizers[col]
                    mask = df[col].notna()
                    df[col] = df[col].astype(object)  # avoid FutureWarning on mixed-dtype assignment
                    if mask.any():
                        binned = kbd.transform(df.loc[mask, [col]])
                        df.loc[mask, col] = binned[:, 0].astype(int).astype(str)
                    # NaN rows stay NaN — filled to 'missing' downstream
                    actual_bins = len(kbd.bin_edges_[0]) - 1
                    logger.info(f"    ✓ '{col}' discretized ({n_unique} unique → {actual_bins} {kbd.strategy} bins, cached)")
                else:
                    sturges_bins = 1 + int(math.log2(n_unique))
                    n_bins = max(MIN_BINS, min(MAX_BINS, sturges_bins, n_unique))

                    # Skew-aware strategy: quantile for heavy skew, kmeans otherwise
                    col_skew = df[col].dropna().skew()
                    strategy = 'quantile' if abs(col_skew) > SKEW_THRESHOLD else 'kmeans'

                    # Separate NaN mask — sklearn cannot handle NaN
                    mask = df[col].notna()
                    if not mask.any():
                        df[col] = df[col].astype(str)
                        logger.info(f"    ⚠️  '{col}' all NaN — skipped")
                        continue
                    col_clean = df.loc[mask, [col]]
                    df[col] = df[col].astype(object)  # avoid FutureWarning on mixed-dtype assignment

                    try:
                        kbd = KBinsDiscretizer(
                            n_bins=n_bins, encode='ordinal', strategy=strategy,
                            random_state=42,
                        )
                        binned = kbd.fit_transform(col_clean)
                        df.loc[mask, col] = binned[:, 0].astype(int).astype(str)
                        actual_bins = len(kbd.bin_edges_[0]) - 1
                        self._cached_discretizers[col] = kbd
                        skew_note = f", skew={col_skew:.1f}" if strategy == 'quantile' else ""
                        logger.info(f"    ✓ '{col}' discretized ({n_unique} unique → {actual_bins} {strategy} bins, Sturges' suggested {sturges_bins}{skew_note})")
                    except Exception:
                        # Primary strategy failed — try uniform (equal-width)
                        # which only needs min/max and almost never fails.
                        if strategy != 'uniform':
                            try:
                                kbd = KBinsDiscretizer(
                                    n_bins=n_bins, encode='ordinal', strategy='uniform',
                                    random_state=42,
                                )
                                binned = kbd.fit_transform(col_clean)
                                df.loc[mask, col] = binned[:, 0].astype(int).astype(str)
                                actual_bins = len(kbd.bin_edges_[0]) - 1
                                self._cached_discretizers[col] = kbd
                                logger.info(f"    ✓ '{col}' discretized ({n_unique} unique → {actual_bins} uniform bins, fallback from {strategy})")
                            except Exception:
                                df[col] = df[col].astype(str)
                                logger.info(f"    ⚠️  '{col}' fallback to string (all binning strategies failed)")
                        else:
                            df[col] = df[col].astype(str)
                            logger.info(f"    ⚠️  '{col}' fallback to string (uniform binning failed)")
            else:
                df[col] = df[col].astype(str)

        return df

    def _group_chunks_by_source(self, chunks: List[KnowledgeBase], doc_token_limit: int = 180000) -> List[KnowledgeBase]:
        """Group RAG chunks by source document into one KnowledgeBase per source.
        
        Per LACR paper, each source document = one independent voter.
        Falls back to per-chunk behavior when source metadata is unavailable.
        """
        
        # Group chunks by source URI
        source_groups = defaultdict(list)
        fallback_pattern = re.compile(r'^doc_\d+$')
        for kb in chunks:
            # Only treat exact "doc_N" pattern as fallback (not real URIs containing "doc_")
            is_fallback = not kb.source or fallback_pattern.match(kb.source) or kb.source == "single_doc"
            if not is_fallback:
                source_groups[kb.source].append(kb)
            else:
                # No real source URI — treat each chunk as its own document (fallback)
                source_groups[f"_unknown_{id(kb)}"] = [kb]
        
        combined_docs = []
        for source, group in sorted(source_groups.items(), key=lambda x: str(x[0])):
            # Sort chunks within group by content hash for deterministic concatenation
            group.sort(key=lambda kb: kb.content or "")
            if len(group) == 1:
                combined_docs.append(group[0])
            else:
                # Multiple chunks from same source — concatenate
                combined_content = "\n\n---\n\n".join(kb.content for kb in group if kb.content)
                estimated_tokens = self._estimate_tokens(combined_content)
                
                if estimated_tokens > doc_token_limit:
                    logger.info(f"  📝 Source document ({source}) has ~{estimated_tokens} tokens across {len(group)} chunks, summarizing...")
                    try:
                        summary_prompt = (
                            f"Summarize the following document concisely, preserving all key facts, "
                            f"relationships, causal claims, and statistical evidence. "
                            f"Keep the summary under {doc_token_limit // 2} tokens.\n\n"
                            f"Document:\n{combined_content}"
                        )
                        response = self.model.invoke(summary_prompt)
                        combined_content = response.content if hasattr(response, 'content') else str(response)
                        logger.info(f"  ✓ Summarized to ~{self._estimate_tokens(combined_content)} tokens")
                    except Exception as e:
                        max_chars = doc_token_limit * 4
                        logger.warning(f"  ⚠️ Summarization failed ({e}), truncating to {max_chars} chars")
                        combined_content = combined_content[:max_chars]
                
                combined_kb = KnowledgeBase(
                    kb_type="DOC",
                    content=combined_content,
                    source=source
                )
                combined_docs.append(combined_kb)
        
        return combined_docs

    def filter_graph_by_degrees(self, graph: Union[nx.Graph, nx.DiGraph], target_factor: str, max_degrees: int = None) -> Union[nx.Graph, nx.DiGraph]:
        """Filter graph to nodes within max_degrees of target_factor. Returns original if max_degrees is None."""
        if max_degrees is None:
            return graph
        if target_factor not in graph.nodes():
            return graph

        visited = set([target_factor])
        queue = deque([(target_factor, 0)])
        nodes_within_degrees = set([target_factor])

        while queue:
            current_node, current_degree = queue.popleft()

            if current_degree >= max_degrees:
                continue

            if isinstance(graph, nx.DiGraph):
                neighbors = set(graph.predecessors(current_node)) | set(graph.successors(current_node))
            else:
                neighbors = set(graph.neighbors(current_node))

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_within_degrees.add(neighbor)
                    queue.append((neighbor, current_degree + 1))

        if isinstance(graph, nx.DiGraph):
            filtered_graph = graph.subgraph(nodes_within_degrees).copy()
        else:
            filtered_graph = graph.subgraph(nodes_within_degrees).copy()

        logger.info(f"Filtered graph to {len(filtered_graph.nodes())} nodes within {max_degrees} degrees of {target_factor}")
        return filtered_graph

    def get_relationship_path(self, graph: Union[nx.Graph, nx.DiGraph], source: str, target: str) -> List[str]:
        try:
            if isinstance(graph, nx.DiGraph):
                undirected = graph.to_undirected()
                path = nx.shortest_path(undirected, source, target)
            else:
                path = nx.shortest_path(graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []

    def analyze_degrees_of_separation(self, graph: Union[nx.Graph, nx.DiGraph], target_factor: str) -> Dict:
        """Analyze degrees of separation from target factor."""
        degrees_analysis = {
            'target_factor': target_factor,
            'factors_by_degree': {},
            'paths': {},
            'max_degree_found': 0
        }

        for factor in graph.nodes():
            if factor == target_factor:
                degrees_analysis['factors_by_degree'][0] = degrees_analysis['factors_by_degree'].get(0, []) + [factor]
                continue

            path = self.get_relationship_path(graph, target_factor, factor)
            if path:
                degree = len(path) - 1
                # Include all degrees if max_degrees is None, otherwise filter
                if self.max_degrees is None or degree <= self.max_degrees:
                    degrees_analysis['factors_by_degree'][degree] = degrees_analysis['factors_by_degree'].get(degree, []) + [factor]
                    degrees_analysis['paths'][factor] = path
                    degrees_analysis['max_degree_found'] = max(degrees_analysis['max_degree_found'], degree)

        return degrees_analysis


    def retrieve_documents(self, factor_a: str, factor_b: str) -> List[KnowledgeBase]:
        """Retrieve relevant documents for factor pair using RAG.
        Prefers raw retriever for metadata; falls back to retriever_tool.
        """
        try:
            query = f"{factor_a} and {factor_b} association relationship"
            logger.info(f"RAG Query: {query}")

            # Prefer raw retriever for metadata access
            if self.retriever:
                retrieved_docs = self.retriever.invoke(query)
                documents = []
                unique_sources = set()

                if isinstance(retrieved_docs, list):
                    for i, doc in enumerate(retrieved_docs):
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        source = f"doc_{i}"
                        if hasattr(doc, 'metadata') and doc.metadata:
                            meta = doc.metadata
                            loc = meta.get('location', {})
                            if isinstance(loc, dict):
                                s3_loc = loc.get('s3Location', {})
                                if isinstance(s3_loc, dict):
                                    source = s3_loc.get('uri', source)
                            if source == f"doc_{i}":
                                source = meta.get('source_uri', meta.get('source', source))
                            unique_sources.add(source)

                        kb = KnowledgeBase(kb_type="DOC", content=content, source=source)
                        documents.append(kb)

                # Store per-edge stats
                self.rag_document_stats[(factor_a, factor_b)] = {
                    'chunks_retrieved': len(documents),
                    'unique_documents': len(unique_sources),
                    'source_uris': list(unique_sources)
                }

                logger.info(f"  Retrieved {len(documents)} chunks from {len(unique_sources)} unique source documents for {factor_a} and {factor_b}")
                # Sort by source URI for deterministic ordering
                documents.sort(key=lambda kb: kb.source or "")
                return documents

            elif self.retriever_tool:
                # Fallback: use retriever_tool (metadata not available)
                retrieved_docs = self.retriever_tool.invoke({"query": query})
                documents = []

                if isinstance(retrieved_docs, list):
                    for i, doc in enumerate(retrieved_docs):
                        if isinstance(doc, dict):
                            content = doc.get('content', '') or doc.get('page_content', '') or str(doc)
                        elif hasattr(doc, 'page_content'):
                            content = doc.page_content
                        elif isinstance(doc, str):
                            content = doc
                        else:
                            content = str(doc)

                        kb = KnowledgeBase(kb_type="DOC", content=content, source=f"doc_{i}")
                        documents.append(kb)
                elif isinstance(retrieved_docs, str):
                    kb = KnowledgeBase(kb_type="DOC", content=retrieved_docs, source="single_doc")
                    documents.append(kb)

                # No metadata available from tool wrapper
                self.rag_document_stats[(factor_a, factor_b)] = {
                    'chunks_retrieved': len(documents),
                    'unique_documents': None,  # Unknown - tool strips metadata
                    'source_uris': []
                }

                logger.info(f"  Retrieved {len(documents)} documents for {factor_a} and {factor_b} (source count unavailable via tool wrapper)")
                # Sort by source for deterministic ordering
                documents.sort(key=lambda kb: kb.source or "")
                return documents
            else:
                logger.info("No retriever tool available for RAG")
                return []

        except Exception as e:
            logger.info(f"Error retrieving documents: {e}")
            return []


    def parallel_llm_query_sync(self, prompts: List[str]) -> List[str]:
        """Execute multiple LLM queries in parallel with retry logic and concurrency limiting."""

        def _invoke(task):
            idx, prompt = task
            def _call():
                response = self.model.invoke(prompt)
                result = response.content if hasattr(response, 'content') else str(response)
                return str(result)
            return self._llm_invoke_with_retry(_call, label="query", index=idx + 1)

        tasks = list(enumerate(prompts))
        return self._run_parallel_with_retry(tasks, _invoke, label="query")

    def execute_parallel_queries(self, prompts: List[str]) -> List[str]:
        """Execute parallel queries in batches to avoid overwhelming the API."""
        if not prompts:
            return []

        # Process in batches to avoid overwhelming API
        BATCH_SIZE = 50  # Process max 50 queries at a time

        if len(prompts) <= BATCH_SIZE:
            return self.parallel_llm_query_sync(prompts)
        else:
            logger.info(f"  Processing {len(prompts)} queries in batches of {BATCH_SIZE}...")
            all_results = []

            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

                logger.info(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} queries...")

                batch_results = self.parallel_llm_query_sync(batch)
                all_results.extend(batch_results)

            return all_results

    def _invoke_with_messages(self, system_content: str, human_content: str) -> str:
        """Invoke LLM with system + human messages (for context-cached doc queries)."""
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]
        response = self.model.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    def _llm_invoke_with_retry(
        self,
        invoke_fn: Callable[[], str],
        label: str = "query",
        index: int = 0,
        max_retries: int = 3,
    ) -> str:
        """Shared retry wrapper for all LLM invocations.

        Args:
            invoke_fn: Zero-arg callable that performs the LLM call and returns a string.
            label: Human-readable label for log messages (e.g. "DOC query", "BG query").
            index: 1-based index used in log messages.
            max_retries: Maximum number of attempts.

        Returns:
            The LLM response string, or ``"UNKNOWN"`` on failure.
        """
        for attempt in range(max_retries):
            try:
                if self.model is None:
                    return "UNKNOWN"
                return invoke_fn()

            except (ValueError, ProtocolError, HTTPException, ConnectionError, ReadTimeoutError) as e:
                error_msg = str(e).lower()
                is_retryable = any(kw in error_msg for kw in
                                   ['connection pool', 'i/o operation', 'closed file',
                                    'connection', 'protocol', 'broken pipe',
                                    'read timeout', 'timed out', 'timeout'])
                if is_retryable and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 2)
                    logger.info(f"  Connection/timeout error on {label} {index}, "
                          f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {str(e)[:80]}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.info(f"  Connection/timeout error in {label} {index} "
                          f"(gave up after {max_retries} attempts): {str(e)[:100]}")
                    return "UNKNOWN"

            except Exception as e:
                error_msg = str(e).lower()
                is_retryable = any(kw in error_msg for kw in
                                   ['throttl', 'rate limit', 'too many requests', '429',
                                    'read timeout', 'timed out', 'timeout', 'endpoint url'])
                if is_retryable and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"  Retryable error on {label} {index}, "
                          f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries}): {str(e)[:80]}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.info(f"  Error in {label} {index}: {str(e)[:100]}")
                    return "UNKNOWN"

        return "UNKNOWN"

    def _run_parallel_with_retry(
        self,
        tasks: List[Any],
        invoke_fn: Callable[[Any], str],
        label: str = "query",
    ) -> List[str]:
        """Run a list of tasks in parallel using the shared retry logic.

        Each task is submitted to a thread pool. Results are returned in the
        same order as *tasks*.

        Args:
            tasks: Ordered list of task payloads (passed to *invoke_fn*).
            invoke_fn: ``(task) -> str`` callable executed per task.
            label: Human-readable label for log messages.

        Returns:
            List of result strings aligned with *tasks*.
        """
        if not tasks:
            return []

        safe_max_workers = min(self.max_parallel_queries, 8)
        results: List[str] = [None] * len(tasks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
            future_to_index = {
                executor.submit(invoke_fn, task): i
                for i, task in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result(timeout=90)
                except concurrent.futures.TimeoutError:
                    logger.info(f"  {label} {idx + 1} timed out after 90 seconds")
                    results[idx] = "UNKNOWN"
                except Exception as e:
                    logger.info(f"  {label} {idx + 1} failed: {str(e)[:100]}")
                    results[idx] = "UNKNOWN"

        return [r if r is not None else "UNKNOWN" for r in results]

    def execute_doc_context_queries(
        self,
        system_context: str,
        human_prompts: List[str],
    ) -> List[str]:
        """Execute edge-batch queries sharing the same document context in parallel."""

        def _invoke(task):
            idx, human_prompt = task
            def _call():
                return self._invoke_with_messages(system_context, human_prompt)
            return self._llm_invoke_with_retry(_call, label="doc-ctx query", index=idx + 1)

        tasks = list(enumerate(human_prompts))
        return self._run_parallel_with_retry(tasks, _invoke, label="doc-ctx query")

    def causalif_1_edge_existence_verification(self, factors: List[str], domains: List[str], target_factor: str = None) -> nx.Graph:
        """CausalIF 1: Edge Existence Verification.
        
        Start with complete graph, query LLM+RAG per edge, remove edges with S ≤ 0.
        """
        
        logger.info("="*80)
        logger.info("CausalIF 1: Edge Existence Verification (Paper Algorithm)")
        logger.info("="*80)
        logger.info("Strategy: Start with complete graph, remove edges via LLM voting")
        
        # Step 1: Initialize complete undirected graph
        G = nx.complete_graph(len(factors))
        mapping = {i: factors[i] for i in range(len(factors))}
        G = nx.relabel_nodes(G, mapping)
        
        logger.info(f"\nInitialized complete graph:")
        logger.info(f"  - Nodes: {len(G.nodes())}")
        logger.info(f"  - Edges: {len(G.edges())} (all possible pairs)")
        
        # Step 2: For each edge in complete graph, run LLM voting
        logger.info(f"\n{'='*80}")
        logger.info(f"Step 2: LLM Voting on All Edges (BATCH MODE)")
        logger.info(f"{'='*80}")
        
        # Seed RNGs for deterministic retry jitter and any random operations
        random.seed(42)
        np.random.seed(42)
        
        edges_to_remove = []
        edges_to_keep = []
        all_edges = sorted(G.edges())
        
        logger.info(f"Testing {len(all_edges)} edges with LLM voting...\n")
        
        # PHASE 1: Retrieve documents per edge
        # Per LACR paper, each edge gets its own RAG retrieval
        logger.info(f"Phase 1: Retrieving documents for all edges...")
        edge_documents = {}  # Maps (factor_a, factor_b) -> List[KnowledgeBase]
        
        for factor_a, factor_b in all_edges:
            doc_kbs = self.retrieve_documents(factor_a, factor_b)
            doc_kbs = self._group_chunks_by_source(doc_kbs)
            edge_documents[(factor_a, factor_b)] = doc_kbs
        
        logger.info(f"✓ Retrieved and grouped documents for {len(all_edges)} edges\n")
        
        # Print RAG document match summary
        if self.rag_document_stats:
            total_chunks = sum(s.get('chunks_retrieved', 0) for s in self.rag_document_stats.values())
            all_sources = set()
            for s in self.rag_document_stats.values():
                all_sources.update(s.get('source_uris', []))
            num_unique_docs = len(all_sources)
            avg_docs_per_edge = sum(
                (s.get('unique_documents', 0) or 0) for s in self.rag_document_stats.values()
            ) / max(1, len(self.rag_document_stats))
            total_doc_kbs = sum(len(edge_documents[e]) for e in all_edges)
            
            logger.info(f"📄 RAG Document Match Summary:")
            logger.info(f"  - Total chunks retrieved across all edges: {total_chunks}")
            logger.info(f"  - Unique source documents across all edges: {num_unique_docs}")
            logger.info(f"  - Avg unique docs per edge: {avg_docs_per_edge:.1f}")
            logger.info(f"  - Total document-level KBs after grouping: {total_doc_kbs} (from {total_chunks} chunks)")
        
        # PHASE 2: Build batched prompts for execution
        logger.info(f"Phase 2: Building BATCHED prompts for LLM execution (batch_size={self.batch_size})...")
        
        # Tracking structures for batch results
        batch_edge_map = []  # (edge, kb_type, kb_index) in result order
        edge_doc_content = {}  # (factor_a, factor_b, kb_index) -> content
        edge_doc_source_key = {}  # (factor_a, factor_b, kb_index) -> source_key
        edge_kb_counter = {}
        
        # BG association batching
        bg_edge_pairs = list(all_edges)
        bg_batches = [bg_edge_pairs[i:i + self.batch_size] for i in range(0, len(bg_edge_pairs), self.batch_size)]
        
        bg_prompts = []  # Human messages only (edge questions)
        bg_batch_sizes = []  # Track how many edges per batch prompt for parsing
        for batch in bg_batches:
            human_prompt = self.prompts.bg_association_edge_batch(batch)
            bg_prompts.append(human_prompt)
            bg_batch_sizes.append(len(batch))
        
        # Record BG edge metadata in order
        for edge in bg_edge_pairs:
            kb_idx = edge_kb_counter.get(edge, 0)
            batch_edge_map.append((edge, 'BG', kb_idx))
            edge_kb_counter[edge] = kb_idx + 1
        
        # --- DOC association batching ---
        # Group DOC edges by source document, then batch up to batch_size per source
        # Cache prepared content per source to avoid redundant _prepare_doc_content calls
        # (especially important when summarization is triggered — ensures consistent
        # content across all edges and phases for the same source document)
        doc_groups = {}  # Maps source_key -> {'content': str, 'edges': [...]}
        prepared_content_cache = {}  # Maps source_key -> prepared_content
        
        for factor_a, factor_b in all_edges:
            edge = (factor_a, factor_b)
            for doc_kb in edge_documents[edge]:
                # Resolve source_key first so we can cache by it
                source_key = getattr(doc_kb, 'source', None) or hash(doc_kb.content)
                
                if source_key not in prepared_content_cache:
                    prepared_content_cache[source_key] = self._prepare_doc_content(doc_kb.content)
                
                prepared_content = prepared_content_cache[source_key]
                kb_idx = edge_kb_counter.get(edge, 0)
                edge_doc_content[(factor_a, factor_b, kb_idx)] = prepared_content
                edge_doc_source_key[(factor_a, factor_b, kb_idx)] = source_key
                if source_key not in doc_groups:
                    doc_groups[source_key] = {'content': prepared_content, 'edges': []}
                doc_groups[source_key]['edges'].append((edge, doc_kb.kb_type, kb_idx))
                edge_kb_counter[edge] = kb_idx + 1
        
        doc_prompts = []
        doc_batch_sizes = []
        doc_batch_edge_entries = []  # Parallel to doc_prompts: list of lists of (edge, kb_type, kb_index)
        # Track which doc group each doc prompt belongs to (for context-cached execution)
        doc_prompt_source_keys = []  # Parallel to doc_prompts: source_key for each prompt
        
        for source_key, group in sorted(doc_groups.items(), key=lambda x: str(x[0])):
            content = group['content']
            edges_in_group = group['edges']
            # Sub-batch within this source document
            for i in range(0, len(edges_in_group), self.batch_size):
                sub_batch = edges_in_group[i:i + self.batch_size]
                edge_pairs_for_prompt = [entry[0] for entry in sub_batch]
                # Build only the edge-question part (no document content)
                human_prompt = self.prompts.doc_association_edge_batch(edge_pairs_for_prompt)
                doc_prompts.append(human_prompt)
                doc_batch_sizes.append(len(sub_batch))
                doc_batch_edge_entries.append(sub_batch)
                doc_prompt_source_keys.append(source_key)
        
        total_prompts = len(bg_prompts) + len(doc_prompts)
        total_edge_queries = sum(bg_batch_sizes) + sum(doc_batch_sizes)
        logger.info(f"✓ Built {total_prompts} batch prompts covering {total_edge_queries} edge queries "
              f"({len(bg_prompts)} BG batches + {len(doc_prompts)} DOC batches)\n")
        logger.info(f"  All prompts use context-cached execution (system context sent once per KB type)\n")
        
        # Execute BG batch prompts with context caching (system context sent once)
        logger.info(f"Executing {len(bg_prompts)} BG batch prompts with context caching...")
        bg_system_ctx = self.prompts.bg_association_system_context(factors, domains)
        bg_responses = self.execute_doc_context_queries(bg_system_ctx, bg_prompts) if bg_prompts else []
        logger.info(f"✓ Completed BG batch LLM queries\n")
        
        # Execute DOC batch prompts across all source documents in parallel
        logger.info(f"Executing {len(doc_prompts)} DOC batch prompts in parallel across all sources...")
        doc_responses = [None] * len(doc_prompts)
        
        # Group doc prompt indices by source_key
        source_to_prompt_indices = defaultdict(list)
        for i, sk in enumerate(doc_prompt_source_keys):
            source_to_prompt_indices[sk].append(i)
        
        # Build a flat list of (global_index, system_ctx, human_prompt) for all DOC batches
        doc_query_tasks = []  # List of (global_idx, system_ctx, human_prompt)
        for source_key, prompt_indices in sorted(source_to_prompt_indices.items(), key=lambda x: str(x[0])):
            content = doc_groups[source_key]['content']
            system_ctx = self.prompts.doc_association_system_context(factors, domains, content)
            logger.info(f"  Source '{str(source_key)[:60]}...': {len(prompt_indices)} batches")
            for global_idx in prompt_indices:
                doc_query_tasks.append((global_idx, system_ctx, doc_prompts[global_idx]))
        
        # Fire all DOC batches in parallel using the shared retry helper
        def _doc_invoke(task_tuple):
            global_idx, sys_ctx, human_msg = task_tuple
            result = self._llm_invoke_with_retry(
                lambda: self._invoke_with_messages(sys_ctx, human_msg),
                label="DOC query", index=global_idx + 1,
            )
            return result

        doc_results_flat = self._run_parallel_with_retry(
            doc_query_tasks, _doc_invoke, label="DOC query"
        )
        # Map flat results back to global indices
        for task, result in zip(doc_query_tasks, doc_results_flat):
            doc_responses[task[0]] = result
        
        doc_responses = [r if r is not None else "UNKNOWN" for r in doc_responses]
        logger.info(f"✓ Completed DOC batch LLM queries\n")
        
        # Combine responses: BG first, then DOC
        all_responses = bg_responses + doc_responses
        logger.info(f"✓ Completed all batch LLM queries\n")
        
        # Parse batch responses back into per-edge results
        logger.info(f"Parsing batch association verification results...\n")
        
        edge_association_status = {}  # Maps (factor_a, factor_b) -> List[(kb_type, kb_index, status, raw_response)]
        
        # Parse BG batch responses
        bg_response_idx = 0
        bg_edge_offset = 0
        for batch_idx, batch_size in enumerate(bg_batch_sizes):
            response_text = all_responses[bg_response_idx] if bg_response_idx < len(all_responses) else ""
            statuses = parse_batch_association_response(response_text, batch_size)
            
            for j, status in enumerate(statuses):
                edge, kb_type, kb_index = batch_edge_map[bg_edge_offset + j]
                if edge not in edge_association_status:
                    edge_association_status[edge] = []
                edge_association_status[edge].append((kb_type, kb_index, status, response_text))
            
            bg_edge_offset += batch_size
            bg_response_idx += 1
        
        # Parse DOC batch responses
        doc_response_start = len(bg_prompts)
        for batch_idx, batch_size in enumerate(doc_batch_sizes):
            resp_idx = doc_response_start + batch_idx
            response_text = all_responses[resp_idx] if resp_idx < len(all_responses) else ""
            statuses = parse_batch_association_response(response_text, batch_size)
            
            entries = doc_batch_edge_entries[batch_idx]
            for j, status in enumerate(statuses):
                edge, kb_type, kb_index = entries[j]
                if edge not in edge_association_status:
                    edge_association_status[edge] = []
                edge_association_status[edge].append((kb_type, kb_index, status, response_text))
        
        # PHASE 3: Association Type Verification (BATCHED)
        logger.info(f"\n{'='*80}")
        logger.info(f"Phase 3: Association Type Verification (BATCHED)")
        logger.info(f"{'='*80}")
        logger.info(f"For edges marked 'associated', determine if direct or indirect...\n")
        
        # Data structure for tracking association type results
        edge_association_type = {}  # Maps (factor_a, factor_b, kb_index) -> Dict
        
        # Collect associated edges by KB type for batching
        bg_type_edges = []  # List of (edge, kb_index) for BG associated edges
        doc_type_edges = {}  # Maps source_key -> list of (edge, kb_type, kb_index)
        
        for edge, status_list in sorted(edge_association_status.items()):
            factor_a, factor_b = edge
            for kb_type, kb_index, status, raw_response in status_list:
                if status == "associated":
                    if kb_type == 'BG':
                        bg_type_edges.append((edge, kb_index))
                    else:
                        doc_content = edge_doc_content.get((factor_a, factor_b, kb_index), "")
                        source_key = edge_doc_source_key.get((factor_a, factor_b, kb_index), hash(doc_content) if doc_content else kb_index)
                        if source_key not in doc_type_edges:
                            doc_type_edges[source_key] = {'content': doc_content, 'edges': []}
                        doc_type_edges[source_key]['edges'].append((edge, kb_type, kb_index))
        
        # Build batched BG association type prompts (context-cached)
        type_bg_prompts = []
        type_bg_batch_sizes = []
        type_bg_batch_entries = []  # Parallel: list of lists of (edge, kb_index)
        
        for i in range(0, len(bg_type_edges), self.batch_size):
            sub_batch = bg_type_edges[i:i + self.batch_size]
            edge_pairs_for_prompt = [entry[0] for entry in sub_batch]
            human_prompt = self.prompts.bg_association_type_edge_batch(edge_pairs_for_prompt, factors)
            type_bg_prompts.append(human_prompt)
            type_bg_batch_sizes.append(len(sub_batch))
            type_bg_batch_entries.append(sub_batch)
        
        # Build batched DOC association type prompts (context-cached)
        type_doc_prompts = []
        type_doc_batch_sizes = []
        type_doc_batch_entries = []
        type_doc_prompt_source_keys = []  # Track source_key per prompt
        
        for source_key, group in sorted(doc_type_edges.items(), key=lambda x: str(x[0])):
            content = group['content']
            edges_in_group = group['edges']
            for i in range(0, len(edges_in_group), self.batch_size):
                sub_batch = edges_in_group[i:i + self.batch_size]
                edge_pairs_for_prompt = [entry[0] for entry in sub_batch]
                # Build only the edge-question part (no document content)
                human_prompt = self.prompts.doc_association_type_edge_batch(edge_pairs_for_prompt, factors)
                type_doc_prompts.append(human_prompt)
                type_doc_batch_sizes.append(len(sub_batch))
                type_doc_batch_entries.append(sub_batch)
                type_doc_prompt_source_keys.append(source_key)
        
        total_type_prompts = len(type_bg_prompts) + len(type_doc_prompts)
        total_type_edges = sum(type_bg_batch_sizes) + sum(type_doc_batch_sizes)
        logger.info(f"Built {total_type_prompts} batch type prompts covering {total_type_edges} associated edges")
        
        # Execute association type queries
        if total_type_prompts > 0:
            # Execute BG type prompts with context caching
            logger.info(f"Executing {len(type_bg_prompts)} BG type batch prompts with context caching...")
            type_bg_system_ctx = self.prompts.bg_association_type_system_context(factors, domains)
            type_bg_responses = self.execute_doc_context_queries(type_bg_system_ctx, type_bg_prompts) if type_bg_prompts else []
            
            # Execute DOC type prompts in parallel across all sources
            logger.info(f"Executing {len(type_doc_prompts)} DOC type batch prompts in parallel across all sources...")
            type_doc_responses = [None] * len(type_doc_prompts)
            
            source_to_type_prompt_indices = defaultdict(list)
            for i, sk in enumerate(type_doc_prompt_source_keys):
                source_to_type_prompt_indices[sk].append(i)
            
            # Build flat list of (global_idx, system_ctx, human_prompt) for all DOC type batches
            doc_type_query_tasks = []
            for source_key, prompt_indices in sorted(source_to_type_prompt_indices.items(), key=lambda x: str(x[0])):
                content = doc_type_edges[source_key]['content']
                system_ctx = self.prompts.doc_association_type_system_context(factors, domains, content)
                logger.info(f"  Source '{str(source_key)[:60]}...': {len(prompt_indices)} batches")
                for global_idx in prompt_indices:
                    doc_type_query_tasks.append((global_idx, system_ctx, type_doc_prompts[global_idx]))
            
            def _doc_type_invoke(task_tuple):
                global_idx, sys_ctx, human_msg = task_tuple
                result = self._llm_invoke_with_retry(
                    lambda: self._invoke_with_messages(sys_ctx, human_msg),
                    label="DOC type query", index=global_idx + 1,
                )
                return result

            doc_type_results_flat = self._run_parallel_with_retry(
                doc_type_query_tasks, _doc_type_invoke, label="DOC type query"
            )
            for task, result in zip(doc_type_query_tasks, doc_type_results_flat):
                type_doc_responses[task[0]] = result
            
            type_doc_responses = [r if r is not None else "UNKNOWN" for r in type_doc_responses]
            
            # Combine responses
            all_type_responses = type_bg_responses + type_doc_responses
            
            logger.info(f"✓ Completed association type queries\n")
            
            # Parse BG type batch responses
            logger.info(f"Parsing association type responses...")
            for batch_idx, batch_size in enumerate(type_bg_batch_sizes):
                response_text = all_type_responses[batch_idx] if batch_idx < len(all_type_responses) else ""
                parsed_results = parse_batch_association_type_response(response_text, batch_size)
                
                entries = type_bg_batch_entries[batch_idx]
                for j, parsed_result in enumerate(parsed_results):
                    edge, kb_index = entries[j]
                    key = (edge[0], edge[1], kb_index)
                    edge_association_type[key] = parsed_result
                    
                    assoc_type = parsed_result.get('type', 'unknown')
                    intermediaries = parsed_result.get('intermediary_factors', [])
                    via_str = f" via {intermediaries}" if intermediaries else ""
                    logger.info(f"  Edge ({edge[0]}, {edge[1]}) KB BG: {assoc_type.upper()}{via_str}")
            
            # Parse DOC type batch responses
            doc_type_resp_start = len(type_bg_prompts)
            for batch_idx, batch_size in enumerate(type_doc_batch_sizes):
                resp_idx = doc_type_resp_start + batch_idx
                response_text = all_type_responses[resp_idx] if resp_idx < len(all_type_responses) else ""
                parsed_results = parse_batch_association_type_response(response_text, batch_size)
                
                entries = type_doc_batch_entries[batch_idx]
                for j, parsed_result in enumerate(parsed_results):
                    edge, kb_type, kb_index = entries[j]
                    key = (edge[0], edge[1], kb_index)
                    edge_association_type[key] = parsed_result
                    
                    assoc_type = parsed_result.get('type', 'unknown')
                    intermediaries = parsed_result.get('intermediary_factors', [])
                    via_str = f" via {intermediaries}" if intermediaries else ""
                    logger.info(f"  Edge ({edge[0]}, {edge[1]}) KB {kb_type}: {assoc_type.upper()}{via_str}")
            
            logger.info(f"✓ Parsed {total_type_edges} association type responses\n")
        else:
            logger.info(f"No edges marked 'associated' - skipping Phase 3\n")
        
        # PHASE 4: Association Rechecker (NEW - from paper)
        logger.info(f"\n{'='*80}")
        logger.info(f"Phase 4: Association Rechecker")
        logger.info(f"{'='*80}")
        logger.info(f"Verifying intermediary factors are from variable set V...\n")
        
        # Data structure for final classification results
        edge_final_classification = {}  # Maps (factor_a, factor_b, kb_index) -> str
        # Values: "directly_associated" | "indirectly_associated" | "independent" | "unknown"
        
        corrections_applied = 0
        final_direct_count = 0
        final_indirect_count = 0
        final_independent_count = 0
        final_unknown_count = 0
        
        # Apply rechecker logic for all edges
        for edge, status_list in sorted(edge_association_status.items()):
            factor_a, factor_b = edge
            
            for kb_type, kb_index, initial_status, raw_response in status_list:
                key = (factor_a, factor_b, kb_index)
                
                if initial_status == "independent":
                    # Independent edges stay independent
                    edge_final_classification[key] = "independent"
                    final_independent_count += 1
                
                elif initial_status == "associated":
                    # Check if we have Phase 3 results
                    if key in edge_association_type:
                        type_result = edge_association_type[key]
                        association_type = type_result.get('type', 'unknown')
                        intermediary_factors = type_result.get('intermediary_factors', [])
                        
                        if association_type == "indirect" and intermediary_factors:
                            # Apply rechecker: check if intermediary factors are in V
                            intermediary_in_v = check_intermediary_factors_in_v(
                                intermediary_factors, factors, factor_a, factor_b
                            )
                            
                            # Apply rechecker logic
                            final_classification = apply_rechecker_logic(association_type, intermediary_in_v)
                            
                            # Log if correction was applied
                            if final_classification == "directly_associated":
                                corrections_applied += 1
                                logger.warning(f"  ⚠️  CORRECTION: Edge ({factor_a}, {factor_b}) KB {kb_type}")
                                logger.info(f"      Indirect via {intermediary_factors} → Corrected to DIRECT (factors not in V)")
                                final_direct_count += 1
                            else:
                                final_indirect_count += 1
                            
                            edge_final_classification[key] = final_classification
                        
                        elif association_type == "indirect" and not intermediary_factors:
                            # LLM said indirect but didn't name intermediaries.
                            # Per paper: ζ=0 (indirectly associated), cannot run rechecker
                            # without intermediaries, so preserve as indirect.
                            edge_final_classification[key] = "indirectly_associated"
                            final_indirect_count += 1
                        
                        elif association_type == "direct":
                            # Direct stays direct
                            edge_final_classification[key] = "directly_associated"
                            final_direct_count += 1
                        
                        else:  # unknown from Phase 3
                            edge_final_classification[key] = "unknown"
                            final_unknown_count += 1
                    
                    else:
                        # No Phase 3 results (shouldn't happen for associated edges)
                        # Default to directly_associated for backward compatibility
                        edge_final_classification[key] = "directly_associated"
                        final_direct_count += 1
                
                else:  # unknown
                    edge_final_classification[key] = "unknown"
                    final_unknown_count += 1
        
        logger.info(f"\n✓ Phase 4 Complete:")
        logger.info(f"  - Corrections applied: {corrections_applied}")
        logger.info(f"  - Final direct associations: {final_direct_count}")
        logger.info(f"  - Final indirect associations: {final_indirect_count}")
        logger.info(f"  - Final independent: {final_independent_count}")
        logger.info(f"  - Final unknown: {final_unknown_count}")
        
        # PHASE 5: Vote Scoring and Edge Decisions (Updated with proper vote mapping)
        logger.info(f"\n{'='*80}")
        logger.info(f"Phase 5: Vote Scoring and Edge Decisions")
        logger.info(f"{'='*80}")
        logger.info(f"Mapping classifications to votes and making edge decisions...\n")
        
        # Track statistics for summary
        vote_stats = {
            'direct_votes': 0,
            'indirect_votes': 0,
            'independent_votes': 0,
            'unknown_votes': 0,
            'corrections_logged': 0
        }
        
        # Process each edge and compute vote scores
        for idx, (factor_a, factor_b) in enumerate(all_edges, 1):
            logger.info(f"[{idx}/{len(all_edges)}] Edge: {factor_a} ↔ {factor_b}")
            
            status_list = edge_association_status.get((factor_a, factor_b), [])
            S = 0  # Voting score (as per Algorithm 1 in paper)
            total_kbs = len(status_list)
            kb_details = []  # Store details for enhanced logging
            
            # Process each KB's vote using Phase 4 final classifications
            for kb_type, kb_index, initial_status, raw_response in status_list:
                # Get final classification from Phase 4
                classification_key = (factor_a, factor_b, kb_index)
                final_classification = edge_final_classification.get(classification_key, "unknown")
                
                # Map final classification to vote using the proper mapping function
                vote = map_classification_to_vote(final_classification, initial_status)
                S += vote
                
                # Track statistics
                if final_classification == "directly_associated":
                    vote_stats['direct_votes'] += 1
                elif final_classification == "indirectly_associated":
                    vote_stats['indirect_votes'] += 1
                elif final_classification == "independent":
                    vote_stats['independent_votes'] += 1
                else:
                    vote_stats['unknown_votes'] += 1
                
                # Build detailed log entry
                kb_detail = {
                    'kb_type': kb_type,
                    'initial_status': initial_status,
                    'final_classification': final_classification,
                    'vote': vote
                }
                
                # Check if rechecker correction was applied
                type_key = (factor_a, factor_b, kb_index)
                if type_key in edge_association_type:
                    type_result = edge_association_type[type_key]
                    association_type = type_result.get('type', 'unknown')
                    intermediaries = type_result.get('intermediary_factors', [])
                    
                    kb_detail['association_type'] = association_type
                    kb_detail['intermediaries'] = intermediaries
                    
                    # Check if correction was applied (indirect → direct)
                    if association_type == "indirect" and final_classification == "directly_associated":
                        kb_detail['corrected'] = True
                        vote_stats['corrections_logged'] += 1
                
                kb_details.append(kb_detail)
            
            # Enhanced logging for each KB's contribution
            for detail in kb_details:
                kb_type = detail['kb_type']
                initial = detail['initial_status']
                final = detail['final_classification']
                vote = detail['vote']
                
                # Format KB type for display
                kb_display = f"KB {kb_type:>4}" if kb_type != 'BG' else "KB   BG"
                
                # Build log message with full chain
                if final == "directly_associated":
                    logger.info(f"  {kb_display}: {initial.upper():>12} → Direct → vote {vote:+2d}")
                    
                elif final == "indirectly_associated":
                    intermediaries = detail.get('intermediaries', [])
                    if intermediaries:
                        intermediaries_str = ', '.join(intermediaries)
                        logger.info(f"  {kb_display}: {initial.upper():>12} → Indirect (via {intermediaries_str}) → vote {vote:+2d}")
                    else:
                        logger.info(f"  {kb_display}: {initial.upper():>12} → Indirect → vote {vote:+2d}")
                    
                elif final == "independent":
                    logger.info(f"  {kb_display}: {initial.upper():>12} → vote {vote:+2d}")
                    
                else:  # unknown
                    logger.info(f"  {kb_display}: {initial.upper():>12} → Unknown → vote {vote:+2d}")
                
                # Log rechecker correction with warning symbol
                if detail.get('corrected', False):
                    intermediaries = detail.get('intermediaries', [])
                    intermediaries_str = ', '.join(intermediaries)
                    logger.info(f"      ⚠️  RECHECKER: Corrected Indirect→Direct (factors {intermediaries_str} not in V)")
            
            # Decision rule from paper: Remove edge if S ≤ 0, keep if S > 0
            if S > 0:
                # Store voting information as edge attributes
                G[factor_a][factor_b]['vote_score'] = S
                G[factor_a][factor_b]['total_kbs'] = total_kbs
                
                edges_to_keep.append((factor_a, factor_b))
                logger.info(f"  → Final Score: S = {S:+d} → KEEP edge")
            else:
                edges_to_remove.append((factor_a, factor_b))
                logger.info(f"  → Final Score: S = {S:+d} → REMOVE edge")
            
            logger.info("")  # Blank line between edges
        
        # Log summary statistics for all edges
        logger.info(f"{'='*80}")
        logger.info(f"Phase 5 Summary Statistics:")
        logger.info(f"{'='*80}")
        logger.info(f"Vote Distribution:")
        logger.info(f"  - Direct associations (vote +1):     {vote_stats['direct_votes']}")
        logger.info(f"  - Indirect associations (vote -1):   {vote_stats['indirect_votes']}")
        logger.info(f"  - Independent (vote -1):             {vote_stats['independent_votes']}")
        logger.info(f"  - Unknown (vote 0):                  {vote_stats['unknown_votes']}")
        logger.info(f"  - Total votes cast:                  {sum(vote_stats.values())}")
        logger.info(f"\nRechecker Corrections:")
        logger.info(f"  - Corrections logged in Phase 5:     {vote_stats['corrections_logged']}")
        logger.info(f"\nEdge Decisions:")
        logger.info(f"  - Edges to keep (S > 0):             {len(edges_to_keep)}")
        logger.info(f"  - Edges to remove (S ≤ 0):           {len(edges_to_remove)}")
        logger.info(f"{'='*80}\n")
        
        # Step 3: Remove edges with S ≤ 0
        logger.info(f"\n{'='*80}")
        logger.info(f"Step 3: Removing Edges")
        logger.info(f"{'='*80}")
        G.remove_edges_from(edges_to_remove)
        logger.info(f"Removed {len(edges_to_remove)} edges (S ≤ 0)")
        logger.info(f"Kept {len(edges_to_keep)} edges (S > 0)")
        
        # Step 4: Compute LLM-based priors for edges in skeleton
        logger.info(f"\n{'='*80}")
        logger.info(f"Step 4: Computing Priors for Skeleton Edges")
        logger.info(f"{'='*80}")
        
        if len(edges_to_keep) > 0:
            logger.info(f"Computing LLM-based priors for {len(edges_to_keep)} skeleton edges...")
            logger.info(f"Using raw vote scores for better discrimination in Bayesian inference")
            
            for factor_a, factor_b in edges_to_keep:
                if G.has_edge(factor_a, factor_b):
                    vote_score = G[factor_a][factor_b].get('vote_score', 1)
                    total_kbs = G[factor_a][factor_b].get('total_kbs', 1)
                    
                    # Use raw vote score as prior strength (better for log space)
                    # vote_score = 1, 2, or 3 typically
                    # log(1) = 0, log(2) ≈ 0.69, log(3) ≈ 1.10
                    # These values meaningfully influence BDeu scores
                    prior_strength = max(vote_score, 0.1)  # Ensure positive for log
                    
                    # Store prior components
                    G[factor_a][factor_b]['prior_strength'] = prior_strength
                    G[factor_a][factor_b]['prior_source'] = "llm_voting_raw"
                    
                    logger.info(f"  {factor_a} - {factor_b}: Prior = {prior_strength:.1f} (raw vote: {vote_score}/{total_kbs} KBs)")

        
        logger.info(f"\n{'='*80}")
        logger.info(f"CausalIF 1 Complete:")
        logger.info(f"  - Initial edges (complete graph): {len(all_edges)}")
        logger.info(f"  - Edges removed: {len(edges_to_remove)}")
        logger.info(f"  - Final edges: {len(G.edges())}")
        logger.info(f"  - Sparsity: {len(G.edges())}/{len(all_edges)} = {len(G.edges())/max(1, len(all_edges))*100:.1f}%")
        logger.info(f"{'='*80}")
        
        return G

    def causalif_2_orientation(self, skeleton: nx.Graph, factors: List[str], domains: List[str], target_factor: str = None) -> nx.DiGraph:
        """CausalIF 2: Bayesian Orientation using Prior from Skeleton.
        
        P(Directed Graph | Data) ∝ P(Data | Directed Graph) × P(Directed Graph | Skeleton)
        """
        
        logger.info("=" * 80)
        logger.info("BAYESIAN CAUSAL INFERENCE FRAMEWORK")
        logger.info("=" * 80)
        logger.info(f"PRIOR: Skeleton graph with {len(skeleton.edges())} undirected edges (from CausalIF 1)")
        logger.info("       Represents prior belief about which variables are associated")
        logger.info(f"LIKELIHOOD: Observational data with {len(self.dataframe) if self.dataframe is not None else 0} samples")
        logger.info("POSTERIOR: Directed causal graph (to be learned)")
        logger.info("=" * 80)
        
        # Use full skeleton — degree filtering applied AFTER orientation
        skeleton_for_orientation = skeleton
        logger.info(f"\nUsing full skeleton ({len(skeleton.nodes())} nodes, {len(skeleton.edges())} edges) for Bayesian orientation")
        logger.info(f"Degree filtering will be applied after orientation for presentation")
        
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(skeleton_for_orientation.nodes())
        
        edges_to_orient = sorted(skeleton_for_orientation.edges())
        
        if not edges_to_orient:
            logger.info("No edges in prior skeleton to orient")
            return directed_graph
        
        logger.info(f"\nOrienting {len(edges_to_orient)} edges from PRIOR using Bayesian inference...")
        
        # Check if we have data for Bayesian structure learning
        if self.dataframe is not None and len(self.dataframe) > 0:
            try:
                nodes_in_skeleton = sorted(skeleton_for_orientation.nodes())
                available_columns = sorted(col for col in nodes_in_skeleton if col in self.dataframe.columns)
                
                if len(available_columns) >= 2:
                    logger.info(f"\nLIKELIHOOD: Using {len(self.dataframe)} data samples for {len(available_columns)} factors")
                    
                    df_for_learning = self.dataframe[available_columns].copy()
                    columns_to_drop = []
                    
                    # Convert datetime columns to numeric; detect date-like object columns
                    for col in df_for_learning.columns:
                        try:
                            if pd.api.types.is_datetime64_any_dtype(df_for_learning[col]):
                                df_for_learning[col] = (df_for_learning[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                                logger.info(f"  ✓ Converted datetime column '{col}' to numeric (days since epoch)")
                            elif df_for_learning[col].dtype == 'object':
                                date_keywords = ['date', 'time', 'timestamp', 'dt', 'day', 'month', 'year']
                                looks_like_date = any(keyword in col.lower() for keyword in date_keywords)
                                
                                if looks_like_date:
                                    try:
                                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y%m%d']:
                                            try:
                                                df_for_learning[col] = pd.to_datetime(df_for_learning[col], format=fmt)
                                                df_for_learning[col] = (df_for_learning[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                                                logger.info(f"  ✓ Parsed '{col}' as date (format: {fmt}) and converted to numeric")
                                                break
                                            except Exception:
                                                continue
                                        else:
                                            df_for_learning[col] = pd.to_datetime(df_for_learning[col], format='mixed')
                                            df_for_learning[col] = (df_for_learning[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                                            logger.info(f"  ✓ Parsed '{col}' as date (auto-detected format) and converted to numeric")
                                    except Exception:
                                        logger.info(f"  ℹ Column '{col}' kept as categorical")
                                else:
                                    logger.info(f"  ℹ Column '{col}' kept as categorical")
                        except Exception as e:
                            # If any conversion fails, mark column for removal
                            logger.error(f"  ✗ Failed to process column '{col}': {str(e)[:100]}")
                            columns_to_drop.append(col)
                    
                    # Drop problematic columns
                    if columns_to_drop:
                        logger.info(f"\n  Dropping {len(columns_to_drop)} problematic columns: {columns_to_drop}")
                        df_for_learning = df_for_learning.drop(columns=columns_to_drop)
                        available_columns = [col for col in available_columns if col not in columns_to_drop]
                    
                    ## Report missing values (pgmpy handles NaN natively)
                    
                    if len(df_for_learning) > 10 and len(df_for_learning.columns) >= 2:
                        logger.info(f"\n✓ Data preparation complete:")
                        logger.info(f"  - {len(df_for_learning)} samples")
                        logger.info(f"  - {len(df_for_learning.columns)} factors: {list(df_for_learning.columns)}")
                        
                        logger.info(f"  - Data types:")
                        for col in df_for_learning.columns:
                            if pd.api.types.is_numeric_dtype(df_for_learning[col]):
                                logger.info(f"    • {col}: numeric")
                            else:
                                unique_vals = df_for_learning[col].nunique()
                                logger.info(f"    • {col}: categorical ({unique_vals} unique values)")
                        
                        self._discretize_dataframe(df_for_learning, label="BDeu scoring")
                        df_for_learning = df_for_learning.fillna('missing')

                        logger.info(f"\nComputing POSTERIOR using Bayesian structure learning...")
                        logger.info(f"  Method: Hill Climbing with Prior-Weighted BDeu score (True Bayesian)")
                        logger.info(f"  Prior constraint: Only orient edges from skeleton graph")
                        logger.info(f"  Prior weighting: ENABLED (LLM voting confidence from CausalIF 1)")
                        
                        try:
                            # Seed RNGs for deterministic Hill Climb results.
                            # pgmpy's HillClimbSearch uses Python/numpy internals
                            # whose iteration order can vary across runs.
                            random.seed(42)
                            np.random.seed(42)

                            scoring_method = PriorWeightedBDeu(
                                data=df_for_learning,
                                skeleton_graph=skeleton_for_orientation,
                                prior_weight='auto',
                                equivalent_sample_size=10
                            )
                            allowed_edges = []
                            fixed_edges_undirected = []
                            
                            for factor_a, factor_b in edges_to_orient:
                                if factor_a in available_columns and factor_b in available_columns:
                                    is_related_factor = (factor_a in self.related_factors or factor_b in self.related_factors)
                                    if is_related_factor:
                                        fixed_edges_undirected.append((factor_a, factor_b))
                                    allowed_edges.append((factor_a, factor_b))
                                    allowed_edges.append((factor_b, factor_a))
                            
                            logger.info(f"  Prior edges (from skeleton): {len(allowed_edges)} possible directed edges from {len(edges_to_orient)} undirected prior edges")
                            
                            # Adaptive max_iter: more nodes need more iterations to converge
                            n_nodes = len(available_columns)
                            adaptive_max_iter = max(100, n_nodes * 20)
                            logger.info(f"  Max iterations: {adaptive_max_iter} (adaptive: max(100, {n_nodes} nodes × 20))")
                            
                            # Adaptive max-indegree to prevent BDeu state-space overflow.
                            # BDeu computes num_parents_states = prod(card(parent_i)).
                            # If that product exceeds ~2^53 (float64 integer range),
                            # pgmpy overflows in scalar multiply.  We find the
                            # largest per-variable cardinality in the discretized
                            # data and pick the highest indegree k such that
                            # max_card^k stays safely below the overflow threshold.
                            _SAFE_LIMIT = 2**53          # float64 exact-integer ceiling
                            max_card = max(df_for_learning[c].nunique() for c in df_for_learning.columns)
                            max_card = max(max_card, 2)  # guard against degenerate 1-state columns
                            max_indegree = max(2, int(math.log(_SAFE_LIMIT) / math.log(max_card)))
                            max_indegree = min(max_indegree, n_nodes - 1)
                            logger.info(f"  Max indegree: {max_indegree} (adaptive: max_card={max_card}, safe limit=2^53)")
                            
                            # True Bayesian approach: only forbid edges involving
                            # user-excluded columns.  All other edges are allowed so
                            # the data likelihood can discover relationships the LLM
                            # prior missed.  Skeleton edges receive a prior score
                            # bonus via PriorWeightedBDeu; non-skeleton edges get
                            # zero prior bonus but can still be added if the data
                            # strongly supports them.
                            forbidden_edges = []
                            
                            # Forbid edges involving excluded columns
                            excluded_columns = set()
                            if self.excluded_target_columns:
                                excluded_columns.update(col for col in self.excluded_target_columns if col in available_columns)
                            if self.excluded_related_columns:
                                excluded_columns.update(col for col in self.excluded_related_columns if col in available_columns)
                            
                            if excluded_columns:
                                for node_a in available_columns:
                                    for node_b in available_columns:
                                        if node_a != node_b and (node_a in excluded_columns or node_b in excluded_columns):
                                            forbidden_edges.append((node_a, node_b))
                                logger.info(f"  Excluded columns from posterior: {excluded_columns}")
                                logger.info(f"  Forbidden edges (excluded columns): {len(forbidden_edges)}")
                            
                            expert_knowledge = None
                            if forbidden_edges:
                                expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
   
                            # Fixed edges: start Hill Climb with them, use tabu_length to prevent removal
                            if fixed_edges_undirected:
                                valid_fixed_edges = []
                                for factor_a, factor_b in fixed_edges_undirected:
                                    if factor_a in df_for_learning.columns and factor_b in df_for_learning.columns:
                                        valid_fixed_edges.append((factor_a, factor_b))
                                    else:
                                        logger.warning(f"  ⚠ Skipping fixed edge {factor_a} - {factor_b} (not in df_for_learning)")
                                
                                if valid_fixed_edges:
                                    # DEBUG: Print what we're putting in start_dag
                                    logger.info(f"\n  DEBUG: Creating start_dag with {len(valid_fixed_edges)} edges")
                                    logger.info(f"  DEBUG: df_for_learning.columns = {list(df_for_learning.columns)}")
                                    logger.info(f"  DEBUG: Nodes in valid_fixed_edges:")
                                    nodes_in_edges = set()
                                    for a, b in valid_fixed_edges:
                                        nodes_in_edges.add(a)
                                        nodes_in_edges.add(b)
                                    logger.info(f"  DEBUG: {sorted(nodes_in_edges)}")
                                    logger.info(f"  DEBUG: Nodes NOT in df_for_learning:")
                                    missing_nodes = [n for n in nodes_in_edges if n not in df_for_learning.columns]
                                    logger.info(f"  DEBUG: {missing_nodes if missing_nodes else 'None - all nodes are valid!'}")
                                    
                                    # start_dag must have ALL variables from df_for_learning
                                    nodes_not_in_edges = [col for col in df_for_learning.columns if col not in nodes_in_edges]
                                    if nodes_not_in_edges:
                                        logger.info(f"  DEBUG: Variables in df_for_learning but NOT in fixed edges: {nodes_not_in_edges}")
                                        logger.info(f"  DEBUG: These will be added as isolated nodes to start_dag")
                                    
                                    start_dag = DiscreteBayesianNetwork(valid_fixed_edges)
                                    # Cycle check: if the fixed edges form a cycle, pgmpy will
                                    # reject the start_dag.  Detect and remove cycle-forming edges
                                    # before passing to Hill Climb.
                                    if not nx.is_directed_acyclic_graph(start_dag):
                                        logger.warning("Fixed edges form a cycle in start_dag — removing cycle-forming edges")
                                        # Rebuild without cycle-forming edges
                                        acyclic_edges = []
                                        temp_dag = nx.DiGraph()
                                        for edge in valid_fixed_edges:
                                            temp_dag.add_edge(*edge)
                                            if not nx.is_directed_acyclic_graph(temp_dag):
                                                temp_dag.remove_edge(*edge)
                                                logger.warning(f"  Removed cycle-forming edge: {edge[0]} → {edge[1]}")
                                            else:
                                                acyclic_edges.append(edge)
                                        valid_fixed_edges = acyclic_edges
                                        start_dag = DiscreteBayesianNetwork(valid_fixed_edges) if valid_fixed_edges else DiscreteBayesianNetwork()

                                    for node in nodes_not_in_edges:
                                        start_dag.add_node(node)
                                    
                                    logger.info(f"\n  Starting Hill Climb with {len(valid_fixed_edges)} fixed edges + {len(nodes_not_in_edges)} isolated nodes")
                                    
                                    best_model = _run_hill_climb(
                                        data=df_for_learning,
                                        scoring_method=scoring_method, 
                                        max_iter=adaptive_max_iter,
                                        max_indegree=max_indegree,
                                        start_dag=start_dag,
                                        tabu_length=len(valid_fixed_edges) * 2,
                                        expert_knowledge=expert_knowledge,
                                        show_progress=False
                                    )
                                else:
                                    logger.info(f"\n  No valid fixed edges (all involve factors not in df_for_learning)")
                                    best_model = _run_hill_climb(
                                        data=df_for_learning,
                                        scoring_method=scoring_method, 
                                        max_iter=adaptive_max_iter,
                                        max_indegree=max_indegree,
                                        expert_knowledge=expert_knowledge,
                                        show_progress=False
                                    )
                            else:
                                best_model = _run_hill_climb(
                                    data=df_for_learning,
                                    scoring_method=scoring_method, 
                                    max_iter=adaptive_max_iter,
                                    max_indegree=max_indegree,
                                    expert_knowledge=expert_knowledge,
                                    show_progress=False
                                )
                            
                            logger.info(f"\nPOSTERIOR: Learned {len(best_model.edges())} directed edges from Bayesian inference")
                            logger.info("  These represent the most probable causal directions given:")
                            logger.info("    1. PRIOR: Skeleton edges from CausalIF 1 (score bonus via PriorWeightedBDeu)")
                            logger.info("    2. LIKELIHOOD: Observational data (can discover edges the prior missed)")
                            
                            bayesian_edge_count = 0
                            data_discovered_count = 0
                            skeleton_confirmed_count = 0
                            logger.info("\nPOSTERIOR Edges (MAP Estimate):")
                            
                            for factor_a, factor_b in best_model.edges():
                                in_skeleton = (skeleton_for_orientation.has_edge(factor_a, factor_b) 
                                             or skeleton_for_orientation.has_edge(factor_b, factor_a))
                                directed_graph.add_edge(factor_a, factor_b)
                                bayesian_edge_count += 1
                                if in_skeleton:
                                    logger.info(f"  ✓ {factor_a} → {factor_b} (prior + data)")
                                    skeleton_confirmed_count += 1
                                else:
                                    # Edge discovered purely by data — not in CausalIF 1 skeleton
                                    directed_graph[factor_a][factor_b]['data_discovered'] = True
                                    logger.info(f"  ★ {factor_a} → {factor_b} (data-discovered, not in prior)")
                                    data_discovered_count += 1
                            
                            # Check for rejected skeleton edges
                            rejected_count = 0
                            for factor_a, factor_b in edges_to_orient:
                                if factor_a in available_columns and factor_b in available_columns:
                                    if not best_model.has_edge(factor_a, factor_b) and not best_model.has_edge(factor_b, factor_a):
                                        logger.info(f"  ✗ Rejected by data: {factor_a} - {factor_b} (was in prior)")
                                        rejected_count += 1
                            
                            logger.info(f"\nPOSTERIOR Summary:")
                            logger.info(f"  Total edges: {bayesian_edge_count}")
                            logger.info(f"  Prior-confirmed: {skeleton_confirmed_count} (in skeleton + supported by data)")
                            logger.info(f"  Data-discovered: {data_discovered_count} (not in skeleton, found by data alone)")
                            logger.info(f"  Prior-rejected: {rejected_count} (in skeleton, rejected by data)")
                            
                            # Bootstrap Stability Validation
                            if self.bootstrap_iterations > 0:
                                logger.info(f"\n{'='*80}")
                                logger.info(f"Bootstrap Stability Validation")
                                logger.info(f"{'='*80}")
                                logger.info(f"Running {self.bootstrap_iterations} bootstrap resamples to assess edge robustness...")
                                logger.info(f"Threshold: edges appearing in <{self.bootstrap_threshold*100:.0f}% of resamples will be pruned\n")
                                
                                edge_counts = defaultdict(int)  # (A, B) directed count
                                edge_pair_counts = defaultdict(int)  # frozenset({A, B}) either-direction count
                                n_rows = len(df_for_learning)
                                successful_iterations = 0
                                
                                # Bootstrap iterations are independent — run in
                                # parallel threads.  pgmpy/numpy release the GIL
                                # for heavy numeric work so threads give real speedup.
                                
                                def _run_single_bootstrap(b):
                                    """Run one bootstrap resample and return its edges."""
                                    np.random.seed(42 + b)
                                    random.seed(42 + b)
                                    boot_indices = np.random.choice(n_rows, size=n_rows, replace=True)
                                    boot_df = df_for_learning.iloc[boot_indices].reset_index(drop=True)
                                    
                                    pgmpy_logger = logging.getLogger('pgmpy')
                                    prev_level = pgmpy_logger.level
                                    pgmpy_logger.setLevel(logging.WARNING)
                                    # Suppress pgmpy's tqdm progress bars during bootstrap
                                    try:
                                        from pgmpy import config as _pgmpy_config
                                        _prev_show_progress = _pgmpy_config.SHOW_PROGRESS
                                        _pgmpy_config.SHOW_PROGRESS = False
                                    except (ImportError, AttributeError):
                                        _pgmpy_config = None
                                        _prev_show_progress = None
                                    try:
                                        boot_scoring = PriorWeightedBDeu(
                                            data=boot_df,
                                            skeleton_graph=skeleton_for_orientation,
                                            prior_weight='auto',
                                            equivalent_sample_size=10,
                                            quiet=True
                                        )
                                        
                                        if fixed_edges_undirected and valid_fixed_edges:
                                            boot_start_dag = DiscreteBayesianNetwork(valid_fixed_edges)
                                            for node in nodes_not_in_edges:
                                                boot_start_dag.add_node(node)
                                            boot_model = _run_hill_climb(
                                                data=boot_df,
                                                scoring_method=boot_scoring,
                                                max_iter=adaptive_max_iter,
                                                max_indegree=max_indegree,
                                                start_dag=boot_start_dag,
                                                tabu_length=len(valid_fixed_edges) * 2,
                                                expert_knowledge=expert_knowledge,
                                                show_progress=False
                                            )
                                        else:
                                            boot_model = _run_hill_climb(
                                                data=boot_df,
                                                scoring_method=boot_scoring,
                                                max_iter=adaptive_max_iter,
                                                max_indegree=max_indegree,
                                                expert_knowledge=expert_knowledge,
                                                show_progress=False
                                            )
                                        return list(boot_model.edges())
                                    except Exception as e:
                                        logger.debug(f"  Bootstrap iteration {b+1} failed: {str(e)[:80]}")
                                        return None
                                    finally:
                                        pgmpy_logger.setLevel(prev_level)
                                        if _pgmpy_config is not None:
                                            _pgmpy_config.SHOW_PROGRESS = _prev_show_progress
                                
                                import os
                                n_workers = min(self.bootstrap_iterations, max(1, os.cpu_count() or 4))
                                logger.info(f"  Using {n_workers} parallel workers\n")
                                
                                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                                    futures = {executor.submit(_run_single_bootstrap, b): b 
                                              for b in range(self.bootstrap_iterations)}
                                    for future in concurrent.futures.as_completed(futures):
                                        result = future.result()
                                        if result is not None:
                                            for edge in result:
                                                edge_counts[edge] += 1
                                                edge_pair_counts[frozenset(edge)] += 1
                                            successful_iterations += 1
                                
                                if successful_iterations == 0:
                                    logger.warning("All bootstrap iterations failed — skipping stability pruning")
                                    self._bootstrap_pruned_edges = set()
                                else:
                                    if successful_iterations < self.bootstrap_iterations:
                                        logger.warning(f"  {self.bootstrap_iterations - successful_iterations} bootstrap iterations failed, using {successful_iterations} successful runs")
                                    
                                    # Compute stability scores and prune
                                    edges_before = list(directed_graph.edges())
                                    pruned_edges = []
                                    
                                    logger.info(f"Edge stability scores ({successful_iterations} successful resamples):")
                                    for factor_a, factor_b in sorted(edges_before):
                                        # Directed stability: exact direction match
                                        dir_freq = edge_counts.get((factor_a, factor_b), 0)
                                        dir_stability = dir_freq / successful_iterations
                                        # Undirected stability: edge present in either direction
                                        pair_freq = edge_pair_counts.get(frozenset({factor_a, factor_b}), 0)
                                        pair_stability = pair_freq / successful_iterations
                                        
                                        directed_graph[factor_a][factor_b]['bootstrap_stability'] = dir_stability
                                        directed_graph[factor_a][factor_b]['bootstrap_pair_stability'] = pair_stability
                                        
                                        # Prune based on directed stability
                                        if dir_stability < self.bootstrap_threshold:
                                            pruned_edges.append((factor_a, factor_b))
                                            logger.info(f"  ✗ {factor_a} → {factor_b}: {dir_stability*100:.0f}% directed, {pair_stability*100:.0f}% either — PRUNED")
                                        else:
                                            logger.info(f"  ✓ {factor_a} → {factor_b}: {dir_stability*100:.0f}% directed, {pair_stability*100:.0f}% either")
                                    
                                    if pruned_edges:
                                        directed_graph.remove_edges_from(pruned_edges)
                                        # Track pruned edges so they aren't re-added as dashed/unoriented
                                        self._bootstrap_pruned_edges = set()
                                        for fa, fb in pruned_edges:
                                            self._bootstrap_pruned_edges.add((fa, fb))
                                            self._bootstrap_pruned_edges.add((fb, fa))
                                        logger.info(f"\nPruned {len(pruned_edges)} unstable edges (below {self.bootstrap_threshold*100:.0f}% threshold)")
                                    else:
                                        self._bootstrap_pruned_edges = set()
                                        logger.info(f"\nAll edges are stable (≥{self.bootstrap_threshold*100:.0f}%)")
                                    
                                    logger.info(f"Final posterior: {len(directed_graph.edges())} edges")
                            
                            # Log unoriented edges (missing data)
                            unoriented_edges = []
                            for factor_a, factor_b in edges_to_orient:
                                factor_a_missing = factor_a not in available_columns
                                factor_b_missing = factor_b not in available_columns
                                
                                if factor_a_missing or factor_b_missing:
                                    missing_factors = []
                                    if factor_a_missing:
                                        missing_factors.append(factor_a)
                                    if factor_b_missing:
                                        missing_factors.append(factor_b)
                                    unoriented_edges.append((factor_a, factor_b, missing_factors))
                            
                            if unoriented_edges:
                                logger.info(f"\n⚠️  {len(unoriented_edges)} skeleton edges could not be oriented (missing data):")
                                for fa, fb, missing in unoriented_edges:
                                    logger.info(f"    {fa} - {fb} (no data for: {', '.join(missing)})")
                                logger.info(f"  These edges remain in skeleton graph but are excluded from causal DAG")
                            
                        except Exception as e:
                            logger.info(f"\nBayesian structure learning failed: {e}")
                            raise RuntimeError(
                                "CausalIF 2 orientation failed: Bayesian structure learning encountered an error.\n"
                                f"Error details: {e}\n"
                                "The original CausalIF algorithm requires either:\n"
                                "  1. Valid observational data for Bayesian structure learning, OR\n"
                                "  2. An LLM model for causal direction queries (not yet implemented)\n"
                                "Please ensure your dataframe has sufficient samples and valid data types."
                            )
                    else:
                        raise ValueError(
                            "CausalIF 2 orientation failed: Insufficient data samples.\n"
                            f"Current samples: {len(df_for_learning)}, Required: >10 samples\n"
                            "The original CausalIF algorithm requires sufficient observational data "
                            "for reliable Bayesian structure learning. Please provide more data samples."
                        )
                else:
                    raise ValueError(
                        "CausalIF 2 orientation failed: Insufficient columns in dataframe.\n"
                        f"Available columns: {len(available_columns)}, Required: ≥2 columns\n"
                        "The original CausalIF algorithm requires at least 2 factors with data "
                        "for Bayesian structure learning. Please ensure your dataframe contains "
                        "the factors specified in the skeleton graph."
                    )
                    
            except Exception as e:
                raise RuntimeError(
                    f"CausalIF 2 orientation failed: Error during Bayesian structure learning.\n"
                    f"Error details: {e}\n"
                    "The original CausalIF algorithm requires valid observational data for orientation. "
                    "Please check that your dataframe:\n"
                    "  - Contains the factors from the skeleton graph\n"
                    "  - Has valid data types (numeric or categorical)\n"
                    "  - Has sufficient samples (>10 recommended)\n"
                    "  - Does not have excessive missing values"
                )
        else:
            raise ValueError(
                "CausalIF 2 orientation failed: No dataframe available.\n"
                "The original CausalIF algorithm requires observational data for Bayesian structure learning "
                "to determine causal directions (POSTERIOR). Please provide a dataframe with:\n"
                "  - Columns matching the factors in your skeleton graph\n"
                "  - Sufficient samples (>10 recommended)\n"
                "  - Valid data types (numeric or categorical)\n"
                "Alternative: The paper describes LLM-based causal direction queries, but this is not yet implemented."
            )
        
        # Add unoriented skeleton edges to the directed graph as dashed lines.
        # ONLY for edges where one or both factors lack data (related factors
        # not in the dataframe). Edges that Hill Climb rejected are dropped.
        skeleton_edges_added = 0
        bootstrap_pruned = getattr(self, '_bootstrap_pruned_edges', set())
        df_columns = set(self.dataframe.columns) if self.dataframe is not None else set()
        for factor_a, factor_b in skeleton_for_orientation.edges():
            if (factor_a, factor_b) in bootstrap_pruned:
                continue  # Edge was pruned by bootstrap — don't re-add as dashed
            # Only add as dashed if at least one factor has no data
            factor_a_in_data = factor_a in df_columns
            factor_b_in_data = factor_b in df_columns
            if factor_a_in_data and factor_b_in_data:
                continue  # Both factors have data — Hill Climb had a chance to evaluate this edge
            if not directed_graph.has_edge(factor_a, factor_b) and not directed_graph.has_edge(factor_b, factor_a):
                directed_graph.add_edge(factor_a, factor_b, undirected=True)
                if skeleton_for_orientation[factor_a][factor_b].get('prior_strength'):
                    ps = skeleton_for_orientation[factor_a][factor_b]['prior_strength']
                    directed_graph[factor_a][factor_b]['prior_strength'] = ps
                skeleton_edges_added += 1
        
        if skeleton_edges_added > 0:
            logger.info(f"\n  Added {skeleton_edges_added} unoriented skeleton edges (LLM-only, no data)")
            logger.info(f"  These will appear as dashed lines in visualization")
        
        # Filter by degrees after orientation if max_degrees is specified
        if target_factor and target_factor in directed_graph.nodes() and self.max_degrees is not None:
            logger.info(f"\nFiltering causal graph to {self.max_degrees} degrees from {target_factor}...")
            logger.info(f"  Before filtering: {len(directed_graph.nodes())} nodes, {len(directed_graph.edges())} edges")
            
            directed_graph = self.filter_graph_by_degrees(directed_graph, target_factor, self.max_degrees)
            
            logger.info(f"  After filtering: {len(directed_graph.nodes())} nodes, {len(directed_graph.edges())} edges")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"BAYESIAN INFERENCE COMPLETE")
        logger.info(f"POSTERIOR: {len(directed_graph.edges())} directed causal edges")
        logger.info("=" * 80)
        return directed_graph


    def run_complete_causalif(self, factors: List[str], domains: List[str], target_factor: str = None) -> Tuple[nx.Graph, nx.DiGraph]:
        """Run complete CausalIF: Prior (skeleton) → Posterior (directed causal graph)."""
        logger.info("\n" + "=" * 100)
        logger.info("BAYESIAN CausalIF: PRIOR → POSTERIOR CAUSAL INFERENCE")
        logger.info("=" * 100)
        logger.info(f"Factors: {factors}")
        logger.info(f"Domains: {domains}")
        logger.info(f"Max degrees of separation: {self.max_degrees if self.max_degrees is not None else 'None (no filtering)'}")
        logger.info(f"Max parallel queries: {self.max_parallel_queries}")
        logger.info(f"RAG retriever available: {self.retriever_tool is not None or self.retriever is not None}")
        logger.info(f"Dataframe available: {self.dataframe is not None}")
        if self.dataframe is not None:
            logger.info(f"Data samples: {len(self.dataframe)}")
        if target_factor:
            logger.info(f"Target factor: {target_factor}")
        logger.info("=" * 100)

        logger.info("\n" + "=" * 100)
        logger.info("STEP 1: CausalIF 1 - Building PRIOR (Edge Existence Verification)")
        logger.info("=" * 100)
        logger.info("Using LLM + RAG to determine which variable pairs are associated")
        logger.info("This creates our PRIOR belief about the causal structure")
        skeleton = self.causalif_1_edge_existence_verification(factors, domains, target_factor)
        logger.info(f"\nPRIOR (Skeleton) edges: {list(skeleton.edges())}")
        logger.info(f"PRIOR contains {len(skeleton.edges())} undirected associations")

        logger.info("\n" + "=" * 100)
        logger.info("STEP 2: CausalIF 2 - Computing POSTERIOR (Bayesian Orientation)")
        logger.info("=" * 100)
        logger.info("Using Bayesian structure learning to orient edges from PRIOR")
        logger.info("This updates our beliefs using observational data")
        causal_graph = self.causalif_2_orientation(skeleton, factors, domains, target_factor)
        logger.info(f"\nPOSTERIOR (Causal) edges: {list(causal_graph.edges())}")
        logger.info(f"POSTERIOR contains {len(causal_graph.edges())} directed causal relationships")

        logger.info("\n" + "=" * 100)
        logger.info("BAYESIAN CausalIF COMPLETE")
        logger.info("=" * 100)
        logger.info(f"Prior → Posterior transformation:")
        logger.info(f"  {len(skeleton.edges())} undirected associations → {len(causal_graph.edges())} directed causal edges")
        logger.info("=" * 100 + "\n")

        # Optionally fit causal model for inference
        if self.enable_causal_estimate:
            logger.info("\n" + "=" * 100)
            logger.info("STEP 3: Fitting Causal Model for Inference (OPTIONAL)")
            logger.info("=" * 100)
            fitted_model = self.fit_causal_model(causal_graph)
            
            if fitted_model and target_factor and self.causal_inference_engine:
                # Automatically get causal summary for target
                logger.info(f"\nGenerating causal summary for target: {target_factor}")
                summary = self.get_causal_summary(target_factor, causal_graph)
                
                if summary['direct_causes']:
                    logger.info(f"\n  Direct causes of {target_factor}:")
                    for cause in summary['direct_causes']:
                        adj_set = summary['adjustment_sets'].get(cause, [])
                        if adj_set:
                            logger.info(f"    • {cause} (adjust for: {adj_set})")
                        else:
                            logger.info(f"    • {cause} (no adjustment needed)")
                
                logger.info("=" * 100)

        return skeleton, causal_graph

    def fit_causal_model(self, causal_graph: nx.DiGraph) -> DiscreteBayesianNetwork:
        """Fit a Bayesian Network with CPDs from the causal DAG for causal inference."""
        if not self.enable_causal_estimate:
            logger.info("⚠️  Causal inference is disabled. Enable with enable_causal_estimate=True")
            return None
        
        if self.dataframe is None:
            logger.info("⚠️  No dataframe available for fitting CPDs")
            return None
        
        logger.info("\n" + "=" * 80)
        logger.info("FITTING CAUSAL MODEL (Learning CPDs for Causal Inference)")
        logger.info("=" * 80)
        
        try:
            # Only include directed edges where both nodes have data
            available_columns = [col for col in causal_graph.nodes() if col in self.dataframe.columns]
            valid_edges = [(u, v) for u, v in causal_graph.edges() 
                          if u in available_columns and v in available_columns
                          and not causal_graph[u][v].get('undirected', False)]
            
            if not valid_edges:
                logger.info("⚠️  No valid directed edges with data for causal inference")
                return None
            
            # Create Bayesian Network
            bn = DiscreteBayesianNetwork(valid_edges)
            
            nodes_in_bn = list(bn.nodes())
            df_for_fitting = self.dataframe[nodes_in_bn].copy()
            
            logger.info(f"  Preparing data for {len(nodes_in_bn)} variables...")
            
            # Reuses cached discretizers from CausalIF 2
            self._discretize_dataframe(df_for_fitting, label="CPD fitting")
            
            # Fill NaN values
            for col in df_for_fitting.columns:
                if df_for_fitting[col].dtype.name == 'category':
                    if 'missing' not in df_for_fitting[col].cat.categories:
                        df_for_fitting[col] = df_for_fitting[col].cat.add_categories(['missing'])
                    df_for_fitting[col] = df_for_fitting[col].fillna('missing')
                else:
                    df_for_fitting[col] = df_for_fitting[col].fillna('missing')
            
            logger.info(f"\n  Fitting CPDs for {len(bn.nodes())} nodes, {len(bn.edges())} edges")
            logger.info(f"  Using {len(df_for_fitting)} data samples")
            
            # Fit CPDs using MLE
            bn.fit(df_for_fitting, estimator=DiscreteMLE())
            
            logger.info(f"  ✓ Successfully fitted {len(bn.get_cpds())} CPDs")
            
            # Store for later use
            self.causal_model = bn
            self.causal_inference_engine = CausalInference(bn)
            
            logger.info("  ✓ Causal inference engine initialized")
            logger.info("=" * 80)
            
            return bn
            
        except Exception as e:
            logger.error(f"  ✗ Failed to fit causal model: {e}")
            logger.exception("Traceback:")
            return None
    
    def do(self, target: str, do_vars: Dict[str, str], causal_graph: nx.DiGraph = None) -> Dict:
        """Perform do-operator: compute P(target | do(X1=x1, X2=x2, ...)).
        
        Returns dict with distribution, most_likely_value, max_probability.
        Raises ValueError if causal inference is disabled or variables not in model.
        """
        if not self.enable_causal_estimate:
            raise ValueError("Causal inference is disabled. Enable with enable_causal_estimate=True")

        # Fit model if needed
        if self.causal_inference_engine is None:
            if causal_graph is not None:
                self.fit_causal_model(causal_graph)
            else:
                raise ValueError("No causal model available. Pass causal_graph or call fit_causal_model() first.")

        if self.causal_inference_engine is None:
            raise ValueError("Failed to fit causal model — cannot perform do-operation.")

        # Validate variables exist in the model
        model_nodes = set(self.causal_model.nodes())
        if target not in model_nodes:
            raise ValueError(f"Target '{target}' not in model. Available: {sorted(model_nodes)}")
        for var in do_vars:
            if var not in model_nodes:
                raise ValueError(f"Intervention variable '{var}' not in model. Available: {sorted(model_nodes)}")

        logger.info(f"\n[do-operator] Computing P({target} | do({do_vars}))")

        # Build bin-label → value-range mapping for human-readable output
        def _bin_range_label(var_name: str, bin_label: str) -> str:
            """Map a discretized bin label like '3' to a human-readable range like '[2.5, 5.0)'."""
            kbd = self._cached_discretizers.get(var_name)
            if kbd is None:
                return bin_label
            try:
                idx = int(bin_label)
                edges = kbd.bin_edges_[0]
                if idx < 0 or idx >= len(edges) - 1:
                    return bin_label
                lo, hi = edges[idx], edges[idx + 1]
                return f"[{lo:.2f}, {hi:.2f})"
            except (ValueError, IndexError):
                return bin_label

        # Build readable do_vars for logging
        do_display_parts = []
        for var, val in do_vars.items():
            range_label = _bin_range_label(var, val)
            if range_label != val:
                do_display_parts.append(f"{var}={val} ({range_label})")
            else:
                do_display_parts.append(f"{var}={val}")
        do_display = ', '.join(do_display_parts)

        try:
            result = self.causal_inference_engine.query(
                variables=[target],
                do=do_vars
            )

            # Extract interventional distribution
            dist = {}
            state_names = result.state_names[target]
            values = result.values
            for state, prob in zip(state_names, values):
                dist[str(state)] = float(prob)

            most_likely = max(dist, key=dist.get)

            # Compute baseline (observational) distribution: P(target) with no intervention
            baseline_dist = {}
            try:
                baseline_result = self.causal_inference_engine.query(
                    variables=[target],
                    do=None
                )
                bl_states = baseline_result.state_names[target]
                bl_values = baseline_result.values
                for state, prob in zip(bl_states, bl_values):
                    baseline_dist[str(state)] = float(prob)
            except Exception:
                baseline_dist = {}

            # Compute weighted means using bin midpoints to determine direction
            def _weighted_mean(distribution: dict, var_name: str) -> float:
                """Compute E[var] from a discrete distribution using bin midpoints."""
                kbd = self._cached_discretizers.get(var_name)
                if kbd is None:
                    # No discretizer — try using bin labels as numeric values
                    try:
                        return sum(float(s) * p for s, p in distribution.items())
                    except (ValueError, TypeError):
                        return float('nan')
                edges = kbd.bin_edges_[0]
                total = 0.0
                for s, p in distribution.items():
                    try:
                        idx = int(s)
                        if 0 <= idx < len(edges) - 1:
                            midpoint = (edges[idx] + edges[idx + 1]) / 2.0
                            total += midpoint * p
                    except (ValueError, IndexError):
                        continue
                return total

            interventional_mean = _weighted_mean(dist, target)
            baseline_mean = _weighted_mean(baseline_dist, target) if baseline_dist else float('nan')

            # Determine relationship direction by comparing cause and effect shifts
            direction = "unknown"
            effect_shift = float('nan')
            cause_direction = float('nan')

            if not (np.isnan(interventional_mean) or np.isnan(baseline_mean)):
                effect_shift = interventional_mean - baseline_mean

                # Compute cause variable's baseline mean and intervention midpoint
                # to determine whether the intervention pushed the cause UP or DOWN
                for do_var, do_val in do_vars.items():
                    cause_baseline = _weighted_mean(baseline_dist, do_var) if baseline_dist else float('nan')
                    # If no baseline for cause, compute it from observational P(cause)
                    if np.isnan(cause_baseline):
                        try:
                            cause_obs = self.causal_inference_engine.query(variables=[do_var], do=None)
                            cause_obs_dist = {str(s): float(p) for s, p in
                                              zip(cause_obs.state_names[do_var], cause_obs.values)}
                            cause_baseline = _weighted_mean(cause_obs_dist, do_var)
                        except Exception:
                            cause_baseline = float('nan')

                    # Get the intervention midpoint for the cause
                    kbd_cause = self._cached_discretizers.get(do_var)
                    if kbd_cause is not None:
                        try:
                            idx = int(do_val)
                            edges = kbd_cause.bin_edges_[0]
                            if 0 <= idx < len(edges) - 1:
                                intervention_midpoint = (edges[idx] + edges[idx + 1]) / 2.0
                            else:
                                intervention_midpoint = float('nan')
                        except (ValueError, IndexError):
                            intervention_midpoint = float('nan')
                    else:
                        try:
                            intervention_midpoint = float(do_val)
                        except (ValueError, TypeError):
                            intervention_midpoint = float('nan')

                    if not (np.isnan(cause_baseline) or np.isnan(intervention_midpoint)):
                        cause_direction = intervention_midpoint - cause_baseline
                    break  # Use first do_var for direction

                # Compare cause and effect directions
                if not (np.isnan(cause_direction) or np.isnan(effect_shift)):
                    if abs(effect_shift) < 1e-6:
                        direction = "neutral"
                    elif (cause_direction > 0 and effect_shift > 0) or (cause_direction < 0 and effect_shift < 0):
                        direction = "positive"  # same direction = directly related
                    else:
                        direction = "negative"  # opposite direction = inversely related

            # Print interventional distribution
            logger.info(f"  ✓ P({target} | do({do_display})):")
            for state, prob in sorted(dist.items(), key=lambda x: -x[1]):
                marker = " ◀" if state == most_likely else ""
                range_label = _bin_range_label(target, state)
                shift_str = ""
                if baseline_dist and state in baseline_dist:
                    delta = prob - baseline_dist[state]
                    shift_str = f"  ({delta:+.4f} vs baseline)"
                if range_label != state:
                    logger.info(f"    bin {state} {range_label}: {prob:.4f}{marker}{shift_str}")
                else:
                    logger.info(f"    {state}: {prob:.4f}{marker}{shift_str}")

            # Print direction summary
            if direction != "unknown":
                do_var_names = list(do_vars.keys())
                arrow = "↑" if direction == "positive" else ("↓" if direction == "negative" else "→")
                relation = "directly" if direction == "positive" else ("inversely" if direction == "negative" else "neutrally")
                cause_arrow = "↑" if not np.isnan(cause_direction) and cause_direction > 0 else "↓"
                effect_arrow = "↑" if effect_shift > 0 else "↓"
                logger.info(f"\n  📊 Direction: {', '.join(do_var_names)} {cause_arrow} → {target} {effect_arrow} ({relation} related)")
                logger.info(f"     Baseline E[{target}] = {baseline_mean:.4f}")
                logger.info(f"     Interventional E[{target}] = {interventional_mean:.4f}")
                logger.info(f"     Effect shift: {effect_shift:+.4f}")

            # Build human-readable distribution with ranges
            dist_with_ranges = {}
            for state, prob in dist.items():
                range_label = _bin_range_label(target, state)
                if range_label != state:
                    dist_with_ranges[f"bin {state} {range_label}"] = prob
                else:
                    dist_with_ranges[state] = prob

            most_likely_range = _bin_range_label(target, most_likely)
            most_likely_display = f"bin {most_likely} ({most_likely_range})" if most_likely_range != most_likely else most_likely

            return {
                "target": target,
                "do": do_vars,
                "distribution": dist,
                "distribution_with_ranges": dist_with_ranges,
                "baseline_distribution": baseline_dist,
                "most_likely_value": most_likely,
                "most_likely_display": most_likely_display,
                "max_probability": dist[most_likely],
                "direction": direction,
                "baseline_mean": baseline_mean if not np.isnan(baseline_mean) else None,
                "interventional_mean": interventional_mean if not np.isnan(interventional_mean) else None,
                "mean_shift": effect_shift if not np.isnan(effect_shift) else None,
            }

        except ValueError as e:
            err_msg = str(e)
            if "Invalid causal query" in err_msg or "descendants" in err_msg.lower():
                # pgmpy rejects do(descendant) → query(ancestor) — explain why
                do_var_names = list(do_vars.keys())
                logger.info(f"\n  ⚠️  Invalid causal direction: cannot intervene on {do_var_names} and query {target}")
                logger.info(f"     The causal graph has {target} → {', '.join(do_var_names)} (or {target} is an ancestor).")
                logger.info(f"     The do-operator only works in the causal direction: cause → effect.")
                logger.info(f"     Try: \"what happens to {do_var_names[0]} if {target} is high\" instead.")
                return {
                    "target": target,
                    "do": do_vars,
                    "error": "invalid_causal_direction",
                    "message": (
                        f"Cannot compute P({target} | do({do_vars})). "
                        f"The causal graph shows {target} causes {', '.join(do_var_names)}, not the reverse. "
                        f"The do-operator only works in the causal direction (ancestor → descendant)."
                    ),
                    "suggestion": f"Try querying {do_var_names[0]} with do({target}=...) instead.",
                    "distribution": {},
                    "most_likely_value": None,
                    "max_probability": None,
                    "direction": None,
                }
            logger.error(f"  ✗ do-operator failed: {e}")
            raise
        except Exception as e:
            logger.error(f"  ✗ do-operator failed: {e}")
            raise

    def compute_ate(self, cause: str, target: str, causal_graph: nx.DiGraph = None) -> Dict:
        """Compute Average Treatment Effect of cause on target via do-calculus."""
        if not self.enable_causal_estimate:
            raise ValueError("Causal inference is disabled. Enable with enable_causal_estimate=True")

        # Ensure model is fitted
        if self.causal_inference_engine is None:
            if causal_graph is not None:
                self.fit_causal_model(causal_graph)
            else:
                raise ValueError("No causal model available.")

        if self.causal_inference_engine is None:
            raise ValueError("Failed to fit causal model.")

        model_nodes = set(self.causal_model.nodes())
        if cause not in model_nodes:
            raise ValueError(f"Cause '{cause}' not in model. Available: {sorted(model_nodes)}")
        if target not in model_nodes:
            raise ValueError(f"Target '{target}' not in model. Available: {sorted(model_nodes)}")

        # Get adjustment set (informational)
        try:
            adj_set = self.causal_inference_engine.get_minimal_adjustment_set(X=cause, Y=target)
            adj_set = list(adj_set) if adj_set else []
        except Exception:
            adj_set = None

        # Get all states of the cause variable from the CPD
        cause_cpd = self.causal_model.get_cpds(cause)
        cause_states = [str(s) for s in cause_cpd.state_names[cause]]

        logger.info(f"\n[ATE] Estimating effect of {cause} on {target}")
        logger.info(f"  Cause states: {cause_states}")
        if adj_set is not None:
            logger.info(f"  Adjustment set: {adj_set if adj_set else 'empty (no confounders)'}")

        # Compute P(target | do(cause=x)) for each x
        interventions = {}
        for state in cause_states:
            try:
                result = self.causal_inference_engine.query(
                    variables=[target],
                    do={cause: state}
                )
                state_names = result.state_names[target]
                values = result.values
                interventions[state] = {str(s): float(p) for s, p in zip(state_names, values)}
            except Exception as e:
                logger.warning(f"  ⚠️  do({cause}={state}) failed: {e}")
                interventions[state] = None

        # Compute ATE summary: max absolute shift per target state
        valid = {k: v for k, v in interventions.items() if v is not None}
        ate_summary = {}
        if len(valid) >= 2:
            target_states = list(next(iter(valid.values())).keys())
            for ts in target_states:
                probs = [v[ts] for v in valid.values()]
                ate_summary[ts] = round(max(probs) - min(probs), 6)

        ate_max = max(ate_summary.values()) if ate_summary else 0.0

        # Compute direction: how E[target] changes as cause increases
        def _weighted_mean_from_dist(distribution: dict, var_name: str) -> float:
            kbd = self._cached_discretizers.get(var_name)
            if kbd is None:
                try:
                    return sum(float(s) * p for s, p in distribution.items())
                except (ValueError, TypeError):
                    return float('nan')
            edges = kbd.bin_edges_[0]
            total = 0.0
            for s, p in distribution.items():
                try:
                    idx = int(s)
                    if 0 <= idx < len(edges) - 1:
                        total += ((edges[idx] + edges[idx + 1]) / 2.0) * p
                except (ValueError, IndexError):
                    continue
            return total

        # Compute E[target] for each intervention level of cause
        cause_means = {}
        for cause_val, dist in valid.items():
            cause_means[cause_val] = _weighted_mean_from_dist(dist, target)

        # Determine direction from lowest to highest cause state
        direction = "unknown"
        mean_shift_low_to_high = float('nan')
        sorted_cause_states = sorted(cause_means.keys(), key=lambda s: int(s) if s.isdigit() else s)
        if len(sorted_cause_states) >= 2:
            low_mean = cause_means[sorted_cause_states[0]]
            high_mean = cause_means[sorted_cause_states[-1]]
            if not (np.isnan(low_mean) or np.isnan(high_mean)):
                mean_shift_low_to_high = high_mean - low_mean
                if abs(mean_shift_low_to_high) < 1e-6:
                    direction = "neutral"
                elif mean_shift_low_to_high > 0:
                    direction = "positive"
                else:
                    direction = "negative"

        # Print summary
        logger.info(f"  Interventional distributions:")
        for cause_val, dist in interventions.items():
            if dist:
                e_val = cause_means.get(cause_val, float('nan'))
                dist_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(dist.items()))
                mean_str = f"  E[{target}]={e_val:.4f}" if not np.isnan(e_val) else ""
                logger.info(f"    do({cause}={cause_val}) → {dist_str}{mean_str}")
            else:
                logger.info(f"    do({cause}={cause_val}) → FAILED")
        logger.info(f"  ATE (max probability shift): {ate_max:.4f}")

        if direction != "unknown":
            arrow = "↑" if direction == "positive" else ("↓" if direction == "negative" else "→")
            relation = "directly" if direction == "positive" else ("inversely" if direction == "negative" else "neutrally")
            logger.info(f"  📊 Direction: {cause} ↑ → {target} {arrow} ({relation} related, shift={mean_shift_low_to_high:+.4f})")

        return {
            "cause": cause,
            "target": target,
            "adjustment_set": adj_set,
            "interventions": interventions,
            "ate_summary": ate_summary,
            "ate_max": ate_max,
        }

    def estimate_causal_effects(self, target_factor: str, causal_graph: nx.DiGraph = None) -> Dict[str, float]:
        """Estimate causal effect of each parent on target using do-operator."""
        if not self.enable_causal_estimate:
            logger.info("⚠️  Causal inference is disabled. Enable with enable_causal_estimate=True")
            return {}
        
        logger.info("\n" + "=" * 80)
        logger.info(f"ESTIMATING CAUSAL EFFECTS ON: {target_factor}")
        logger.info("=" * 80)
        
        if causal_graph:
            if self.causal_model is None:
                self.fit_causal_model(causal_graph)
            causes = [p for p in causal_graph.predecessors(target_factor)]
        else:
            if self.causal_inference_engine is None:
                logger.info("⚠️  No causal model available. Run fit_causal_model() first.")
                return {}
            causes = list(self.causal_model.get_parents(target_factor))
        
        if not causes:
            logger.info(f"  No direct causes found for {target_factor}")
            return {}
        
        logger.info(f"  Found {len(causes)} direct causes: {causes}")
        
        effects = {}
        
        for cause in causes:
            try:
                ate_result = self.compute_ate(cause, target_factor)
                effects[cause] = ate_result
            except Exception as e:
                logger.error(f"  ✗ {cause} → {target_factor}: Failed ({str(e)[:100]})")
                effects[cause] = None
        
        logger.info("=" * 80)
        return effects
    
    def estimate_downstream_effects(self, target_factor: str, causal_graph: nx.DiGraph = None) -> Dict[str, any]:
        """Estimate causal effect of target on its children using do-operator."""
        if not self.enable_causal_estimate:
            logger.info("⚠️  Causal inference is disabled. Enable with enable_causal_estimate=True")
            return {}
        
        logger.info("\n" + "=" * 80)
        logger.info(f"ESTIMATING DOWNSTREAM EFFECTS FROM: {target_factor}")
        logger.info("=" * 80)
        
        if causal_graph:
            if self.causal_model is None:
                self.fit_causal_model(causal_graph)
            children = [c for c in causal_graph.successors(target_factor)]
        else:
            if self.causal_inference_engine is None:
                logger.info("⚠️  No causal model available. Run fit_causal_model() first.")
                return {}
            children = [node for node in self.causal_model.nodes()
                        if self.causal_model.has_edge(target_factor, node)]
        
        if not children:
            logger.info(f"  No downstream effects found for {target_factor}")
            logger.info(f"  ℹ️  {target_factor} does not directly cause any other variables in the model")
            return {}
        
        logger.info(f"  Found {len(children)} downstream effects: {children}")
        
        downstream = {}
        
        for child in children:
            try:
                ate_result = self.compute_ate(target_factor, child)
                downstream[child] = ate_result
            except Exception as e:
                logger.error(f"  ✗ {target_factor} → {child}: Failed ({str(e)[:100]})")
                downstream[child] = None
        
        logger.info("=" * 80)
        return downstream
    
    def get_causal_summary(self, target_factor: str, causal_graph: nx.DiGraph) -> Dict:
        """Get comprehensive causal summary: direct causes/effects, ATEs, adjustment sets."""
        summary = {
            'target': target_factor,
            'direct_causes': [],
            'direct_effects': [],
            'causal_effects': {},
            'downstream_effects': {},
            'adjustment_sets': {},
            'has_causal_inference': self.enable_causal_estimate
        }
        
        # Get direct causes from graph
        if target_factor in causal_graph.nodes():
            summary['direct_causes'] = list(causal_graph.predecessors(target_factor))
            summary['direct_effects'] = list(causal_graph.successors(target_factor))
        
        # If causal inference enabled, compute effects
        if self.enable_causal_estimate and self.causal_inference_engine:
            summary['causal_effects'] = self.estimate_causal_effects(target_factor, causal_graph)
            summary['downstream_effects'] = self.estimate_downstream_effects(target_factor, causal_graph)
            
            # Store do-operator probabilities and direction on ALL directed edges
            # so visualization can display them at every degree, not just 1st degree.
            logger.info(f"\n  Computing do-operator ATE for all {len(list(causal_graph.edges()))} edges...")
            edges_computed = 0
            edges_failed = 0
            for cause, effect in list(causal_graph.edges()):
                if causal_graph[cause][effect].get('undirected', False):
                    continue  # Skip undirected/dashed edges
                try:
                    ate_result = self.compute_ate(cause, effect)
                    if ate_result and ate_result.get('ate_max', 0.0) > 0:
                        causal_graph[cause][effect]['do_probability'] = ate_result['ate_max']
                        interventions = ate_result.get('interventions', {})
                        valid = {k: v for k, v in interventions.items() if v is not None}
                        sorted_states = sorted(valid.keys(), key=lambda s: int(s) if s.isdigit() else s)
                        if len(sorted_states) >= 2:
                            low_dist = valid[sorted_states[0]]
                            high_dist = valid[sorted_states[-1]]
                            low_mean = self._weighted_mean_from_dist_static(low_dist, effect)
                            high_mean = self._weighted_mean_from_dist_static(high_dist, effect)
                            if not (np.isnan(low_mean) or np.isnan(high_mean)):
                                shift = high_mean - low_mean
                                if abs(shift) < 1e-6:
                                    causal_graph[cause][effect]['do_direction'] = 'neutral'
                                elif shift > 0:
                                    causal_graph[cause][effect]['do_direction'] = 'positive'
                                else:
                                    causal_graph[cause][effect]['do_direction'] = 'negative'
                            else:
                                # Fallback for categorical effects: compare most-likely states
                                # If the most probable outcome shifts, the relationship is non-neutral
                                low_most_likely = max(low_dist, key=low_dist.get)
                                high_most_likely = max(high_dist, key=high_dist.get)
                                if low_most_likely == high_most_likely:
                                    causal_graph[cause][effect]['do_direction'] = 'neutral'
                                else:
                                    # Try numeric comparison of state labels
                                    try:
                                        if float(high_most_likely) > float(low_most_likely):
                                            causal_graph[cause][effect]['do_direction'] = 'positive'
                                        else:
                                            causal_graph[cause][effect]['do_direction'] = 'negative'
                                    except (ValueError, TypeError):
                                        # Truly categorical — just mark as "shifts" (positive by convention)
                                        causal_graph[cause][effect]['do_direction'] = 'positive'
                        edges_computed += 1
                    else:
                        # ATE is zero — no measurable effect
                        causal_graph[cause][effect]['do_probability'] = 0.0
                        causal_graph[cause][effect]['do_direction'] = 'neutral'
                        edges_computed += 1
                except Exception as e:
                    edges_failed += 1
                    logger.warning(f"  ⚠ ATE failed for {cause} → {effect}: {str(e)[:100]}")
            
            logger.info(f"  ✓ Do-operator annotation: {edges_computed} edges computed, {edges_failed} failed")

            # Prune edges with negligible causal effect (ATE below threshold).
            # If the do-operator shows no measurable interventional effect,
            # the edge is likely noise from structure learning.
            # Also prune edges where ATE computation failed entirely.
            ate_threshold = 0.01
            edges_to_remove = []
            for cause, effect in list(causal_graph.edges()):
                if causal_graph[cause][effect].get('undirected', False):
                    continue
                do_prob = causal_graph[cause][effect].get('do_probability', None)
                # Remove if: ATE is below threshold OR ATE couldn't be computed
                if do_prob is None or do_prob < ate_threshold:
                    edges_to_remove.append((cause, effect))

            if edges_to_remove:
                causal_graph.remove_edges_from(edges_to_remove)
                logger.info(f"  ✂ Pruned {len(edges_to_remove)} edges with ATE < {ate_threshold} (no measurable causal effect)")
                # Remove isolated nodes left after pruning
                isolated = list(nx.isolates(causal_graph))
                if isolated:
                    causal_graph.remove_nodes_from(isolated)
                    logger.info(f"  ✂ Removed {len(isolated)} isolated nodes after pruning")

            # Get adjustment sets
            for cause in summary['direct_causes']:
                try:
                    adj_set = self.causal_inference_engine.get_minimal_adjustment_set(X=cause, Y=target_factor)
                    summary['adjustment_sets'][cause] = list(adj_set) if adj_set else []
                except Exception:
                    summary['adjustment_sets'][cause] = None
        
        return summary

    def _weighted_mean_from_dist_static(self, distribution: dict, var_name: str) -> float:
        """Compute E[var] from a discrete distribution using bin midpoints (for edge annotation)."""
        kbd = self._cached_discretizers.get(var_name)
        if kbd is None:
            try:
                return sum(float(s) * p for s, p in distribution.items())
            except (ValueError, TypeError):
                return float('nan')
        edges = kbd.bin_edges_[0]
        total = 0.0
        for s, p in distribution.items():
            try:
                idx = int(s)
                if 0 <= idx < len(edges) - 1:
                    total += ((edges[idx] + edges[idx + 1]) / 2.0) * p
            except (ValueError, IndexError):
                continue
        return total

    def get_causal_summary_lightweight(self, target_factor: str, causal_graph: nx.DiGraph) -> Dict:
        """Lightweight causal summary: direct causes/effects and adjustment sets only.

        Unlike :meth:`get_causal_summary`, this does NOT run the expensive
        ``estimate_causal_effects`` / ``estimate_downstream_effects`` do-operator
        computations.  Use this when you only need graph structure and adjustment
        sets (e.g. for the result dict returned by ``causalif()``).
        """
        summary = {
            'target': target_factor,
            'direct_causes': [],
            'direct_effects': [],
            'adjustment_sets': {},
            'has_causal_inference': self.enable_causal_estimate,
        }

        if target_factor in causal_graph.nodes():
            summary['direct_causes'] = list(causal_graph.predecessors(target_factor))
            summary['direct_effects'] = list(causal_graph.successors(target_factor))

        if self.enable_causal_estimate and self.causal_inference_engine:
            for cause in summary['direct_causes']:
                try:
                    adj_set = self.causal_inference_engine.get_minimal_adjustment_set(X=cause, Y=target_factor)
                    summary['adjustment_sets'][cause] = list(adj_set) if adj_set else []
                except Exception:
                    summary['adjustment_sets'][cause] = None

        return summary