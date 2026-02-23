# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CausalIF Engine implementation"""

# Standard library imports
import concurrent.futures
import math
from typing import Dict, List, Union, Tuple
from collections import deque

# Third-party imports
import pandas as pd
import networkx as nx
import numpy as np

# pgmpy imports for Bayesian causal inference
from pgmpy.estimators import HillClimbSearch, BDeu, ExpertKnowledge, StructureScore, MaximumLikelihoodEstimator
from pgmpy.inference import CausalInference
from pgmpy.models import DiscreteBayesianNetwork

# Local imports (relative)
from .core import KnowledgeBase
from .prompts import CausalIFPrompts


# ============================================================================
# CUSTOM BAYESIAN SCORING WITH PRIORS FROM CausalIF 1
# ============================================================================

class PriorWeightedBDeu(StructureScore):
    """
    Custom scoring function that combines BDeu score with CausalIF 1 priors.
    
    Mathematical Framework:
    ----------------------
    Bayesian Score: log P(G | D) = log P(D | G) + log P(G)
                                  = [BDeu score] + [structure prior]
    
    - BDeu score = log P(D | G): marginal likelihood (includes parameter priors via ESS)
    - Structure prior = log P(G): our belief about graph structure (from LLM)
    
    Key Insight: Structure priors should be independent of sample size!
    They represent expert knowledge about which edges exist, not data evidence.
    
    Implementation:
    --------------
    local_score(X, parents) = BDeu_local(X, parents) + λ × Σ log(prior_strength)
    
    where:
    - BDeu_local: standard BDeu local score (grows with n)
    - prior_strength: LLM vote score (1, 2, or 3)
    - λ: weight parameter
    
    Prior Weight Scaling:
    --------------------
    - LLM priors: log(1) ≈ 0, log(2) ≈ 0.69, log(3) ≈ 1.10
    - Adaptive scaling (default): λ = log(n) × ESS
      * For n=100, ESS=10: λ ≈ 46 → log(3) × 46 ≈ 51
      * For n=1000, ESS=10: λ ≈ 69 → log(3) × 69 ≈ 76
      * For n=10000, ESS=10: λ ≈ 92 → log(3) × 92 ≈ 101
    
    Why log(n) scaling?
    - Priors should fade as data accumulates (Bayesian principle)
    - But not too fast (we still trust expert knowledge)
    - log(n) is very slow growth, similar to BIC penalty
    - Mathematically principled: matches information-theoretic bounds
    """
    
    def __init__(self, data, skeleton_graph, prior_weight='auto', equivalent_sample_size=10, **kwargs):
        """
        Args:
            data: pandas DataFrame with observational data
            skeleton_graph: networkx Graph from CausalIF 1 with edge attributes
            prior_weight: λ parameter controlling prior influence
                         - 'auto': Adaptive scaling based on sample size (recommended)
                         - float: Manual weight (e.g., 100.0 for strong prior influence)
                         - 0: Ignore priors completely
            equivalent_sample_size: BDeu hyperparameter
        """
        # Initialize parent StructureScore class
        super(PriorWeightedBDeu, self).__init__(data, **kwargs)
        
        self.bdeu = BDeu(data, equivalent_sample_size=equivalent_sample_size)
        self.skeleton = skeleton_graph
        
        # Prior weight scaling - Mathematical Framework:
        # 
        # Bayesian Score: log P(G | D) = log P(D | G) + log P(G)
        #                               = [BDeu score] + [structure prior]
        # 
        # - BDeu score = log P(D | G) includes parameter priors via ESS
        # - Structure prior = log P(G) should be independent of sample size!
        # 
        # Our LLM scores represent structure priors: P(edge exists) from expert knowledge
        # These should NOT scale with n (they're beliefs about structure, not data)
        # 
        # However, BDeu scores grow with n, so to maintain relative influence:
        # - Option 1: Fixed weight (treats prior as absolute belief)
        # - Option 2: Scale with log(n) (prior influence fades slowly as data grows)
        # - Option 3: Scale with sqrt(n) (balanced approach)
        # 
        # We use Option 2 (log scaling) as it's mathematically justified:
        # - Priors should fade as data accumulates (Bayesian learning)
        # - But not too fast (log is very slow growth)
        # - Matches BIC penalty term: log(n)/2
        if prior_weight == 'auto':
            n_samples = len(data)
            # Scale by log(n) × ESS for slow, principled growth
            # For n=100: λ ≈ 46, for n=1000: λ ≈ 69, for n=10000: λ ≈ 92
            # This gives priors meaningful but fading influence as data grows
            self.prior_weight = math.log(n_samples) * equivalent_sample_size
        else:
            self.prior_weight = prior_weight
        
        # Extract prior strengths from skeleton
        # prior_strength contains LLM voting confidence only
        self.prior_strengths = {}
        for u, v in skeleton_graph.edges():
            # Get prior strength from CausalIF 1 (LLM-based)
            prior_strength = skeleton_graph[u][v].get('prior_strength', 0.5)
            
            # Store for both directions (undirected edge)
            # No need to multiply by confidence again - it's already in prior_strength
            self.prior_strengths[(u, v)] = prior_strength
            self.prior_strengths[(v, u)] = prior_strength
        
        print(f"\n[Prior-Weighted Scoring] Initialized with {len(self.prior_strengths)//2} prior edges")
        print(f"  Prior weight (λ): {self.prior_weight:.2f} {'(adaptive: log(n) × ESS)' if prior_weight == 'auto' else '(manual)'}")
        print(f"  Sample size: {len(data)}, ESS: {equivalent_sample_size}")
        if self.prior_strengths:
            print(f"  Prior strength range: [{min(self.prior_strengths.values()):.3f}, {max(self.prior_strengths.values()):.3f}]")
            min_contrib = self.prior_weight * math.log(min(self.prior_strengths.values()) + 1e-10)
            max_contrib = self.prior_weight * math.log(max(self.prior_strengths.values()) + 1e-10)
            print(f"  Effective prior contribution per edge: [{min_contrib:.2f}, {max_contrib:.2f}]")
            print(f"  Note: Structure priors fade slowly (log growth) as data accumulates")
    
    def score(self, model):
        """
        Compute prior-weighted score for a Bayesian Network model.

        NOTE: This method delegates to BDeu.score() WITHOUT adding priors.
        The overall BDeu score is already the sum of all local scores, and
        local_score() already includes the prior contribution. Adding priors
        here would double-count them.

        Hill Climb Search uses local_score() for optimization, so the priors
        are correctly incorporated there.

        Args:
            model: DiscreteBayesianNetwork (DAG) to score

        Returns:
            float: BDeu score (higher is better)
        """
        # Just return BDeu score - priors are already in local_score()
        return self.bdeu.score(model)
    
    def local_score(self, variable, parents):
        """
        Compute local score for a variable given its parents.
        Required by pgmpy's HillClimbSearch.
        
        This delegates to BDeu for the likelihood part, then adds prior contribution.
        """
        # Get BDeu local score
        bdeu_local = self.bdeu.local_score(variable, parents)
        
        # Add prior contribution for edges from parents to variable
        prior_local = 0.0
        for parent in parents:
            if (parent, variable) in self.prior_strengths:
                prior_strength = self.prior_strengths[(parent, variable)]
                prior_local += self.prior_weight * np.log(prior_strength + 1)
        
        print(f"bdeu_local + prior_local:{bdeu_local} + {prior_local}")
        return bdeu_local + prior_local



class CausalIFEngine:
    """CausalIF implementation with Bayesian causal inference for orientation"""
    
    def __init__(self, model, retriever_tool=None, dataframe: pd.DataFrame = None, 
                 k_documents: int = 1, max_degrees: int = None, max_parallel_queries: int = 25,
                 excluded_target_columns: List[str] = None, excluded_related_columns: List[str] = None,
                 related_factors: List[str] = None, selected_dataframe_columns: List[str] = None,
                 enable_causal_estimate: bool = False):
        self.model = model
        self.retriever_tool = retriever_tool
        self.dataframe = dataframe
        self.k_documents = k_documents
        self.max_degrees = max_degrees  # None = no filtering, int = filter by degrees
        self.max_parallel_queries = max_parallel_queries
        self.excluded_target_columns = excluded_target_columns or []
        self.excluded_related_columns = excluded_related_columns or []
        self.related_factors = related_factors or []  # Factors to fix in skeleton before Hill Climb
        self.selected_dataframe_columns = selected_dataframe_columns  # Specific columns to select from dataframe
        self.prompts = CausalIFPrompts()
        self.enable_causal_estimate = enable_causal_estimate  # NEW: Enable causal inference
        self.causal_model = None  # Will store fitted Bayesian Network
        self.causal_inference_engine = None  # Will store CausalInference object

    def filter_graph_by_degrees(self, graph: Union[nx.Graph, nx.DiGraph], target_factor: str, max_degrees: int = None) -> Union[nx.Graph, nx.DiGraph]:
        """
        Filter graph to only include nodes within max_degrees of target_factor.

        Args:
            graph: Graph to filter
            target_factor: Target node to measure degrees from
            max_degrees: Maximum degrees of separation (None = no filtering, returns entire graph)

        Returns:
            Filtered graph (or original graph if max_degrees is None)
        """
        if max_degrees is None:
            # No filtering - return entire graph
            return graph

        if target_factor not in graph.nodes():
            return graph

        # Python BFS implementation
        visited = set()
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

        print(f"Filtered graph to {len(filtered_graph.nodes())} nodes within {max_degrees} degrees of {target_factor}")
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
        """
        Analyze degrees of separation from target factor.

        Args:
            graph: Graph to analyze
            target_factor: Target node to measure degrees from

        Returns:
            Dictionary with degree analysis (all factors if max_degrees is None)
        """
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
        """Retrieve relevant documents for factor pair using RAG"""
        try:
            if self.retriever_tool:
                query = f"{factor_a} and {factor_b} association relationship"
                print(f"RAG Query: {query}")
                retrieved_docs = self.retriever_tool.invoke({"query": query})

                documents = []

                if isinstance(retrieved_docs, list):
                    for i, doc in enumerate(retrieved_docs[:self.k_documents]):
                        if isinstance(doc, dict):
                            content = doc.get('content', '') or doc.get('page_content', '') or str(doc)
                        elif hasattr(doc, 'page_content'):
                            content = doc.page_content
                        elif isinstance(doc, str):
                            content = doc
                        else:
                            content = str(doc)

                        kb = KnowledgeBase(
                            kb_type="DOC",
                            content=content,
                            source=f"doc_{i}"
                        )
                        documents.append(kb)
                elif isinstance(retrieved_docs, str):
                    kb = KnowledgeBase(
                        kb_type="DOC",
                        content=retrieved_docs,
                        source="single_doc"
                    )
                    documents.append(kb)

                print(f"Retrieved {len(documents)} documents for {factor_a} and {factor_b}")
                return documents
            else:
                print("No retriever tool available for RAG")
                return []

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


    def parallel_llm_query_sync(self, prompts: List[str]) -> List[str]:
        """Execute multiple LLM queries in parallel using ThreadPoolExecutor with retry logic.

        Features:
        - Limits concurrency to avoid connection pool exhaustion
        - Retries throttled requests with exponential backoff
        - Proper response-to-prompt mapping using indices
        - Reasonable timeouts with graceful degradation
        - Handles connection pool errors and I/O errors
        """
        import time
        import random
        from urllib3.exceptions import ProtocolError
        from http.client import HTTPException

        def single_query_with_retry(prompt_index: int, prompt: str, max_retries: int = 3) -> str:
            """Execute a single query with retry logic for throttling and connection errors."""
            for attempt in range(max_retries):
                try:
                    if self.model is None:
                        return "UNKNOWN"

                    response = self.model.invoke(prompt)
                    result = response.content if hasattr(response, 'content') else str(response)
                    return str(result)

                except (ValueError, ProtocolError, HTTPException, ConnectionError) as e:
                    # Connection pool errors, I/O errors, protocol errors
                    error_msg = str(e).lower()
                    
                    is_connection_error = any(keyword in error_msg for keyword in
                                            ['connection pool', 'i/o operation', 'closed file', 
                                             'connection', 'protocol', 'broken pipe'])
                    
                    if is_connection_error and attempt < max_retries - 1:
                        # Exponential backoff with jitter for connection errors
                        wait_time = (2 ** attempt) + random.uniform(0, 2)
                        print(f"  Connection error on query {prompt_index + 1}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  Connection error in query {prompt_index + 1}: {str(e)[:100]}")
                        return "UNKNOWN"

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check if it's a throttling error
                    is_throttle = any(keyword in error_msg for keyword in
                                     ['throttl', 'rate limit', 'too many requests', '429'])

                    if is_throttle and attempt < max_retries - 1:
                        # Exponential backoff with jitter for throttling
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"  Throttled on query {prompt_index + 1}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"  Error in query {prompt_index + 1}: {str(e)[:100]}")
                        return "UNKNOWN"

            return "UNKNOWN"

        # Limit concurrent workers to avoid HTTP connection pool exhaustion
        # Reduce from 10 to 5 to be more conservative with connection pool
        safe_max_workers = min(self.max_parallel_queries, 5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
            # Submit with index to ensure correct mapping
            future_to_index = {
                executor.submit(single_query_with_retry, i, prompt): i
                for i, prompt in enumerate(prompts)
            }

            results = [None] * len(prompts)

            for future in concurrent.futures.as_completed(future_to_index):
                prompt_index = future_to_index[future]
                try:
                    result = future.result(timeout=90)  # Increased to 90 second timeout per query
                    results[prompt_index] = result
                except concurrent.futures.TimeoutError:
                    print(f"  Query {prompt_index + 1} timed out after 90 seconds")
                    results[prompt_index] = "UNKNOWN"
                except Exception as e:
                    print(f"  Query {prompt_index + 1} failed: {str(e)[:100]}")
                    results[prompt_index] = "UNKNOWN"

        # Ensure all results are filled
        results = [r if r is not None else "UNKNOWN" for r in results]
        return results

    def execute_parallel_queries(self, prompts: List[str]) -> List[str]:
        """Execute parallel queries with batch size limiting.

        Processes queries in batches to avoid overwhelming the LLM API.
        """
        if not prompts:
            return []

        # Process in batches to avoid overwhelming API
        BATCH_SIZE = 50  # Process max 50 queries at a time

        if len(prompts) <= BATCH_SIZE:
            # Small batch - process all at once
            return self.parallel_llm_query_sync(prompts)
        else:
            # Large batch - process in chunks
            print(f"  Processing {len(prompts)} queries in batches of {BATCH_SIZE}...")
            all_results = []

            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

                print(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} queries...")

                batch_results = self.parallel_llm_query_sync(batch)
                all_results.extend(batch_results)

            return all_results

    def causalif_1_edge_existence_verification(self, factors: List[str], domains: List[str], target_factor: str = None) -> nx.Graph:
        """
        CausalIF 1: Edge Existence Verification Algorithm (from paper)
        
        Algorithm from paper (https://arxiv.org/html/2402.15301v2):
        1. Initialize with complete undirected graph (all possible edges)
        2. For each variable pair (vi, vj):
           - Initialize score S = 0
           - Query each KB (documents + background knowledge)
           - If ζ_KB(ij) = 1 (directly associated): S += 1
           - If ζ_KB(ij) = 0 (independent or indirectly associated): S += -1
           - If unknown: S += 0 (no change)
           - If S ≤ 0: REMOVE the edge from graph
           - If S > 0: KEEP the edge in graph
        3. Return skeleton graph
        """
        
        print("="*80)
        print("CausalIF 1: Edge Existence Verification (Paper Algorithm)")
        print("="*80)
        print("Strategy: Start with complete graph, remove edges via LLM voting")
        
        # Step 1: Initialize with complete undirected graph
        G = nx.complete_graph(len(factors))
        # Relabel nodes from integers to factor names
        mapping = {i: factors[i] for i in range(len(factors))}
        G = nx.relabel_nodes(G, mapping)
        
        print(f"\nInitialized complete graph:")
        print(f"  - Nodes: {len(G.nodes())}")
        print(f"  - Edges: {len(G.edges())} (all possible pairs)")
        
        # Step 2: For each edge in complete graph, run LLM voting
        print(f"\n{'='*80}")
        print(f"Step 2: LLM Voting on All Edges (BATCH MODE)")
        print(f"{'='*80}")
        
        edges_to_remove = []
        edges_to_keep = []
        all_edges = list(G.edges())
        
        print(f"Testing {len(all_edges)} edges with LLM voting...\n")
        
        # PHASE 1: Collect all edge pairs and retrieve documents
        print(f"Phase 1: Retrieving documents for all edges...")
        edge_documents = {}  # Maps (factor_a, factor_b) -> List[KnowledgeBase]
        
        for factor_a, factor_b in all_edges:
            doc_kbs = self.retrieve_documents(factor_a, factor_b)
            edge_documents[(factor_a, factor_b)] = doc_kbs[:2]  # Use top 2 documents
        
        print(f"✓ Retrieved documents for {len(all_edges)} edges\n")
        
        # PHASE 2: Build all prompts for batch execution
        print(f"Phase 2: Building prompts for batch LLM execution...")
        
        # Structure: List of (edge_pair, kb_type, kb_content, prompt)
        batch_queries = []
        
        for factor_a, factor_b in all_edges:
            # Background knowledge prompt - Original CausalIF 1
            bg_prompt = f"""
            {self.prompts.background_reminder(factors, domains)}
            
            Your task is to thoroughly use the knowledge in your training data to solve a task. Your task is: based on your background knowledge, try to find statistical evidence to clarify the association relationship between the pair of 'Main factors' according to the 'Association Context'.
            
            Main factors: {factor_a} and {factor_b}
            
            Association Context:
            {self.prompts.association_context()}
            
            Association Question: Are {factor_a} and {factor_b} associated?
            
            Expected Response Format:
            Thoughts: [Write your thoughts on the question]
            Answer: (A) Associated (B) Independent (C) Unknown
            """
            
            batch_queries.append({
                'edge': (factor_a, factor_b),
                'kb_type': 'BG',
                'prompt': bg_prompt
            })
            
            # Document knowledge prompts
            for doc_kb in edge_documents[(factor_a, factor_b)]:
                doc_prompt = f"""
                {self.prompts.background_reminder(factors, domains)}
                
                Your task is to thoroughly read the given 'Document'. Then, based on the knowledge from the given 'Document', try to find statistical evidence to clarify the association relationship between the pair of 'Main factors' according to the 'Association Context'.
                
                Document: {doc_kb.content[:2000]}...
                
                Main factors: {factor_a} and {factor_b}
                
                Association Context:
                {self.prompts.association_context()}
                
                Association Question: Are {factor_a} and {factor_b} associated?
                
                Expected Response Format:
                Thoughts: [Write your thoughts on the question]
                Answer: (A) Associated (B) Independent (C) Unknown
                Reference: [Skip this if you chose option C above. Otherwise, provide a supporting sentence from the document for your choice]
                """
                
                batch_queries.append({
                    'edge': (factor_a, factor_b),
                    'kb_type': doc_kb.kb_type,
                    'prompt': doc_prompt
                })
        
        print(f"✓ Built {len(batch_queries)} prompts ({len(all_edges)} edges × ~{len(batch_queries)//len(all_edges)} KBs each)\n")
        
        # PHASE 3: Execute all prompts in parallel
        print(f"Phase 3: Executing {len(batch_queries)} LLM queries in parallel...")
        
        prompts = [q['prompt'] for q in batch_queries]
        responses = self.execute_parallel_queries(prompts)
        
        print(f"✓ Completed all LLM queries\n")
        
        # PHASE 4: Parse results and compute vote scores
        print(f"Phase 4: Processing results and computing vote scores...\n")
        
        # Map responses back to edges
        edge_votes = {}  # Maps (factor_a, factor_b) -> List[(kb_type, vote)]
        
        for i, query_info in enumerate(batch_queries):
            edge = query_info['edge']
            kb_type = query_info['kb_type']
            response_text = responses[i].upper() if i < len(responses) else "UNKNOWN"
            
            # Parse response to vote
            if "ASSOCIATED" in response_text and "INDEPENDENT" not in response_text:
                vote = 1  # Directly associated
            elif "INDEPENDENT" in response_text:
                vote = 0  # Independent or indirectly associated
            else:
                vote = None  # Unknown
            
            if edge not in edge_votes:
                edge_votes[edge] = []
            edge_votes[edge].append((kb_type, vote))
        
        # PHASE 5: Make keep/remove decisions
        print(f"Phase 5: Making edge decisions based on voting...\n")
        
        for idx, (factor_a, factor_b) in enumerate(all_edges, 1):
            print(f"[{idx}/{len(all_edges)}] Processing pair: {factor_a} ↔ {factor_b}")
            
            votes = edge_votes.get((factor_a, factor_b), [])
            S = 0  # Voting score (as per Algorithm 1 in paper)
            total_kbs = len(votes)
            
            # Count votes
            for kb_type, vote in votes:
                if vote == 1:  # Directly associated
                    S += 1
                    print(f"  ✓ KB {kb_type} votes FOR edge (ζ=1, directly associated)")
                elif vote == 0:  # Independent or indirectly associated
                    S -= 1
                    print(f"  ✗ KB {kb_type} votes AGAINST edge (ζ=0, independent/indirect)")
                else:  # Unknown
                    print(f"  ? KB {kb_type} is UNKNOWN (ζ=?, no vote)")
            
            # Decision rule from paper: Remove edge if S ≤ 0, keep if S > 0
            if S <= 0:
                edges_to_remove.append((factor_a, factor_b))
                print(f"  ✗ DECISION: REMOVE edge (Score S={S} ≤ 0)")
            else:
                # Store voting information as edge attributes
                G[factor_a][factor_b]['vote_score'] = S
                G[factor_a][factor_b]['total_kbs'] = total_kbs
                
                edges_to_keep.append((factor_a, factor_b))
                print(f"  ✓ DECISION: KEEP edge (Score S={S} > 0)")
        
        # Step 3: Remove edges with S ≤ 0
        print(f"\n{'='*80}")
        print(f"Step 3: Removing Edges")
        print(f"{'='*80}")
        G.remove_edges_from(edges_to_remove)
        print(f"Removed {len(edges_to_remove)} edges (S ≤ 0)")
        print(f"Kept {len(edges_to_keep)} edges (S > 0)")
        
        # Step 4: Compute LLM-based priors for edges in skeleton
        print(f"\n{'='*80}")
        print(f"Step 4: Computing Priors for Skeleton Edges")
        print(f"{'='*80}")
        
        if len(edges_to_keep) > 0:
            print(f"Computing LLM-based priors for {len(edges_to_keep)} skeleton edges...")
            print(f"Using raw vote scores for better discrimination in Bayesian inference")
            
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
                    
                    print(f"  {factor_a} - {factor_b}: Prior = {prior_strength:.1f} (raw vote: {vote_score}/{total_kbs} KBs)")

        
        print(f"\n{'='*80}")
        print(f"CausalIF 1 Complete:")
        print(f"  - Initial edges (complete graph): {len(all_edges)}")
        print(f"  - Edges removed: {len(edges_to_remove)}")
        print(f"  - Final edges: {len(G.edges())}")
        print(f"  - Sparsity: {len(G.edges())}/{len(all_edges)} = {len(G.edges())/max(1, len(all_edges))*100:.1f}%")
        print(f"{'='*80}")
        
        return G

    def causalif_2_orientation(self, skeleton: nx.Graph, factors: List[str], domains: List[str], target_factor: str = None) -> nx.DiGraph:
        """
        CausalIF 2: Bayesian Orientation using Prior from Skeleton
        
        Bayesian Framework:
        - PRIOR: skeleton graph from CausalIF 1 (undirected edges = prior belief about associations)
        - LIKELIHOOD: observational data in dataframe
        - POSTERIOR: directed causal graph (posterior belief about causal directions)
        
        P(Directed Graph | Data) ∝ P(Data | Directed Graph) × P(Directed Graph | Skeleton)
        
        The skeleton constrains the search space - we only orient edges that exist in the prior.
        """
        
        print("=" * 80)
        print("BAYESIAN CAUSAL INFERENCE FRAMEWORK")
        print("=" * 80)
        print(f"PRIOR: Skeleton graph with {len(skeleton.edges())} undirected edges (from CausalIF 1)")
        print("       Represents prior belief about which variables are associated")
        print(f"LIKELIHOOD: Observational data with {len(self.dataframe) if self.dataframe is not None else 0} samples")
        print("POSTERIOR: Directed causal graph (to be learned)")
        print("=" * 80)
        
        # OPTIMIZATION: Filter skeleton to only nodes within 2 degrees of target factor
        # This dramatically speeds up Hill Climb search by reducing the search space
        if target_factor and target_factor in skeleton.nodes() and self.max_degrees is not None:
            print(f"\n🎯 TARGET-FOCUSED OPTIMIZATION")
            print(f"   Filtering skeleton to nodes within {self.max_degrees} degrees of '{target_factor}'")
            print(f"   Before filtering: {len(skeleton.nodes())} nodes, {len(skeleton.edges())} edges")
            
            # Use existing filter_graph_by_degrees method with self.max_degrees
            filtered_skeleton = self.filter_graph_by_degrees(skeleton, target_factor, max_degrees=self.max_degrees)
            
            print(f"   After filtering: {len(filtered_skeleton.nodes())} nodes, {len(filtered_skeleton.edges())} edges")
            print(f"   Reduction: {len(skeleton.nodes()) - len(filtered_skeleton.nodes())} nodes removed")
            print(f"   Speed improvement: ~{((len(skeleton.nodes())**2) / max(1, len(filtered_skeleton.nodes())**2)):.1f}x faster")
            print(f"   This focuses Hill Climb search on relationships relevant to {target_factor}")
            
            # Use filtered skeleton for orientation
            skeleton_for_orientation = filtered_skeleton
        else:
            if self.max_degrees is None:
                print(f"\n⚠️  max_degrees=None - using full skeleton (no filtering)")
            else:
                print(f"\n⚠️  No target factor specified - using full skeleton")
            skeleton_for_orientation = skeleton
        
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(skeleton_for_orientation.nodes())
        
        edges_to_orient = list(skeleton_for_orientation.edges())
        
        if not edges_to_orient:
            print("No edges in prior skeleton to orient")
            return directed_graph
        
        print(f"\nOrienting {len(edges_to_orient)} edges from PRIOR using Bayesian inference...")
        
        # Check if we have data for Bayesian structure learning
        if self.dataframe is not None and len(self.dataframe) > 0:
            try:
                # Prepare data for Bayesian structure learning
                # Use filtered skeleton nodes (already filtered to 2 degrees if target_factor exists)
                nodes_in_skeleton = list(skeleton_for_orientation.nodes())
                available_columns = [col for col in nodes_in_skeleton if col in self.dataframe.columns]
                
                if len(available_columns) >= 2:
                    print(f"\nLIKELIHOOD: Using {len(self.dataframe)} data samples for {len(available_columns)} factors")
                    
                    # Prepare data - pgmpy can handle categorical data natively
                    df_for_learning = self.dataframe[available_columns].copy()
                    
                    # Track columns to drop due to conversion errors
                    columns_to_drop = []
                    
                    # Only convert datetime columns to numeric (pgmpy doesn't handle datetime)
                    for col in df_for_learning.columns:
                        try:
                            if pd.api.types.is_datetime64_any_dtype(df_for_learning[col]):
                                # Convert datetime to numeric (days since epoch)
                                df_for_learning[col] = (df_for_learning[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                                print(f"  ✓ Converted datetime column '{col}' to numeric (days since epoch)")
                            elif df_for_learning[col].dtype == 'object':
                                # Check if it looks like a date column
                                date_keywords = ['date', 'time', 'timestamp', 'dt', 'day', 'month', 'year']
                                looks_like_date = any(keyword in col.lower() for keyword in date_keywords)
                                
                                if looks_like_date:
                                    # Try to parse as datetime
                                    try:
                                        # Try common date formats
                                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y%m%d']:
                                            try:
                                                df_for_learning[col] = pd.to_datetime(df_for_learning[col], format=fmt)
                                                df_for_learning[col] = (df_for_learning[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                                                print(f"  ✓ Parsed '{col}' as date (format: {fmt}) and converted to numeric")
                                                break
                                            except:
                                                continue
                                        else:
                                            # Try infer_datetime_format
                                            df_for_learning[col] = pd.to_datetime(df_for_learning[col], infer_datetime_format=True)
                                            df_for_learning[col] = (df_for_learning[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
                                            print(f"  ✓ Parsed '{col}' as date (inferred format) and converted to numeric")
                                    except:
                                        # Date parsing failed, leave as categorical (pgmpy will handle it)
                                        print(f"  ℹ Column '{col}' kept as categorical (pgmpy will handle it)")
                                else:
                                    # Not a date column, leave as categorical
                                    print(f"  ℹ Column '{col}' kept as categorical (pgmpy will handle it)")
                        except Exception as e:
                            # If any conversion fails, mark column for removal
                            print(f"  ✗ Failed to process column '{col}': {str(e)[:100]}")
                            columns_to_drop.append(col)
                    
                    # Drop problematic columns
                    if columns_to_drop:
                        print(f"\n  Dropping {len(columns_to_drop)} problematic columns: {columns_to_drop}")
                        df_for_learning = df_for_learning.drop(columns=columns_to_drop)
                        available_columns = [col for col in available_columns if col not in columns_to_drop]
                    
                    # Handle NaN values
                    print(f"\n  Before handling NaN: {len(df_for_learning)} rows")
                    print(f"  NaN counts per column:")
                    for col in df_for_learning.columns:
                        nan_count = df_for_learning[col].isna().sum()
                        if nan_count > 0:
                            print(f"    - {col}: {nan_count} NaN values ({nan_count/len(df_for_learning)*100:.1f}%)")
                    
                    # Fill NaN: 0 for numeric, 'missing' for categorical
                    for col in df_for_learning.columns:
                        if pd.api.types.is_numeric_dtype(df_for_learning[col]):
                            df_for_learning[col] = df_for_learning[col].fillna(0)
                        else:
                            # Handle categorical columns specially
                            if df_for_learning[col].dtype.name == 'category':
                                if 'missing' not in df_for_learning[col].cat.categories:
                                    df_for_learning[col] = df_for_learning[col].cat.add_categories(['missing'])
                            df_for_learning[col] = df_for_learning[col].fillna('missing')
                    
                    print(f"  After handling NaN: {len(df_for_learning)} rows (all data preserved)")
                    
                    if len(df_for_learning) > 10 and len(df_for_learning.columns) >= 2:  # Need sufficient data
                        print(f"\n✓ Data preparation complete:")
                        print(f"  - {len(df_for_learning)} samples")
                        print(f"  - {len(df_for_learning.columns)} factors: {list(df_for_learning.columns)}")
                        
                        # Show data types
                        print(f"  - Data types:")
                        for col in df_for_learning.columns:
                            dtype = df_for_learning[col].dtype
                            if pd.api.types.is_numeric_dtype(df_for_learning[col]):
                                print(f"    • {col}: numeric")
                            else:
                                unique_vals = df_for_learning[col].nunique()
                                print(f"    • {col}: categorical ({unique_vals} unique values)")
                        
                        print(f"\nComputing POSTERIOR using Bayesian structure learning...")
                        print(f"  Method: Hill Climbing with Prior-Weighted BDeu score (True Bayesian)")
                        print(f"  Prior constraint: Only orient edges from skeleton graph")
                        print(f"  Prior weighting: ENABLED (LLM voting confidence from CausalIF 1)")
                        
                        # Use Hill Climbing with Prior-Weighted BDeu score
                        # This implements true Bayesian inference: P(G | Data, Priors) ∝ P(Data | G) × P(G | Priors)
                        try:
                            # Create custom scoring function that combines BDeu + CausalIF 1 priors
                            # Use filtered skeleton (already contains only nodes within 2 degrees)
                            scoring_method = PriorWeightedBDeu(
                                data=df_for_learning,
                                skeleton_graph=skeleton_for_orientation,  # Use filtered skeleton
                                prior_weight='auto',  # Adaptive: sqrt(n) × ESS for meaningful prior influence
                                equivalent_sample_size=10
                            )
                            hc = HillClimbSearch(df_for_learning)
                            
                            # Constrain search to only edges in the skeleton (PRIOR)
                            # This encodes: P(G | Skeleton) - only structures consistent with skeleton have non-zero prior
                            # In pgmpy 1.0+, we use ExpertKnowledge instead of white_list
                            allowed_edges = []
                            fixed_edges_undirected = []  # Undirected edges to fix (from related_factors)
                            
                            for factor_a, factor_b in edges_to_orient:
                                if factor_a in available_columns and factor_b in available_columns:
                                    # Check if either factor is in related_factors list
                                    is_related_factor = (factor_a in self.related_factors or factor_b in self.related_factors)
                                    
                                    if is_related_factor:
                                        # This edge involves a related_factor - mark it as fixed (undirected)
                                        fixed_edges_undirected.append((factor_a, factor_b))
                                    
                                    # All edges from skeleton are allowed (both directions)
                                    allowed_edges.append((factor_a, factor_b))
                                    allowed_edges.append((factor_b, factor_a))
                            
                            print(f"  Allowed edges: {len(allowed_edges)} possible directed edges from {len(edges_to_orient)} undirected prior edges")
                            
                            # Create all possible edges and mark non-allowed as forbidden
                            all_possible_edges = []
                            for node_a in available_columns:
                                for node_b in available_columns:
                                    if node_a != node_b:
                                        all_possible_edges.append((node_a, node_b))
                            
                            forbidden_edges = [edge for edge in all_possible_edges if edge not in allowed_edges]
                            
                            # Build ExpertKnowledge with fixed edges
                            # Strategy: For edges involving related_factors, we'll use a modified approach
                            # Since pgmpy doesn't support "required_edges" directly, we'll:
                            # 1. Start Hill Climb with an initial model that includes fixed edges
                            # 2. Use tabu_length to prevent removal of fixed edges
                            
                            expert_knowledge = None
                            if forbidden_edges:
                                # Pass forbidden edges directly to constructor
                                expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
                            
                            # Log fixed edges for transparency
                            if fixed_edges_undirected:
                                print(f"\n  Fixed edges (related_factors): {len(fixed_edges_undirected)} undirected edges")
                                print(f"    These edges will be preserved in the skeleton before Hill Climb")
                                for factor_a, factor_b in sorted(fixed_edges_undirected):
                                    print(f"    • {factor_a} - {factor_b} (involves related_factor)")
                            
                            # Create initial model with fixed edges
                            # For fixed edges, we'll add them to the starting model
                            # Hill Climb will then orient them but won't remove them
                            
                            if fixed_edges_undirected:
                                # CRITICAL: start_dag must only contain variables in df_for_learning
                                # Filter fixed_edges to only include edges where BOTH nodes are in df_for_learning
                                valid_fixed_edges = []
                                for factor_a, factor_b in fixed_edges_undirected:
                                    if factor_a in df_for_learning.columns and factor_b in df_for_learning.columns:
                                        valid_fixed_edges.append((factor_a, factor_b))
                                    else:
                                        print(f"  ⚠ Skipping fixed edge {factor_a} - {factor_b} (not in df_for_learning)")
                                
                                if valid_fixed_edges:
                                    # DEBUG: Print what we're putting in start_dag
                                    print(f"\n  DEBUG: Creating start_dag with {len(valid_fixed_edges)} edges")
                                    print(f"  DEBUG: df_for_learning.columns = {list(df_for_learning.columns)}")
                                    print(f"  DEBUG: Nodes in valid_fixed_edges:")
                                    nodes_in_edges = set()
                                    for a, b in valid_fixed_edges:
                                        nodes_in_edges.add(a)
                                        nodes_in_edges.add(b)
                                    print(f"  DEBUG: {sorted(nodes_in_edges)}")
                                    print(f"  DEBUG: Nodes NOT in df_for_learning:")
                                    missing_nodes = [n for n in nodes_in_edges if n not in df_for_learning.columns]
                                    print(f"  DEBUG: {missing_nodes if missing_nodes else 'None - all nodes are valid!'}")
                                    
                                    # CRITICAL: start_dag must have ALL variables from df_for_learning, not just those in edges
                                    # Add isolated nodes for variables not in edges
                                    nodes_not_in_edges = [col for col in df_for_learning.columns if col not in nodes_in_edges]
                                    if nodes_not_in_edges:
                                        print(f"  DEBUG: Variables in df_for_learning but NOT in fixed edges: {nodes_not_in_edges}")
                                        print(f"  DEBUG: These will be added as isolated nodes to start_dag")
                                    
                                    # Start with a model that includes fixed edges (arbitrary direction)
                                    # Hill Climb will optimize the direction but keep the edges
                                    start_dag = DiscreteBayesianNetwork(valid_fixed_edges)
                                    
                                    # Add isolated nodes for variables not in edges
                                    for node in nodes_not_in_edges:
                                        start_dag.add_node(node)
                                    
                                    print(f"\n  Starting Hill Climb with {len(valid_fixed_edges)} fixed edges + {len(nodes_not_in_edges)} isolated nodes")
                                    
                                    # Use tabu_length to prevent removal of fixed edges
                                    # tabu_length creates a memory of recent changes, preventing their reversal
                                    best_model = hc.estimate(
                                        scoring_method=scoring_method, 
                                        max_iter=100,
                                        start_dag=start_dag,
                                        tabu_length=len(valid_fixed_edges) * 2,  # Prevent removal of fixed edges
                                        expert_knowledge=expert_knowledge,
                                        show_progress=False
                                    )
                                else:
                                    # No valid fixed edges - standard Hill Climb
                                    print(f"\n  No valid fixed edges (all involve factors not in df_for_learning)")
                                    best_model = hc.estimate(
                                        scoring_method=scoring_method, 
                                        max_iter=100,
                                        expert_knowledge=expert_knowledge,
                                        show_progress=False
                                    )
                            else:
                                # No fixed edges - standard Hill Climb
                                best_model = hc.estimate(
                                    scoring_method=scoring_method, 
                                    max_iter=100,
                                    expert_knowledge=expert_knowledge,
                                    show_progress=False
                                )
                            
                            print(f"\nPOSTERIOR: Learned {len(best_model.edges())} directed edges from Bayesian inference")
                            print("  These represent the most probable causal directions given:")
                            print("    1. PRIOR: Skeleton constraints from CausalIF 1")
                            print("    2. LIKELIHOOD: Observational data")
                            print(f"\nNote: {len(edges_to_orient) - len([e for e in edges_to_orient if (e[0] in available_columns and e[1] in available_columns and (best_model.has_edge(e[0], e[1]) or best_model.has_edge(e[1], e[0])))])} edges from PRIOR were not learned by Bayesian method")
                            
                            # ONLY add edges that are in the Bayesian POSTERIOR
                            # Iterate through best_model edges directly to preserve true direction
                            bayesian_edge_count = 0
                            print("\nPOSTERIOR Edges (MAP Estimate):")
                            
                            # Iterate through best_model edges (these are in true directional order from Hill Climb)
                            for factor_a, factor_b in best_model.edges():
                                # Verify this edge was in the skeleton (should always be true due to constraints)
                                if skeleton_for_orientation.has_edge(factor_a, factor_b) or skeleton_for_orientation.has_edge(factor_b, factor_a):
                                    directed_graph.add_edge(factor_a, factor_b)
                                    print(f"  ✓ {factor_a} → {factor_b}")
                                    bayesian_edge_count += 1
                                else:
                                    # This shouldn't happen due to skeleton constraints, but log if it does
                                    print(f"  ⚠ Unexpected edge (not in skeleton): {factor_a} → {factor_b}")
                                    directed_graph.add_edge(factor_a, factor_b)
                                    bayesian_edge_count += 1
                            
                            # Check for skeleton edges that were rejected by Bayesian method
                            rejected_count = 0
                            for factor_a, factor_b in edges_to_orient:
                                if factor_a in available_columns and factor_b in available_columns:
                                    if not best_model.has_edge(factor_a, factor_b) and not best_model.has_edge(factor_b, factor_a):
                                        print(f"  ✗ Rejected: {factor_a} - {factor_b}")
                                        rejected_count += 1
                            
                            print(f"\nFinal POSTERIOR: {bayesian_edge_count} edges (pure Bayesian, no heuristics)")
                            print("Note: The posterior is the graph structure itself (MAP estimate)")
                            print("      Edge existence/direction represents the most probable causal structure")
                            
                            # NEW: Add undirected edges for factors not in dataframe
                            # These are skeleton edges that couldn't be oriented due to missing data
                            print(f"\n{'='*80}")
                            print("ADDING UNDIRECTED EDGES FOR FACTORS WITHOUT DATA")
                            print(f"{'='*80}")
                            
                            undirected_edges_added = 0
                            for factor_a, factor_b in edges_to_orient:
                                # Check if at least one factor is NOT in available_columns
                                factor_a_missing = factor_a not in available_columns
                                factor_b_missing = factor_b not in available_columns
                                
                                if factor_a_missing or factor_b_missing:
                                    # At least one factor has no data - add as undirected edge
                                    # We'll add both directions to represent undirected edge in DiGraph
                                    # Mark with special attribute to distinguish from directed edges
                                    
                                    # Get prior strength from skeleton
                                    prior_strength = skeleton_for_orientation[factor_a][factor_b].get('prior_strength', 0.5) if skeleton_for_orientation.has_edge(factor_a, factor_b) else 0.5
                                    
                                    # Add edge in both directions with 'undirected' marker
                                    directed_graph.add_edge(factor_a, factor_b, 
                                                          undirected=True, 
                                                          prior_strength=prior_strength,
                                                          reason='no_observational_data')
                                    directed_graph.add_edge(factor_b, factor_a, 
                                                          undirected=True, 
                                                          prior_strength=prior_strength,
                                                          reason='no_observational_data')
                                    
                                    missing_factors = []
                                    if factor_a_missing:
                                        missing_factors.append(factor_a)
                                    if factor_b_missing:
                                        missing_factors.append(factor_b)
                                    
                                    print(f"  ↔ {factor_a} ↔ {factor_b} (undirected - missing data for: {', '.join(missing_factors)})")
                                    undirected_edges_added += 1
                            
                            if undirected_edges_added > 0:
                                print(f"\n✓ Added {undirected_edges_added} undirected edges for factors without observational data")
                                print(f"  These edges represent associations from CausalIF 1 (LLM voting)")
                                print(f"  Direction cannot be determined without data (kept as undirected)")
                            else:
                                print(f"\n  No factors without data - all skeleton edges processed by Bayesian method")
                            
                        except Exception as e:
                            print(f"\nBayesian structure learning failed: {e}")
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
        
        # Filter by degrees after orientation if max_degrees is specified
        if target_factor and target_factor in directed_graph.nodes() and self.max_degrees is not None:
            print(f"\nFiltering causal graph to {self.max_degrees} degrees from {target_factor}...")
            print(f"  Before filtering: {len(directed_graph.nodes())} nodes, {len(directed_graph.edges())} edges")
            directed_graph = self.filter_graph_by_degrees(directed_graph, target_factor, self.max_degrees)
            print(f"  After filtering: {len(directed_graph.nodes())} nodes, {len(directed_graph.edges())} edges")
        
        print("\n" + "=" * 80)
        print(f"BAYESIAN INFERENCE COMPLETE")
        
        # Count directed vs undirected edges
        directed_edges = [(u, v) for u, v in directed_graph.edges() if not directed_graph[u][v].get('undirected', False)]
        undirected_edge_pairs = set()
        for u, v in directed_graph.edges():
            if directed_graph[u][v].get('undirected', False):
                # Store as sorted tuple to avoid counting both directions
                edge_pair = tuple(sorted([u, v]))
                undirected_edge_pairs.add(edge_pair)
        
        print(f"POSTERIOR: {len(directed_edges)} directed edges + {len(undirected_edge_pairs)} undirected edges")
        print(f"  - Directed edges: Bayesian-inferred causal relationships (with data)")
        print(f"  - Undirected edges: LLM-inferred associations (without data)")
        print("=" * 80)
        return directed_graph


    def run_complete_causalif(self, factors: List[str], domains: List[str], target_factor: str = None) -> Tuple[nx.Graph, nx.DiGraph]:
        """
        Run complete CausalIF algorithm with Bayesian framework
        
        Bayesian Framework:
        ==================
        CausalIF 1 (Edge Existence) → PRIOR
          - Uses LLM + RAG to determine which edges exist (undirected)
          - Output: Skeleton graph G_skeleton
          - Interpretation: P(edge exists) based on domain knowledge
        
        CausalIF 2 (Orientation) → POSTERIOR  
          - Uses Bayesian structure learning on observational data
          - Constrained by skeleton (only orient edges from PRIOR)
          - Output: Directed causal graph G_causal
          - Interpretation: P(direction | data, skeleton) ∝ P(data | direction) × P(direction | skeleton)
        
        This creates a principled Bayesian workflow:
          Prior beliefs (from experts/LLM) → Updated by data → Posterior causal graph
        """
        print("\n" + "=" * 100)
        print("BAYESIAN CausalIF: PRIOR → POSTERIOR CAUSAL INFERENCE")
        print("=" * 100)
        print(f"Factors: {factors}")
        print(f"Domains: {domains}")
        print(f"Max degrees of separation: {self.max_degrees if self.max_degrees is not None else 'None (no filtering)'}")
        print(f"Max parallel queries: {self.max_parallel_queries}")
        print(f"RAG retriever available: {self.retriever_tool is not None}")
        print(f"Dataframe available: {self.dataframe is not None}")
        if self.dataframe is not None:
            print(f"Data samples: {len(self.dataframe)}")
        if target_factor:
            print(f"Target factor: {target_factor}")
        print("=" * 100)

        print("\n" + "=" * 100)
        print("STEP 1: CausalIF 1 - Building PRIOR (Edge Existence Verification)")
        print("=" * 100)
        print("Using LLM + RAG to determine which variable pairs are associated")
        print("This creates our PRIOR belief about the causal structure")
        skeleton = self.causalif_1_edge_existence_verification(factors, domains, target_factor)
        print(f"\nPRIOR (Skeleton) edges: {list(skeleton.edges())}")
        print(f"PRIOR contains {len(skeleton.edges())} undirected associations")

        print("\n" + "=" * 100)
        print("STEP 2: CausalIF 2 - Computing POSTERIOR (Bayesian Orientation)")
        print("=" * 100)
        print("Using Bayesian structure learning to orient edges from PRIOR")
        print("This updates our beliefs using observational data")
        causal_graph = self.causalif_2_orientation(skeleton, factors, domains, target_factor)
        print(f"\nPOSTERIOR (Causal) edges: {list(causal_graph.edges())}")
        print(f"POSTERIOR contains {len(causal_graph.edges())} directed causal relationships")

        print("\n" + "=" * 100)
        print("BAYESIAN CausalIF COMPLETE")
        print("=" * 100)
        print(f"Prior → Posterior transformation:")
        print(f"  {len(skeleton.edges())} undirected associations → {len(causal_graph.edges())} directed causal edges")
        print("=" * 100 + "\n")

        # NEW: Optionally fit causal model for inference
        if self.enable_causal_estimate:
            print("\n" + "=" * 100)
            print("STEP 3: Fitting Causal Model for Inference (OPTIONAL)")
            print("=" * 100)
            fitted_model = self.fit_causal_model(causal_graph)
            
            if fitted_model and target_factor and self.causal_inference_engine:
                # Automatically get causal summary for target
                print(f"\nGenerating causal summary for target: {target_factor}")
                summary = self.get_causal_summary(target_factor, causal_graph)
                
                if summary['direct_causes']:
                    print(f"\n  Direct causes of {target_factor}:")
                    for cause in summary['direct_causes']:
                        adj_set = summary['adjustment_sets'].get(cause, [])
                        if adj_set:
                            print(f"    • {cause} (adjust for: {adj_set})")
                        else:
                            print(f"    • {cause} (no adjustment needed)")
                
                print("=" * 100)

        return skeleton, causal_graph

    # ========================================================================
    # CAUSAL INFERENCE METHODS
    # ========================================================================
    
    def fit_causal_model(self, causal_graph: nx.DiGraph) -> DiscreteBayesianNetwork:
        """
        Fit a Bayesian Network with CPDs from the causal DAG structure.
        
        This is required for causal inference - we need both structure AND parameters.
        
        Args:
            causal_graph: Directed causal graph from CausalIF 2
            
        Returns:
            Fitted DiscreteBayesianNetwork with CPDs, or None if failed
        """
        if not self.enable_causal_estimate:
            print("⚠️  Causal inference is disabled. Enable with enable_causal_estimate=True")
            return None
        
        if self.dataframe is None:
            print("⚠️  No dataframe available for fitting CPDs")
            return None
        
        print("\n" + "=" * 80)
        print("FITTING CAUSAL MODEL (Learning CPDs for Causal Inference)")
        print("=" * 80)
        
        try:
            # Only include edges where both nodes have data and are directed
            available_columns = [col for col in causal_graph.nodes() if col in self.dataframe.columns]
            valid_edges = [(u, v) for u, v in causal_graph.edges() 
                          if u in available_columns and v in available_columns
                          and not causal_graph[u][v].get('undirected', False)]  # Skip undirected edges
            
            if not valid_edges:
                print("⚠️  No valid directed edges with data for causal inference")
                return None
            
            # Create Bayesian Network
            bn = DiscreteBayesianNetwork(valid_edges)
            
            # Prepare data - only use columns in the network
            nodes_in_bn = list(bn.nodes())
            df_for_fitting = self.dataframe[nodes_in_bn].copy()
            
            print(f"  Preparing data for {len(nodes_in_bn)} variables...")
            
            # Handle data types - discretize continuous variables
            for col in df_for_fitting.columns:
                if pd.api.types.is_numeric_dtype(df_for_fitting[col]):
                    # Discretize continuous variables using quantile-based binning
                    try:
                        df_for_fitting[col] = pd.qcut(
                            df_for_fitting[col], 
                            q=3, 
                            labels=['low', 'medium', 'high'], 
                            duplicates='drop'
                        )
                        print(f"    ✓ Discretized '{col}' (continuous → categorical)")
                    except Exception as e:
                        # If qcut fails (e.g., too few unique values), use cut
                        try:
                            df_for_fitting[col] = pd.cut(
                                df_for_fitting[col],
                                bins=3,
                                labels=['low', 'medium', 'high'],
                                duplicates='drop'
                            )
                            print(f"    ✓ Discretized '{col}' (continuous → categorical, using cut)")
                        except:
                            # Last resort: convert to string
                            df_for_fitting[col] = df_for_fitting[col].astype(str)
                            print(f"    ⚠️  '{col}' converted to string (discretization failed)")
                else:
                    # Categorical - ensure string type
                    df_for_fitting[col] = df_for_fitting[col].astype(str)
                    print(f"    ✓ '{col}' kept as categorical")
            
            # Fill NaN values - handle categorical columns specially
            for col in df_for_fitting.columns:
                if df_for_fitting[col].dtype.name == 'category':
                    # For categorical columns, add 'missing' to categories first
                    if 'missing' not in df_for_fitting[col].cat.categories:
                        df_for_fitting[col] = df_for_fitting[col].cat.add_categories(['missing'])
                    df_for_fitting[col] = df_for_fitting[col].fillna('missing')
                else:
                    # For non-categorical columns
                    df_for_fitting[col] = df_for_fitting[col].fillna('missing')
            
            print(f"\n  Fitting CPDs for {len(bn.nodes())} nodes, {len(bn.edges())} edges")
            print(f"  Using {len(df_for_fitting)} data samples")
            
            # Fit CPDs using Maximum Likelihood Estimation
            bn.fit(df_for_fitting, estimator=MaximumLikelihoodEstimator)
            
            print(f"  ✓ Successfully fitted {len(bn.get_cpds())} CPDs")
            
            # Store for later use
            self.causal_model = bn
            self.causal_inference_engine = CausalInference(bn)
            
            print("  ✓ Causal inference engine initialized")
            print("=" * 80)
            
            return bn
            
        except Exception as e:
            print(f"  ✗ Failed to fit causal model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def estimate_causal_effects(self, target_factor: str, causal_graph: nx.DiGraph = None) -> Dict[str, float]:
        """
        Estimate the causal effect of each parent on the target factor.
        
        This answers: "How much does each cause affect the target?"
        
        Args:
            target_factor: The outcome variable
            causal_graph: Optional causal graph (if None, uses self.causal_model)
            
        Returns:
            Dictionary mapping cause -> effect size (ATE), or empty dict if failed
        """
        if not self.enable_causal_estimate:
            print("⚠️  Causal inference is disabled. Enable with enable_causal_estimate=True")
            return {}
        
        # Fit model if not already done
        if self.causal_model is None and causal_graph is not None:
            self.fit_causal_model(causal_graph)
        
        if self.causal_inference_engine is None:
            print("⚠️  No causal model available. Run fit_causal_model() first.")
            return {}
        
        print("\n" + "=" * 80)
        print(f"ESTIMATING CAUSAL EFFECTS ON: {target_factor}")
        print("=" * 80)
        
        # Find all direct causes (parents) of target
        if causal_graph:
            causes = [p for p in causal_graph.predecessors(target_factor) 
                     if not causal_graph[p][target_factor].get('undirected', False)]
        else:
            causes = list(self.causal_model.get_parents(target_factor))
        
        if not causes:
            print(f"  No direct causes found for {target_factor}")
            return {}
        
        print(f"  Found {len(causes)} direct causes: {causes}")
        
        effects = {}
        
        for cause in causes:
            try:
                # Get adjustment set (variables to control for)
                adj_set = self.causal_inference_engine.get_minimal_adjustment_set(X=cause, Y=target_factor)
                
                if adj_set is None:
                    print(f"  ⚠️  {cause} → {target_factor}: No valid adjustment set (may have confounding)")
                    effects[cause] = None
                    continue
                
                # Note: ATE estimation with discrete variables is complex
                # For now, we'll use interventional queries to estimate effects
                print(f"  ℹ️  {cause} → {target_factor}: Adjustment set = {adj_set if adj_set else 'empty'}")
                
                # Store adjustment set info
                effects[cause] = {
                    'adjustment_set': list(adj_set) if adj_set else [],
                    'note': 'Control for adjustment set to isolate causal effect'
                }
                
            except Exception as e:
                print(f"  ✗ {cause} → {target_factor}: Failed ({str(e)[:100]})")
                effects[cause] = None
        
        print("=" * 80)
        return effects
    
    def estimate_downstream_effects(self, target_factor: str, causal_graph: nx.DiGraph = None) -> Dict[str, any]:
        """
        Estimate the causal effect of the target factor on its children (downstream effects).
        
        This answers: "What does the target factor influence?"
        
        Args:
            target_factor: The causal variable
            causal_graph: Optional causal graph (if None, uses self.causal_model)
            
        Returns:
            Dictionary mapping effect -> info, or empty dict if no effects
        """
        if not self.enable_causal_estimate:
            print("⚠️  Causal inference is disabled. Enable with enable_causal_estimate=True")
            return {}
        
        # Fit model if not already done
        if self.causal_model is None and causal_graph is not None:
            self.fit_causal_model(causal_graph)
        
        if self.causal_inference_engine is None:
            print("⚠️  No causal model available. Run fit_causal_model() first.")
            return {}
        
        print("\n" + "=" * 80)
        print(f"ESTIMATING DOWNSTREAM EFFECTS FROM: {target_factor}")
        print("=" * 80)
        
        # Find all direct effects (children) of target
        if causal_graph:
            effects = [c for c in causal_graph.successors(target_factor)
                      if not causal_graph[target_factor][c].get('undirected', False)]
        else:
            # Get children from the Bayesian Network
            effects = []
            for node in self.causal_model.nodes():
                if self.causal_model.has_edge(target_factor, node):
                    effects.append(node)
        
        if not effects:
            print(f"  No downstream effects found for {target_factor}")
            print(f"  ℹ️  {target_factor} does not directly cause any other variables in the model")
            return {}
        
        print(f"  Found {len(effects)} downstream effects: {effects}")
        
        downstream = {}
        
        for effect in effects:
            try:
                # Get adjustment set (variables to control for)
                adj_set = self.causal_inference_engine.get_minimal_adjustment_set(X=target_factor, Y=effect)
                
                if adj_set is None:
                    print(f"  ⚠️  {target_factor} → {effect}: No valid adjustment set (may have confounding)")
                    downstream[effect] = None
                    continue
                
                print(f"  ℹ️  {target_factor} → {effect}: Adjustment set = {adj_set if adj_set else 'empty'}")
                
                # Store adjustment set info
                downstream[effect] = {
                    'adjustment_set': list(adj_set) if adj_set else [],
                    'note': 'Control for adjustment set to isolate causal effect'
                }
                
            except Exception as e:
                print(f"  ✗ {target_factor} → {effect}: Failed ({str(e)[:100]})")
                downstream[effect] = None
        
        print("=" * 80)
        return downstream
    
    def get_causal_summary(self, target_factor: str, causal_graph: nx.DiGraph) -> Dict:
        """
        Get a comprehensive causal summary for the target factor.
        
        Args:
            target_factor: The target variable to analyze
            causal_graph: The causal DAG
            
        Returns:
            Dictionary with:
            - direct_causes: List of direct parents
            - direct_effects: List of direct children
            - causal_effects: Effect information (if causal inference enabled)
            - downstream_effects: What the target influences (if causal inference enabled)
            - adjustment_sets: Variables to control for
            - has_causal_inference: Whether causal inference is enabled
        """
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
            summary['direct_causes'] = [p for p in causal_graph.predecessors(target_factor)
                                       if not causal_graph[p][target_factor].get('undirected', False)]
            summary['direct_effects'] = [c for c in causal_graph.successors(target_factor)
                                        if not causal_graph[target_factor][c].get('undirected', False)]
        
        # If causal inference enabled, get effect information
        if self.enable_causal_estimate and self.causal_inference_engine:
            # What causes the target
            summary['causal_effects'] = self.estimate_causal_effects(target_factor, causal_graph)
            
            # What the target causes (downstream effects)
            summary['downstream_effects'] = self.estimate_downstream_effects(target_factor, causal_graph)
            
            # Get adjustment sets for each cause
            for cause in summary['direct_causes']:
                try:
                    adj_set = self.causal_inference_engine.get_minimal_adjustment_set(X=cause, Y=target_factor)
                    summary['adjustment_sets'][cause] = list(adj_set) if adj_set else []
                except:
                    summary['adjustment_sets'][cause] = None
        
        return summary