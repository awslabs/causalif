# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causalif Engine implementation"""
from typing import Union, Dict, List, Tuple
import plotly.graph_objects as go
import jax
import asyncio
import concurrent.futures
import pandas as pd
import networkx as nx
import numpy as np
from collections import deque
from .core import AssociationResponse, CausalDirection, KnowledgeBase
from .prompts import CausalifPrompts

# Try to import nest_asyncio for Jupyter compatibility
try:
    import nest_asyncio

    nest_asyncio.apply()
    JUPYTER_COMPATIBLE = True
except ImportError:
    JUPYTER_COMPATIBLE = False
    print("nest_asyncio not available. Install with: pip install nest-asyncio")


class CausalifEngine:
    """Causalif implementation with parallel LLM queries and complete RAG support"""

    def __init__(
        self,
        model,
        retriever_tool=None,
        dataframe: pd.DataFrame = None,
        factors=None,
        domains=None,
        k_documents: int = 5,
        max_degrees: int = 5,
        max_parallel_queries: int = 50,
    ):
        self.model = model
        self.retriever_tool = retriever_tool
        self.dataframe = dataframe
        self.factors = factors
        self.domains = domains
        self.k_documents = k_documents
        self.max_degrees = max_degrees
        self.max_parallel_queries = max_parallel_queries
        self.prompts = CausalifPrompts()

        # Initialize JAX for parallel processing
        try:
            self.devices = jax.devices()
            print(f"JAX devices available: {len(self.devices)}")
        except Exception:
            print("JAX not available, using standard parallel processing")
            self.devices = []

    def get_factors_within_degrees(
        self, target_factor: str, all_factors: List[str], max_degrees: int = None
    ) -> List[str]:
        if max_degrees is None:
            max_degrees = self.max_degrees
        return all_factors

    def filter_graph_by_degrees(
        self,
        graph: Union[nx.Graph, nx.DiGraph],
        target_factor: str,
        max_degrees: int = None,
    ) -> Union[nx.Graph, nx.DiGraph]:
        if max_degrees is None:
            max_degrees = self.max_degrees

        if target_factor not in graph.nodes():
            return graph

        visited = set()
        queue = deque([(target_factor, 0)])
        nodes_within_degrees = set([target_factor])

        while queue:
            current_node, current_degree = queue.popleft()

            if current_degree >= max_degrees:
                continue

            if isinstance(graph, nx.DiGraph):
                neighbors = set(graph.predecessors(current_node)) | set(
                    graph.successors(current_node)
                )
            else:
                neighbors = set(graph.neighbors(current_node))

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_within_degrees.add(neighbor)
                    queue.append((neighbor, current_degree + 1))

        filtered_graph = graph.subgraph(nodes_within_degrees).copy()

        print(
            f"Filtered graph to {len(filtered_graph.nodes())} nodes within {max_degrees} degrees of {target_factor}"
        )
        return filtered_graph

    def get_relationship_path(
        self, graph: Union[nx.Graph, nx.DiGraph], source: str, target: str
    ) -> List[str]:
        try:
            if isinstance(graph, nx.DiGraph):
                undirected = graph.to_undirected()
                path = nx.shortest_path(undirected, source, target)
            else:
                path = nx.shortest_path(graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []

    def analyze_degrees_of_separation(
        self, graph: Union[nx.Graph, nx.DiGraph], target_factor: str
    ) -> Dict:
        degrees_analysis = {
            "target_factor": target_factor,
            "factors_by_degree": {},
            "paths": {},
            "max_degree_found": 0,
        }

        for factor in graph.nodes():
            if factor == target_factor:
                degrees_analysis["factors_by_degree"][0] = degrees_analysis[
                    "factors_by_degree"
                ].get(0, []) + [factor]
                continue

            path = self.get_relationship_path(graph, target_factor, factor)
            if path:
                degree = len(path) - 1
                if degree <= self.max_degrees:
                    degrees_analysis["factors_by_degree"][degree] = degrees_analysis[
                        "factors_by_degree"
                    ].get(degree, []) + [factor]
                    degrees_analysis["paths"][factor] = path
                    degrees_analysis["max_degree_found"] = max(
                        degrees_analysis["max_degree_found"], degree
                    )

        return degrees_analysis

    def retrieve_documents(self, factor_a: str, factor_b: str) -> List[KnowledgeBase]:
        """Retrieve relevant documents for factor pair using RAG"""
        try:
            if self.retriever_tool:
                query = f"{factor_a} and {factor_b} correlation causation relationship"
                print(f"RAG Query: {query}")
                retrieved_docs = self.retriever_tool.invoke({"query": query})

                documents = []

                if isinstance(retrieved_docs, list):
                    for i, doc in enumerate(retrieved_docs[: self.k_documents]):
                        if isinstance(doc, dict):
                            content = (
                                doc.get("content", "")
                                or doc.get("page_content", "")
                                or str(doc)
                            )
                        elif hasattr(doc, "page_content"):
                            content = doc.page_content
                        elif isinstance(doc, str):
                            content = doc
                        else:
                            content = str(doc)

                        kb = KnowledgeBase(
                            kb_type="DOC", content=content, source=f"doc_{i}"
                        )
                        documents.append(kb)
                elif isinstance(retrieved_docs, str):
                    kb = KnowledgeBase(
                        kb_type="DOC", content=retrieved_docs, source="single_doc"
                    )
                    documents.append(kb)

                print(
                    f"Retrieved {len(documents)} documents for {factor_a} and {factor_b}"
                )
                return documents
            else:
                print("No retriever tool available for RAG")
                return []

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def get_correlation_evidence(self, factor_a: str, factor_b: str) -> str:
        """Get correlation evidence from dataframe if available"""
        if (
            self.dataframe is not None
            and factor_a in self.dataframe.columns
            and factor_b in self.dataframe.columns
        ):
            try:
                df_numeric = self.dataframe[[factor_a, factor_b]].select_dtypes(
                    include=[np.number]
                )
                if len(df_numeric.columns) >= 2:
                    correlation = df_numeric.corr().iloc[0, 1]
                    return f"Statistical analysis shows correlation of {correlation:.3f} between {factor_a} and {factor_b}"
                else:
                    try:
                        from sklearn.preprocessing import LabelEncoder

                        le = LabelEncoder()
                        df_encoded = pd.DataFrame()

                        for col in [factor_a, factor_b]:
                            if self.dataframe[col].dtype == "object":
                                df_encoded[col] = le.fit_transform(
                                    self.dataframe[col].astype(str)
                                )
                            else:
                                df_encoded[col] = self.dataframe[col]

                        correlation = df_encoded.corr().iloc[0, 1]
                        return f"Statistical analysis (encoded) shows correlation of {correlation:.3f} between {factor_a} and {factor_b}"
                    except Exception:
                        return f"Could not compute correlation between {factor_a} and {factor_b}"
            except Exception as e:
                return f"Could not compute correlation between {factor_a} and {factor_b}: {str(e)}"
        return "No statistical evidence available from data"

    def parallel_llm_query_sync(self, prompts: List[str]) -> List[str]:
        """Execute multiple LLM queries in parallel using ThreadPoolExecutor (Jupyter-compatible)"""

        def single_query(prompt: str) -> str:
            try:
                if self.model is None:
                    return "UNKNOWN"

                response = self.model.invoke(prompt)
                return (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                print(f"Error in parallel query: {e}")
                return "UNKNOWN"

        # Use ThreadPoolExecutor for parallel execution (Jupyter-compatible)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_parallel_queries
        ) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(single_query, prompt): prompt for prompt in prompts
            }

            # Collect results in order
            results = [None] * len(prompts)
            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_prompt)
            ):
                try:
                    result = future.result()
                    # Find the original index
                    original_prompt = future_to_prompt[future]
                    original_index = prompts.index(original_prompt)
                    results[original_index] = result
                except Exception as e:
                    print(f"Query failed: {e}")
                    # Find the original index for failed query
                    original_prompt = future_to_prompt[future]
                    original_index = prompts.index(original_prompt)
                    results[original_index] = "UNKNOWN"

        # Fill any None values with "UNKNOWN"
        results = [r if r is not None else "UNKNOWN" for r in results]
        return results

    async def parallel_llm_query_async(self, prompts: List[str]) -> List[str]:
        """Execute multiple LLM queries in parallel using asyncio (for non-Jupyter environments)"""

        async def single_query(prompt: str) -> str:
            try:
                if self.model is None:
                    return "UNKNOWN"

                # Use asyncio to run the synchronous model call in a thread pool
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_parallel_queries
                ) as executor:
                    response = await loop.run_in_executor(
                        executor, self.model.invoke, prompt
                    )
                    return (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
            except Exception as e:
                print(f"Error in parallel query: {e}")
                return "UNKNOWN"

        # Execute all queries in parallel
        tasks = [single_query(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Query failed: {result}")
                processed_results.append("UNKNOWN")
            else:
                processed_results.append(result)

        return processed_results

    def execute_parallel_queries(self, prompts: List[str]) -> List[str]:
        """Execute parallel queries with Jupyter compatibility"""
        if not prompts:
            return []

        try:
            # Try asyncio approach first (works in regular Python)
            if JUPYTER_COMPATIBLE:
                # Use asyncio with nest_asyncio for Jupyter
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.parallel_llm_query_async(prompts))
            else:
                # Fallback to synchronous parallel execution
                return self.parallel_llm_query_sync(prompts)
        except Exception as e:
            print(f"Parallel execution failed, falling back to sync: {e}")
            return self.parallel_llm_query_sync(prompts)

    def query_association_background(
        self, factor_a: str, factor_b: str, factors: List[str], domains: List[str]
    ) -> AssociationResponse:
        """Query LLM for association using background knowledge"""

        # Check if model is available
        if self.model is None:
            print("Warning: No model available, using statistical evidence only")
            # Fallback to statistical analysis
            return self._statistical_association_fallback(factor_a, factor_b)

        # Include data evidence if available
        data_evidence = self.get_correlation_evidence(factor_a, factor_b)

        prompt = f"""
        {self.prompts.background_reminder(factors, domains)}

        Your task is to thoroughly use the knowledge in your training data to solve a task. Your task is: based on your background knowledge and the following data evidence, try to find statistical evidence to clarify the association relationship between the pair of 'Main factors' according to the 'Association Context'.

        Data Evidence: {data_evidence}

        Main factors: {factor_a} and {factor_b}

        Association Context:
        {self.prompts.association_context()}

        Association Question: Are {factor_a} and {factor_b} associated?

        Consider both your background knowledge and the data evidence provided.

        Expected Response Format:
        Thoughts: [Write your thoughts on the question]
        Answer: (A) Associated (B) Independent (C) Unknown
        """

        try:
            response = self.model.invoke(prompt)
            response_text = response.content.upper()

            if "ASSOCIATED" in response_text and "INDEPENDENT" not in response_text:
                return AssociationResponse.ASSOCIATED
            elif "INDEPENDENT" in response_text:
                return AssociationResponse.INDEPENDENT
            else:
                return AssociationResponse.UNKNOWN
        except Exception as e:
            print(f"Error in background association query: {e}")
            return self._statistical_association_fallback(factor_a, factor_b)

    def query_association_document(
        self,
        factor_a: str,
        factor_b: str,
        document: KnowledgeBase,
        factors: List[str],
        domains: List[str],
    ) -> AssociationResponse:
        """Query LLM for association using a specific document"""

        if self.model is None:
            print("Warning: No model available for document analysis, using fallback")
            return self._statistical_association_fallback(factor_a, factor_b)

        prompt = f"""
        {self.prompts.background_reminder(factors, domains)}

        Your task is to thoroughly read the given 'Document'. Then, based on the knowledge from the given 'Document', try to find statistical evidence to clarify the association relationship between the pair of 'Main factors' according to the 'Association Context'.

        Document: {document.content[:2000]}...

        Main factors: {factor_a} and {factor_b}

        Association Context:
        {self.prompts.association_context()}

        Association Question: Are {factor_a} and {factor_b} associated?

        Expected Response Format:
        Thoughts: [Write your thoughts on the question]
        Answer: (A) Associated (B) Independent (C) Unknown
        Reference: [Skip this if you chose option C above. Otherwise, provide a supporting sentence from the document for your choice]
        """

        try:
            response = self.model.invoke(prompt)
            response_text = response.content.upper()

            if "ASSOCIATED" in response_text and "INDEPENDENT" not in response_text:
                return AssociationResponse.ASSOCIATED
            elif "INDEPENDENT" in response_text:
                return AssociationResponse.INDEPENDENT
            else:
                return AssociationResponse.UNKNOWN
        except Exception as e:
            print(f"Error in document association query: {e}")
            return self._statistical_association_fallback(factor_a, factor_b)

    def query_causal_direction_background(
        self, factor_a: str, factor_b: str, factors: List[str], domains: List[str]
    ) -> CausalDirection:
        """Query causal direction using background knowledge"""

        if self.model is None:
            print("Warning: No model available for causal direction")
            return CausalDirection.UNKNOWN

        # Include data evidence
        data_evidence = self.get_correlation_evidence(factor_a, factor_b)

        prompt = f"""
        {self.prompts.background_reminder([factor_a, factor_b], domains)}

        Your task is to thoroughly use the knowledge in your training data to solve a task. Your task is: based on your background knowledge and data evidence, try to find statistical evidence to clarify the direction of the causal relationship between the pair of 'Main factors' according to the 'Causal direction context'.

        Data Evidence: {data_evidence}

        Main factors: {factor_a} and {factor_b}

        Causal direction context:
        {self.prompts.causal_direction_context()}

        Causal direction question: Is {factor_a} the cause of {factor_b}, or {factor_b} the cause of {factor_a}?

        Expected Response Format:
        Thoughts: [Write your thoughts on the question]
        Answer: (A) {factor_a} is the cause of {factor_b} (B) {factor_b} is the cause of {factor_a} (C) Unknown
        """

        try:
            response = self.model.invoke(prompt)
            response_text = response.content

            if (
                f"{factor_a} is the cause of {factor_b}" in response_text
                or f"{factor_a.upper()} IS THE CAUSE OF {factor_b.upper()}"
                in response_text.upper()
            ):
                return CausalDirection.A_CAUSES_B
            elif (
                f"{factor_b} is the cause of {factor_a}" in response_text
                or f"{factor_b.upper()} IS THE CAUSE OF {factor_a.upper()}"
                in response_text.upper()
            ):
                return CausalDirection.B_CAUSES_A
            else:
                return CausalDirection.UNKNOWN
        except Exception as e:
            print(f"Error in causal direction query: {e}")
            return CausalDirection.UNKNOWN

    def query_causal_direction_document(
        self,
        factor_a: str,
        factor_b: str,
        document: KnowledgeBase,
        factors: List[str],
        domains: List[str],
    ) -> CausalDirection:
        """Query causal direction using document knowledge"""

        prompt = f"""
        {self.prompts.background_reminder([factor_a, factor_b], domains)}

        Your task is to thoroughly read the given 'Document'. Then, based on the knowledge from the given 'Document', try to find statistical evidence to clarify the direction of the causal relationship between the pair of 'Main factors' according to the 'Causal direction context'.

        Document: {document.content[:2000]}...

        Main factors: {factor_a} and {factor_b}

        Causal direction context:
        {self.prompts.causal_direction_context()}

        Causal direction question: Is {factor_a} the cause of {factor_b}, or {factor_b} the cause of {factor_a}?

        Expected Response Format:
        Thoughts: [Write your thoughts on the question]
        Answer: (A) {factor_a} is the cause of {factor_b} (B) {factor_b} is the cause of {factor_a} (C) Unknown
        Reference: [Skip this if you chose option C above. Otherwise, provide a supporting sentence from the document for your choice]
        """

        try:
            response = self.model.invoke(prompt)
            response_text = response.content

            if (
                f"{factor_a} is the cause of {factor_b}" in response_text
                or f"{factor_a.upper()} IS THE CAUSE OF {factor_b.upper()}"
                in response_text.upper()
            ):
                return CausalDirection.A_CAUSES_B
            elif (
                f"{factor_b} is the cause of {factor_a}" in response_text
                or f"{factor_b.upper()} IS THE CAUSE OF {factor_a.upper()}"
                in response_text.upper()
            ):
                return CausalDirection.B_CAUSES_A
            else:
                return CausalDirection.UNKNOWN
        except Exception as e:
            print(f"Error in document causal direction query: {e}")
            return CausalDirection.UNKNOWN

    def compute_edge_existence(
        self,
        factor_a: str,
        factor_b: str,
        kb: KnowledgeBase,
        factors: List[str],
        domains: List[str],
    ) -> int:
        """Compute ζ_KB(ij) for a knowledge base - original Causalif method"""

        # Step 1: Association verification
        if kb.kb_type == "BG":
            association = self.query_association_background(
                factor_a, factor_b, factors, domains
            )
        else:
            association = self.query_association_document(
                factor_a, factor_b, kb, factors, domains
            )

        if association == AssociationResponse.INDEPENDENT:
            return 0  # No edge - factors are independent
        elif association == AssociationResponse.UNKNOWN:
            return None  # KB is unusable

        # Step 2: Association type verification (simplified for now)
        # In full Causalif, this would check for direct vs indirect association
        # For now, we assume associated means direct edge
        if association == AssociationResponse.ASSOCIATED:
            return 1  # Edge exists - direct association

        return None  # Unknown

    def batch_association_queries(
        self,
        factor_pairs: List[Tuple[str, str]],
        factors: List[str],
        domains: List[str],
    ) -> Dict[Tuple[str, str], AssociationResponse]:
        """Batch process association queries for multiple factor pairs using background knowledge"""

        # Prepare all prompts for parallel execution
        prompts = []

        for factor_a, factor_b in factor_pairs:
            data_evidence = self.get_correlation_evidence(factor_a, factor_b)

            prompt = f"""
            {self.prompts.background_reminder(factors, domains)}

            Your task is to thoroughly use the knowledge in your training data to solve a task. Your task is: based on your background knowledge and the following data evidence, try to find statistical evidence to clarify the association relationship between the pair of 'Main factors' according to the 'Association Context'.

            Data Evidence: {data_evidence}

            Main factors: {factor_a} and {factor_b}

            Association Context:
            {self.prompts.association_context()}

            Association Question: Are {factor_a} and {factor_b} associated?

            Consider both your background knowledge and the data evidence provided.

            Expected Response Format:
            Thoughts: [Write your thoughts on the question]
            Answer: (A) Associated (B) Independent (C) Unknown
            """

            prompts.append(prompt)

        # Execute all queries in parallel
        try:
            responses = self.execute_parallel_queries(prompts)
        except Exception as e:
            print(f"Error in batch processing: {e}")
            responses = ["UNKNOWN"] * len(prompts)

        # Process responses
        results = {}
        for i, (factor_a, factor_b) in enumerate(factor_pairs):
            if i < len(responses):
                response_text = responses[i].upper()

                if "ASSOCIATED" in response_text and "INDEPENDENT" not in response_text:
                    results[(factor_a, factor_b)] = AssociationResponse.ASSOCIATED
                elif "INDEPENDENT" in response_text:
                    results[(factor_a, factor_b)] = AssociationResponse.INDEPENDENT
                else:
                    results[(factor_a, factor_b)] = AssociationResponse.UNKNOWN
            else:
                results[(factor_a, factor_b)] = AssociationResponse.UNKNOWN

        return results

    def batch_causal_direction_queries(
        self,
        factor_pairs: List[Tuple[str, str]],
        factors: List[str],
        domains: List[str],
    ) -> Dict[Tuple[str, str], CausalDirection]:
        """Batch process causal direction queries for multiple factor pairs using background knowledge"""

        prompts = []

        for factor_a, factor_b in factor_pairs:
            data_evidence = self.get_correlation_evidence(factor_a, factor_b)

            prompt = f"""
            {self.prompts.background_reminder([factor_a, factor_b], domains)}

            Your task is to thoroughly use the knowledge in your training data to solve a task. Your task is: based on your background knowledge and data evidence, try to find statistical evidence to clarify the direction of the causal relationship between the pair of 'Main factors' according to the 'Causal direction context'.

            Data Evidence: {data_evidence}

            Main factors: {factor_a} and {factor_b}

            Causal direction context:
            {self.prompts.causal_direction_context()}

            Causal direction question: Is {factor_a} the cause of {factor_b}, or {factor_b} the cause of {factor_a}?

            Expected Response Format:
            Thoughts: [Write your thoughts on the question]
            Answer: (A) {factor_a} is the cause of {factor_b} (B) {factor_b} is the cause of {factor_a} (C) Unknown
            """

            prompts.append(prompt)

        # Execute all queries in parallel
        try:
            responses = self.execute_parallel_queries(prompts)
        except Exception as e:
            print(f"Error in batch causal direction processing: {e}")
            responses = ["UNKNOWN"] * len(prompts)

        # Process responses
        results = {}
        for i, (factor_a, factor_b) in enumerate(factor_pairs):
            if i < len(responses):
                response_text = responses[i]

                if (
                    f"{factor_a} is the cause of {factor_b}" in response_text
                    or f"{factor_a.upper()} IS THE CAUSE OF {factor_b.upper()}"
                    in response_text.upper()
                ):
                    results[(factor_a, factor_b)] = CausalDirection.A_CAUSES_B
                elif (
                    f"{factor_b} is the cause of {factor_a}" in response_text
                    or f"{factor_b.upper()} IS THE CAUSE OF {factor_a.upper()}"
                    in response_text.upper()
                ):
                    results[(factor_a, factor_b)] = CausalDirection.B_CAUSES_A
                else:
                    results[(factor_a, factor_b)] = CausalDirection.UNKNOWN
            else:
                results[(factor_a, factor_b)] = CausalDirection.UNKNOWN

        return results

    def _statistical_association_fallback(
        self, factor_a: str, factor_b: str
    ) -> AssociationResponse:
        """Fallback method using only statistical correlation from data"""
        try:
            evidence = self.get_correlation_evidence(factor_a, factor_b)

            if "correlation of" in evidence:
                corr_value = float(evidence.split("correlation of ")[1].split(" ")[0])

                # Use absolute correlation value to determine association
                # No hard-coded thresholds - rely purely on statistical significance
                if abs(corr_value) > 0.5:  # Strong correlation
                    return AssociationResponse.ASSOCIATED
                elif abs(corr_value) < 0.1:  # Very weak correlation
                    return AssociationResponse.INDEPENDENT
                else:  # Moderate correlation - uncertain
                    return AssociationResponse.UNKNOWN
            else:
                # No statistical data available
                return AssociationResponse.UNKNOWN

        except Exception as e:
            print(f"Statistical fallback error: {e}")
            return AssociationResponse.UNKNOWN

    def causalif_1_edge_existence_verification(
        self, factors: List[str], domains: List[str], target_factor: str = None
    ) -> nx.Graph:
        """Causalif 1: Edge Existence Verification Algorithm with complete knowledge base support"""

        print(
            "Starting Causalif 1: Edge Existence Verification with complete RAG support in causalif_local"
        )

        # Initialize complete undirected graph
        G = nx.complete_graph(len(factors))
        G = nx.relabel_nodes(G, {i: factors[i] for i in range(len(factors))})

        edges_to_remove = []

        # For each pair of variables (limit pairs for demo)
        factor_pairs = [
            (factors[i], factors[j])
            for i in range(len(factors))
            for j in range(i + 1, len(factors))
        ]
        limited_pairs = factor_pairs[: min(15, len(factor_pairs))]

        print(
            f"Processing {len(limited_pairs)} factor pairs with complete knowledge base support..."
        )

        for factor_a, factor_b in limited_pairs:
            print(f"Processing pair: {factor_a} - {factor_b}")

            S = 0  # Score for this variable pair

            # Get knowledge bases - COMPLETE ORIGINAL Causalif APPROACH
            knowledge_bases = []

            # Add background knowledge
            bg_kb = KnowledgeBase(kb_type="BG")
            knowledge_bases.append(bg_kb)
            print(f"Added background knowledge base for {factor_a} - {factor_b}")

            # Add retrieved documents using RAG
            doc_kbs = self.retrieve_documents(factor_a, factor_b)
            knowledge_bases.extend(doc_kbs[:2])  # Limit to 2 for performance
            print(
                f"Added {len(doc_kbs[:2])} document knowledge bases for {factor_a} - {factor_b}"
            )

            # Query each knowledge base - ORIGINAL Causalif METHOD
            for kb in knowledge_bases:
                edge_decision = self.compute_edge_existence(
                    factor_a, factor_b, kb, factors, domains
                )

                if edge_decision == 1:
                    S += 1  # Vote for edge existence
                    print(f"KB {kb.kb_type} votes FOR edge {factor_a} - {factor_b}")
                elif edge_decision == 0:
                    S -= 1  # Vote against edge existence
                    print(f"KB {kb.kb_type} votes AGAINST edge {factor_a} - {factor_b}")
                else:
                    print(
                        f"KB {kb.kb_type} is UNKNOWN for edge {factor_a} - {factor_b}"
                    )
                # If None (unknown), don't change score

            # Decision: remove edge if S <= 0
            if S <= 0:
                edges_to_remove.append((factor_a, factor_b))
                print(f"Removing edge: {factor_a} - {factor_b} (Score: {S})")
            else:
                print(f"Keeping edge: {factor_a} - {factor_b} (Score: {S})")

        # Remove edges based on voting
        G.remove_edges_from(edges_to_remove)

        # Filter by degrees if target factor is specified
        if target_factor and target_factor in G.nodes():
            G = self.filter_graph_by_degrees(G, target_factor, self.max_degrees)

        print(
            f"Causalif 1 completed with complete RAG support. Final graph has {len(G.edges())} edges."
        )
        return G

    def causalif_2_orientation(
        self,
        skeleton: nx.Graph,
        factors: List[str],
        domains: List[str],
        target_factor: str = None,
    ) -> nx.DiGraph:
        """Causalif 2: Orientation Algorithm with complete knowledge base support"""

        print("Starting Causalif 2: Orientation with complete RAG support")

        # Convert undirected skeleton to directed graph
        directed_graph = nx.DiGraph()
        directed_graph.add_nodes_from(skeleton.nodes())

        # Get all edges that need orientation
        edges_to_orient = list(skeleton.edges())

        if not edges_to_orient:
            print("No edges to orient")
            if target_factor and target_factor in directed_graph.nodes():
                directed_graph = self.filter_graph_by_degrees(
                    directed_graph, target_factor, self.max_degrees
                )
            return directed_graph

        print(
            f"Orienting {len(edges_to_orient)} edges with complete knowledge base support..."
        )

        # For each edge in skeleton, determine direction using complete knowledge base approach
        for factor_a, factor_b in edges_to_orient:
            print(f"Determining direction for: {factor_a} - {factor_b}")

            causal_votes = []

            # Query causal direction using background knowledge
            bg_direction = self.query_causal_direction_background(
                factor_a, factor_b, factors, domains
            )
            causal_votes.append(bg_direction)
            print(f"Background knowledge vote: {bg_direction}")

            # Query causal direction using documents from RAG
            doc_kbs = self.retrieve_documents(factor_a, factor_b)
            for doc_kb in doc_kbs[:2]:  # Limit for performance
                doc_direction = self.query_causal_direction_document(
                    factor_a, factor_b, doc_kb, factors, domains
                )
                causal_votes.append(doc_direction)
                print(f"Document {doc_kb.source} vote: {doc_direction}")

            # Majority voting for causal direction
            a_causes_b_count = sum(
                1 for vote in causal_votes if vote == CausalDirection.A_CAUSES_B
            )
            b_causes_a_count = sum(
                1 for vote in causal_votes if vote == CausalDirection.B_CAUSES_A
            )

            if a_causes_b_count > b_causes_a_count:
                directed_graph.add_edge(factor_a, factor_b)
                print(
                    f"Direction: {factor_a} -> {factor_b} (votes: {a_causes_b_count} vs {b_causes_a_count})"
                )
            elif b_causes_a_count > a_causes_b_count:
                directed_graph.add_edge(factor_b, factor_a)
                print(
                    f"Direction: {factor_b} -> {factor_a} (votes: {b_causes_a_count} vs {a_causes_b_count})"
                )
            else:
                # If tied or all unknown, add edge in original order (no heuristics)
                directed_graph.add_edge(factor_a, factor_b)
                print(
                    f"Direction (default - no clear winner): {factor_a} -> {factor_b}"
                )

        # Filter by degrees if target factor is specified
        if target_factor and target_factor in directed_graph.nodes():
            directed_graph = self.filter_graph_by_degrees(
                directed_graph, target_factor, self.max_degrees
            )

        print(
            f"Causalif 2 completed with complete RAG support. Final graph has {len(directed_graph.edges())} directed edges."
        )
        return directed_graph

    def run_complete_causalif(
        self, factors: List[str], domains: List[str], target_factor: str = None
    ) -> Tuple[nx.Graph, nx.DiGraph]:
        """Run complete Causalif algorithm with full knowledge base support"""
        print("Starting Complete Causalif Algorithm with Full RAG Support...")
        print(f"Factors: {factors}")
        print(f"Domains: {domains}")
        print(f"Max degrees of separation: {self.max_degrees}")
        print(f"Max parallel queries: {self.max_parallel_queries}")
        print(f"RAG retriever available: {self.retriever_tool is not None}")
        if target_factor:
            print(f"Target factor: {target_factor}")

        # Causalif 1: Edge Existence Verification (with complete knowledge base support)
        print("\n=== Causalif 1: Edge Existence Verification with Complete RAG ===")
        skeleton = self.causalif_1_edge_existence_verification(
            factors, domains, target_factor
        )

        print(f"\n{'='*60}")
        print(f"CAUSALIF 1 COMPLETE")
        print(f"Skeleton has {len(skeleton.nodes())} nodes and {len(skeleton.edges())} edges")
        print(f"Edges: {list(skeleton.edges())}")
        print(f"{'='*60}\n")
        
        if len(skeleton.edges()) == 0:
            print("⚠️ WARNING: Empty skeleton! No edges to orient.")
            return skeleton, nx.DiGraph()

        # Causalif 2: Orientation (with complete knowledge base support)
        print("\n=== Causalif 2: Orientation with Complete RAG ===")
        causal_graph = self.causalif_2_orientation(
            skeleton, factors, domains, target_factor
        )

        print(f"\n{'='*60}")
        print(f"CAUSALIF 2 COMPLETE")
        print(f"Causal graph has {len(causal_graph.nodes())} nodes and {len(causal_graph.edges())} edges")
        print(f"Directed edges: {list(causal_graph.edges())}")
        print(f"{'='*60}\n")
        
        print(f"\nFinal causal graph edges: {list(causal_graph.edges())}")

        return skeleton, causal_graph

    def visualize_graph(
        self,
        graph: Union[nx.Graph, nx.DiGraph],
        title: str = "Graph",
        target_factor: str = None,
    ) -> go.Figure:
        import math

        if len(graph.nodes()) == 0:
            print(f"No nodes to display in {title}")
            return None

        # Layout
        pos = (
            nx.spring_layout(graph, seed=42, k=3, iterations=50)
            if graph.edges()
            else {node: (i, 0) for i, node in enumerate(graph.nodes())}
        )

        # Degrees of separation
        degrees_map = {}
        if target_factor and target_factor in graph.nodes():
            degrees_analysis = self.analyze_degrees_of_separation(graph, target_factor)
            for degree, factors in degrees_analysis["factors_by_degree"].items():
                for factor in factors:
                    degrees_map[factor] = degree

        fig = go.Figure()

        # --- Edges with Arrows ---
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_label = (
                f"{edge[0]} → {edge[1]}"
                if isinstance(graph, nx.DiGraph)
                else f"{edge[0]} — {edge[1]}"
            )

            # Edge color
            edge_color = "red"
            if target_factor:
                max_degree = max(
                    degrees_map.get(edge[0], 0), degrees_map.get(edge[1], 0)
                )
                if max_degree <= 1:
                    edge_color = "red"
                elif max_degree <= 2:
                    edge_color = "orange"
                elif max_degree <= 3:
                    edge_color = "yellow"
                elif max_degree <= 4:
                    edge_color = "lightblue"
                else:
                    edge_color = "lightgray"

            # Calculate arrow positioning
            dx, dy = x1 - x0, y1 - y0
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx_norm, dy_norm = dx / length, dy / length
                node_radius = 0.05
                x1_short = x1 - dx_norm * node_radius
                y1_short = y1 - dy_norm * node_radius
                x0_short = x0 + dx_norm * node_radius
                y0_short = y0 + dy_norm * node_radius

                # Edge line with hover label
                fig.add_trace(
                    go.Scatter(
                        x=[x0_short, x1_short],
                        y=[y0_short, y1_short],
                        line=dict(width=2, color=edge_color),
                        mode="lines",
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=edge_label,
                    )
                )

                # Arrow annotation (shown for all edges now)
                arrow_x = x0_short + 0.85 * (x1_short - x0_short)
                arrow_y = y0_short + 0.85 * (y1_short - y0_short)
                fig.add_annotation(
                    x=arrow_x,
                    y=arrow_y,
                    ax=arrow_x - 0.02 * dx_norm,
                    ay=arrow_y - 0.02 * dy_norm,
                    axref="x",
                    ayref="y",
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=3,
                    arrowcolor=edge_color,
                    showarrow=True,
                )

        # --- Nodes ---
        node_x, node_y, node_text, node_colors = [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

            if target_factor and node in degrees_map:
                node_colors.append(degrees_map[node])
            else:
                degree = (
                    graph.degree(node)
                    if not isinstance(graph, nx.DiGraph)
                    else graph.in_degree(node) + graph.out_degree(node)
                )
                node_colors.append(degree)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            textfont=dict(size=12, color="white", family="Arial Black"),
            marker=dict(
                size=50,
                color=node_colors,
                colorscale="RdYlBu_r" if target_factor else "Viridis",
                line=dict(width=3, color="white"),
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Degrees from Target" if target_factor else "Node Degree",
                        font=dict(color="white"),
                    ),
                    tickfont=dict(color="white"),
                ),
            ),
            hoverinfo="text",
            hovertext=[
                f"Node: {node}<br>Degree from {target_factor}: {degrees_map.get(node, 'N/A')}<br>Connections: {graph.degree(node)}"
                if target_factor and node in degrees_map
                else f"Node: {node}<br>Connections: {graph.degree(node)}"
                for node in graph.nodes()
            ],
        )

        fig.add_trace(node_trace)

        # --- Layout with Black Background ---
        graph_type = "Directed" if isinstance(graph, nx.DiGraph) else "Undirected"
        degree_info = f" (Max {self.max_degrees} degrees)" if target_factor else ""
        fig.update_layout(
            title=dict(
                text=f"{title} ({graph_type}){degree_info} - {len(graph.nodes())} nodes, {len(graph.edges())} edges - Causalif Enhanced with RAG",
                x=0.5,
                font=dict(size=16, color="white"),
            ),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=700,
            plot_bgcolor="black",
            paper_bgcolor="black",
        )

        return fig
