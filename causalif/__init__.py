                                                            # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CausalIF: Language-Augmented Causal Reasoning with Bayesian Inference
"""

import logging

__version__ = "0.1.10"
__author__ = "Subhro Bose"
__email__ = "bossubhr@amazon.co.uk"

# Auto-configure logging so output is visible in notebooks and scripts.
# Only adds a handler if the root logger (or causalif logger) has none,
# to avoid duplicate output if the user already configured logging.
_logger = logging.getLogger("causalif")
if not _logger.handlers and not logging.root.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger.setLevel(logging.INFO)

from .core import (
    KnowledgeBase
)

from .engine import (
    CausalIFEngine,
    PriorWeightedBDeu
)

from .prompts import CausalIFPrompts, format_causal_graph_for_llm, generate_llm_interpretation

from .tool import (
    causalif_tool,
    set_causalif_engine,
    extract_factors_from_query,
    causalif,
    causalif_intervene,
    parse_intervention_query
)

from .visualization import visualize_causalif_results, visualize_graph

from .benchmarks import (
    evaluate_graph,
    evaluate_causalif_result,
    run_benchmark,
    sensitivity_analysis,
    generate_data_from_dag,
    generate_binary_data_from_dag,
    run_baselines,
    compare_all_with_baselines,
    BenchmarkSuite,
    BenchmarkResult,
    GraphMetrics,
    BENCHMARK_REGISTRY,
)

__all__ = [
    'KnowledgeBase',
    'CausalIFEngine',
    'PriorWeightedBDeu',
    'CausalIFPrompts',
    'causalif_tool',
    'set_causalif_engine',
    'extract_factors_from_query',
    'format_causal_graph_for_llm',
    'generate_llm_interpretation',
    'causalif',
    'causalif_intervene',
    'parse_intervention_query',
    'visualize_causalif_results',
    'visualize_graph',
    # Benchmarking
    'evaluate_graph',
    'evaluate_causalif_result',
    'run_benchmark',
    'sensitivity_analysis',
    'generate_data_from_dag',
    'generate_binary_data_from_dag',
    'run_baselines',
    'compare_all_with_baselines',
    'BenchmarkSuite',
    'BenchmarkResult',
    'GraphMetrics',
    'BENCHMARK_REGISTRY',
]
