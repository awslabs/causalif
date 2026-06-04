# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CausalIF Benchmarking Module

Provides accuracy evaluation against known ground-truth causal graphs.
Supports standard benchmark networks (ASIA, Sachs, ALARM) and custom graphs.

Metrics computed:
- Precision, Recall, F1 (edge-level)
- Structural Hamming Distance (SHD)
- Structural Intervention Distance (SID) approximation
- Per-edge breakdown (TP, FP, FN, reversed)
- Bootstrap stability correlation with ground truth

Usage:
    from causalif.benchmarks import evaluate_graph, run_benchmark, BenchmarkSuite

    # Quick evaluation against a known graph
    metrics = evaluate_graph(
        discovered_edges=[('A', 'B'), ('B', 'C')],
        true_edges=[('A', 'B'), ('A', 'C'), ('B', 'C')]
    )

    # Full benchmark against standard networks
    results = run_benchmark("asia", model=your_llm_model, n_samples=1000)

    # Run all benchmarks
    suite = BenchmarkSuite(model=your_llm_model)
    all_results = suite.run_all(n_samples=1000)
    suite.summary_table(all_results)
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@dataclass
class GraphMetrics:
    """Container for causal graph evaluation metrics."""
    precision: float
    recall: float
    f1: float
    shd: int
    true_positives: int
    false_positives: int
    false_negatives: int
    reversed_edges: int
    total_discovered: int
    total_true: int
    tp_edges: List[Tuple[str, str]] = field(default_factory=list)
    fp_edges: List[Tuple[str, str]] = field(default_factory=list)
    fn_edges: List[Tuple[str, str]] = field(default_factory=list)
    reversed_edge_list: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert metrics to a flat dictionary for reporting."""
        return {
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'shd': self.shd,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'reversed_edges': self.reversed_edges,
            'total_discovered': self.total_discovered,
            'total_true': self.total_true,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "CAUSAL GRAPH EVALUATION METRICS",
            "=" * 60,
            f"  Precision:        {self.precision:.4f}",
            f"  Recall:           {self.recall:.4f}",
            f"  F1 Score:         {self.f1:.4f}",
            f"  SHD:              {self.shd}",
            "-" * 60,
            f"  True Positives:   {self.true_positives}",
            f"  False Positives:  {self.false_positives}",
            f"  False Negatives:  {self.false_negatives}",
            f"  Reversed Edges:   {self.reversed_edges}",
            "-" * 60,
            f"  Discovered Edges: {self.total_discovered}",
            f"  Ground Truth:     {self.total_true}",
            "=" * 60,
        ]
        return "\n".join(lines)


def evaluate_graph(
    discovered_edges: List[Tuple[str, str]],
    true_edges: List[Tuple[str, str]],
    check_orientation: bool = True,
) -> GraphMetrics:
    """Evaluate a discovered causal graph against a ground-truth DAG.

    Args:
        discovered_edges: List of (cause, effect) tuples from CausalIF output.
        true_edges: List of (cause, effect) tuples from ground truth.
        check_orientation: If True, edge direction matters (A→B ≠ B→A).
                          If False, treats edges as undirected for comparison.

    Returns:
        GraphMetrics dataclass with all evaluation metrics.
    """
    if check_orientation:
        discovered_set = set(discovered_edges)
        true_set = set(true_edges)
    else:
        discovered_set = set(frozenset(e) for e in discovered_edges)
        true_set = set(frozenset(e) for e in true_edges)

    tp_edges = discovered_set & true_set
    fp_edges = discovered_set - true_set
    fn_edges = true_set - discovered_set

    true_positives = len(tp_edges)
    false_positives = len(fp_edges)
    false_negatives = len(fn_edges)

    # Count reversed edges (discovered A→B when truth is B→A)
    reversed_edges = []
    if check_orientation:
        for edge in fp_edges:
            reversed = (edge[1], edge[0])
            if reversed in true_set:
                reversed_edges.append(edge)
    reversed_count = len(reversed_edges)

    # SHD = additions + deletions + reversals
    # Each reversed edge counts as 1 (not 2, since it's a single reversal operation)
    shd = false_positives + false_negatives - reversed_count

    # Precision, Recall, F1
    precision = true_positives / len(discovered_set) if discovered_set else 0.0
    recall = true_positives / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Convert frozensets back to tuples for reporting
    if not check_orientation:
        tp_list = [tuple(sorted(e)) for e in tp_edges]
        fp_list = [tuple(sorted(e)) for e in fp_edges]
        fn_list = [tuple(sorted(e)) for e in fn_edges]
    else:
        tp_list = list(tp_edges)
        fp_list = list(fp_edges)
        fn_list = list(fn_edges)

    return GraphMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        shd=shd,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        reversed_edges=reversed_count,
        total_discovered=len(discovered_set),
        total_true=len(true_set),
        tp_edges=sorted(tp_list),
        fp_edges=sorted(fp_list),
        fn_edges=sorted(fn_list),
        reversed_edge_list=sorted(reversed_edges),
    )


def evaluate_causalif_result(result: Dict, true_edges: List[Tuple[str, str]], check_orientation: bool = True) -> GraphMetrics:
    """Evaluate a CausalIF result dict against ground truth.

    Convenience wrapper that extracts edges from the CausalIF output format.

    Args:
        result: Dict returned by causalif() function.
        true_edges: Ground-truth edges as (cause, effect) tuples.
        check_orientation: Whether edge direction matters.

    Returns:
        GraphMetrics with full evaluation.
    """
    if not result.get('success', False):
        logger.warning("CausalIF result indicates failure — returning zero metrics")
        return GraphMetrics(
            precision=0.0, recall=0.0, f1=0.0, shd=len(true_edges),
            true_positives=0, false_positives=0, false_negatives=len(true_edges),
            reversed_edges=0, total_discovered=0, total_true=len(true_edges),
        )

    # Extract edges from CausalIF result format
    causal_graph_data = result.get('causal_graph', {})
    edges_raw = causal_graph_data.get('edges', [])

    # CausalIF stores edges as (u, v, edge_data_dict) or (u, v)
    discovered_edges = []
    for edge in edges_raw:
        if len(edge) >= 2:
            discovered_edges.append((edge[0], edge[1]))

    return evaluate_graph(discovered_edges, true_edges, check_orientation)


# ---------------------------------------------------------------------------
# Standard benchmark networks
# ---------------------------------------------------------------------------

# ASIA network (Lauritzen & Spiegelhalter, 1988)
# 8 nodes, 8 edges — small but well-known benchmark
ASIA_EDGES = [
    ('asia', 'tub'),
    ('smoking', 'lung'),
    ('smoking', 'bronc'),
    ('tub', 'either'),
    ('lung', 'either'),
    ('either', 'xray'),
    ('either', 'dysp'),
    ('bronc', 'dysp'),
]

ASIA_NODES = ['asia', 'tub', 'smoking', 'lung', 'bronc', 'either', 'xray', 'dysp']

ASIA_DESCRIPTIONS = """# ASIA Network Factor Definitions
- asia: recent visit to Asia (yes/no)
- tub: has tuberculosis (yes/no)
- smoking: is a smoker (yes/no)
- lung: has lung cancer (yes/no)
- bronc: has bronchitis (yes/no)
- either: has either tuberculosis or lung cancer (yes/no)
- xray: positive X-ray result (yes/no)
- dysp: has dyspnoea/shortness of breath (yes/no)
"""

# Sachs protein signaling network (Sachs et al., 2005)
# 11 nodes, 17 edges — biological signaling network
SACHS_EDGES = [
    ('Raf', 'Mek'),
    ('Mek', 'Erk'),
    ('PLCg', 'PIP2'),
    ('PLCg', 'PIP3'),
    ('PIP3', 'PIP2'),
    ('PKC', 'Mek'),
    ('PKC', 'Raf'),
    ('PKC', 'PKA'),
    ('PKC', 'Jnk'),
    ('PKC', 'P38'),
    ('PKA', 'Raf'),
    ('PKA', 'Mek'),
    ('PKA', 'Erk'),
    ('PKA', 'Akt'),
    ('PKA', 'Jnk'),
    ('PKA', 'P38'),
    ('Erk', 'Akt'),
]

SACHS_NODES = ['Raf', 'Mek', 'Erk', 'PLCg', 'PIP2', 'PIP3', 'PKC', 'PKA', 'Jnk', 'P38', 'Akt']

SACHS_DESCRIPTIONS = """# Sachs Protein Signaling Network Factor Definitions
- Raf: Raf kinase - serine/threonine protein kinase in MAPK/ERK pathway
- Mek: MEK kinase - MAP kinase kinase, downstream of Raf
- Erk: ERK kinase - extracellular signal-regulated kinase, downstream of MEK
- PLCg: Phospholipase C gamma - enzyme that cleaves PIP2
- PIP2: Phosphatidylinositol 4,5-bisphosphate - membrane phospholipid
- PIP3: Phosphatidylinositol 3,4,5-trisphosphate - signaling lipid
- PKC: Protein Kinase C - calcium-activated kinase
- PKA: Protein Kinase A - cAMP-dependent kinase
- Jnk: JNK kinase - c-Jun N-terminal kinase, stress response
- P38: p38 MAPK - stress-activated protein kinase
- Akt: Akt/PKB - serine/threonine kinase in PI3K pathway
"""

# ALARM network (Beinlich et al., 1989) - subset of 10 most connected nodes
# Full ALARM has 37 nodes, 46 edges — we use a manageable subset
ALARM_SUBSET_EDGES = [
    ('HYPOVOLEMIA', 'LVEDVOLUME'),
    ('LVEDVOLUME', 'STROKEVOLUME'),
    ('LVEDVOLUME', 'CVP'),
    ('STROKEVOLUME', 'CO'),
    ('CO', 'BP'),
    ('INSUFFANESTH', 'CATECHOL'),
    ('CATECHOL', 'HR'),
    ('CATECHOL', 'TPR'),
    ('TPR', 'BP'),
    ('HR', 'CO'),
    ('INTUBATION', 'VENTLUNG'),
    ('VENTLUNG', 'MINVOL'),
]

ALARM_SUBSET_NODES = [
    'HYPOVOLEMIA', 'LVEDVOLUME', 'STROKEVOLUME', 'CVP', 'CO',
    'BP', 'INSUFFANESTH', 'CATECHOL', 'HR', 'TPR',
    'INTUBATION', 'VENTLUNG', 'MINVOL',
]

ALARM_DESCRIPTIONS = """# ALARM Network Factor Definitions (Subset)
- HYPOVOLEMIA: low blood volume condition
- LVEDVOLUME: left ventricular end-diastolic volume
- STROKEVOLUME: volume of blood pumped per heartbeat
- CVP: central venous pressure
- CO: cardiac output (blood pumped per minute)
- BP: blood pressure
- INSUFFANESTH: insufficient anesthesia
- CATECHOL: catecholamine level (stress hormones)
- HR: heart rate
- TPR: total peripheral resistance
- INTUBATION: patient intubation status
- VENTLUNG: ventilation to lungs
- MINVOL: minute ventilation volume
"""

# Simple synthetic benchmarks with domain-meaningful variable names
SIMPLE_CHAIN_EDGES = [
    ('infection', 'fever'),
    ('fever', 'hospital_visit'),
    ('hospital_visit', 'recovery_time'),
]

SIMPLE_FORK_EDGES = [
    ('obesity', 'diabetes'),
    ('obesity', 'hypertension'),
    ('obesity', 'joint_pain'),
]

SIMPLE_COLLIDER_EDGES = [
    ('genetic_risk', 'heart_disease'),
    ('smoking', 'heart_disease'),
    ('heart_disease', 'mortality'),
]

SIMPLE_DIAMOND_EDGES = [
    ('education', 'income'),
    ('education', 'job_satisfaction'),
    ('income', 'quality_of_life'),
    ('job_satisfaction', 'quality_of_life'),
]


BENCHMARK_REGISTRY: Dict[str, Dict] = {
    'asia': {
        'name': 'ASIA (Lauritzen & Spiegelhalter 1988)',
        'edges': ASIA_EDGES,
        'nodes': ASIA_NODES,
        'descriptions': ASIA_DESCRIPTIONS,
        'domains': ['medicine', 'respiratory_disease', 'diagnostics'],
        'n_nodes': 8,
        'n_edges': 8,
        'description': 'Lung disease diagnosis network. 8 binary variables.',
    },
    'sachs': {
        'name': 'Sachs Protein Signaling (Sachs et al. 2005)',
        'edges': SACHS_EDGES,
        'nodes': SACHS_NODES,
        'descriptions': SACHS_DESCRIPTIONS,
        'domains': ['biology', 'cell_signaling', 'proteomics'],
        'n_nodes': 11,
        'n_edges': 17,
        'description': 'Protein signaling network from flow cytometry. 11 continuous variables.',
    },
    'alarm_subset': {
        'name': 'ALARM Subset (Beinlich et al. 1989)',
        'edges': ALARM_SUBSET_EDGES,
        'nodes': ALARM_SUBSET_NODES,
        'descriptions': ALARM_DESCRIPTIONS,
        'domains': ['medicine', 'anesthesia', 'patient_monitoring'],
        'n_nodes': 13,
        'n_edges': 12,
        'description': 'Medical monitoring network (13-node subset of 37-node ALARM).',
    },
    'chain': {
        'name': 'Causal Chain (infection→fever→hospital→recovery)',
        'edges': SIMPLE_CHAIN_EDGES,
        'nodes': ['infection', 'fever', 'hospital_visit', 'recovery_time'],
        'descriptions': """# Causal Chain Factor Definitions
- infection: severity of bacterial or viral infection (mild to severe)
- fever: body temperature elevation in degrees above normal
- hospital_visit: whether the patient visits hospital for treatment
- recovery_time: number of days until full recovery
""",
        'domains': ['medicine', 'epidemiology', 'patient_outcomes'],
        'n_nodes': 4,
        'n_edges': 3,
        'description': 'Linear causal chain: infection causes fever, fever leads to hospital visit, hospital visit affects recovery time.',
    },
    'fork': {
        'name': 'Common Cause (obesity→diabetes, hypertension, joint_pain)',
        'edges': SIMPLE_FORK_EDGES,
        'nodes': ['obesity', 'diabetes', 'hypertension', 'joint_pain'],
        'descriptions': """# Common Cause Factor Definitions
- obesity: body mass index (BMI) indicating overweight/obese status
- diabetes: blood glucose level indicating type 2 diabetes risk
- hypertension: systolic blood pressure indicating high blood pressure
- joint_pain: severity of weight-bearing joint pain and inflammation
""",
        'domains': ['medicine', 'metabolic_disease', 'public_health'],
        'n_nodes': 4,
        'n_edges': 3,
        'description': 'Common cause structure: obesity causes diabetes, hypertension, and joint pain independently.',
    },
    'collider': {
        'name': 'Collider (genetic_risk→heart_disease←smoking, heart_disease→mortality)',
        'edges': SIMPLE_COLLIDER_EDGES,
        'nodes': ['genetic_risk', 'smoking', 'heart_disease', 'mortality'],
        'descriptions': """# Collider Factor Definitions
- genetic_risk: inherited genetic predisposition to cardiovascular disease
- smoking: tobacco smoking frequency (packs per day)
- heart_disease: presence and severity of coronary heart disease
- mortality: risk of death from cardiovascular causes
""",
        'domains': ['medicine', 'cardiology', 'epidemiology'],
        'n_nodes': 4,
        'n_edges': 3,
        'description': 'Collider structure: genetic risk and smoking independently cause heart disease, which causes mortality.',
    },
    'diamond': {
        'name': 'Diamond (education→income→QoL, education→satisfaction→QoL)',
        'edges': SIMPLE_DIAMOND_EDGES,
        'nodes': ['education', 'income', 'job_satisfaction', 'quality_of_life'],
        'descriptions': """# Diamond Factor Definitions
- education: years of formal education completed
- income: annual household income in dollars
- job_satisfaction: self-reported job satisfaction score
- quality_of_life: overall quality of life index combining health, wealth, and happiness
""",
        'domains': ['economics', 'sociology', 'well_being'],
        'n_nodes': 4,
        'n_edges': 4,
        'description': 'Diamond structure: education affects quality of life through two paths — income and job satisfaction.',
    },
}


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_data_from_dag(
    edges: List[Tuple[str, str]],
    nodes: List[str],
    n_samples: int = 1000,
    noise_std: float = 0.5,
    edge_weights: Optional[Dict[Tuple[str, str], float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate observational data from a known DAG using linear Gaussian model.

    Each node X_i = Σ_{parent j} w_ji * X_j + ε_i, where ε_i ~ N(0, noise_std²).
    Edge weights default to uniform random in [0.5, 1.5] with random sign.

    Args:
        edges: Ground-truth edges as (cause, effect) tuples.
        nodes: All node names.
        n_samples: Number of data samples to generate.
        noise_std: Standard deviation of noise term.
        edge_weights: Optional dict mapping edges to specific weights.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with n_samples rows and len(nodes) columns.
    """
    rng = np.random.default_rng(seed)

    # Build adjacency for topological ordering
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Ground-truth graph must be a DAG (no cycles)")

    topo_order = list(nx.topological_sort(G))

    # Assign edge weights
    weights = {}
    if edge_weights:
        weights = edge_weights.copy()
    for edge in edges:
        if edge not in weights:
            # Random weight between 0.5 and 1.5, with random sign
            w = rng.uniform(0.5, 1.5)
            if rng.random() < 0.4:
                w = -w
            weights[edge] = w

    # Generate data in topological order
    data = {}
    for node in topo_order:
        parents = list(G.predecessors(node))
        noise = rng.normal(0, noise_std, n_samples)

        if not parents:
            # Root node: pure noise centered at random mean
            data[node] = rng.normal(rng.uniform(-2, 2), 1.0, n_samples) + noise
        else:
            # Sum of weighted parent values + noise
            value = noise.copy()
            for parent in parents:
                w = weights.get((parent, node), 1.0)
                value += w * data[parent]
            data[node] = value

    return pd.DataFrame(data, columns=nodes)


def generate_binary_data_from_dag(
    edges: List[Tuple[str, str]],
    nodes: List[str],
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate binary observational data using a noisy-OR model.

    Suitable for networks like ASIA that have binary variables.
    P(X=1 | parents) = 1 - Π_{active parents} (1 - strength_i)

    Args:
        edges: Ground-truth edges.
        nodes: All node names.
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        DataFrame with binary (0/1) values.
    """
    rng = np.random.default_rng(seed)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Ground-truth graph must be a DAG (no cycles)")

    topo_order = list(nx.topological_sort(G))

    # Assign causal strengths to edges
    strengths = {}
    for edge in edges:
        strengths[edge] = rng.uniform(0.4, 0.8)

    data = {}
    for node in topo_order:
        parents = list(G.predecessors(node))

        if not parents:
            # Root node: base probability
            base_prob = rng.uniform(0.2, 0.5)
            data[node] = rng.binomial(1, base_prob, n_samples)
        else:
            # Noisy-OR: P(X=1) = 1 - prod(1 - s_i * parent_i) for active parents
            prob = np.zeros(n_samples)
            leak = rng.uniform(0.05, 0.15)  # leak probability (spontaneous activation)

            for i in range(n_samples):
                prod_term = 1.0 - leak
                for parent in parents:
                    if data[parent][i] == 1:
                        s = strengths[(parent, node)]
                        prod_term *= (1.0 - s)
                prob[i] = 1.0 - prod_term

            data[node] = rng.binomial(1, prob)

    return pd.DataFrame(data, columns=nodes)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for a single benchmark run result."""
    benchmark_name: str
    metrics: GraphMetrics
    skeleton_metrics: Optional[GraphMetrics] = None
    n_samples: int = 0
    elapsed_seconds: float = 0.0
    config: Dict = field(default_factory=dict)
    causalif_result: Optional[Dict] = None

    def __str__(self) -> str:
        lines = [
            f"\n{'=' * 70}",
            f"BENCHMARK: {self.benchmark_name}",
            f"{'=' * 70}",
            f"  Samples: {self.n_samples} | Time: {self.elapsed_seconds:.1f}s",
            f"",
            f"  DIRECTED GRAPH (orientation matters):",
            f"    Precision: {self.metrics.precision:.4f}",
            f"    Recall:    {self.metrics.recall:.4f}",
            f"    F1:        {self.metrics.f1:.4f}",
            f"    SHD:       {self.metrics.shd}",
            f"    Reversed:  {self.metrics.reversed_edges}",
        ]
        if self.skeleton_metrics:
            lines += [
                f"",
                f"  SKELETON (undirected, edge existence only):",
                f"    Precision: {self.skeleton_metrics.precision:.4f}",
                f"    Recall:    {self.skeleton_metrics.recall:.4f}",
                f"    F1:        {self.skeleton_metrics.f1:.4f}",
            ]
        lines.append(f"{'=' * 70}")
        return "\n".join(lines)


def run_benchmark(
    benchmark_name: str,
    model,
    n_samples: int = 1000,
    retriever_tool=None,
    retriever=None,
    enable_causal_estimate: bool = False,
    bootstrap_iterations: int = 50,
    bootstrap_threshold: float = 0.7,
    max_parallel_queries: int = 50,
    seed: int = 42,
    binary_data: bool = None,
    target_factor: Optional[str] = None,
) -> BenchmarkResult:
    """Run CausalIF on a standard benchmark network and compute accuracy.

    Args:
        benchmark_name: Key from BENCHMARK_REGISTRY (e.g., 'asia', 'sachs').
        model: LangChain-compatible LLM model.
        n_samples: Number of synthetic data samples to generate.
        retriever_tool: Optional RAG retriever tool.
        retriever: Optional raw retriever.
        enable_causal_estimate: If True, fit CPDs and enable do-operator.
        bootstrap_iterations: Number of bootstrap resamples.
        bootstrap_threshold: Stability threshold for edge pruning.
        max_parallel_queries: Max parallel LLM queries.
        seed: Random seed for data generation.
        binary_data: If True, generate binary data. If None, auto-detect from benchmark.
        target_factor: Target factor for the query. If None, picks a leaf node.

    Returns:
        BenchmarkResult with metrics and details.
    """
    from .tool import set_causalif_engine, causalif

    if benchmark_name not in BENCHMARK_REGISTRY:
        available = ', '.join(sorted(BENCHMARK_REGISTRY.keys()))
        raise ValueError(f"Unknown benchmark: '{benchmark_name}'. Available: {available}")

    bench = BENCHMARK_REGISTRY[benchmark_name]
    true_edges = bench['edges']
    nodes = bench['nodes']
    descriptions = bench['descriptions']
    domains = bench['domains']

    logger.info(f"\n{'#' * 70}")
    logger.info(f"# BENCHMARK: {bench['name']}")
    logger.info(f"# Nodes: {bench['n_nodes']}, True edges: {bench['n_edges']}")
    logger.info(f"# Samples: {n_samples}, Seed: {seed}")
    logger.info(f"{'#' * 70}\n")

    # Generate synthetic data
    if binary_data is None:
        # Auto-detect: ASIA uses binary, others use continuous
        binary_data = benchmark_name == 'asia'

    if binary_data:
        df = generate_binary_data_from_dag(true_edges, nodes, n_samples=n_samples, seed=seed)
    else:
        df = generate_data_from_dag(true_edges, nodes, n_samples=n_samples, seed=seed)

    logger.info(f"Generated {len(df)} samples for {len(nodes)} variables")
    logger.info(f"Data types: {'binary' if binary_data else 'continuous'}")

    # Select target factor (pick a node with most incoming edges if not specified)
    if target_factor is None:
        G_true = nx.DiGraph()
        G_true.add_edges_from(true_edges)
        # Pick node with highest in-degree as target
        in_degrees = dict(G_true.in_degree())
        target_factor = max(in_degrees, key=in_degrees.get)
    logger.info(f"Target factor: {target_factor}")

    # Configure and run CausalIF
    set_causalif_engine(
        model=model,
        retriever_tool=retriever_tool,
        retriever=retriever,
        dataframe=df,
        max_parallel_queries=max_parallel_queries,
        enable_causal_estimate=enable_causal_estimate,
        domains=domains,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_threshold=bootstrap_threshold,
        factor_descriptions=descriptions,
    )

    start_time = time.time()
    result = causalif(f"what causes {target_factor}")
    elapsed = time.time() - start_time

    # Compute directed metrics
    metrics = evaluate_causalif_result(result, true_edges, check_orientation=True)

    # Compute skeleton (undirected) metrics
    skeleton_metrics = evaluate_causalif_result(result, true_edges, check_orientation=False)

    logger.info(f"\n{metrics}")
    logger.info(f"\nSkeleton (undirected) F1: {skeleton_metrics.f1:.4f}")

    return BenchmarkResult(
        benchmark_name=bench['name'],
        metrics=metrics,
        skeleton_metrics=skeleton_metrics,
        n_samples=n_samples,
        elapsed_seconds=elapsed,
        config={
            'bootstrap_iterations': bootstrap_iterations,
            'bootstrap_threshold': bootstrap_threshold,
            'binary_data': binary_data,
            'seed': seed,
            'target_factor': target_factor,
        },
        causalif_result=result,
    )


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Run multiple benchmarks and produce a summary report.

    Usage:
        suite = BenchmarkSuite(model=your_llm)
        results = suite.run_all(n_samples=1000)
        print(suite.summary_table(results))
        suite.to_dataframe(results).to_csv('benchmark_results.csv')
    """

    def __init__(self, model, retriever_tool=None, retriever=None, **default_kwargs):
        """Initialize with shared config.

        Args:
            model: LangChain-compatible LLM.
            retriever_tool: Optional RAG retriever tool.
            retriever: Optional raw retriever.
            **default_kwargs: Default kwargs passed to run_benchmark.
        """
        self.model = model
        self.retriever_tool = retriever_tool
        self.retriever = retriever
        self.default_kwargs = default_kwargs

    def run_all(
        self,
        benchmarks: Optional[List[str]] = None,
        n_samples: int = 1000,
        **kwargs,
    ) -> List[BenchmarkResult]:
        """Run all (or specified) benchmarks sequentially.

        Args:
            benchmarks: List of benchmark names. None = run all.
            n_samples: Samples per benchmark.
            **kwargs: Override default_kwargs.

        Returns:
            List of BenchmarkResult objects.
        """
        if benchmarks is None:
            benchmarks = list(BENCHMARK_REGISTRY.keys())

        merged_kwargs = {**self.default_kwargs, **kwargs}
        results = []

        for name in benchmarks:
            logger.info(f"\n{'━' * 70}")
            logger.info(f"Running benchmark: {name}")
            logger.info(f"{'━' * 70}")
            try:
                result = run_benchmark(
                    benchmark_name=name,
                    model=self.model,
                    n_samples=n_samples,
                    retriever_tool=self.retriever_tool,
                    retriever=self.retriever,
                    **merged_kwargs,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark '{name}' failed: {e}")
                results.append(BenchmarkResult(
                    benchmark_name=name,
                    metrics=GraphMetrics(
                        precision=0, recall=0, f1=0, shd=-1,
                        true_positives=0, false_positives=0, false_negatives=0,
                        reversed_edges=0, total_discovered=0, total_true=0,
                    ),
                    n_samples=n_samples,
                    elapsed_seconds=0,
                    config={'error': str(e)},
                ))

        return results

    def summary_table(self, results: List[BenchmarkResult]) -> str:
        """Produce a formatted summary table of all benchmark results.

        Args:
            results: List of BenchmarkResult from run_all().

        Returns:
            Formatted string table.
        """
        lines = [
            "",
            "=" * 90,
            "CAUSALIF BENCHMARK RESULTS SUMMARY",
            "=" * 90,
            f"{'Benchmark':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'SHD':>5} {'Rev':>4} {'Skel-F1':>8} {'Time':>7}",
            "-" * 90,
        ]

        f1_scores = []
        for r in results:
            skel_f1 = f"{r.skeleton_metrics.f1:.3f}" if r.skeleton_metrics else "N/A"
            lines.append(
                f"{r.benchmark_name:<30} "
                f"{r.metrics.f1:>6.3f} "
                f"{r.metrics.precision:>6.3f} "
                f"{r.metrics.recall:>6.3f} "
                f"{r.metrics.shd:>5d} "
                f"{r.metrics.reversed_edges:>4d} "
                f"{skel_f1:>8} "
                f"{r.elapsed_seconds:>6.1f}s"
            )
            f1_scores.append(r.metrics.f1)

        lines.append("-" * 90)
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        lines.append(f"{'AVERAGE':<30} {avg_f1:>6.3f}")
        lines.append("=" * 90)

        table = "\n".join(lines)
        logger.info(table)
        return table

    def to_dataframe(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis and export.

        Args:
            results: List of BenchmarkResult.

        Returns:
            DataFrame with one row per benchmark.
        """
        rows = []
        for r in results:
            row = {
                'benchmark': r.benchmark_name,
                'n_samples': r.n_samples,
                'elapsed_seconds': round(r.elapsed_seconds, 1),
                **r.metrics.to_dict(),
            }
            if r.skeleton_metrics:
                row['skeleton_f1'] = round(r.skeleton_metrics.f1, 4)
                row['skeleton_precision'] = round(r.skeleton_metrics.precision, 4)
                row['skeleton_recall'] = round(r.skeleton_metrics.recall, 4)
            row.update({f'config_{k}': v for k, v in r.config.items()})
            rows.append(row)

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: compare multiple runs (e.g., different sample sizes)
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    benchmark_name: str,
    model,
    sample_sizes: List[int] = None,
    n_repeats: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Run a benchmark at multiple sample sizes to measure sensitivity.

    Args:
        benchmark_name: Which benchmark to run.
        model: LLM model.
        sample_sizes: List of sample sizes to test. Default: [200, 500, 1000, 2000, 5000].
        n_repeats: Number of repeats per sample size (different seeds).
        **kwargs: Additional kwargs for run_benchmark.

    Returns:
        DataFrame with columns: n_samples, seed, f1, precision, recall, shd, elapsed_seconds.
    """
    if sample_sizes is None:
        sample_sizes = [200, 500, 1000, 2000, 5000]

    rows = []
    for n in sample_sizes:
        for rep in range(n_repeats):
            seed = 42 + rep
            logger.info(f"\n--- Sensitivity: {benchmark_name}, n={n}, seed={seed} ---")
            try:
                result = run_benchmark(
                    benchmark_name=benchmark_name,
                    model=model,
                    n_samples=n,
                    seed=seed,
                    **kwargs,
                )
                rows.append({
                    'n_samples': n,
                    'seed': seed,
                    'f1': result.metrics.f1,
                    'precision': result.metrics.precision,
                    'recall': result.metrics.recall,
                    'shd': result.metrics.shd,
                    'reversed_edges': result.metrics.reversed_edges,
                    'skeleton_f1': result.skeleton_metrics.f1 if result.skeleton_metrics else None,
                    'elapsed_seconds': result.elapsed_seconds,
                })
            except Exception as e:
                logger.error(f"Failed: n={n}, seed={seed}: {e}")
                rows.append({
                    'n_samples': n, 'seed': seed,
                    'f1': 0, 'precision': 0, 'recall': 0,
                    'shd': -1, 'reversed_edges': 0, 'skeleton_f1': 0,
                    'elapsed_seconds': 0, 'error': str(e),
                })

    df = pd.DataFrame(rows)
    logger.info(f"\n--- Sensitivity Analysis Summary ---")
    logger.info(df.groupby('n_samples')[['f1', 'precision', 'recall', 'shd']].mean().to_string())
    return df


# ---------------------------------------------------------------------------
# Baseline comparison: run PC and GES on the same data
# ---------------------------------------------------------------------------

def run_baselines(
    benchmark_name: str,
    n_samples: int = 1000,
    seed: int = 42,
    binary_data: bool = None,
) -> pd.DataFrame:
    """Run PC and Hill Climb (GES-equivalent) baselines on a benchmark dataset.

    Uses pgmpy's built-in constraint-based (PC) and score-based (HillClimbSearch
    with BDeu) algorithms on the exact same synthetic data that CausalIF would use.
    This gives a fair apples-to-apples comparison.

    Args:
        benchmark_name: Key from BENCHMARK_REGISTRY.
        n_samples: Number of synthetic data samples.
        seed: Random seed for data generation.
        binary_data: If True, generate binary data. If None, auto-detect.

    Returns:
        DataFrame with one row per algorithm (PC, HillClimb-BDeu) with metrics.
    """
    from pgmpy.causal_discovery import HillClimbSearch, PC
    from pgmpy.structure_score import BDeu
    from sklearn.preprocessing import KBinsDiscretizer

    if benchmark_name not in BENCHMARK_REGISTRY:
        available = ', '.join(sorted(BENCHMARK_REGISTRY.keys()))
        raise ValueError(f"Unknown benchmark: '{benchmark_name}'. Available: {available}")

    bench = BENCHMARK_REGISTRY[benchmark_name]
    true_edges = bench['edges']
    nodes = bench['nodes']

    # Generate data
    if binary_data is None:
        binary_data = benchmark_name == 'asia'

    if binary_data:
        df = generate_binary_data_from_dag(true_edges, nodes, n_samples=n_samples, seed=seed)
    else:
        df = generate_data_from_dag(true_edges, nodes, n_samples=n_samples, seed=seed)

    # Discretize continuous data for BDeu scoring (same as CausalIF does)
    df_discrete = df.copy()
    for col in df_discrete.columns:
        if df_discrete[col].nunique() > 5:
            n_bins = min(5, max(2, int(np.log2(len(df_discrete)) + 1)))
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            df_discrete[col] = kbd.fit_transform(df_discrete[[col]]).astype(int).astype(str)
        else:
            df_discrete[col] = df_discrete[col].astype(str)

    rows = []

    # --- Baseline 1: Hill Climb with BDeu (score-based, no prior) ---
    logger.info(f"\nRunning Hill Climb (BDeu) baseline on {benchmark_name}...")
    try:
        start = time.time()
        scoring = BDeu(df_discrete, equivalent_sample_size=10)
        hc = HillClimbSearch(
            scoring_method=scoring,
            max_indegree=4,
            max_iter=200,
            epsilon=1e-4,
            show_progress=False,
        )
        hc.fit(df_discrete)
        hc_edges = list(hc.causal_graph_.edges())
        elapsed = time.time() - start

        hc_metrics = evaluate_graph(hc_edges, true_edges, check_orientation=True)
        hc_skeleton = evaluate_graph(hc_edges, true_edges, check_orientation=False)

        rows.append({
            'algorithm': 'HillClimb-BDeu',
            'benchmark': bench['name'],
            'n_samples': n_samples,
            'precision': hc_metrics.precision,
            'recall': hc_metrics.recall,
            'f1': hc_metrics.f1,
            'shd': hc_metrics.shd,
            'reversed_edges': hc_metrics.reversed_edges,
            'skeleton_f1': hc_skeleton.f1,
            'elapsed_seconds': round(elapsed, 1),
        })
        logger.info(f"  HillClimb-BDeu: F1={hc_metrics.f1:.4f}, SHD={hc_metrics.shd}")
    except Exception as e:
        logger.error(f"  HillClimb-BDeu failed: {e}")
        rows.append({
            'algorithm': 'HillClimb-BDeu',
            'benchmark': bench['name'],
            'n_samples': n_samples,
            'precision': 0, 'recall': 0, 'f1': 0,
            'shd': -1, 'reversed_edges': 0, 'skeleton_f1': 0,
            'elapsed_seconds': 0, 'error': str(e),
        })

    # --- Baseline 2: PC algorithm (constraint-based) ---
    logger.info(f"Running PC algorithm baseline on {benchmark_name}...")
    try:
        start = time.time()
        # pgmpy >= 1.1: significance_level is a constructor param, not fit() param
        # See: https://pgmpy.org/api/generated/structure_learning/pgmpy.causal_discovery.PC.html
        pc = PC(
            ci_test="chi_square",
            significance_level=0.05,
            return_type="dag",
            show_progress=False,
        )
        pc.fit(df_discrete)
        pc_edges = list(pc.causal_graph_.edges())
        elapsed = time.time() - start

        pc_metrics = evaluate_graph(pc_edges, true_edges, check_orientation=True)
        pc_skeleton = evaluate_graph(pc_edges, true_edges, check_orientation=False)

        rows.append({
            'algorithm': 'PC',
            'benchmark': bench['name'],
            'n_samples': n_samples,
            'precision': pc_metrics.precision,
            'recall': pc_metrics.recall,
            'f1': pc_metrics.f1,
            'shd': pc_metrics.shd,
            'reversed_edges': pc_metrics.reversed_edges,
            'skeleton_f1': pc_skeleton.f1,
            'elapsed_seconds': round(elapsed, 1),
        })
        logger.info(f"  PC: F1={pc_metrics.f1:.4f}, SHD={pc_metrics.shd}")
    except Exception as e:
        logger.error(f"  PC failed: {e}")
        rows.append({
            'algorithm': 'PC',
            'benchmark': bench['name'],
            'n_samples': n_samples,
            'precision': 0, 'recall': 0, 'f1': 0,
            'shd': -1, 'reversed_edges': 0, 'skeleton_f1': 0,
            'elapsed_seconds': 0, 'error': str(e),
        })

    return pd.DataFrame(rows)


def compare_all_with_baselines(
    benchmarks: Optional[List[str]] = None,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run PC and HillClimb-BDeu baselines on all benchmarks.

    Use this alongside your CausalIF results for a side-by-side comparison table.

    Args:
        benchmarks: List of benchmark names. None = run all.
        n_samples: Number of samples per benchmark.
        seed: Random seed.

    Returns:
        DataFrame with rows for each (algorithm, benchmark) combination.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARK_REGISTRY.keys())

    all_rows = []
    for name in benchmarks:
        logger.info(f"\n{'━' * 50}")
        logger.info(f"Baselines for: {name}")
        logger.info(f"{'━' * 50}")
        try:
            df = run_baselines(name, n_samples=n_samples, seed=seed)
            all_rows.append(df)
        except Exception as e:
            logger.error(f"Baselines for '{name}' failed: {e}")

    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
        logger.info(f"\n{'=' * 70}")
        logger.info("BASELINE COMPARISON SUMMARY")
        logger.info(f"{'=' * 70}")
        logger.info(result[['algorithm', 'benchmark', 'f1', 'shd', 'skeleton_f1']].to_string(index=False))
        return result
    return pd.DataFrame()
