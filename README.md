# Causal Inference Framework for AWS (causalif)

[![PyPI version](https://badge.fury.io/py/causalif.svg)](https://pypi.org/project/causalif/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)


---

## Table of Contents

1. [Overview](#overview)
2. [Logical Flow](#logical-flow)
3. [Why Hill Climb and BDeu Score?](#why-hill-climb-and-bdeu-score)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Usage Examples](#usage-examples)
7. [Architecture](#architecture)
8. [Limitations](#limitations)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

CausalIF combines LLMs with Bayesian causal inference to discover causal relationships from both qualitative documents and quantitative data. It leverages:

- **Background Knowledge**: LLM's pre-trained causal understanding
- **Document Knowledge**: Domain documents via RAG retrieval
- **Bayesian Structure Learning**: Hill Climbing + BDeu scoring for causal orientation
- **Do-Calculus**: Interventional queries via pgmpy's do-operator (`causalif_intervene`)

Best used as a tool in agentic systems for interpreting causal relationships.

**GitHub**: [awslabs/causalif](https://github.com/awslabs/causalif) | **PyPI**: [causalif](https://pypi.org/project/causalif/)

The direct, indirect and independent association algorithm (causalif_1_edge_existence_verification) is inspired by LACR 1 algorithm: https://arxiv.org/html/2402.15301v2

Note: It is an experimental project which is dependent on quality RAG documents, model knowledge and data size for its analysis.
---

## Ideal Use Cases

CausalIF works best when you have both qualitative domain knowledge and quantitative observational data.

**What You Need**:
1. **Qualitative**: Documents with formulae, relationships, and domain expertise
2. **Quantitative**: Observational data (even if noisy)

**Example**: Financial institution analyzing derived metrics using research papers + historical market data.

**When to Use**:
✅ Rich document corpus + observational data  
✅ Understanding derived metrics  
✅ "What causes what" questions  

**When Not to Use**:
⚠️ No domain documents  
⚠️ Real-time requirements  
⚠️ <100 data samples  
⚠️ Purely experimental data (use RCTs)

---

## Logical Flow

CausalIF implements a 3-stage algorithm:

![Library Architecture](docs/causalif_flow_arch.png)

### Stage 1: Edge Existence (CausalIF 1)

**Goal**: Identify direct causal associations

**5 Phases**:
1. **Document Retrieval**: Get k_documents from RAG per edge
2. **Association Verification**: LLM votes (1 BG + k DOC votes per edge) → Associated/Independent/Unknown
3. **Type Classification**: Direct/Indirect/Unknown for associated edges
4. **Rechecker**: Validate intermediaries are in variable set V; reclassify if not
5. **Vote Scoring**: Direct: +1, Indirect/Independent: -1, Unknown: 0 → Keep if S > 0

**Output**: Skeleton graph with only direct associations

### Stage 2: Causal Orientation (CausalIF 2)

**Goal**: Determine causal direction (A → B or B ← A) and validate edge robustness

**Process**:
1. **Hill Climbing + BDeu**: Orient skeleton edges using `PriorWeightedBDeu` scoring on observational data
2. **Bootstrap Stability**: Resample data N times (default 50), re-run Hill Climb on each resample, compute per-edge directed stability (% of resamples where exact edge direction appeared)
3. **Pruning**: Remove edges with directed stability below threshold (default 70%)

**Output**: Directed Acyclic Graph (DAG) with bootstrap-validated edges

### Stage 3: Causal Inference (Optional)

**Goal**: Quantify causal effects and enable interventional queries

**Process**: Fit CPDs → Compute Average Treatment Effects → Direction analysis → Enable do-operator queries

**Enable with**: `enable_causal_estimate=True`

The do-operator computes `P(target | do(cause=value))` using pgmpy's backdoor adjustment. Direction analysis compares the intervention direction (cause pushed above/below its baseline) with the effect shift to determine if variables are directly or inversely related. The do-operator only works in the causal direction (ancestor → descendant); querying the reverse returns a helpful error with a suggestion.

---

## Why Hill Climb and BDeu Score?

### Hill Climbing
Local search algorithm that iteratively improves graph structure. Advantages: incorporates prior knowledge, computationally efficient (10-20 variables), interpretable steps.

### BDeu Score
Bayesian scoring function measuring how well a graph explains data. Advantages: combines priors with data, score equivalence, built-in regularization.

**CausalIF Enhancement**: `Score(G) = BDeu(G | Data) + λ × Prior(G | LLM)`, validated by bootstrap stability

Implements Bayesian inference: **P(G | Data, LLM) ∝ P(Data | G) × P(G | LLM)**

---

## Prerequisites

1. **AWS Bedrock Knowledge Base**: [Setup guide](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html)
2. **LLM Model**: Any LangChain-compatible LLM (Bedrock, OpenAI, etc.)
3. **Observational Data**: Pandas DataFrame with 100+ samples

### Quick Setup

```python
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws import ChatBedrockConverse

# Retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="your-kb-id",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 20}}
)

# LLM
model = ChatBedrockConverse(
    model_id="global.anthropic.claude-sonnet-4-6",
    temperature=0.0,
    region_name="us-west-2"
)
```

---

## Installation

```bash
pip install causalif
```

---

## Usage Examples

### Basic Usage

```python
from causalif import set_causalif_engine, causalif_tool, visualize_causalif_results
from langchain_aws import ChatBedrockConverse
import pandas as pd

# 1. Prepare your data
df = pd.DataFrame({
    'sleep_hours': [7, 6, 8, 5, 7, 9, 6, 8, 7, 5],
    'exercise_minutes': [30, 20, 45, 10, 35, 60, 25, 50, 40, 15],
    'stress_level': [5, 7, 3, 8, 4, 2, 6, 3, 5, 8],
    'productivity': [8, 6, 9, 4, 7, 10, 6, 9, 8, 5]
})

# 2. Initialize LLM
model=ChatBedrockConverse(model_id="<model_id>",temperature=0.0,region_name="<region_id>")

# 3. Configure Causalif engine
# Configure with financial data

set_causalif_engine(
            model=<your_bedrock_model>,
            retriever_tool=retriever_tool,
            dataframe=<dataframe_name>, 
            max_degrees=<degree of edges>,  # None = no filtering (show entire graph), or set to int (e.g., 2) to filter.
            max_parallel_queries=50, #This is variable but the code is tested with 50.
            excluded_target_columns=None, # This a list of factors that shouldn't be target columns
            excluded_related_columns=None, # This a list of factors that shouldn't be related columns
            related_factors=None,  # Add custom related factors here (will be appended with dataframe columns). Mostly derived columns from documents
            selected_dataframe_columns=None, # list of columns from your dataframe if you dont want the whole dataframe to be analyzed.
            enable_causal_estimate = True,  #Causal inference to find upstream or downstream direct effects of the target factor.
            domains = <list of industry domains>, # Consider this mandatory for the model to apply adequate background knowledge
            bootstrap_iterations=50, # Number of bootstrap resamples for edge stability validation (0 to disable)
            bootstrap_threshold=0.7, # Prune edges with directed stability below this threshold
        )

# 4. Run causal analysis
result = causalif.causalif("<query>") # example: Why is interest_rate so low in week 3?

# 5. Visualize results
fig = visualize_causalif_results(result)
fig.show()

```


### Query Formats

Causalif supports natural language queries in various formats. The `<target_factor>` is the column or factor whose dependencies with other variables you want to analyze:

```python
"""
Allowed query formats (where <target_factor> is the variable to analyze):

1. why (is|are) <target_factor> so (low|high|poor|bad|good)
2. what (causes|affects|influences) <target_factor>
3. <target_factor> (is|are) too (low|high)
4. analyze the causes (of|for) <target_factor>
5. dependencies (of|for) <target_factor>
6. factors (affecting|influencing) <target_factor>
"""

# Format 1: Why questions
result = causalif.causalif("Why is stress_level so high?")
result = causalif.causalif("Why are sales so low?")

# Format 2: What causes questions
result = causalif.causalif("What causes low productivity?")
result = causalif.causalif("What affects customer satisfaction?")

# Format 3: Direct statements
result = causalif.causalif("productivity is too low")
result = causalif.causalif("revenue is too high")

# Format 4: Analysis requests
result = causalif.causalif("analyze the causes of high stress_level")
result = causalif.causalif("analyze the causes for poor performance")

# Format 5: Dependency queries
result = causalif.causalif("dependencies of productivity")
result = causalif.causalif("dependencies for stock_price")

# Format 6: Factor influence queries
result = causalif.causalif("factors affecting sleep_hours")
result = causalif.causalif("factors influencing market_volatility")
```

### Interventional Queries (do-operator)

Once the causal model is fitted (`enable_causal_estimate=True` and a causal discovery query has been run), you can ask interventional questions using `causalif_intervene`:

```python
from causalif import causalif_intervene

"""
Allowed intervention formats (where X is cause, Y is effect):

1. what happens to Y if X is (high|low|medium)
2. what would Y be if X is (high|low|medium)
3. how does Y change if X is (high|low|medium)
4. effect of setting X to (high|low|medium) on Y
5. what happens to Y if X is (high|low|medium) and Z is (high|low|medium)
"""

# Format 1: What happens questions
result = causalif_intervene("what happens to asp if our_price is high")
print(result['summary'])

# Format 2: What would questions
result = causalif_intervene("what would productivity be if stress_level is low")

# Format 3: How does questions
result = causalif_intervene("how does revenue change if marketing_spend is high")

# Format 4: Effect of setting
result = causalif_intervene("effect of setting interest_rate to low on bond_price")

# Format 5: Multiple interventions
result = causalif_intervene("what happens to Y if X is low and Z is high")
```

Note: The do-operator only works in the causal direction. If `A → B` in the graph, you can query `do(A)` on `B`, but not `do(B)` on `A`.


### Visualization Features

The interactive visualization includes:

- **Node Colors**: Degree of separation from target factor (red = direct, blue = distant)
- **Edge Colors**: Same color scheme as nodes
- **Arrows**: Direction of causality
- **Hover Information**: Detailed relationship information
- **Interactive**: Zoom, pan, and click for details

```python
fig = visualize_causalif_results(result)
```

---

## Architecture

![Overall Architecture](docs/overall_design.png)

**Layers**: Agent → CausalIF Tool → Engine → Knowledge (RAG + LLM) → Data

**Components**:
```
causalif/
├── core.py           # Data structures
├── engine.py         # CausalIF algorithm
├── prompts.py        # LLM prompts
├── tool.py           # API & LangChain integration
└── visualization.py  # Plotly graphs
```

---

## Limitations

**Not ideal for**: Pure quantitative data or feedback-loop driven inference. Built for hybrid qualitative + quantitative analysis.

**Data**: Min 100 samples recommended, 10-20 variables max run at a time, Complexity is O(n² × k)

**LLM**: May hallucinate, reflects training biases, 2-5 calls per variable pair

**Assumptions**: DAG structure (no cycles), no unmeasured confounders, conditional independence

**Do-operator**: Only works in causal direction (ancestor → descendant), not reverse

**Mitigation**: Use `max_degrees` for filtering, `temperature=0` for consistency, validate with domain expertise

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Reporting Issues

Please report bugs and feature requests on [GitHub Issues](https://github.com/awslabs/causalif/issues).

---

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) for details.


## Version History

- **v0.1.9.7**: Improved numerical stability in discretization pipeline, refined prior contribution diagnostics, and adaptive graph visualization for larger causal structures.
- **v0.1.9.6**: Bootstrap stability validation in CausalIF 2 (resample + re-run Hill Climb, prune edges below 70% directed stability).
- **v0.1.9.5**: LACR 1 direct/indirect association algorithm, do-operator with direction analysis, interventional queries via `causalif_intervene`.
- **v0.1.9**: Removed LLM-based causal directions, introduced Bayesian-based causal direction with Hill Climb search and immediate upstream/downstream effects. Hybrid graph with associations and causal directions.
- **v0.1.6**: Removed directed graph dependencies, added example notebook.
- **v0.1.5**: README updates.
- **v0.1.4**: Base version with complete Causalif algorithm.

---

## Support

- **Documentation**: [GitHub README](https://github.com/awslabs/causalif/blob/main/README.md)
- **Issues**: [GitHub Issues](https://github.com/awslabs/causalif/issues)
- **Email**: bossubhr@amazon.co.uk

---

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - LLM orchestration
- [NetworkX](https://networkx.org/) - Graph algorithms
- [Plotly](https://plotly.com/) - Interactive visualization
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - LLM and RAG infrastructure
