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

Best used as a tool in agentic systems for interpreting causal relationships.

**GitHub**: [awslabs/causalif](https://github.com/awslabs/causalif) | **PyPI**: [causalif](https://pypi.org/project/causalif/)

The association algorithm (causalif 1) is Inspired by LACR 1 algorithm: https://arxiv.org/html/2402.15301v2

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
⚠️ <10 data samples  
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

**Goal**: Determine causal direction (A → B or B ← A)

**Process**: Hill Climbing + BDeu scoring constrained by skeleton graph

**Output**: Directed Acyclic Graph (DAG)

### Stage 3: Causal Inference (Optional)

**Goal**: Quantify causal effects

**Process**: Fit CPDs → Compute Average Treatment Effects → Enable interventional queries

**Enable with**: `enable_causal_estimate=True`

---

## Why Hill Climb and BDeu Score?

### Hill Climbing
Local search algorithm that iteratively improves graph structure. Advantages: incorporates prior knowledge, computationally efficient (10-20 variables), interpretable steps.

### BDeu Score
Bayesian scoring function measuring how well a graph explains data. Advantages: combines priors with data, score equivalence, built-in regularization.

**CausalIF Enhancement**: `Score(G) = BDeu(G | Data) + λ × Prior(G | LLM)`

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
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
            domains = <lsit of industry domains> # Consider this manadatory for the model to apply adequate background knowledge
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


### Visualization Features

The interactive visualization includes:

- **Node Colors**: Degree of separation from target factor (red = direct, blue = distant)
- **Edge Colors**: Same color scheme as nodes
- **Arrows**: Direction of causality
- **Hover Information**: Detailed relationship information
- **Interactive**: Zoom, pan, and click for details

```python
fig = visualize_causalif_results(result)

# Customize visualization
fig.update_layout(
    title="Custom Title",
    width=1200,
    height=800
)

# Save to file
fig.write_html("causal_graph.html")
fig.write_image("causal_graph.png")  # Requires kaleido
```

---

## Architecture

![Library Architecture](docs/overall_design.png)

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

**Data**: Min 100 samples recommended, 10-20 variables max without filtering, O(n² × k) complexity

**LLM**: May hallucinate, reflects training biases, 2-5 calls per variable pair

**Assumptions**: DAG structure (no cycles), no unmeasured confounders, conditional independence

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

- **v0.1.9.5**:Allowing LLM to implement indirect and direct associations following LACR 1 algorithm.
- **v0.1.9**: Remeved LLM based causal directions and introduced bayesian based causal direction with hill climb search and immediate upstream and downstream effects. Building a hybrid graph with associations and causal directions.
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
