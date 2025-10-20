# Causal Inference Framework for AWS (causalif)

LLM assisted Causal Reasoning with JAX and RAG

# Architecture and usage in applications

![Library architecture:](docs/library_integrations.png)
![Overall design where causalif integrates with agentic applciations:](docs/overall_design.png)

# prerequisites

This library is based on RAG; it requires a knowledgebase to be ready before using this library. Below steps are required to be performed before using this library.

### Step-1:

You can set up a bedrock knowledgebase following the [instructions](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html).

### Step-2:

After setting up a knowledgebase, you can create a retriever tool like the following:

<pre>```python
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.tools.retriever import create_retriever_tool

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="<knowledge-base-id>",
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 20 #it could be any of your desired number
        }
    },
)

retriever_tool = create_retriever_tool(
    retriever,
    "<name of the retriever tool>",
    "<Description of the retriever tool>",
)```</pre>

# Installation

<pre>```bash
pip install causalif
```</pre>

# QuickStart

<pre>```python
from causalif import set_causalif_engine, causalif_tool, visualize_causalif_results
```</pre>

## Configure the engine

<pre>```python
set_causalif_engine(
model=your_model,       #Bedrock or any other provider. Please import the provider if it is other than Bedrock.
retriever_tool=retriever_tool,      # retriever_tool from prerequisites.
dataframe=your_dataframe,       # if you want all columns of your dataframe to be considered in causal analysis. Otherwise, leave it as 'None'.
factors = list of factors,      # list of your factors, e.g., ['water', 'food', 'exercise'].
domains = list of domains,      # list of your domains, e.g., ['life', 'health', 'well being'].
max_degrees=1,      # degrees of relationships that you wish to check.
max_parallel_queries=10         # it could be between 4 to 50 but it depends on the model throughput.
)
```</pre>

## Use the tool

<pre>```python
result = causalif_tool("Why is water so low in body after we wake up?")
```</pre>

<pre>```python
"""
Allowed formation of enqueries:

why (is|are) <target factor> so (low|high|poor|bad|good)",
what (causes|affects|influences) <target factor>",
<target factor> (is|are) too (low|high)",
analyze the causes (of|for) <target factor>",
dependencies (of|for) <target factor>,
factors (affecting|influencing) <target factor>"
"""
```</pre>

## Visualize results

<pre>```python
fig = visualize_causalif_results(result)
fig.show()
```</pre>

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
