# OpenAI Knowledge-Graph RAG Streamlit App

An advanced RAG chatbot that combines OpenAI models, a generated knowledge graph, and a Streamlit interface for exploring entity relationships.

The app uses data related to the OpenAI leadership saga as a concrete knowledge source, then lets users ask questions while inspecting the graph structure behind the retrieval context.

## What This Demonstrates

- A working Streamlit app around OpenAI-powered retrieval.
- Knowledge-graph construction from text with SpaCy-LLM-style extraction.
- Neo4j-style graph exploration alongside a conversational interface.
- Advanced RAG patterns adapted from LangChain's Neo4j template.

## Project Flow

1. Collect source text about OpenAI stakeholders.
2. Extract entities and relationships into a knowledge graph.
3. Load graph data into the app layer.
4. Retrieve graph-aware context for user questions.
5. Display both the chatbot answer and the relationship graph.

## Files

- `App.py` - Streamlit application entry point.
- `openaiKG.ipynb` - exploratory notebook for building the OpenAI knowledge graph.
- `ingest.py` - ingestion flow.
- `retrievers.py`, `chain.py` - advanced RAG retriever and chain logic.
- `functions.py`, `utils.py` - app helpers.
- `requirements.txt` - Python dependencies.

## Running Locally

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set any required API/database credentials in your shell, then start Streamlit:

```bash
streamlit run App.py
```

The exploratory notebook can be opened with:

```bash
jupyter notebook openaiKG.ipynb
```

## Attribution

`retrievers.py` and `chain.py` are adapted from LangChain's [`neo4j-advanced-rag`](https://github.com/langchain-ai/langchain/tree/master/templates/neo4j-advanced-rag?ref=blog.langchain.dev) template.

## Portfolio Note

This repo is a practical example of developer-facing AI education: it turns RAG, knowledge graphs, graph visualization, and OpenAI model usage into an inspectable app rather than a static explanation.
