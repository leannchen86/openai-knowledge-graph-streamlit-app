from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain.graphs import Neo4jGraph
import streamlit as st
import os


# NEO4J_URI= st.secrets["NEO4J_URI"]
# NEO4J_USERNAME= st.secrets["NEO4J_USERNAME"]
# NEO4J_PASSWORD= st.secrets["NEO4J_PASSWORD"]

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
import os

# Declare global variables for the retrievers
# typical_rag = None
# parent_vectorstore = None
# hypothetic_question_vectorstore = None
# summary_vectorstore = None

def initialize_retrievers(openai_api_key):
    # global typical_rag, parent_vectorstore, hypothetic_question_vectorstore, summary_vectorstore

    os.environ["OPENAI_API_KEY"] = openai_api_key

    # NEO4J_URI= st.secrets["NEO4J_URI"]
    # NEO4J_USERNAME= st.secrets["NEO4J_USERNAME"]
    # NEO4J_PASSWORD= st.secrets["NEO4J_PASSWORD"]
   
    # graph = Neo4jGraph(
    #     url=os.environ["NEO4J_URI"],
    #     username=os.environ["NEO4J_USERNAME"],
    #     password=os.environ["NEO4J_PASSWORD"])

    # Initialize typical_rag
    typical_rag = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(), index_name="typical_rag")

    # Initialize parent_vectorstore
    parent_query = """
    MATCH (node)<-[:HAS_CHILD]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata LIMIT 1
    """
    parent_vectorstore = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        index_name="parent_document",
        retrieval_query=parent_query,
    )

    # Initialize hypothetic_question_vectorstore
    hypothetic_question_query = """
    MATCH (node)<-[:HAS_QUESTION]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """
    hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        index_name="hypothetical_questions",
        retrieval_query=hypothetic_question_query,
    )

    # Initialize summary_vectorstore
    summary_query = """
    MATCH (node)<-[:HAS_SUMMARY]-(parent)
    WITH parent, max(score) AS score // deduplicate parents
    RETURN parent.text AS text, score, {} AS metadata
    """
    summary_vectorstore = Neo4jVector.from_existing_index(
        OpenAIEmbeddings(),
        index_name="summary",
        retrieval_query=summary_query,
    )

    return typical_rag, parent_vectorstore, hypothetic_question_vectorstore, summary_vectorstore

