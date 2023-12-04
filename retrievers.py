from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
import os
# Typical RAG retriever
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["NEO4J_URI"] = st.secrets["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = st.secrets["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = st.secrets["NEO4J_PASSWORD"]

typical_rag = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(), index_name="typical_rag"
)

# Parent retriever

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

# Hypothetic questions retriever

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
# Summary retriever

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
