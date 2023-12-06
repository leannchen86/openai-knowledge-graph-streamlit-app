from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import ConfigurableField, RunnableParallel
import os

from retrievers import (
    hypothetic_question_vectorstore,
    parent_vectorstore,
    summary_vectorstore,
    typical_rag,
)

def initialize_chain(openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()

    retriever = typical_rag.as_retriever().configurable_alternatives(
        ConfigurableField(id="strategy"),
        default_key="typical_rag",
        parent_strategy=parent_vectorstore.as_retriever(),
        hypothetical_questions=hypothetic_question_vectorstore.as_retriever(),
        summary_strategy=summary_vectorstore.as_retriever(),
    )

    chain = (
        RunnableParallel(
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )


    # Add typing for input
    class Question(BaseModel):
        question: str


    chain = chain.with_types(input_type=Question)
    return chain
