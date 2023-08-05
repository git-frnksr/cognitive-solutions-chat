"""Create a ChatVectorDBChain for question/answering."""
import logging
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStore

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log_file.log"),
        logging.StreamHandler()
    ]
)

def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        max_tokens=2000
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
        max_tokens=2000
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )
    
    template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer. Try to add the page number of the document context. Always answer in German.

    QUESTION: {question}
    =========
    {summaries}
    =========
    {chat_history}
    =========
    FINAL ANSWER:"""
    
    SOURCES_QA_PROMPT = PromptTemplate(
        template=template, input_variables=["summaries", "question", "chat_history"]
    )

    doc_chain_sources = load_qa_with_sources_chain(
        llm=streaming_llm, chain_type="stuff", prompt=SOURCES_QA_PROMPT, callback_manager=manager
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain=doc_chain_sources,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True
    )
    return qa
