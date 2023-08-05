"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
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

def ingest_docs():
    """Get documents from web pages."""
    
    logging.info("Trying to ingest documents")
    
    loader = PyPDFLoader("in/import.pdf")
    raw_documents = loader.load()
    
    logging.info("Splitting documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    
    logging.info("Creating embeddings...")

    vectorstore = FAISS.from_documents(documents, embeddings)

    logging.info("Saving embeddings to file...")

    # Save vectorstore
    with open("db/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    logging.info("Done.")


if __name__ == "__main__":
    ingest_docs()
