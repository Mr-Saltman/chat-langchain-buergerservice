"""Load html from files, clean up, split, ingest into FAISS."""
import logging
from parser import langchain_docs_extractor

from langchain.document_loaders import AsyncHtmlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_transformers import BeautifulSoupTransformer

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_web():
    return AsyncHtmlLoader("https://www.schrems.at/-/antraege-formulare").load()


def ingest_docs():
    docs_from_webpage = load_web()

    docs_transformed = BeautifulSoupTransformer().transform_documents(
        docs_from_webpage,
        unwanted_tags=[
            "script",
            "style",
        ],
        tags_to_extract=[
            "a",
            "li",
            "p",
        ],
    )
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs_transformed)

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    embedding = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(docs, embedding)

    vectorstore.save_local("data/schrems_index")


if __name__ == "__main__":
    ingest_docs()
