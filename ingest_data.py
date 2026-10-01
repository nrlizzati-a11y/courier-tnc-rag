import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.docstore.document import Document

load_dotenv()
logging.basicConfig(level=logging.INFO)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")

# ~1000 characters is roughly 250 tokens, so the strategies are comparable
CHAR_CHUNK, CHAR_OVERLAP = 1000, 100
TOKEN_CHUNK, TOKEN_OVERLAP = 250, 25


def getconn():
    with Connector() as connector:
        return connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            ip_type=IPTypes.PUBLIC,
        )


def get_splitter(strategy: str):
    if strategy == "character":
        return CharacterTextSplitter(chunk_size=CHAR_CHUNK, chunk_overlap=CHAR_OVERLAP, separator="\n")
    if strategy == "token":
        return TokenTextSplitter(chunk_size=TOKEN_CHUNK, chunk_overlap=TOKEN_OVERLAP)
    if strategy == "section":
        # Split before clause numbers like "5.4 " first, then fall back to lines/words
        return RecursiveCharacterTextSplitter(
            chunk_size=CHAR_CHUNK,
            chunk_overlap=0,
            separators=[r"\n(?=\d+\.\d+\s)", "\n", " ", ""],
            is_separator_regex=True,
        )
    return RecursiveCharacterTextSplitter(chunk_size=CHAR_CHUNK, chunk_overlap=CHAR_OVERLAP)

def prepare_chunks(docs: list, strategy: str) -> list:
    splitter = get_splitter(strategy)
    documents = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": f"{doc['carrier']} - {doc['title']}",
                    "carrier": doc["carrier"],
                    "url": doc["url"],
                    "retrieved": doc["retrieved"],
                    "chunk_index": i,
                    "strategy": strategy,
                },
            ))
    return documents


def main():
    docs = json.loads(Path("data/docs.json").read_text())
    logging.info(f"Loaded {len(docs)} documents")

    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    for strategy in ["character", "recursive", "token", "section"]:
        collection_name = f"{BASE_COLLECTION_NAME}_{strategy}"
        chunks = prepare_chunks(docs, strategy)
        logging.info(f"--- {strategy.upper()}: {len(chunks)} chunks -> {collection_name}")

        store = PGVector(
            collection_name=collection_name,
            embedding_function=embeddings,
            connection_string=pg_conn_str,
            engine_args={"creator": getconn},
            pre_delete_collection=True,
        )
        store.add_documents(chunks)
        logging.info(f"Finished {strategy}\n")


if __name__ == "__main__":
    main()