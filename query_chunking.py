import os
import logging
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")

QUERIES = [
    "What happens if my parcel is lost or damaged?",
    "Which items are prohibited from being shipped?",
]


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


def main():
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    for query in QUERIES:
        print(f"\n{'=' * 70}\nQUERY: {query}\n{'=' * 70}")
        for strategy in ["character", "recursive", "token", "section"]:
            store = PGVector(
                collection_name=f"{BASE_COLLECTION_NAME}_{strategy}",
                embedding_function=embeddings,
                connection_string=pg_conn_str,
                engine_args={"creator": getconn},
            )
            print(f"\n[{strategy.upper()}]")
            for i, (doc, score) in enumerate(store.similarity_search_with_score(query, k=2)):
                text = doc.page_content[:300].replace("\n", " ")
                print(f"  {i + 1}. ({doc.metadata['carrier']}, distance {score:.4f}) {text}...")


if __name__ == "__main__":
    main()