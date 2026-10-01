import os
import sys
import logging
import time
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_community.vertex_rank import VertexAIRank

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
RANKING_LOCATION = os.getenv("RANKING_LOCATION_ID")

STRATEGY = sys.argv[1] if len(sys.argv) > 1 else "section"
COLLECTION_NAME = f"{os.getenv('PGVECTOR_COLLECTION_NAME')}_{STRATEGY}"

QUERIES = [
    "What happens if my parcel is lost or damaged?",
    "Which items are prohibited from being shipped?",
    "How long do I have to make a claim?",
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


def preview(doc, n=170):
    return doc.page_content[:n].replace("\n", " ")


def with_retry(fn, tries=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < tries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  (rate limited, retrying in {wait}s...)")
                time.sleep(wait)
            else:
                raise


def main():
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    store = PGVector(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        connection_string=pg_conn_str,
        engine_args={"creator": getconn},
    )

    base_retriever = store.as_retriever(search_kwargs={"k": 10})
    reranker = VertexAIRank(
        project_id=PROJECT_ID,
        location_id=RANKING_LOCATION,
        ranking_config="default_ranking_config",
        title_field="source",
        top_n=3,
    )
    rerank_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

    print(f"Collection: {COLLECTION_NAME}")
    for query in QUERIES:
        time.sleep(2)
        print(f"\n{'=' * 70}\nQUERY: {query}\n{'=' * 70}")

        print("\nBEFORE reranking (vector search top 3 of 10):")
        for i, doc in enumerate(with_retry(lambda: base_retriever.invoke(query))[:3]):
            print(f"  {i + 1}. {preview(doc)}...")

        print("\nAFTER reranking (top 3 of 10):")
        try:
            for i, doc in enumerate(with_retry(lambda: rerank_retriever.invoke(query))):
                score = doc.metadata.get("relevance_score", "N/A")
                print(f"  {i + 1}. (score {score}) {preview(doc)}...")
        except Exception as e:
            print(f"  Reranking error: {e}")


if __name__ == "__main__":
    main()