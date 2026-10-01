import os
import json
import time
import logging
from pathlib import Path
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
BASE_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME")
RANKING_LOCATION = os.getenv("RANKING_LOCATION_ID")
STRATEGIES = ["token", "section"]
TOP_K = 3


def getconn():
    with Connector() as connector:
        return connector.connect(
            INSTANCE_CONNECTION_NAME, "pg8000",
            user=DB_USER, password=DB_PASS, db=DB_NAME,
            ip_type=IPTypes.PUBLIC,
        )


def with_retry(fn, tries=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < tries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise


def hit(chunks, keywords):
    if not keywords:
        return None
    text = " ".join(c.page_content for c in chunks).lower()
    return any(kw.lower() in text for kw in keywords)


def main():
    questions = json.loads(Path("eval_questions.json").read_text())
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    results = {s: {"hits": 0, "total": 0} for s in STRATEGIES}
    misses = {s: [] for s in STRATEGIES}

    for strategy in STRATEGIES:
        store = PGVector(
            collection_name=f"{BASE_COLLECTION_NAME}_{strategy}",
            embedding_function=embeddings,
            connection_string=pg_conn_str,
            engine_args={"creator": getconn},
        )
        base_retriever = store.as_retriever(search_kwargs={"k": 10})
        reranker = VertexAIRank(
            project_id=PROJECT_ID, location_id=RANKING_LOCATION,
            ranking_config="default_ranking_config", title_field="source", top_n=TOP_K,
        )
        rerank_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)

        for q in questions:
            if not q["answerable"]:
                continue
            chunks = with_retry(lambda: rerank_retriever.invoke(q["question"]))
            is_hit = hit(chunks, q["expected_keywords"])
            results[strategy]["total"] += 1
            results[strategy]["hits"] += int(is_hit)
            if not is_hit:
                misses[strategy].append(q["id"])
            time.sleep(1)

    print(f"\n{'Strategy':<12} {'Hit Rate (reranked)':<22} {'Hits/Total'}")
    print("-" * 48)
    for s in STRATEGIES:
        r = results[s]
        rate = r["hits"] / r["total"] if r["total"] else 0
        print(f"{s:<12} {rate:.0%}                   {r['hits']}/{r['total']}")
    print("\nMisses:", misses)


if __name__ == "__main__":
    main()