import os
import json
import time
import logging
from pathlib import Path
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
STRATEGIES = ["character", "recursive", "token", "section"]
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
        return None  # unanswerable question, scored separately
    text = " ".join(c.page_content for c in chunks).lower()
    return any(kw.lower() in text for kw in keywords)


def main():
    questions = json.loads(open("eval_questions.json").read())
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"

    results = {s: {"hits": 0, "total": 0} for s in STRATEGIES}
    rows = []

    for strategy in STRATEGIES:
        store = PGVector(
            collection_name=f"{BASE_COLLECTION_NAME}_{strategy}",
            embedding_function=embeddings,
            connection_string=pg_conn_str,
            engine_args={"creator": getconn},
        )
        for q in questions:
            if not q["answerable"]:
                continue
            chunks = with_retry(lambda: store.similarity_search(q["question"], k=TOP_K))
            is_hit = hit(chunks, q["expected_keywords"])
            results[strategy]["total"] += 1
            results[strategy]["hits"] += int(is_hit)
            rows.append({"strategy": strategy, "id": q["id"], "hit": is_hit})
            time.sleep(0.5)

    print(f"\n{'Strategy':<12} {'Hit Rate':<10} {'Hits/Total'}")
    print("-" * 36)
    for s in STRATEGIES:
        r = results[s]
        rate = r["hits"] / r["total"] if r["total"] else 0
        print(f"{s:<12} {rate:.0%}       {r['hits']}/{r['total']}")

    Path("eval_results.json").write_text(json.dumps(rows, indent=2))
    print("\nSaved per-question results to eval_results.json")


if __name__ == "__main__":
    main()