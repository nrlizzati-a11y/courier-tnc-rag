import os
import sys
import time
import logging
from dotenv import load_dotenv
from google.cloud.sql.connector import Connector, IPTypes
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate

load_dotenv()
logging.basicConfig(level=logging.ERROR)

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
DB_USER = os.getenv("SQL_USER")
DB_PASS = os.getenv("SQL_PASSWORD")
DB_NAME = os.getenv("SQL_DATABASE_NAME")
INSTANCE_CONNECTION_NAME = f"{PROJECT_ID}:{REGION}:{os.getenv('SQL_INSTANCE_NAME')}"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_LOCATION = os.getenv("LLM_LOCATION", REGION)

STRATEGY = sys.argv[1] if len(sys.argv) > 1 else "section"
COLLECTION_NAME = f"{os.getenv('PGVECTOR_COLLECTION_NAME')}_{STRATEGY}"

# Written the way a customer would ask, not the way the terms are worded
QUERIES = [
    "My package never arrived, can I get my money back?",
    "Can I send a bottle of perfume?",
    "How many days do I have to complain?",
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


HYDE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Write a short passage (2-3 sentences) in the style of a courier company's "
        "terms and conditions that would answer the question below. "
        "Use formal contract wording.\n\nQuestion: {question}\n\nPassage:"
    ),
)

STEPBACK_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "Rewrite the question below as a more general question about the underlying "
        "topic in courier terms and conditions. Reply with the question only.\n\n"
        "Original: {question}\n\nGeneral question:"
    ),
)


def show(label, text, store):
    print(f"\n  [{label}]")
    results = with_retry(lambda: store.similarity_search(text, k=2))
    for i, doc in enumerate(results):
        print(f"    {i + 1}. {doc.page_content[:200].replace(chr(10), ' ')}...")


def main():
    embeddings = VertexAIEmbeddings(model_name="gemini-embedding-001", project=PROJECT_ID, location=REGION)
    llm = VertexAI(model_name=LLM_MODEL, project=PROJECT_ID, location=LLM_LOCATION, temperature=0.3)

    pg_conn_str = f"postgresql+pg8000://{DB_USER}:{DB_PASS}@placeholder/{DB_NAME}"
    store = PGVector(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        connection_string=pg_conn_str,
        engine_args={"creator": getconn},
    )

    print(f"Collection: {COLLECTION_NAME} | LLM: {LLM_MODEL}")
    for query in QUERIES:
        print(f"\n{'=' * 70}\nORIGINAL QUERY: {query}\n{'=' * 70}")

        hyde_doc = with_retry(lambda: (HYDE_PROMPT | llm).invoke({"question": query})).strip()
        step_back = with_retry(lambda: (STEPBACK_PROMPT | llm).invoke({"question": query})).strip()
        print(f"\n  HyDE passage: {hyde_doc[:300]}...")
        print(f"  Step-back question: {step_back}")

        show("Original query", query, store)
        show("HyDE", hyde_doc, store)
        show("Step-back", step_back, store)
        time.sleep(2)


if __name__ == "__main__":
    main()