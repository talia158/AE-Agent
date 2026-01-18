import argparse
import os
import sys
from typing import Optional

import chromadb
from fastapi import FastAPI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import uvicorn


_reranker: Optional[CrossEncoder] = None


def get_reranker(model_name: str) -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank_documents(query: str, docs, top_k: int, model_name: str):
    if not docs:
        return docs
    reranker = get_reranker(model_name)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


class CrossEncoderRerankRetriever(BaseRetriever):
    vectorstore: Chroma
    search_k: int = 20
    rerank_k: int = 8
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def _get_relevant_documents(self, query: str):
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_k})
        docs = base_retriever.invoke(query)
        return rerank_documents(query, docs, self.rerank_k, self.model_name)

    async def _aget_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)


def get_retriever(vectorstore: Chroma) -> CrossEncoderRerankRetriever:
    search_k = int(os.getenv("RETRIEVAL_K", "20"))
    rerank_k = int(os.getenv("RERANK_TOP_K", "8"))
    model_name = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return CrossEncoderRerankRetriever(
        vectorstore=vectorstore,
        search_k=search_k,
        rerank_k=rerank_k,
        model_name=model_name,
    )


def build_chain(persist_dir: str, collection_name: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    retriever = get_retriever(vectorstore)

    system_prompt = (
        "You are the Agent-AE Technical Director. Use the provided documentation "
        "to write After Effects ExtendScript (JSX).\n"
        "STRICT RULES:\n"
        "- Use ONLY 'var'. NEVER use 'const' or 'let'.\n"
        "- Use 'function name() {{}}' syntax. NEVER use arrow functions.\n"
        "- If the context doesn't contain the answer, state that you don't know based on the docs.\n"
        "- Context: {context}\n"
        "- Question: {input}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, docs_chain), vectorstore


@traceable(name="AE_Technical_Director_RAG", run_type="chain")
def generate_ae_script(user_query: str) -> dict:
    validate_tracing_env()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
    collection_name = os.getenv("CHROMA_COLLECTION", "ae-scripting-guide")

    chain, vectorstore = build_chain(persist_dir, collection_name)
    docs = get_retriever(vectorstore).invoke(user_query)
    result = chain.invoke({"input": user_query})
    return {
        "answer": result["answer"],
        "source_documents": [doc.page_content for doc in docs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RetrievalQA over the AE docs ChromaDB."
    )
    parser.add_argument("--query", help="User query for the assistant.")
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "/data/chroma"),
        help="Path to the ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "ae-scripting-guide"),
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print top-k retrieved chunk metadata before answering.",
    )
    parser.add_argument(
        "--debug-snippet-chars",
        type=int,
        default=240,
        help="Number of characters to show from each retrieved chunk.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as an HTTP service.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for the HTTP service.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the HTTP service.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    validate_tracing_env()

    if args.serve:
        run_server(args.host, args.port)
        return

    user_query: Optional[str] = args.query
    if not user_query:
        user_query = sys.stdin.read().strip()

    if not user_query:
        raise SystemExit("Provide --query or pass a prompt via stdin.")

    chain, vectorstore = build_chain(args.persist_dir, args.collection)
    if args.debug:
        docs = get_retriever(vectorstore).invoke(user_query)
        print("Top-k retrieved chunks:")
        for idx, doc in enumerate(docs, start=1):
            snippet = doc.page_content[: args.debug_snippet_chars].replace("\n", " ")
            print(f"{idx}. {doc.metadata} | snippet: {snippet}")
    result = generate_ae_script(user_query)
    print(result["answer"])


class QueryRequest(BaseModel):
    query: str


app = FastAPI()


@app.post("/query")
def query_endpoint(payload: QueryRequest):
    result = generate_ae_script(payload.query)
    return result


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


def run_server(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


def validate_tracing_env() -> None:
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")
    if not tracing:
        return
    missing = [
        name
        for name in ("LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


if __name__ == "__main__":
    main()
