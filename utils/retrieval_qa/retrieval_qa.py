import argparse
import os
import sys
from typing import Optional

import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_chain(persist_dir: str, collection_name: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

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


def generate_ae_script(user_query: str) -> str:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "/data/chroma")
    collection_name = os.getenv("CHROMA_COLLECTION", "ae-scripting-guide")

    chain, _ = build_chain(persist_dir, collection_name)
    result = chain.invoke({"input": user_query})
    return result["answer"]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    user_query: Optional[str] = args.query
    if not user_query:
        user_query = sys.stdin.read().strip()

    if not user_query:
        raise SystemExit("Provide --query or pass a prompt via stdin.")

    chain, vectorstore = build_chain(args.persist_dir, args.collection)
    if args.debug:
        docs = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(user_query)
        print("Top-k retrieved chunks:")
        for idx, doc in enumerate(docs, start=1):
            snippet = doc.page_content[: args.debug_snippet_chars].replace("\n", " ")
            print(f"{idx}. {doc.metadata} | snippet: {snippet}")
    result = chain.invoke({"input": user_query})
    print(result["answer"])


if __name__ == "__main__":
    main()
