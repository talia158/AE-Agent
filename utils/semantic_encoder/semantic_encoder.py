import argparse
import json
import os
from typing import List, Tuple

import chromadb
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed Markdown content and store vectors in ChromaDB."
    )
    parser.add_argument(
        "--input",
        default="utils/data/ae_docs_chunks.jsonl",
        help="Path to the JSONL chunk file.",
    )
    parser.add_argument(
        "--persist-dir",
        default="/data/chroma",
        help="Directory for ChromaDB persistence.",
    )
    parser.add_argument(
        "--collection",
        default="ae-scripting-guide",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of chunks per embedding batch.",
    )
    return parser.parse_args()


def load_chunks(path: str) -> List[Tuple[str, dict]]:
    chunks: List[Tuple[str, dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if not metadata:
                metadata = {"chunk_index": idx}
            chunks.append((record["text"], metadata))
    return chunks


def batched(items: List[Tuple[str, dict]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    chunks = load_chunks(args.input)
    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]

    client = OpenAI(api_key=api_key)

    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() > 0:
        print(
            "Collection already contains embeddings; skipping re-embed to preserve data."
        )
        return

    for batch in batched(list(zip(texts, metadatas)), args.batch_size):
        batch_texts = [item[0] for item in batch]
        batch_metadatas = [item[1] for item in batch]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch_texts,
        )
        embeddings = [item.embedding for item in response.data]
        base_id = collection.count()
        ids = [str(i) for i in range(base_id, base_id + len(batch))]
        collection.add(ids=ids, documents=batch_texts, metadatas=batch_metadatas, embeddings=embeddings)


if __name__ == "__main__":
    main()
