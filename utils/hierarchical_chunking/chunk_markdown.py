import argparse
import json
import os
from typing import List, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Markdown into hierarchical chunks using headers."
    )
    parser.add_argument(
        "--input",
        default="utils/data/ae_docs_cleaned.md",
        help="Path to the Markdown input file.",
    )
    parser.add_argument(
        "--output",
        default="ae_docs_chunks.jsonl",
        help="Path to the JSONL output file.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=120,
        help="Minimum token count per chunk before writing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Approximate target token size for each chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Approximate token overlap between adjacent chunks.",
    )
    return parser.parse_args()


def build_header_rules() -> List[Tuple[str, str]]:
    return [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
    ]


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        print("Output already exists; skipping chunking.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=build_header_rules())
    sections = header_splitter.split_text(markdown_text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    semantic_splitter = SemanticChunker(embeddings)

    semantic_chunks: List[Tuple[str, dict]] = []
    for section in sections:
        section_text = section.page_content
        header_path = " > ".join(
            value for _, value in sorted(section.metadata.items())
        ).strip()
        if header_path:
            section_text = f"{header_path}\n\n{section_text}"

        for chunk in semantic_splitter.split_text(section_text):
            semantic_chunks.append((chunk, dict(section.metadata)))

    merged_sections: List[Tuple[str, dict]] = []
    buffer_text = ""
    buffer_metadata = {}
    buffer_count = 0

    for chunk_text, chunk_metadata in semantic_chunks:
        if not buffer_text:
            buffer_text = chunk_text
            buffer_metadata = chunk_metadata
            buffer_count = 1
        else:
            buffer_text = f"{buffer_text}\n\n{chunk_text}"
            buffer_count += 1

        token_count = len(buffer_text.split())
        if token_count >= args.chunk_size:
            if buffer_count > 1:
                buffer_metadata = dict(buffer_metadata)
                buffer_metadata["merged_chunks"] = buffer_count
            merged_sections.append((buffer_text, buffer_metadata))
            if args.overlap > 0:
                overlap_tokens = buffer_text.split()[-args.overlap :]
                buffer_text = " ".join(overlap_tokens)
                buffer_metadata = dict(chunk_metadata)
                buffer_count = 1
            else:
                buffer_text = ""
                buffer_metadata = {}
                buffer_count = 0

    if buffer_text:
        if len(buffer_text.split()) >= args.min_tokens:
            if buffer_count > 1:
                buffer_metadata = dict(buffer_metadata)
                buffer_metadata["merged_chunks"] = buffer_count
            merged_sections.append((buffer_text, buffer_metadata))

    with open(args.output, "w", encoding="utf-8") as f:
        for text, metadata in merged_sections:
            record = {
                "text": text,
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
