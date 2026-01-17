import argparse
import json
import os
from typing import List, Tuple

from langchain_text_splitters import MarkdownHeaderTextSplitter


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

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=build_header_rules())
    sections = splitter.split_text(markdown_text)

    with open(args.output, "w", encoding="utf-8") as f:
        for section in sections:
            record = {
                "text": section.page_content,
                "metadata": section.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
