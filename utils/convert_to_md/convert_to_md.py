import argparse
import re

from bs4 import BeautifulSoup
from markdownify import markdownify as md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an HTML file to Markdown.")
    parser.add_argument(
        "--input",
        default="Raw Guide.html",
        help="Path to the HTML input file.",
    )
    parser.add_argument(
        "--output",
        default="ae_docs_cleaned.md",
        help="Path to the Markdown output file.",
    )
    return parser.parse_args()


def clean_ae_markdown(text: str) -> str:
    # Remove "Permanent link" icons and anchors.
    text = re.sub(r'\[¶\]\(#[^\)]+\s"Permanent link"\)', "", text)

    # Remove "Skip to content" style links.
    text = re.sub(r'\[Skip to content\]\(#index\)', "", text)

    # Fix multiple newlines created by the cleaning.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove scripts, styles, or navigation if they exist in the file.
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()

    markdown_text = md(str(soup), heading_style="ATX")
    markdown_text = clean_ae_markdown(markdown_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown_text)


if __name__ == "__main__":
    main()
