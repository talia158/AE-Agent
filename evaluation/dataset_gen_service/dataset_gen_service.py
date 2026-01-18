import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)


def load_documents(path: str) -> List[Document]:
    documents: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text")
            if not text:
                continue
            metadata = record.get("metadata", {})
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def filter_short_documents(documents: List[Document], min_tokens: int) -> List[Document]:
    filtered = []
    for doc in documents:
        token_count = len(doc.page_content.split())
        if token_count >= min_tokens:
            filtered.append(doc)
    return filtered


class TemperatureFreeChatOpenAI(ChatOpenAI):
    @property
    def _default_params(self):
        params = super()._default_params
        params.pop("temperature", None)
        return params

    def _get_invocation_params(self, **kwargs):
        params = super()._get_invocation_params(**kwargs)
        params.pop("temperature", None)
        return params

    def bind(self, **kwargs):
        kwargs.pop("temperature", None)
        return super().bind(**kwargs)

    def _get_request_payload(self, *args, **kwargs):
        payload = super()._get_request_payload(*args, **kwargs)
        payload.pop("temperature", None)
        return payload


def build_generator() -> TestsetGenerator:
    llm = TemperatureFreeChatOpenAI(
        model="gpt-5-mini",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return TestsetGenerator.from_langchain(llm, embeddings)


def generate_testset(generator: TestsetGenerator, docs: List[Document]):
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator.llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 0.25),
    ]
    return generator.generate_with_langchain_docs(
        docs,
        testset_size=20,
        query_distribution=query_distribution,
    )


def testset_to_rows(testset) -> List[dict]:
    if hasattr(testset, "to_list"):
        return testset.to_list()
    if hasattr(testset, "to_pandas"):
        return testset.to_pandas().to_dict(orient="records")
    if hasattr(testset, "to_dict"):
        return testset.to_dict(orient="records")
    return list(testset)


def normalize_question(value) -> str:
    if isinstance(value, list):
        # Multi-turn samples are lists of messages; keep the last user input if possible.
        for item in value:
            if isinstance(item, dict) and item.get("type") == "human":
                return item.get("content", "")
        return str(value[-1]) if value else ""
    return value or ""


def generate_testset_with_pacing(
    generator: TestsetGenerator,
    docs: List[Document],
    total: int,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
    retry_backoff: float,
) -> List[dict]:
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator.llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 0.25),
    ]
    rows: List[dict] = []
    remaining = total
    while remaining > 0:
        current_size = min(batch_size, remaining)
        print(f"Generating {current_size} questions (remaining: {remaining})...")
        attempt = 0
        while True:
            try:
                testset = generator.generate_with_langchain_docs(
                    docs,
                    testset_size=current_size,
                    query_distribution=query_distribution,
                )
                break
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = "rate limit" in message or "429" in message
                is_transient = any(
                    keyword in message
                    for keyword in (
                        "timeout",
                        "connection",
                        "temporarily unavailable",
                        "internal server error",
                        "service unavailable",
                        "upstream connect error",
                    )
                )
                if not (is_rate_limit or is_transient) or attempt >= max_retries:
                    raise
                sleep_for = retry_sleep_seconds * (retry_backoff ** attempt)
                print(
                    f"Rate limit hit; sleeping {sleep_for:.1f}s before retry "
                    f"({attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_for)
                attempt += 1
        rows.extend(testset_to_rows(testset))
        remaining -= current_size
        if remaining > 0 and sleep_seconds > 0:
            print(f"Sleeping {sleep_seconds} seconds to respect rate limits...")
            time.sleep(sleep_seconds)
    return rows


def get_dataset(client: Client, name: str):
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        datasets = list(client.list_datasets())
        for dataset in datasets:
            if dataset.name == name:
                return dataset
    return None


def upload_to_langsmith(dataset_name: str, rows: List[dict]) -> None:
    client = Client()
    dataset = get_dataset(client, dataset_name)
    if not dataset:
        dataset = client.create_dataset(dataset_name)

    for row in rows:
        question = (
            row.get("question")
            or row.get("user_input")
            or row.get("query")
        )
        question = normalize_question(question)
        ground_truth = (
            row.get("ground_truth")
            or row.get("reference")
            or row.get("response")
        )
        inputs = {"question": question}
        outputs = {"answer": ground_truth}
        metadata = {
            "contexts": row.get("contexts")
            or row.get("reference_contexts")
            or row.get("retrieved_contexts"),
            "type": row.get("evolution_type") or row.get("synthesizer_name"),
        }
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_id=dataset.id,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a golden dataset from AE docs and upload to LangSmith."
    )
    parser.add_argument(
        "--input",
        default="utils/data/ae_docs_chunks.jsonl",
        help="Path to the JSONL chunk file.",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("DATASET_NAME", ""),
        help="Optional fixed dataset name to reuse across runs.",
    )
    skip_default = os.getenv("DATASET_SKIP_IF_EXISTS", "").lower() in ("1", "true", "yes")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        default=skip_default,
        help="Skip generation if the dataset already exists.",
    )
    parser.add_argument(
        "--cache-path",
        default=os.getenv(
            "DATASET_CACHE_PATH",
            "evaluation/dataset_gen_service/cache/ae_scripting_golden_set.jsonl",
        ),
        help="Path to save/load a local JSONL cache of generated questions.",
    )
    parser.add_argument(
        "--doc-fraction",
        type=float,
        default=float(os.getenv("DATASET_DOC_FRACTION", "1.0")),
        help="Fraction of documents to sample for generation (0-1].",
    )
    parser.add_argument(
        "--doc-seed",
        type=int,
        default=int(os.getenv("DATASET_DOC_SEED", "42")),
        help="Random seed used for document sampling.",
    )
    parser.add_argument(
        "--min-doc-tokens",
        type=int,
        default=int(os.getenv("DATASET_MIN_DOC_TOKENS", "120")),
        help="Minimum token count per document to keep.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=int(os.getenv("DATASET_TOTAL", "20")),
        help="Total number of questions to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("DATASET_BATCH_SIZE", "2")),
        help="Questions to generate per batch.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("DATASET_SLEEP_SECONDS", "5.0")),
        help="Seconds to wait between batches.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("DATASET_MAX_RETRIES", "5")),
        help="Max retries per batch on rate limit errors.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=float(os.getenv("DATASET_RETRY_SLEEP_SECONDS", "15.0")),
        help="Base sleep time for rate limit retries.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=float(os.getenv("DATASET_RETRY_BACKOFF", "2.0")),
        help="Backoff multiplier for rate limit retries.",
    )
    return parser.parse_args()


def validate_env() -> None:
    missing = [
        name
        for name in ("OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    validate_env()

    print("Loading documents...")
    docs = load_documents(args.input)
    if not docs:
        raise RuntimeError("No documents loaded from input file.")
    print(f"Loaded {len(docs)} documents.")

    docs = filter_short_documents(docs, args.min_doc_tokens)
    if not docs:
        raise RuntimeError("No documents left after filtering short chunks.")
    print(f"Retained {len(docs)} documents after filtering.")

    if not (0.0 < args.doc_fraction <= 1.0):
        raise ValueError("--doc-fraction must be between 0 and 1.")

    if args.doc_fraction < 1.0:
        random.seed(args.doc_seed)
        sample_size = max(1, int(len(docs) * args.doc_fraction))
        docs = random.sample(docs, sample_size)
        print(f"Sampled {len(docs)} documents for generation.")

    print("Building testset generator...")
    generator = build_generator()

    cache_path = args.cache_path
    dataset_name = args.dataset_name.strip()
    if os.path.exists(cache_path):
        print(f"Found cached dataset at {cache_path}; loading.")
        rows = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        print("Generating testset...")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1.")
        client = Client()
        if dataset_name:
            existing = get_dataset(client, dataset_name)
            if existing and args.skip_if_exists:
                print(f"Dataset '{dataset_name}' already exists; skipping generation.")
                return
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"AE_Scripting_Guide_Golden_Set_{timestamp}"

        rows = generate_testset_with_pacing(
            generator,
            docs,
            total=args.total,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            retry_backoff=args.retry_backoff,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"Saved local cache to {cache_path}")

    if not dataset_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"AE_Scripting_Guide_Golden_Set_{timestamp}"

    print(f"Uploading to LangSmith dataset: {dataset_name}")
    upload_to_langsmith(dataset_name, rows)
    print("Upload complete.")


if __name__ == "__main__":
    main()
