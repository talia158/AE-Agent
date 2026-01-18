import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas import evaluate
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._faithfulness import faithfulness
from datasets import Dataset


def add_repo_to_path(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.append(repo_root)


def build_rag_runner(persist_dir: str, collection_name: str) -> Any:
    service_url = os.getenv("RETRIEVAL_QA_URL", "http://retrieval_qa:8000/query")

    def run_qa(query: str) -> Tuple[str, List[str]]:
        response = requests.post(service_url, json={"query": query}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        return payload["answer"], payload.get("source_documents", [])

    return run_qa


def get_dataset_by_name(client: Client, name: str):
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        datasets = list(client.list_datasets())
        for dataset in datasets:
            if dataset.name == name:
                return dataset
    raise RuntimeError(f"Dataset not found: {name}")


def parse_example(example) -> Tuple[str, str]:
    inputs = example.inputs or {}
    if hasattr(inputs, "model_dump"):
        inputs = inputs.model_dump()
    elif hasattr(inputs, "dict"):
        inputs = inputs.dict()
    outputs = example.outputs or {}
    if hasattr(outputs, "model_dump"):
        outputs = outputs.model_dump()
    elif hasattr(outputs, "dict"):
        outputs = outputs.dict()
    question = (
        inputs.get("question")
        or inputs.get("query")
        or inputs.get("input")
        or inputs.get("user_query")
    )
    ground_truth = (
        outputs.get("answer")
        or outputs.get("ground_truth")
        or outputs.get("response")
    )
    if not question:
        raise ValueError("Example is missing a question input.")
    if not ground_truth:
        raise ValueError("Example is missing a ground truth answer.")
    return question, ground_truth


def evaluate_example(
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, float]:
    record = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }
    dataset = Dataset.from_list([record])
    column_map = {
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
        "reference": "ground_truth",
    }
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        column_map=column_map,
    )
    df = result.to_pandas()
    scores = df.iloc[0].to_dict()
    return {
        "faithfulness": float(scores.get("faithfulness", 0.0)),
        "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
        "context_precision": float(scores.get("context_precision", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG pipeline against a LangSmith dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="AE_Scripting_Guide_Golden_Set",
        help="LangSmith dataset name to evaluate.",
    )
    parser.add_argument(
        "--repo-root",
        default="/data",
        help="Repo root for importing retrieval_qa.",
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "/data/chroma"),
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION", "ae-scripting-guide"),
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--debug-example",
        action="store_true",
        help="Print the first example's input/output keys for debugging.",
    )
    parser.add_argument(
        "--skip-langsmith-logging",
        action="store_true",
        default=os.getenv("SKIP_LANGSMITH_LOGGING", "").lower() in ("1", "true", "yes"),
        help="Skip creating LangSmith runs/feedback to avoid rate limits.",
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
    add_repo_to_path(args.repo_root)

    client = Client()
    dataset = get_dataset_by_name(client, args.dataset_name)
    examples = list(client.list_examples(dataset_id=dataset.id))
    if not examples:
        raise RuntimeError("No examples found in the dataset.")
    if args.debug_example:
        sample = examples[0]
        print(f"Sample example id: {sample.id}")
        raw_inputs = sample.inputs or {}
        raw_outputs = sample.outputs or {}
        print(f"Sample inputs raw: {raw_inputs!r}")
        print(f"Sample outputs raw: {raw_outputs!r}")
        if hasattr(raw_inputs, "model_dump"):
            raw_inputs = raw_inputs.model_dump()
        elif hasattr(raw_inputs, "dict"):
            raw_inputs = raw_inputs.dict()
        if hasattr(raw_outputs, "model_dump"):
            raw_outputs = raw_outputs.model_dump()
        elif hasattr(raw_outputs, "dict"):
            raw_outputs = raw_outputs.dict()
        print(f"Sample inputs keys: {list(raw_inputs.keys())}")
        print(f"Sample outputs keys: {list(raw_outputs.keys())}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    run_qa = build_rag_runner(args.persist_dir, args.collection)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"AE_RAG_Eval_{timestamp}"

    rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(examples, start=1):
        try:
            question, ground_truth = parse_example(example)
        except ValueError as exc:
            print(f"Skipping example {example.id}: {exc}")
            continue

        print(f"[{idx}/{len(examples)}] Evaluating: {question[:80]}")
        try:
            answer, contexts = run_qa(question)
            scores = evaluate_example(
                llm,
                embeddings,
                question,
                answer,
                contexts,
                ground_truth,
            )

            if not args.skip_langsmith_logging:
                run = client.create_run(
                    name="rag-eval",
                    run_type="chain",
                    inputs={"question": question},
                    outputs={"answer": answer},
                    project_name=experiment_name,
                    reference_example_id=example.id,
                )
                for key, value in scores.items():
                    client.create_feedback(
                        run_id=run.id,
                        key=key,
                        score=value,
                    )

            rows.append({"question": question, **scores})
        except Exception as exc:
            print(f"Error on example {example.id}: {exc}")
            continue

    if not rows:
        print("No evaluation results were produced. Check dataset inputs.")
        return

    df = pd.DataFrame(rows)
    summary = df[["faithfulness", "answer_relevancy", "context_precision"]].mean()
    print("\nAverage scores:")
    print(summary.to_frame(name="score"))


if __name__ == "__main__":
    main()
