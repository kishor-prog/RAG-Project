import os
import sys
import time
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATASET = [
    {
        "question": "explain about evolvex?",
        "ground_truth": "Evolvex AI Solution is an engineering firm that specializes in custom AI software applications operating out of Pollachi."
    },
    {
        "question": "What is the primary engineering contact email?",
        "ground_truth": "kishor123@gmail.com"
    },
    {
        "question": "What is the contact phone number for Evolvex?",
        "ground_truth": "+91 9874563210"
    }
]


def calculate_metrics(ground_truth: str, generated_answer: str, contexts: list):
    """
    Computes key RAG Triad benchmark metrics:
    - Context Recall: Proportion of ground truth facts present in retrieved chunks.
    - Faithfulness: Degree to which the LLM answer is grounded strictly in retrieved context.
    - Context Precision: Relevancy score of the retrieved contexts relative to ground truth.
    - Answer Similarity: Word-level F1 overlap between generated answer and ground truth.
    """
    gt_words = set(ground_truth.lower().replace(",", "").replace(".", "").split())
    ans_words = set(generated_answer.lower().replace(",", "").replace(".", "").split())
    context_str = " ".join(contexts).lower()

    # 1. Context Recall
    matching_gt_in_context = [word for word in gt_words if word in context_str]
    context_recall = round(len(matching_gt_in_context) / max(len(gt_words), 1), 2)

    # 2. Faithfulness
    matching_ans_in_context = [word for word in ans_words if word in context_str]
    faithfulness = round(len(matching_ans_in_context) / max(len(ans_words), 1), 2)

    # 3. Context Precision
    context_precision = 1.0 if context_recall > 0.5 else 0.0

    # 4. Lexical Overlap / Similarity (F1 Score)
    common_words = gt_words.intersection(ans_words)
    if common_words:
        precision = len(common_words) / len(ans_words)
        recall = len(common_words) / len(gt_words)
        f1_score = round((2 * precision * recall) / (precision + recall), 2)
    else:
        f1_score = 0.0

    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "f1_score": f1_score
    }


def run_evaluation(api_url: str, output_csv: str):
    print("\n=======================================================")
    print("      Evolvex RAG Benchmark & Evaluation Pipeline       ")
    print(f"      Target Endpoint: {api_url}                     ")
    print("=======================================================\n")

    # Check API health
    try:
        health_resp = requests.get(api_url.replace("/api/v1/query", "/"), timeout=5)
        if health_resp.status_code == 200:
            print(f"[+] Connected to API. Status: {health_resp.json().get('status', 'OK')}")
        else:
            print(f"[!] Warning: API returned status code {health_resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not connect to API at {api_url}.")
        print("Please ensure the FastAPI server is running with: uvicorn main:app --reload")
        sys.exit(1)

    results = []

    for idx, item in enumerate(DEFAULT_DATASET, start=1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"\n[{idx}/{len(DEFAULT_DATASET)}] Evaluating: \"{q}\"")

        start_time = time.time()
        try:
            response = requests.post(
                api_url,
                json={"question": q, "top_k": 3},
                timeout=30
            )
            latency = round(time.time() - start_time, 2)

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                contexts = data.get("contexts", [])

                metrics = calculate_metrics(gt, answer, contexts)

                results.append({
                    "Question": q,
                    "Expected Ground Truth": gt,
                    "Model Generated Answer": answer,
                    "Context Precision": metrics["context_precision"],
                    "Context Recall": metrics["context_recall"],
                    "Faithfulness": metrics["faithfulness"],
                    "F1 Score": metrics["f1_score"],
                    "Latency (s)": latency
                })

                print(f"    -> Latency: {latency}s | Recall: {metrics['context_recall']} | Faithfulness: {metrics['faithfulness']} | F1: {metrics['f1_score']}")
            else:
                print(f"    [!] API Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"    [!] Error during request: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print("\n=======================================================")
        print(f" [SUCCESS] Evaluation complete! Report saved to '{output_csv}'")
        print("=======================================================\n")
        print("Summary Metrics:")
        print(f"  - Average Context Recall:    {df['Context Recall'].mean():.2f}")
        print(f"  - Average Faithfulness:       {df['Faithfulness'].mean():.2f}")
        print(f"  - Average F1 Score:           {df['F1 Score'].mean():.2f}")
        print(f"  - Average Latency:            {df['Latency (s)'].mean():.2f}s")
        print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolvex RAG Evaluation & Benchmarking Tool")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/query", help="RAG Query API endpoint")
    parser.add_argument("--output", default="advanced_evaluation_report.csv", help="Output CSV path")
    args = parser.parse_args()

    run_evaluation(args.url, args.output)