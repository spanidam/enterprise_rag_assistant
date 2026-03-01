import csv
import requests

API_URL = "http://127.0.0.1:8000/ask"
DATASET_PATH = "evaluation/evaluation_dataset.csv"

total = 0
retrieval_hits = 0
faithful_answers = 0

def check_faithfulness(answer, ground_truth):
    # Treat safe abstentions as faithful
    if "do not fully support a confident answer" in answer.lower():
        return True

    gt_words = set(ground_truth.lower().split())
    ans_words = set(answer.lower().split())
    overlap = gt_words.intersection(ans_words)

    return len(overlap) / max(len(gt_words), 1) >= 0.3

with open(DATASET_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        question = row["question"]
        ground_truth = row["ground_truth_answer"]
        expected_chunk = row["source_chunk_file"]

        response = requests.post(API_URL, json={"question": question})
        if response.status_code != 200:
            print(f"❌ API error for question: {question}")
            continue

        data = response.json()
        answer = data["answer"]
        sources = [s["chunk_file"] for s in data["sources"]]

        if any(expected_chunk in src for src in sources):
            retrieval_hits += 1

        if check_faithfulness(answer, ground_truth):
            faithful_answers += 1

        print("✅ Question:", question)
        print("Answer:", answer)
        print("Sources:", sources)
        print("-" * 50)

print("\n===== FINAL METRICS =====")
print("Total Questions:", total)
print("Recall@5:", retrieval_hits / total)
print("Faithfulness Score:", faithful_answers / total)
print("Hallucination Rate:", 1 - (faithful_answers / total))