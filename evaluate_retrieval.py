
from rag_engine import chunk_document, retrieve_relevant_chunks
from semantic_rag import (
    index_document,
    search_document,
    hybrid_search_document,
)
from ollama import Client

OLLAMA_HOST = "http://10.22.39.192:11434"
OLLAMA_MODEL = "qwen2.5vl:latest"

client = Client(host=OLLAMA_HOST)

# Synthetic financial document for evaluation
document = """
--- PAGE 1 ---
Customer: Rahul Sharma
Account Type: Savings Account
Monthly Salary: INR 85000
Monthly Rent: INR 18000
Monthly Food Expense: INR 12000

--- PAGE 2 ---
Personal Loan Outstanding: INR 450000
Personal Loan Interest Rate: 11.5 percent
Monthly EMI: INR 12500
Loan Tenure Remaining: 42 months

--- PAGE 3 ---
Credit Card Outstanding: INR 65000
Credit Card Interest Rate: 36 percent
Minimum Due: INR 3200
Credit Card Payment Due Date: 15 November 2026

--- PAGE 4 ---
Monthly Investment: INR 10000
Emergency Fund: INR 150000
Investment Type: Mutual Funds
"""

chunks = chunk_document(document)

# Index the document in ChromaDB
document_id = index_document(client, chunks, document)

# Questions and expected source pages
test_cases = [
    {
        "question": "What is the monthly salary?",
        "expected_pages": [1],
    },
    {
        "question": "How much is owed on the personal loan?",
        "expected_pages": [2],
    },
    {
        "question": "What is the credit card interest rate?",
        "expected_pages": [3],
    },
    {
        "question": "How much money is kept as an emergency reserve?",
        "expected_pages": [4],
    },
]

def precision_at_k(results, expected_pages, k=3):
    retrieved_pages = [
        item["page"] for item in results[:k]
    ]

    if not retrieved_pages:
        return 0.0

    relevant = sum(
        1 for page in retrieved_pages
        if page in expected_pages
    )

    return relevant / len(retrieved_pages)

def recall_at_k(results, expected_pages, k=3):
    retrieved_pages = {item["page"] for item in results[:k]}
    relevant = sum(
        1 for page in expected_pages
        if page in retrieved_pages
    )
    return relevant / len(expected_pages)

systems = {
    "TF-IDF": lambda q: retrieve_relevant_chunks(
        q, chunks, top_k=3
    ),
    "Semantic": lambda q: search_document(
        client, q, document_id, top_k=3
    ),
    "Hybrid": lambda q: hybrid_search_document(
        client, q, document_id, chunks, top_k=3
    ),
}

totals = {
    name: {"precision": 0, "recall": 0}
    for name in systems
}

for test in test_cases:
    print("\nQuestion:", test["question"])
    expected = test["expected_pages"]

    for name, search_function in systems.items():
        results = search_function(test["question"])

        p = precision_at_k(results, expected)
        r = recall_at_k(results, expected)

        totals[name]["precision"] += p
        totals[name]["recall"] += r

        print(
            f"{name}: "
            f"Pages {[x['page'] for x in results[:3]]} | "
            f"Precision@3={p:.2f} | Recall@3={r:.2f}"
        )

        for item in results[:3]:
            print("  Page:", item["page"])
            print("  Text:", item["text"][:200])
            if "distance" in item:
                print("  Distance:", round(item["distance"], 4))
            if "rrf_score" in item:
                print("  RRF score:", round(item["rrf_score"], 4))
            if "methods" in item:
                print("  Methods:", item["methods"])

print("\n========== AVERAGE RESULTS ==========")

for name, scores in totals.items():
    print(
        f"{name}: "
        f"Precision@3={scores['precision'] / len(test_cases):.2f} | "
        f"Recall@3={scores['recall'] / len(test_cases):.2f}"
    )
