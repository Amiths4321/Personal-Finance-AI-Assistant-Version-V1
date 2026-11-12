
from ollama import Client
from semantic_rag import (
    expanded_hybrid_search,
    index_document
)
from rag_engine import chunk_document

client = Client(
    host="http://10.22.39.192:11434"
)

model = "qwen2.5vl:latest"

document = """
--- PAGE 1 ---
Customer: Rahul Sharma
Account Type: Savings Account
Monthly Salary: INR 85000

--- PAGE 2 ---
Personal Loan Outstanding: INR 450000
Personal Loan Interest Rate: 11.5 percent

--- PAGE 3 ---
Credit Card Outstanding: INR 65000
Credit Card Interest Rate: 36 percent

--- PAGE 4 ---
Monthly Investment: INR 10000
Emergency Fund: INR 150000
Investment Type: Mutual Funds
"""

chunks = chunk_document(document)

document_id = index_document(
    client,
    chunks,
    document
)

question = (
    "How much money is kept as an emergency reserve?"
)

results = expanded_hybrid_search(
    client,
    model,
    question,
    document_id,
    chunks,
    top_k=3
)

print("\nExpanded Hybrid Results:")

for item in results:
    print("\nPage:", item["page"])
    print("Text:", item["text"])
    print("RRF score:", round(item["rrf_score"], 5))
    print("Methods:", item["methods"])
