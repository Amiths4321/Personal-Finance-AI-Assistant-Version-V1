
from rag_engine import (
    chunk_document,
    retrieve_relevant_chunks
)

sample_document = """
--- PAGE 1 ---
Customer: Demo Customer
Monthly salary: INR 80000
Monthly rent: INR 20000
Food expenses: INR 10000

--- PAGE 2 ---
Loan EMI: INR 15000
Loan interest rate: 9 percent
Outstanding loan balance: INR 450000

--- PAGE 3 ---
Insurance premium: INR 3000
Electricity charges: INR 2500
"""

chunks = chunk_document(
    sample_document,
    chunk_size=120,
    overlap=20
)

print("Total chunks:", len(chunks))

question = "What is the outstanding loan balance?"

results = retrieve_relevant_chunks(
    question,
    chunks,
    top_k=3
)

for result in results:
    print("\nPage:", result["page"])
    print("Score:", round(result["score"], 3))
    print("Text:", result["text"])
