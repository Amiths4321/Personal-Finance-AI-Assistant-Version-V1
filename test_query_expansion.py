
from ollama import Client
from query_expansion import expand_query

client = Client(
    host="http://10.22.39.192:11434"
)

model = "qwen2.5vl:latest"

question = (
    "How much money is kept as an emergency reserve?"
)

queries = expand_query(client, model, question)

print("Original question:")
print(question)

print("\nExpanded queries:")

for query in queries:
    print("-", query)
