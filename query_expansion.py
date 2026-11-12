
import json

def expand_query(client, model, question):
    prompt = f"""
Generate 4 alternative search queries for this
financial document question.

Preserve the original meaning.
Use synonyms and alternative financial terminology.
Do not answer the question.

Question: {question}

Return only valid JSON:
{{
    "queries": [
        "alternative query 1",
        "alternative query 2",
        "alternative query 3",
        "alternative query 4"
    ]
}}
"""

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    result = json.loads(response["message"]["content"])

    return result["queries"]
