
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunk_document(document_text, chunk_size=1200, overlap=200):
    """
    Split a document into overlapping chunks.
    Preserve page numbers for citations.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []

    # Split using page markers from document_processor.py
    pages = re.split(r"--- PAGE (\d+) ---", document_text)

    for i in range(1, len(pages), 2):
        page_number = int(pages[i])
        page_text = pages[i + 1].strip()

        if not page_text:
            continue

        step = chunk_size - overlap

        for start in range(0, len(page_text), step):
            chunk_text = page_text[start:start + chunk_size]

            if not chunk_text.strip():
                continue

            chunks.append({
                "page": page_number,
                "text": chunk_text.strip()
            })

            if start + chunk_size >= len(page_text):
                break

    return chunks


def retrieve_relevant_chunks(question, chunks, top_k=3):
    """
    Retrieve document chunks relevant to the question.
    Uses TF-IDF and cosine similarity.
    """

    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()

    try:
        document_vectors = vectorizer.fit_transform(texts)
        question_vector = vectorizer.transform([question])
    except ValueError:
        return []

    scores = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:
        if scores[index] <= 0:
            continue

        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results
