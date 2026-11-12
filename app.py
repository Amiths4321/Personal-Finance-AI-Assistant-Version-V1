
import streamlit as st
import os
from semantic_rag import expanded_hybrid_search

from rag_engine import chunk_document
from semantic_rag import hybrid_search_document

from semantic_rag import (
    index_document,
    answer_with_semantic_rag
)

from ollama import Client

from document_processor import (
    extract_pdf_text,
    answer_document_question,
    extract_financial_data
)

from document_processor import (
    extract_pdf_text,
    answer_document_question
)


# Ollama configuration

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://10.22.39.192:11434"
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5vl:latest"
)

client = Client(host=OLLAMA_HOST)


# Session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "hybrid_results" not in st.session_state:
    st.session_state.hybrid_results = None

if "semantic_document_id" not in st.session_state:
    st.session_state.semantic_document_id = ""

if "semantic_result" not in st.session_state:
    st.session_state.semantic_result = None

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "document_answer" not in st.session_state:
    st.session_state.document_answer = ""


# Application interface

st.title("Personal Finance AI Assistant")

st.write(
    "AI-powered personal finance and document analysis."
)


# PDF upload

st.sidebar.header("Document Intelligence")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a financial PDF",
    type=["pdf"]
)

if st.sidebar.button("Extract PDF Text"):

    if uploaded_pdf is None:
        st.sidebar.warning("Please upload a PDF first.")

    elif uploaded_pdf.size > 10 * 1024 * 1024:
        st.sidebar.error("Maximum PDF size is 10 MB.")

    else:
        try:
            with st.spinner("Extracting PDF text..."):

                extracted_text = extract_pdf_text(
                    uploaded_pdf
                )

                st.session_state.document_text = extracted_text
                st.session_state.document_name = uploaded_pdf.name

                st.session_state.document_answer = ""

            st.success("PDF extracted successfully!")

        except Exception as error:
            st.error(f"PDF extraction failed: {error}")
        # Create document chunks

        chunks = chunk_document(
            extracted_text,
            chunk_size=1200,
            overlap=200
        )

        # Index document in ChromaDB

        with st.spinner("Creating semantic embeddings..."):

            document_id = index_document(
                client,
                chunks,
                extracted_text
            )

        st.session_state.semantic_document_id = document_id
        st.session_state.document_chunks = chunks
        st.session_state.document_answer = ""
        st.session_state.financial_data = None
        st.session_state.rag_result = None

# Document preview
st.session_state.financial_data = None
st.session_state.hybrid_results = None

if st.session_state.document_text:

    st.subheader("Uploaded Document")

    st.caption(
        f"Document: {st.session_state.document_name}"
    )

    with st.expander("View extracted document text"):

        st.text_area(
            "Extracted content",
            value=st.session_state.document_text[:10000],
            height=300,
            disabled=True
        )

    # Structured financial data extraction

st.divider()

st.subheader("📊 Extract Financial Information")

if st.button(
    "Extract Financial Data",
    use_container_width=True
):

    try:
        with st.spinner(
            "Qwen is extracting financial information..."
        ):

            financial_data = extract_financial_data(
                client,
                MODEL,
                st.session_state.document_text
            )

            st.session_state.financial_data = financial_data

    except Exception as error:
        st.error(f"Financial extraction failed: {error}")


if st.session_state.get("financial_data"):

    st.markdown("### Extracted Financial Data")

    st.json(st.session_state.financial_data)
    # Document question answering

    st.subheader("Ask Questions About Your Document")

    with st.form("document_question_form"):

        document_question = st.text_input(
            "Your question",
            placeholder="What is the total monthly expenditure?"
        )

        ask_button = st.form_submit_button("Ask AI")

    if ask_button:

        if not document_question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Qwen is analysing your document..."):

                    answer = answer_document_question(
                        client,
                        MODEL,
                        st.session_state.document_text,
                        document_question
                    )

                    st.session_state.document_answer = answer

            except Exception as error:
                st.error(
                    f"Document analysis failed: {error}"
                )

    if st.session_state.document_answer:

        st.markdown("### AI Analysis")

        st.write(
            st.session_state.document_answer
        )

if "financial_data" not in st.session_state:
    st.session_state.financial_data = None

# Semantic RAG

st.divider()

st.subheader("🧠 Semantic Financial Document Search")

semantic_question = st.text_input(
    "Ask a question using semantic search",
    key="semantic_question"
)

if st.button("Search with Semantic RAG"):

    if not semantic_question.strip():

        st.warning("Please enter a question.")

    else:

        try:
            with st.spinner(
                "Searching semantic vectors and analysing..."
            ):

                semantic_result = answer_with_semantic_rag(
                    client,
                    MODEL,
                    semantic_question,
                    st.session_state.semantic_document_id,
                    top_k=3
                )

                st.session_state.semantic_result = semantic_result

        except Exception as error:
            st.error(f"Semantic RAG failed: {error}")


if st.session_state.get("semantic_result"):

    st.markdown("### Semantic AI Answer")

    st.write(
        st.session_state.semantic_result["answer"]
    )

    with st.expander("View semantic search sources"):

        for source in st.session_state.semantic_result["sources"]:

            st.write(f"Page: {source['page']}")

            st.write(
                f"Cosine distance: {source['distance']:.4f}"
            )

            st.write(source["text"])

            st.divider()

# Hybrid retrieval

st.divider()

st.subheader("🔀 Hybrid Financial Document Search")

hybrid_question = st.text_input(
    "Ask a question using keyword + semantic search",
    key="hybrid_question"
)

if st.button("Run Hybrid Search"):

    if not hybrid_question.strip():

        st.warning("Please enter a question.")

    else:

        try:
            with st.spinner("Combining retrieval results..."):

                hybrid_results = hybrid_search_document(
                    client,
                    hybrid_question,
                    st.session_state.semantic_document_id,
                    st.session_state.document_chunks,
                    top_k=3
                )

                st.session_state.hybrid_results = hybrid_results

        except Exception as error:
            st.error(f"Hybrid search failed: {error}")


if st.session_state.get("hybrid_results"):

    st.markdown("### Hybrid Search Results")

    for result in st.session_state.hybrid_results:

        st.write(f"Page: {result['page']}")

        st.write(
            f"RRF score: {result['rrf_score']:.5f}"
        )

        st.write(
            "Retrieved by: " + ", ".join(result["methods"])
        )

        st.write(result["text"])

        st.divider()


st.divider()
st.header("Advanced Financial Document Q&A")

if (
    "document_chunks" in st.session_state
    and st.session_state.document_chunks
    and "semantic_document_id" in st.session_state
    and st.session_state.semantic_document_id
):
    advanced_question = st.text_input(
        "Ask a question using expanded hybrid retrieval",
        key="advanced_question"
    )

    if st.button("Search and Answer") and advanced_question:
        with st.spinner("Searching financial document..."):

            results = expanded_hybrid_search(
                client,
                MODEL,
                advanced_question,
                st.session_state.semantic_document_id,
                st.session_state.document_chunks,
                top_k=3
            )

            context = "\n\n".join(
                [
                    f"Page {item['page']}:\n{item['text']}"
                    for item in results
                ]
            )

            response = client.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial document assistant. "
                            "Answer only using the supplied document passages. "
                            "Do not invent financial figures. "
                            "If the information is unavailable, say so. "
                            "Mention the relevant page number."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {advanced_question}\n\n"
                            f"Document passages:\n{context}"
                        )
                    }
                ]
            )

        st.subheader("Answer")
        st.write(response["message"]["content"])

        with st.expander("Retrieved evidence"):
            for item in results:
                st.write(f"**Page {item['page']}**")
                st.write(item["text"])
                st.caption(
                    f"RRF: {item['rrf_score']:.5f} | "
                    f"Methods: {item['methods']}"
                )
