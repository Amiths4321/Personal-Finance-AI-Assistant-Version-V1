
import os
import streamlit as st

from ollama import Client

from document_processor import (
    extract_pdf_text,
    answer_document_question,
    extract_financial_data,
)

from rag_engine import chunk_document

from semantic_rag import (
    index_document,
    answer_with_semantic_rag,
    hybrid_search_document,
    expanded_hybrid_search,
)

from financial_data_mapper import map_extracted_financial_data
from financial_analyzer import analyze_finances
from financial_insights import generate_financial_insights
from financial_dashboard import display_financial_dashboard


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Personal Finance AI Assistant",
    page_icon="💰",
    layout="wide",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://10.22.39.192:11434"
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5vl:latest"
)

client = Client(host=OLLAMA_HOST)


# --------------------------------------------------
# 2. SESSION STATE INITIALISATION
# --------------------------------------------------

DEFAULT_STATE = {
    "document_text": "",
    "document_name": "",
    "document_chunks": [],
    "semantic_document_id": "",
    "financial_data": None,
    "document_answer": "",
    "semantic_result": None,
    "hybrid_results": None,
    "advanced_answer": None,
    "advanced_sources": [],
    "messages": [],
    "conversation_summary": "",
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("💰 Personal Finance AI Assistant")

st.write(
    "AI-powered personal finance and financial document intelligence."
)

st.caption(
    f"AI Model: {MODEL} | Ollama Server: {OLLAMA_HOST}"
)


# --------------------------------------------------
# 4. PDF UPLOAD AND DOCUMENT PROCESSING
# --------------------------------------------------

st.sidebar.header("📄 Document Intelligence")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a financial PDF",
    type=["pdf"],
)

if st.sidebar.button(
    "Extract and Index PDF",
    use_container_width=True,
):

    if uploaded_pdf is None:

        st.sidebar.warning(
            "Please upload a PDF first."
        )

    elif uploaded_pdf.size > 10 * 1024 * 1024:

        st.sidebar.error(
            "Maximum PDF size is 10 MB."
        )

    else:

        try:

            with st.spinner("Extracting PDF text..."):

                extracted_text = extract_pdf_text(
                    uploaded_pdf
                )

                if not extracted_text.strip():
                    raise ValueError(
                        "No readable text was found in this PDF."
                    )

            with st.spinner("Creating document chunks..."):

                chunks = chunk_document(
                    extracted_text,
                    chunk_size=1200,
                    overlap=200,
                )

                if not chunks:
                    raise ValueError(
                        "No document chunks were created."
                    )

            with st.spinner(
                "Creating semantic embeddings and indexing..."
            ):

                document_id = index_document(
                    client,
                    chunks,
                    extracted_text,
                )

            # Update document state only after successful indexing

            st.session_state.document_text = extracted_text
            st.session_state.document_name = uploaded_pdf.name
            st.session_state.document_chunks = chunks
            st.session_state.semantic_document_id = document_id

            # Reset results belonging to the previous document

            st.session_state.financial_data = None
            st.session_state.document_answer = ""
            st.session_state.semantic_result = None
            st.session_state.hybrid_results = None
            st.session_state.advanced_answer = None
            st.session_state.advanced_sources = []

            st.success(
                "PDF extracted and indexed successfully!"
            )

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )


# --------------------------------------------------
# 5. DOCUMENT PREVIEW
# --------------------------------------------------

if st.session_state.document_text:

    st.divider()

    st.subheader("📄 Uploaded Document")

    st.caption(
        f"Document: {st.session_state.document_name}"
    )

    st.caption(
        f"Indexed chunks: {len(st.session_state.document_chunks)}"
    )

    with st.expander("View extracted document text"):

        st.text_area(
            "Extracted content",
            value=st.session_state.document_text[:10000],
            height=300,
            disabled=True,
        )


# --------------------------------------------------
# 6. FINANCIAL INFORMATION EXTRACTION
# --------------------------------------------------

st.divider()

st.header("📊 Financial Information Extraction")

if st.button(
    "Extract Financial Data",
    use_container_width=True,
    disabled=not bool(st.session_state.document_text),
):

    try:

        with st.spinner(
            "Qwen is extracting financial information..."
        ):

            financial_data = extract_financial_data(
                client,
                MODEL,
                st.session_state.document_text,
            )

            st.session_state.financial_data = financial_data

    except Exception as error:

        st.error(
            f"Financial extraction failed: {error}"
        )


if st.session_state.financial_data:

    st.subheader("Extracted Financial Data")

    st.json(
        st.session_state.financial_data
    )


# --------------------------------------------------
# 7. DOCUMENT QUESTION ANSWERING
# --------------------------------------------------

st.divider()

st.header("💬 Ask Questions About Your Document")

with st.form("document_question_form"):

    document_question = st.text_input(
        "Your question",
        placeholder="What is the personal loan balance?",
    )

    ask_document_button = st.form_submit_button(
        "Ask AI",
        disabled=not bool(st.session_state.document_text),
    )


if ask_document_button:

    if not document_question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Qwen is analysing your document..."
            ):

                answer = answer_document_question(
                    client,
                    MODEL,
                    st.session_state.document_text,
                    document_question,
                )

                st.session_state.document_answer = answer

        except Exception as error:

            st.error(
                f"Document analysis failed: {error}"
            )


if st.session_state.document_answer:

    st.subheader("AI Document Analysis")

    st.write(
        st.session_state.document_answer
    )


# --------------------------------------------------
# 8. FINANCIAL INTELLIGENCE DASHBOARD
# --------------------------------------------------

st.divider()

st.header("📈 Financial Intelligence Dashboard")

if st.session_state.financial_data:

    try:

        mapped_data = map_extracted_financial_data(
            st.session_state.financial_data
        )

        analysis = analyze_finances(
            mapped_data
        )

        insights = generate_financial_insights(
            analysis
        )

        display_financial_dashboard(
            analysis,
            insights,
        )

        with st.expander("View mapped financial data"):

            st.json(mapped_data)

        with st.expander("View calculated financial analysis"):

            st.json(analysis)

    except Exception as error:

        st.error(
            f"Financial dashboard failed: {error}"
        )

else:

    st.info(
        "Extract financial information from a PDF "
        "to generate your financial dashboard."
    )


# --------------------------------------------------
# 9. SEMANTIC RAG
# --------------------------------------------------

st.divider()

st.header("🧠 Semantic Financial Document Search")

semantic_question = st.text_input(
    "Ask a question using semantic search",
    key="semantic_question",
)

if st.button(
    "Search with Semantic RAG",
    disabled=not bool(st.session_state.semantic_document_id),
):

    if not semantic_question.strip():

        st.warning(
            "Please enter a question."
        )

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
                    top_k=3,
                )

                st.session_state.semantic_result = semantic_result

        except Exception as error:

            st.error(
                f"Semantic RAG failed: {error}"
            )


if st.session_state.semantic_result:

    st.subheader("Semantic AI Answer")

    st.write(
        st.session_state.semantic_result["answer"]
    )

    with st.expander("View semantic search sources"):

        for source in st.session_state.semantic_result["sources"]:

            st.write(
                f"Page: {source['page']}"
            )

            st.write(
                f"Cosine distance: {source['distance']:.4f}"
            )

            st.write(
                source["text"]
            )

            st.divider()


# --------------------------------------------------
# 10. HYBRID RETRIEVAL
# --------------------------------------------------

st.divider()

st.header("🔀 Hybrid Financial Document Search")

hybrid_question = st.text_input(
    "Ask a question using keyword + semantic search",
    key="hybrid_question",
)

if st.button(
    "Run Hybrid Search",
    disabled=not bool(st.session_state.semantic_document_id),
):

    if not hybrid_question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Combining keyword and semantic retrieval..."
            ):

                hybrid_results = hybrid_search_document(
                    client,
                    hybrid_question,
                    st.session_state.semantic_document_id,
                    st.session_state.document_chunks,
                    top_k=3,
                )

                st.session_state.hybrid_results = hybrid_results

        except Exception as error:

            st.error(
                f"Hybrid search failed: {error}"
            )


if st.session_state.hybrid_results:

    st.subheader("Hybrid Search Results")

    for result in st.session_state.hybrid_results:

        st.write(
            f"Page: {result['page']}"
        )

        st.write(
            f"RRF score: {result['rrf_score']:.5f}"
        )

        st.write(
            "Retrieved by: " + ", ".join(result["methods"])
        )

        st.write(
            result["text"]
        )

        st.divider()


# --------------------------------------------------
# 11. EXPANDED HYBRID RETRIEVAL + AI ANSWERING
# --------------------------------------------------

st.divider()

st.header("🚀 Advanced Financial Document Q&A")

advanced_question = st.text_input(
    "Ask a question using expanded hybrid retrieval",
    key="advanced_question",
)

if st.button(
    "Search and Answer",
    disabled=not bool(st.session_state.semantic_document_id),
):

    if not advanced_question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Expanding query and searching financial document..."
            ):

                results = expanded_hybrid_search(
                    client,
                    MODEL,
                    advanced_question,
                    st.session_state.semantic_document_id,
                    st.session_state.document_chunks,
                    top_k=3,
                )

                if not results:
                    raise ValueError(
                        "No relevant passages were retrieved."
                    )

                context = "\n\n".join(
                    [
                        f"Page {item['page']}:\n{item['text']}"
                        for item in results
                    ]
                )

            with st.spinner(
                "Qwen is preparing your answer..."
            ):

                response = client.chat(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a financial document assistant. "
                                "Answer only using the supplied document passages. "
                                "Do not invent financial figures. "
                                "If information is unavailable, say so. "
                                "Distinguish balances, income, expenses, "
                                "interest rates and percentages. "
                                "Mention the relevant page numbers. "
                                "Do not provide unsupported financial advice."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {advanced_question}\n\n"
                                f"Document passages:\n{context}"
                            ),
                        },
                    ],
                )

                st.session_state.advanced_answer = (
                    response["message"]["content"]
                )

                st.session_state.advanced_sources = results

        except Exception as error:

            st.error(
                f"Advanced document analysis failed: {error}"
            )


if st.session_state.advanced_answer:

    st.subheader("AI Financial Analysis")

    st.write(
        st.session_state.advanced_answer
    )

    with st.expander("View retrieved evidence"):

        for item in st.session_state.advanced_sources:

            st.write(
                f"Page: {item['page']}"
            )

            st.write(
                item["text"]
            )

            st.caption(
                f"RRF score: {item['rrf_score']:.5f} | "
                f"Methods: {item['methods']}"
            )

            st.divider()


# --------------------------------------------------
# 12. SIDEBAR STATUS
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("Application Status")

st.sidebar.write(
    f"Model: {MODEL}"
)

st.sidebar.write(
    f"Document loaded: {'Yes' if st.session_state.document_text else 'No'}"
)

st.sidebar.write(
    f"Financial data extracted: {'Yes' if st.session_state.financial_data else 'No'}"
)

if st.sidebar.button(
    "Clear Current Document",
    use_container_width=True,
):

    for key, default_value in DEFAULT_STATE.items():
        st.session_state[key] = default_value

    st.rerun()
