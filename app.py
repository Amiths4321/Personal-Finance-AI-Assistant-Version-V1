
import streamlit as st
import os

from ollama import Client

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


# Document preview

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
