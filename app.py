import streamlit as st
from rag_chain import create_vectorstore, ask_question

st.title("AI Document Assistant")

uploaded_files = st.file_uploader(
    "Upload your PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Process documents
if uploaded_files:

    if "vector_retriever" not in st.session_state:

        with st.spinner("Processing documents..."):

            vector_retriever, bm25_retriever = create_vectorstore(uploaded_files)

            st.session_state.vector_retriever = vector_retriever
            st.session_state.bm25_retriever = bm25_retriever

        st.success("Documents processed!")

# Ask question
if "vector_retriever" in st.session_state:

    query = st.text_input("Ask a question about your documents")

    if query:

        response, sources, rewritten_query = ask_question(
            st.session_state.vector_retriever,
            st.session_state.bm25_retriever,
            query
        )

        st.write("### Rewritten Query")
        st.write(rewritten_query)

        st.write("### Answer")
        st.write(response)

        st.write("### Sources")

        for s in set(sources):
            st.write(s)