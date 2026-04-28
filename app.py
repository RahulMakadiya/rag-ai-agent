import streamlit as st
import logging
from rag_chain import (
    create_vectorstore,
    agent_decision,
    rewrite_query,
    hybrid_retrieval,
    rerank_documents,
    verify_answer,
    get_memory_context,
    build_doc_context,
    build_answer_prompt,
    extract_sources,
    stream_response,
    save_to_memory,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if "vector_retriever" not in st.session_state:
    st.session_state.vector_retriever = None
    st.session_state.bm25_retriever = None
    st.session_state.documents_processed = False
    st.session_state.processing_error = None

st.set_page_config(page_title="AI Document Assistant", layout="wide")
st.title("AI Document Assistant")

# Sidebar
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if not st.session_state.documents_processed:
            with st.spinner("🔄 Processing documents..."):
                try:
                    logger.info(f"Processing {len(uploaded_files)} files")
                    vector_retriever, bm25_retriever = create_vectorstore(uploaded_files)
                    st.session_state.vector_retriever = vector_retriever
                    st.session_state.bm25_retriever = bm25_retriever
                    st.session_state.documents_processed = True
                    st.session_state.processing_error = None
                    st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                except ValueError as e:
                    error_msg = f"Invalid documents: {str(e)}"
                    st.session_state.processing_error = error_msg
                    st.error(f"❌ {error_msg}")
                except Exception as e:
                    error_msg = f"Error processing documents: {str(e)}"
                    st.session_state.processing_error = error_msg
                    st.error(f"❌ {error_msg}")
        else:
            st.info(f"✅ {len(uploaded_files)} document(s) uploaded and ready!")

    if st.button("🔄 Upload New Documents"):
        st.session_state.vector_retriever = None
        st.session_state.bm25_retriever = None
        st.session_state.documents_processed = False
        st.session_state.processing_error = None
        st.rerun()

# Main area
if st.session_state.vector_retriever and st.session_state.documents_processed:
    st.header("🤖 Ask Questions")

    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic of the document?",
        help="Type your question and press Enter"
    )

    if query and query.strip():
        query = query.strip()

        try:
            action = None
            rewritten_query = query
            docs = []
            sources = []
            doc_context = ""
            prompt = None

            col1, col2 = st.columns([3, 1])

            with col1:
                # --- Pipeline progress ---
                with st.status("Analyzing your question...", expanded=True) as status:

                    st.write("🤔 Deciding how to answer...")
                    action = agent_decision(query)
                    logger.info(f"Agent action: {action}")

                    if "ASK_CLARIFICATION" in action:
                        status.update(label="Need clarification", state="complete")

                    elif "DIRECT_ANSWER" in action:
                        status.update(label="Answering directly...", state="complete")

                    else:
                        st.write("✓ Will search documents")

                        st.write("✏️ Rewriting query for better search...")
                        rewritten_query = rewrite_query(query)
                        st.write(f"✓ Optimized: _{rewritten_query[:80]}_")

                        st.write("📚 Retrieving relevant chunks...")
                        docs = hybrid_retrieval(
                            st.session_state.vector_retriever,
                            st.session_state.bm25_retriever,
                            rewritten_query
                        )
                        st.write(f"✓ Found {len(docs)} chunks")

                        if docs:
                            st.write("🎯 Reranking by relevance...")
                            docs = rerank_documents(rewritten_query, docs)
                            st.write(f"✓ Top {len(docs)} chunks selected")

                            sources = extract_sources(docs)
                            memory_context = get_memory_context(query)
                            doc_context = build_doc_context(docs)
                            prompt = build_answer_prompt(query, doc_context, memory_context)

                        status.update(label="✅ Context ready — generating answer", state="complete")

                # --- Answer output ---
                if "ASK_CLARIFICATION" in action:
                    st.info("Could you clarify your question?")

                elif "DIRECT_ANSWER" in action:
                    st.subheader("💡 Answer")
                    direct_prompt = f"Answer this question:\n\n{query}\n\nAnswer:"
                    full_response = st.write_stream(stream_response(direct_prompt))
                    save_to_memory(query, full_response)

                elif not docs:
                    st.warning("No relevant documents found for your query.")

                else:
                    st.subheader("💡 Answer")
                    full_response = st.write_stream(stream_response(prompt))

                    # Hallucination check after answer is shown
                    with st.status("🔍 Verifying answer...", expanded=False) as verify_status:
                        verification = verify_answer(query, doc_context, full_response)
                        if "UNSUPPORTED" in verification:
                            verify_status.update(label="⚠️ Verification failed", state="error")
                            st.warning("This answer could not be fully verified against the source documents.")
                        else:
                            verify_status.update(label="✅ Answer verified", state="complete")

                    save_to_memory(query, full_response)

            with col2:
                if sources:
                    st.subheader("📚 Sources")
                    for source in sorted(set(sources)):
                        st.caption(source)
                else:
                    st.info("No sources found")

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(f"❌ {error_msg}")
            logger.error(error_msg)

else:
    if st.session_state.processing_error:
        st.error("⚠️ Please fix the error above and try again.")
    else:
        st.info("👈 Upload PDF documents in the sidebar to get started!")
