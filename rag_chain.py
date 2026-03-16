from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
import tempfile

# -------------------------
# Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Local LLM
# -------------------------

llm = Ollama(model="llama3")

# -------------------------
# Memory Vector Store
# -------------------------

memory_store = FAISS.from_texts(
    ["initial memory"],
    embedding=embeddings
)

memory_retriever = memory_store.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Create vectorstore
# -------------------------

def create_vectorstore(uploaded_files):

    docs = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name

        docs.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 40}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    return vector_retriever, bm25_retriever


# -------------------------
# Query Rewriting
# -------------------------

def rewrite_query(query):

    memory_docs = memory_retriever.invoke(query)

    history = "\n".join([doc.page_content for doc in memory_docs])

    prompt = f"""
Rewrite the user's question so it becomes a clear standalone search query.

Conversation History:
{history}

User Question:
{query}

Rewritten Search Query:
"""

    rewritten = llm.invoke(prompt)

    return str(rewritten).strip()


# -------------------------
# Agent Decision
# -------------------------

def agent_decision(query):

    prompt = f"""
You are an AI assistant deciding how to answer a question.

Choose one action:

SEARCH_DOCUMENTS
USE_MEMORY
ASK_CLARIFICATION
DIRECT_ANSWER

Question:
{query}

Action:
"""

    decision = llm.invoke(prompt)

    return str(decision).strip()


# -------------------------
# Hybrid Retrieval
# -------------------------

def hybrid_retrieval(vector_retriever, bm25_retriever, query):

    vector_docs = vector_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    docs = vector_docs + bm25_docs

    unique_docs = []
    seen = set()

    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs


# -------------------------
# Reranking
# -------------------------

def rerank_documents(query, docs):

    prompt = f"""
Rank the following document chunks by relevance.

Question:
{query}

Documents:
"""

    for i, doc in enumerate(docs):
        prompt += f"\n{i+1}. {doc.page_content[:300]}"

    prompt += "\nReturn the numbers of the 3 most relevant documents."

    result = llm.invoke(prompt)

    text = str(result)

    indices = []

    for char in text:
        if char.isdigit():
            indices.append(int(char)-1)

    top_docs = []

    for i in indices[:3]:
        if i < len(docs):
            top_docs.append(docs[i])

    if not top_docs:
        top_docs = docs[:3]

    return top_docs


# -------------------------
# Hallucination Check
# -------------------------

def verify_answer(question, context, answer):

    prompt = f"""
Check if the answer is supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond only with:
SUPPORTED
or
UNSUPPORTED
"""

    result = llm.invoke(prompt)

    return str(result).strip()


# -------------------------
# Ask Question
# -------------------------

def ask_question(vector_retriever, bm25_retriever, query):

    action = agent_decision(query)

    # -------------------------
    # Direct Answer
    # -------------------------

    if "DIRECT_ANSWER" in action:

        response = llm.invoke(query)

        return str(response), [], query

    # -------------------------
    # Clarification
    # -------------------------

    if "ASK_CLARIFICATION" in action:

        return "Please clarify your question.", [], query

    # -------------------------
    # Retrieval Flow
    # -------------------------

    rewritten_query = rewrite_query(query)

    docs = hybrid_retrieval(
        vector_retriever,
        bm25_retriever,
        rewritten_query
    )

    docs = rerank_documents(rewritten_query, docs)

    # Build context
    docs_by_source = {}

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")

        if source not in docs_by_source:
            docs_by_source[source] = []

        docs_by_source[source].append(doc.page_content)

    doc_context = ""

    for source, contents in docs_by_source.items():

        doc_context += f"\nDocument: {source}\n"
        doc_context += "\n".join(contents)

    # Conversation memory
    memory_docs = memory_retriever.invoke(query)

    memory_context = "\n\n".join([doc.page_content for doc in memory_docs])

    prompt = f"""
You are a helpful AI assistant.
Cite the document name when answering.
You must answer only using the provided document context or memory context

If multiple documents contain relevant information,
summarize each document separately.
Always mention which document the information comes from.

If the answer is not found in the document, then say:
'I can't find the answer in the uploaded documents.'
Do not use outside knowledge.

Conversation History:
{memory_context}

Document Context:
{doc_context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    response_text = str(response)

    # Self reflection
    verification = verify_answer(query, doc_context, response_text)

    if "UNSUPPORTED" in verification:
        response_text = "I could not verify this answer from the provided documents."

    # Save to memory
    memory_text = f"User: {query}\nAI: {response_text}"

    memory_store.add_documents(
        [Document(page_content=memory_text)]
    )

    # Extract sources
    sources = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        sources.append(f"{source} (Page {page})")

    return response_text, list(set(sources)), rewritten_query