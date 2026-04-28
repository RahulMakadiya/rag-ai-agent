from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
import requests
import json
import tempfile
import os
import logging
import re
from functools import wraps
import time

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Local LLM
# -------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

class OllamaLLM:
    def __init__(self, model, host):
        self.model = model
        self.host = host.rstrip("/")

    def invoke(self, prompt):
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, prompt):
        with requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            },
            stream=True,
            timeout=300
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if chunk.get("message", {}).get("content"):
                        yield chunk["message"]["content"]

llm = OllamaLLM(model=OLLAMA_MODEL, host=OLLAMA_BASE_URL)

# -------------------------
# Retry Decorator
# -------------------------

def retry_on_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator

# -------------------------
# Persistence Helpers
# -------------------------

VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
MEMORY_STORE_PATH = os.getenv("MEMORY_STORE_PATH", "./data/memory_store")

def save_vectorstore(vectorstore, path=VECTORSTORE_PATH):
    """Save vectorstore to disk."""
    try:
        os.makedirs(path, exist_ok=True)
        vectorstore.save_local(path)
        logger.info(f"Vectorstore saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save vectorstore: {str(e)}")

def load_vectorstore(path=VECTORSTORE_PATH):
    """Load vectorstore from disk."""
    try:
        if os.path.exists(path):
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vectorstore loaded from {path}")
            return vectorstore
    except Exception as e:
        logger.error(f"Failed to load vectorstore: {str(e)}")
    return None

def save_memory_store(memory_store, path=MEMORY_STORE_PATH):
    """Save memory store to disk."""
    try:
        os.makedirs(path, exist_ok=True)
        memory_store.save_local(path)
        logger.info(f"Memory store saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save memory store: {str(e)}")

def load_memory_store(path=MEMORY_STORE_PATH):
    """Load memory store from disk or create new."""
    try:
        if os.path.exists(path):
            memory_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Memory store loaded from {path}")
            return memory_store
    except Exception as e:
        logger.error(f"Failed to load memory store: {str(e)}")
    
    # Create new if doesn't exist
    memory_store = FAISS.from_texts(
        ["initial memory"],
        embedding=embeddings
    )
    return memory_store

# -------------------------
# Memory Vector Store
# -------------------------

memory_store = load_memory_store()
memory_retriever = memory_store.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Create vectorstore
# -------------------------

@retry_on_error(max_retries=3, delay=1)
def create_vectorstore(uploaded_files):
    """Create and persist vectorstore from uploaded PDF files."""
    
    try:
        docs = []
        temp_paths = []

        for uploaded_file in uploaded_files:
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                try:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                finally:
                    tmp.close()

                temp_paths.append(tmp_path)

                loader = PyPDFLoader(tmp_path)
                loaded_docs = loader.load()

                for doc in loaded_docs:
                    doc.metadata["source"] = uploaded_file.name

                docs.extend(loaded_docs)
            except Exception as e:
                logger.error(f"Failed to process file {uploaded_file.name}: {str(e)}")

        if not docs:
            raise ValueError("No valid documents were loaded")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )
        
        # Save vectorstore to disk
        save_vectorstore(vectorstore)

        vector_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 40}
        )

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10

        return vector_retriever, bm25_retriever
    
    finally:
        # Clean up temp files
        for tmp_path in temp_paths:
            try:
                os.unlink(tmp_path)
                logger.info(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {str(e)}")


# -------------------------
# Query Rewriting
# -------------------------

@retry_on_error(max_retries=3, delay=1)
def rewrite_query(query):
    """Rewrite query using conversation history."""
    
    try:
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
    
    except Exception as e:
        logger.error(f"Failed to rewrite query: {str(e)}")
        return query


# -------------------------
# Agent Decision
# -------------------------

@retry_on_error(max_retries=3, delay=1)
def agent_decision(query):
    """Decide how to answer the query."""
    
    try:
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

        decision = str(llm.invoke(prompt)).strip().upper()
        
        # Validate response
        valid_actions = ["SEARCH_DOCUMENTS", "USE_MEMORY", "ASK_CLARIFICATION", "DIRECT_ANSWER"]
        if any(action in decision for action in valid_actions):
            return decision
        
        # Default to SEARCH_DOCUMENTS if unclear
        logger.warning(f"Unclear decision response: {decision}. Defaulting to SEARCH_DOCUMENTS")
        return "SEARCH_DOCUMENTS"
    
    except Exception as e:
        logger.error(f"Failed to determine agent action: {str(e)}")
        return "SEARCH_DOCUMENTS"


# -------------------------
# Hybrid Retrieval
# -------------------------

@retry_on_error(max_retries=3, delay=1)
def hybrid_retrieval(vector_retriever, bm25_retriever, query):
    """Combine vector and BM25 retrieval results."""
    
    try:
        vector_docs = vector_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        docs = vector_docs + bm25_docs

        unique_docs = []
        seen = set()

        for doc in docs:
            doc_id = (doc.page_content, doc.metadata.get("source", ""))
            if doc_id not in seen:
                unique_docs.append(doc)
                seen.add(doc_id)

        logger.info(f"Retrieved {len(unique_docs)} unique documents for query")
        return unique_docs
    
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {str(e)}")
        return []


# -------------------------
# Reranking
# -------------------------

@retry_on_error(max_retries=3, delay=1)
def rerank_documents(query, docs):
    """Rerank documents by relevance using LLM."""
    
    if not docs:
        return []
    
    try:
        prompt = f"""
Rank the following document chunks by relevance to the question.

Question:
{query}

Documents:
"""

        for i, doc in enumerate(docs):
            prompt += f"\n{i+1}. {doc.page_content[:300]}"

        prompt += "\nReturn the numbers of the 3 most relevant documents as a comma-separated list (e.g., 1,3,2)."

        result = str(llm.invoke(prompt))

        # Extract numbers more robustly
        indices = re.findall(r'\d+', result)
        indices = [int(idx) - 1 for idx in indices if idx.isdigit()]

        top_docs = []
        for i in indices[:3]:
            if 0 <= i < len(docs):
                top_docs.append(docs[i])

        if not top_docs:
            top_docs = docs[:3]

        logger.info(f"Reranked to {len(top_docs)} documents")
        return top_docs
    
    except Exception as e:
        logger.error(f"Failed to rerank documents: {str(e)}")
        return docs[:3]


# -------------------------
# Hallucination Check
# -------------------------

@retry_on_error(max_retries=3, delay=1)
def verify_answer(question, context, answer):
    """Verify if the answer is supported by the context."""
    
    try:
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

        result = str(llm.invoke(prompt)).strip().upper()
        
        return result if result in ["SUPPORTED", "UNSUPPORTED"] else "SUPPORTED"
    
    except Exception as e:
        logger.error(f"Failed to verify answer: {str(e)}")
        return "SUPPORTED"


# -------------------------
# Pipeline Helpers
# -------------------------

def get_memory_context(query):
    """Retrieve relevant conversation history."""
    try:
        memory_docs = memory_retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in memory_docs])
    except Exception as e:
        logger.warning(f"Failed to retrieve memory: {str(e)}")
        return ""

def build_doc_context(docs):
    """Build formatted context string from docs grouped by source."""
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
    return doc_context

def build_answer_prompt(query, doc_context, memory_context):
    """Build the final answer prompt."""
    return f"""
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

def extract_sources(docs):
    """Extract source filenames and page numbers from docs."""
    sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        sources.add(f"{source} (Page {page})")
    return list(sources)

def stream_response(prompt):
    """Generator: stream LLM response, fall back to invoke if streaming fails."""
    try:
        for chunk in llm.stream(prompt):
            yield str(chunk)
    except Exception as e:
        logger.error(f"Streaming failed ({type(e).__name__}: {str(e)}), falling back to invoke")
        try:
            response = llm.invoke(prompt)
            yield str(response)
        except Exception as e2:
            logger.error(f"Invoke also failed: {str(e2)}")
            yield "Unable to generate answer at this time."

def save_to_memory(query, response):
    """Save Q&A pair to conversation memory."""
    try:
        memory_text = f"User: {query}\nAI: {response}"
        memory_store.add_documents([Document(page_content=memory_text)])
        save_memory_store(memory_store)
    except Exception as e:
        logger.warning(f"Failed to save to memory: {str(e)}")