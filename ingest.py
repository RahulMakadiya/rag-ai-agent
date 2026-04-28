import os
import sys
import logging
import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
DATA_DIR = os.getenv("DATA_DIR", "./data")


def validate_pdf(pdf_path):
    """Validate that the PDF file exists and is readable."""
    try:
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        if path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")
        
        logger.info(f"✓ PDF validated: {pdf_path} ({path.stat().st_size / 1024:.2f} KB)")
        return True
    
    except Exception as e:
        logger.error(f"✗ PDF validation failed: {str(e)}")
        return False


def load_pdf(pdf_path):
    """Load and parse PDF document."""
    try:
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No documents loaded from PDF")
        
        logger.info(f"✓ Loaded {len(documents)} pages from PDF")
        return documents
    
    except Exception as e:
        logger.error(f"✗ Failed to load PDF: {str(e)}")
        raise


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks."""
    try:
        logger.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    except Exception as e:
        logger.error(f"✗ Failed to split documents: {str(e)}")
        raise


def create_embeddings():
    """Create embeddings model."""
    try:
        logger.info("Loading embeddings model: sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("✓ Embeddings model loaded")
        return embeddings
    
    except Exception as e:
        logger.error(f"✗ Failed to load embeddings: {str(e)}")
        raise


def create_vectorstore(chunks, embeddings):
    """Create FAISS vectorstore from chunks."""
    try:
        logger.info(f"Creating vectorstore with {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        logger.info("✓ Vectorstore created successfully")
        return vectorstore
    
    except Exception as e:
        logger.error(f"✗ Failed to create vectorstore: {str(e)}")
        raise


def save_vectorstore(vectorstore, path):
    """Save vectorstore to disk."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving vectorstore to {path}...")
        
        vectorstore.save_local(path)
        logger.info(f"✓ Vectorstore saved successfully to {path}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to save vectorstore: {str(e)}")
        raise


def merge_vectorstore(existing_vectorstore, new_chunks, embeddings):
    """Merge new chunks into existing vectorstore."""
    try:
        logger.info(f"Merging {len(new_chunks)} new chunks into existing vectorstore...")
        
        # Create new vectorstore from chunks
        new_vectorstore = FAISS.from_documents(new_chunks, embeddings)
        
        # Merge
        existing_vectorstore.merge_from(new_vectorstore)
        logger.info("✓ Vectorstore merged successfully")
        
        return existing_vectorstore
    
    except Exception as e:
        logger.error(f"✗ Failed to merge vectorstore: {str(e)}")
        raise


def ingest_pdf(pdf_path, vectorstore_path, merge=False, chunk_size=500, chunk_overlap=50):
    """Main ingestion pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting PDF ingestion...")
        logger.info("=" * 60)
        
        # Validate PDF
        if not validate_pdf(pdf_path):
            sys.exit(1)
        
        # Load PDF
        documents = load_pdf(pdf_path)
        
        # Split documents
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        
        # Create embeddings
        embeddings = create_embeddings()
        
        # Check if vectorstore exists and merge or overwrite
        vectorstore_exists = os.path.exists(vectorstore_path)
        
        if vectorstore_exists and merge:
            logger.info("Loading existing vectorstore for merging...")
            try:
                existing_vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                vectorstore = merge_vectorstore(existing_vectorstore, chunks, embeddings)
            except Exception as e:
                logger.warning(f"Failed to load existing vectorstore: {str(e)}. Creating new one.")
                vectorstore = create_vectorstore(chunks, embeddings)
        else:
            if vectorstore_exists:
                logger.info("Existing vectorstore found. Overwriting...")
            vectorstore = create_vectorstore(chunks, embeddings)
        
        # Save vectorstore
        save_vectorstore(vectorstore, vectorstore_path)
        
        logger.info("=" * 60)
        logger.info("✓ PDF ingestion completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ PDF ingestion failed: {str(e)}")
        logger.error("=" * 60)
        sys.exit(1)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into a FAISS vectorstore"
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to ingest"
    )
    
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing vectorstore instead of overwriting"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Size of text chunks (default: 500)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )
    
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        default=VECTORSTORE_PATH,
        help=f"Path to save vectorstore (default: {VECTORSTORE_PATH})"
    )
    
    args = parser.parse_args()
    
    # Run ingestion with specified vectorstore path
    ingest_pdf(
        args.pdf_path,
        args.vectorstore_path,
        merge=args.merge,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )


if __name__ == "__main__":
    main()