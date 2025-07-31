import re
from typing import Tuple, List

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_optimal_chunk_size(content_length: int) -> Tuple[int, int]:
    """Dynamically determine optimal chunk size based on content length."""
    try:
        if not isinstance(content_length, (int, float)) or content_length < 0:
            print(f"WARNING: Invalid content length: {content_length}, using default values")
            return 800, 150

        if content_length < 5000:
            return 800, 150  # Smaller chunks for short content
        elif content_length < 20000:
            return 1200, 250  # Medium chunks for medium content
        else:
            return 1500, 300  # Larger chunks for long content
    except Exception as e:
        print(f"ERROR: Failed to determine optimal chunk size: {e}")
        return 800, 150  # Default fallback values

def load_document(file_path: str, file_type: str) -> List[Document]:
    """Loads a document based on its type."""
    if file_type.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_type.lower() == ".csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and CSV are supported.")
    return loader.load()

def load_website_content(url: str) -> List[Document]:
    """Loads content from a given website URL."""
    loader = WebBaseLoader(web_paths=[url])
    return loader.load()

def split_documents(docs: List[Document], total_content_length: int) -> List[Document]:
    """Splits documents into chunks using a RecursiveCharacterTextSplitter."""
    chunk_size, chunk_overlap = get_optimal_chunk_size(total_content_length)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(docs)