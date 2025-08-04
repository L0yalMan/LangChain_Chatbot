from typing import Tuple, List

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader, UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Try to import docx2txt, but don't fail if it's not available
try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False
    print("WARNING: docx2txt not available. Will use fallback method for Word documents.")

def load_word_document_custom(file_path: str) -> List[Document]:
    """Custom loader for Word documents that tries multiple methods."""
    try:
        # Method 1: Try docx2txt if available
        if DOCX2TXT_AVAILABLE:
            try:
                print(f"Attempting to load {file_path} with docx2txt...")
                text = docx2txt.process(file_path)
                if text and text.strip():
                    return [Document(page_content=text.strip(), metadata={"source": file_path})]
            except Exception as e:
                print(f"docx2txt failed: {e}")
        
        # Method 2: Try Docx2txtLoader
        try:
            print(f"Attempting to load {file_path} with Docx2txtLoader...")
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            if docs:
                print(f"Successfully loaded {len(docs)} documents with Docx2txtLoader")
                return docs
        except Exception as e:
            print(f"Docx2txtLoader failed: {e}")
        
        # Method 3: Try UnstructuredWordDocumentLoader
        try:
            print(f"Attempting to load {file_path} with UnstructuredWordDocumentLoader...")
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            if docs:
                print(f"Successfully loaded {len(docs)} documents with UnstructuredWordDocumentLoader")
                return docs
        except Exception as e:
            print(f"UnstructuredWordDocumentLoader failed: {e}")
        
        # Method 4: Simple text extraction as last resort
        try:
            print(f"Attempting simple text extraction for {file_path}...")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if text and text.strip():
                return [Document(page_content=text.strip(), metadata={"source": file_path})]
        except Exception as e:
            print(f"Simple text extraction failed: {e}")
        
        raise Exception(f"All methods failed to load Word document: {file_path}")
        
    except Exception as e:
        print(f"Error in custom Word document loader: {e}")
        raise

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
    try:
        if file_type.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_type.lower() == ".csv":
            loader = CSVLoader(file_path)
            return loader.load()
        elif file_type.lower() in [".docx", ".doc"]:
            # Use our custom Word document loader
            return load_word_document_custom(file_path)
        else:
            raise ValueError("Unsupported file type. Only PDF, CSV, DOCX, and DOC are supported.")
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        raise

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