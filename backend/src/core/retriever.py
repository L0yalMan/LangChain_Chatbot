import os
import re
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

from src.core.config import RETRIEVAL_CONFIG, CHROMA_PATH, get_embeddings, update_vector_store_globals

def create_advanced_retriever(vectorstore_instance):
    """Create an advanced retriever with multiple search strategies."""
    from src.core.config import llm
    try:
        if vectorstore_instance is None:
            raise ValueError("Vector store instance is None")

        base_retriever = vectorstore_instance.as_retriever(
            search_type="mmr",  # Use Max Marginal Relevance for diversity
            search_kwargs={
                'k': RETRIEVAL_CONFIG['mmr_k'],
                'fetch_k': RETRIEVAL_CONFIG['mmr_k'] * 2,  # Fetch more for MMR selection
                'lambda_mult': RETRIEVAL_CONFIG['mmr_lambda'],
                'score_threshold': RETRIEVAL_CONFIG['similarity_threshold']
            }
        )

        # Create a multi-query retriever for query expansion
        if llm is None:
            print("WARNING: LLM is None, using base retriever only")
            return base_retriever

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        return multi_query_retriever
    except Exception as e:
        print(f"ERROR: Failed to create advanced retriever: {e}")
        # Fallback to basic retriever
        try:
            fallback_retriever = vectorstore_instance.as_retriever(
                search_type="similarity",
                search_kwargs={'k': RETRIEVAL_CONFIG['default_k']}
            )
            return fallback_retriever
        except Exception as fallback_error:
            print(f"ERROR: Failed to create fallback retriever: {fallback_error}")
            return None

def preprocess_query(query: str) -> str:
    """Preprocess the query to improve retrieval accuracy."""
    try:
        if not query or not isinstance(query, str):
            return ""

        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())

        if not query:
            return ""

        # Extract key terms (simple approach - can be enhanced with NLP)
        # Remove common stop words and focus on content words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

        # Keep important words and phrases
        words = query.lower().split()
        important_words = [word for word in words if word not in stop_words and len(word) > 2]

        # If we have important words, use them to enhance the query
        if important_words:
            enhanced_query = f"{query} {' '.join(important_words[:3])}"  # Add top 3 important words
            return enhanced_query

        return query
    except Exception as e:
        print(f"ERROR: Failed to preprocess query: {e}")
        return query if isinstance(query, str) else ""

def evaluate_retrieval_quality(query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    """Evaluate the quality of retrieved documents and provide feedback."""
    try:
        evaluation = {
            'query_length': len(query) if query else 0,
            'num_docs_retrieved': len(retrieved_docs) if retrieved_docs else 0,
            'avg_doc_length': 0,
            'coverage_score': 0,
            'diversity_score': 0,
            'recommendations': []
        }

        if not retrieved_docs:
            evaluation['recommendations'].append("No documents retrieved - consider lowering similarity threshold")
            return evaluation

        # Calculate average document length
        try:
            doc_lengths = [len(doc.page_content) for doc in retrieved_docs if hasattr(doc, 'page_content')]
            if doc_lengths:
                evaluation['avg_doc_length'] = sum(doc_lengths) / len(doc_lengths)
        except Exception as e:
            print(f"ERROR: Failed to calculate document lengths: {e}")
            evaluation['avg_doc_length'] = 0

        # Calculate coverage (total content retrieved)
        try:
            total_content = sum(doc_lengths) if doc_lengths else 0
            evaluation['coverage_score'] = min(total_content / 5000, 1.0)  # Normalize to 0-1
        except Exception as e:
            print(f"ERROR: Failed to calculate coverage score: {e}")
            evaluation['coverage_score'] = 0

        # Calculate diversity (unique content)
        try:
            unique_content = set()
            for doc in retrieved_docs:
                if hasattr(doc, 'page_content'):
                    words = doc.page_content.lower().split()
                    unique_content.update(words[:50])  # First 50 words per doc

            evaluation['diversity_score'] = len(unique_content) / (len(retrieved_docs) * 50) if retrieved_docs else 0
        except Exception as e:
            print(f"ERROR: Failed to calculate diversity score: {e}")
            evaluation['diversity_score'] = 0

        # Generate recommendations
        if evaluation['num_docs_retrieved'] < 3:
            evaluation['recommendations'].append("Consider increasing k value for better coverage")

        if evaluation['avg_doc_length'] < 500:
            evaluation['recommendations'].append("Documents seem short - consider increasing chunk size")

        if evaluation['coverage_score'] < 0.3:
            evaluation['recommendations'].append("Low coverage - consider adjusting retrieval parameters")

        if evaluation['diversity_score'] < 0.5:
            evaluation['recommendations'].append("Low diversity - consider using MMR search")

        return evaluation
    except Exception as e:
        print(f"ERROR: Failed to evaluate retrieval quality: {e}")
        return {
            'query_length': 0,
            'num_docs_retrieved': 0,
            'avg_doc_length': 0,
            'coverage_score': 0,
            'diversity_score': 0,
            'recommendations': ["Error occurred during evaluation"]
        }

def initialize_vector_store():
    print(f"------->>>>> Initialize_Vector Store")
    print(f"Chroma path exists: {os.path.exists(CHROMA_PATH)}")
    try:
        if os.path.exists(CHROMA_PATH):
            print(f"Files in vector store directory: {os.listdir(CHROMA_PATH)}")
        else:
            print(f"--- NO EXISTING VECTOR STORE FOUND AT {CHROMA_PATH} ---")
            update_vector_store_globals(None, None)
            return
    except Exception as e:
        print(f"ERROR: Could not check Chroma path or list files: {e}")
        update_vector_store_globals(None, None)
        return

    try:
        embeddings_instance = get_embeddings()
        if embeddings_instance is None:
            print("ERROR: Embeddings not initialized, cannot create vector store")
            print("Please ensure GOOGLE_API_KEY environment variable is set and initialize_llm_embeddings() was called successfully.")
            update_vector_store_globals(None, None)
            return

        print(f"------->>>>> Loading vectorstore")
        try:
            new_vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_instance)
            print(f"VECTORSTORE WAS INITIALIZED SUCCESSFULLY!!!!!")
        except Exception as e:
            print(f"ERROR: Failed to load Chroma vector store: {e}")
            update_vector_store_globals(None, None)
            return

        try:
            collection_count = new_vectorstore._collection.count()
            print(f"..........>>>>>>>>>>>>>There are {collection_count} chunks in the vector store and {RETRIEVAL_CONFIG['default_k']} chunks~~~~<<<<<<<<<-------")
        except Exception as e:
            print(f"ERROR: Failed to get collection count: {e}")
            update_vector_store_globals(None, None)
            return

        new_retriever = create_advanced_retriever(new_vectorstore)
        if new_retriever is None:
            print("ERROR: Failed to create retriever during initialization")
            update_vector_store_globals(None, None) # Reset vectorstore if retriever can't be made
            return

        # Update global variables
        update_vector_store_globals(new_vectorstore, new_retriever)

        print(f"--- VECTOR STORE INITIALIZED ---")
        print(f"Total chunks in vector store: {collection_count}")
        print(f"Advanced retriever configured with MMR search")

    except Exception as e:
        print(f"--- CRITICAL ERROR LOADING VECTOR STORE: {e} ---")
        update_vector_store_globals(None, None)