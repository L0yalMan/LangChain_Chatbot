import os
import re
from typing import List, Dict, Any, Optional

from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

from src.core.config_pinecone import (
    RETRIEVAL_CONFIG,
    get_embeddings,
    get_pinecone_client,
    update_vector_store_globals,
    Settings
)

index = None

def load_existing_vector_store(user_id: str, index_name: str, embeddings_instance):
    """
    Load existing vector store from Pinecone for a specific user.
    Returns the vector store if it exists, None otherwise.
    """
    try:
        pinecone_client = get_pinecone_client()
        if not pinecone_client:
            print("ERROR: Pinecone client not available for loading vector store")
            return None

        # Try to get the index directly - this will fail if it doesn't exist
        try:
            index = pinecone_client.Index(index_name)
            print(f"Successfully connected to existing index '{index_name}'")
        except Exception as e:
            print(f"Index '{index_name}' does not exist or is not accessible: {e}")
            return None
        
        # Check if there are any vectors in the user's namespace
        try:
            stats = index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            user_namespace_stats = namespaces.get(user_id, {})
            vector_count = user_namespace_stats.get('vector_count', 0)
            
            if vector_count == 0:
                print(f"No existing vectors found for user {user_id} in namespace {user_id}")
                return None
                
            print(f"Found {vector_count} existing vectors for user {user_id}")
        except Exception as e:
            print(f"WARNING: Could not check namespace stats: {e}")
            # Continue anyway, as the vector store might still work

        # Create vector store instance
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings_instance,
            text_key="text",
            namespace=user_id
        )
        
        print(f"Successfully loaded existing vector store for user {user_id}")
        return vectorstore
        
    except Exception as e:
        print(f"ERROR: Failed to load existing vector store for user {user_id}: {e}")
        return None

def check_vector_store_exists(user_id: str, index_name: str) -> bool:
    """
    Check if a vector store exists for a specific user.
    Returns True if vectors exist for the user, False otherwise.
    """
    try:
        pinecone_client = get_pinecone_client()
        if not pinecone_client:
            return False

        # Try to get the index directly - this will fail if it doesn't exist
        try:
            index = pinecone_client.Index(index_name)
        except Exception as e:
            print(f"Index '{index_name}' does not exist or is not accessible: {e}")
            return False
        
        # Check if there are any vectors in the user's namespace
        try:
            stats = index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            user_namespace_stats = namespaces.get(user_id, {})
            vector_count = user_namespace_stats.get('vector_count', 0)
            
            return vector_count > 0
            
        except Exception as e:
            print(f"WARNING: Could not check namespace stats: {e}")
            return False
        
    except Exception as e:
        print(f"ERROR: Failed to check vector store existence for user {user_id}: {e}")
        return False

def ensure_index_exists(index_name: str, pinecone_client, settings) -> bool:
    """
    Ensures that a Pinecone index exists, creating it if necessary.
    Returns True if the index exists and is accessible, False otherwise.
    """
    try:
        # First, try to connect to the index
        try:
            index = pinecone_client.Index(index_name)
            print(f"Successfully connected to existing index '{index_name}'")
            return True
        except Exception as e:
            print(f"Index '{index_name}' does not exist, creating new one...")
        
        # Create the index
        try:
            cloud = "aws"
            region = settings.PINECONE_ENVIRONMENT
            
            pinecone_client.create_index(
                name=index_name,
                dimension=768,  # Gemini embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"Index '{index_name}' created successfully.")
            
            # Wait a moment for the index to be ready
            import time
            time.sleep(2)
            
            # Verify the index was created
            index = pinecone_client.Index(index_name)
            print(f"Successfully verified index '{index_name}' creation")
            return True
            
        except Exception as create_error:
            if "ALREADY_EXISTS" in str(create_error) or "409" in str(create_error):
                print(f"Index '{index_name}' already exists, connecting to it...")
                index = pinecone_client.Index(index_name)
                return True
            else:
                print(f"ERROR: Failed to create index '{index_name}': {create_error}")
                return False
                
    except Exception as e:
        print(f"ERROR: Failed to ensure index exists: {e}")
        return False

def create_advanced_retriever(vectorstore_instance: PineconeVectorStore):
    """Create an advanced retriever with multiple search strategies."""
    from src.core.config_pinecone import llm
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
        except Exception as e_fallback:
            print(f"CRITICAL ERROR: Failed to create even a basic retriever: {e_fallback}")
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

def initialize_vector_store(user_id: str):
    """
    Initializes a Pinecone vector store and retriever for a specific user.
    First tries to load existing vector store, then creates new one if needed.
    """
    settings = Settings()
    index_name = settings.PINECONE_INDEX_NAME
    pinecone_client = get_pinecone_client()
    embeddings_instance = get_embeddings()

    if not pinecone_client or not embeddings_instance or not index_name:
        print("ERROR: Pinecone client, embeddings, or index name not available.")
        update_vector_store_globals(user_id, None, None)
        return

    try:
        # First, try to load existing vector store
        print(f"Attempting to load existing vector store for user {user_id}...")
        existing_vectorstore = load_existing_vector_store(user_id, index_name, embeddings_instance)
        
        if existing_vectorstore:
            print(f"Successfully loaded existing vector store for user {user_id}")
            new_vectorstore = existing_vectorstore
        else:
            print(f"No existing vector store found for user {user_id}. Creating new one...")
            
            # Ensure the index exists
            if not ensure_index_exists(index_name, pinecone_client, settings):
                print(f"ERROR: Failed to ensure index '{index_name}' exists")
                update_vector_store_globals(user_id, None, None)
                return
            
            # Get the index
            global index
            index = pinecone_client.Index(index_name)
            
            # Create new vector store
            new_vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings_instance,
                text_key="text",
                namespace=user_id
            )
            print(f"New vector store created for user {user_id}")

        # Get collection count
        try:
            # Get the index for stats
            stats_index = pinecone_client.Index(index_name)
            stats = stats_index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            user_namespace_stats = namespaces.get(user_id, {})
            collection_count = user_namespace_stats.get('vector_count', 0)
            print(f"..........>>>>>>>>>>>>>There are {collection_count} chunks in the vector store for user {user_id} and {RETRIEVAL_CONFIG['default_k']} chunks~~~~<<<<<<<<<-------")
        except Exception as e:
            print(f"ERROR: Failed to get collection count: {e}")
            collection_count = 0

        # Create retriever
        new_retriever = create_advanced_retriever(new_vectorstore)
        if new_retriever is None:
            print(f"ERROR: Failed to create retriever for user {user_id}")
            update_vector_store_globals(user_id, new_vectorstore, None)
            return

        update_vector_store_globals(user_id, new_vectorstore, new_retriever)
        print(f"--- VECTOR STORE INITIALIZED for user {user_id} ---")
        print(f"Total chunks in vector store: {collection_count}")
        print(f"Advanced retriever configured with MMR search")

    except Exception as e:
        print(f"--- CRITICAL ERROR LOADING VECTOR STORE for user {user_id}: {e} ---")
        update_vector_store_globals(user_id, None, None)
