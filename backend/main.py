import os
from dotenv import load_dotenv
import shutil
import gc
from typing import List, TypedDict, Optional, Dict, Any
import json
from botocore.exceptions import ClientError

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain components
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,  MessagesPlaceholder

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Replace with ...
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph components
from langgraph.graph import END, StateGraph

# Supabase client and s3 client
from src.utils.supabase_client import supabase
from src.utils.s3_client import s3_client

# Additional imports for improved retrieval
import re
from langchain_core.documents import Document
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# --- Global Variables & Setup ---
load_dotenv()

# Set default environment variables to prevent warnings
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "LangChain-RAG-Bot/1.0.0"

# Set additional environment variables to prevent warnings
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
os.environ.setdefault("CURL_CA_BUNDLE", "")

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*USER_AGENT.*")
warnings.filterwarnings("ignore", message=".*unclosed.*")

# You might need to define a Pydantic model for incoming chat messages
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Message] = [] # Defaults to an empty list if not provided

# Define the persistent path for the Chroma database
CHROMA_PATH = "chroma_db"

# LLM and Embeddings with Gemini API
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # You can also use "gemini-1.5-pro" or "gemini-2.0-flash" if available and suitable
        temperature=0,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Recommended embedding model for Gemini
    )
    print("LLM and embeddings initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to initialize LLM or embeddings: {e}")
    llm = None
    embeddings = None

if os.getenv("GOOGLE_API_KEY") is None:
    print("WARNING: GOOGLE_API_KEY environment variable is not set. The application might not work correctly without it.")

# Initialize global variables for the vector store and retriever
vectorstore = None
retriever = None

# Advanced retrieval configuration
RETRIEVAL_CONFIG = {
    'default_k': 8,  # Increased from 6 for better coverage
    'mmr_k': 12,     # Fetch more documents for MMR diversity
    'mmr_lambda': 0.7,  # Balance between relevance and diversity
    'similarity_threshold': 0.5,  # Minimum similarity score
    'max_chunk_size': 1500,  # Larger chunks for better context
    'chunk_overlap': 300,    # Increased overlap for better continuity
}

def create_advanced_retriever(vectorstore_instance):
    """Create an advanced retriever with multiple search strategies."""
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
            return vectorstore_instance.as_retriever(
                search_type="similarity",
                search_kwargs={'k': RETRIEVAL_CONFIG['default_k']}
            )
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

def get_optimal_chunk_size(content_length: int) -> tuple:
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

# --- Startup Logic to Load Existing DB ---
# Try to load the vector store from the persistent directory on startup
def initialize_vector_store():
    global vectorstore, retriever
    print(f"------->>>>> Initialize_Vector Store")
    print(os.path.exists(CHROMA_PATH))
    print(os.listdir(CHROMA_PATH))
    try:
        # Check if embeddings are available
        if embeddings is None:
            print("ERROR: Embeddings not initialized, cannot create vector store")
            vectorstore = None
            retriever = None
            return
        
        # Check if chroma path exists
        if not os.path.exists(CHROMA_PATH):
            print(f"--- NO EXISTING VECTOR STORE FOUND AT {CHROMA_PATH} ---")
            vectorstore = None
            retriever = None
            return
        
        print(f"Vector store directory exists: {os.path.exists(CHROMA_PATH)}")
        
        # List files in the directory
        try:
            files = os.listdir(CHROMA_PATH)
            print(f"Files in vector store directory: {files}")
        except PermissionError as e:
            print(f"PERMISSION ERROR: Cannot access vector store directory: {e}")
            vectorstore = None
            retriever = None
            return
        except Exception as e:
            print(f"ERROR: Could not list files in vector store directory: {e}")
            files = []
        
        # Try to load the vector store
        print(f"------->>>>> Loading vectorstore")
        try:
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        except Exception as e:
            print(f"ERROR: Failed to load Chroma vector store: {e}")
            vectorstore = None
            retriever = None
            return
        
        # Verify the vector store is working
        try:
            collection_count = vectorstore._collection.count()
            print(f"..........>>>>>>>>>>>>>There are {collection_count} chunks in the vector store and {RETRIEVAL_CONFIG['default_k']} chunks~~~~<<<<<<<<<-------")
        except Exception as e:
            print(f"ERROR: Failed to get collection count: {e}")
            vectorstore = None
            retriever = None
            return
        
        # Create retriever
        try:
            retriever = create_advanced_retriever(vectorstore)
            if retriever is None:
                print("ERROR: Failed to create retriever")
                vectorstore = None
                return
        except Exception as e:
            print(f"ERROR: Failed to create retriever: {e}")
            vectorstore = None
            retriever = None
            return
        
        print(f"--- VECTOR STORE INITIALIZED ---")
        print(f"Total chunks in vector store: {collection_count}")
        print(f"Advanced retriever configured with MMR search")
        
    except Exception as e:
        print(f"--- CRITICAL ERROR LOADING VECTOR STORE: {e} ---")
        vectorstore = None
        retriever = None

initialize_vector_store()

# --- Define the State for our RAG Graph ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generation.
        documents: A list of retrieved documents.
    """
    question: str
    generation: str 
    chat_history: List[BaseMessage]
    documents: List[str]

# --- RAG Graph Nodes ---
def retrieve_documents(state):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    # Chat history is not directly used for retrieval, but can be for rephrasing queries
    chat_history = state.get("chat_history", []) 

    global vectorstore, retriever # Ensure we use the global instance

    if not vectorstore:
        print("ERROR: No vector store available")
        return {"documents": [], "question": question, "chat_history": chat_history}

    try:
        # Check if the collection actually exists and has data
        collection_count = vectorstore._collection.count()
        print(f"Collection Count: {collection_count}")

        if collection_count == 0:
            print("No document exists in the collection.")
            return {"documents": [], "question": question, "chat_history": chat_history}
    except Exception as e:
        print(f"ERROR: Failed to get collection count: {e}")
        return {"documents": [], "question": question, "chat_history": chat_history}

    # Preprocess the query for better retrieval
    try:
        processed_question = preprocess_query(question)
        print(f"Original question: '{question}'")
        print(f"Processed question: '{processed_question}'")
    except Exception as e:
        print(f"ERROR: Failed to preprocess question: {e}")
        processed_question = question
    
    # Try advanced retrieval first
    retrieved_docs = []
    try:
        if retriever:
            retrieved_docs = retriever.invoke(processed_question)
            print(f"Advanced retrieval successful with {len(retrieved_docs)} documents")
        else:
            print("WARNING: No retriever available, using basic similarity search")
            retrieved_docs = vectorstore.similarity_search(
                processed_question, 
                k=RETRIEVAL_CONFIG['default_k']
            )
    except Exception as e:
        print(f"Advanced retrieval failed, falling back to basic retrieval: {e}")
    try:
        # Fallback to basic similarity search (this block is redundant if the above `try` block handles the fallback properly)
        retrieved_docs = vectorstore.similarity_search(
            processed_question, 
            k=RETRIEVAL_CONFIG['default_k']
        )
    except Exception as fallback_error:
        print(f"ERROR: Basic retrieval also failed: {fallback_error}")
        return {"documents": [], "question": question, "chat_history": chat_history}
    
    # Filter and rank documents by relevance
    if retrieved_docs:
        try:
            # Get similarity scores for ranking
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                processed_question, 
                k=len(retrieved_docs) * 2  # Get more for better filtering
            )
            
            # Filter by similarity threshold and sort by score
            filtered_docs = [
                (doc, score) for doc, score in docs_with_scores 
                if score >= RETRIEVAL_CONFIG['similarity_threshold']
            ]
            
            # Sort by relevance score (higher is better)
            filtered_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Take top documents
            top_docs = [doc for doc, score in filtered_docs[:RETRIEVAL_CONFIG['default_k']]]
            
            if top_docs:
                retrieved_docs = top_docs
                print(f"Filtered to {len(retrieved_docs)} high-quality documents")
            else:
                print("No documents met similarity threshold, using all retrieved docs")
        except Exception as e:
            print(f"ERROR: Failed to filter documents by relevance: {e}")
            # Continue with unfiltered docs
    
    # Print chunks when retrieved
    print(f"--- CHUNKS RETRIEVED FOR QUESTION: '{question}' ---")
    print(f"Total chunks retrieved: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        try:
            print(f"\n--- RETRIEVED CHUNK {i+1}/{len(retrieved_docs)} ---")
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            print(f"Content: {content[:200]}...")  # Show first 200 chars
            print(f"Metadata: {getattr(doc, 'metadata', {})}")
            print("-" * 50)
        except Exception as e:
            print(f"ERROR: Failed to print chunk {i+1}: {e}")
    
    # Evaluate retrieval quality
    try:
        quality_evaluation = evaluate_retrieval_quality(processed_question, retrieved_docs)
        print(f"--- RETRIEVAL QUALITY EVALUATION ---")
        print(f"Query length: {quality_evaluation['query_length']}")
        print(f"Documents retrieved: {quality_evaluation['num_docs_retrieved']}")
        print(f"Average document length: {quality_evaluation['avg_doc_length']:.0f} chars")
        print(f"Coverage score: {quality_evaluation['coverage_score']:.2f}")
        print(f"Diversity score: {quality_evaluation['diversity_score']:.2f}")
        if quality_evaluation['recommendations']:
            print(f"Recommendations: {', '.join(quality_evaluation['recommendations'])}")
    except Exception as e:
        print(f"ERROR: Failed to evaluate retrieval quality: {e}")
    
    # Ensure documents are strings or have a 'page_content' attribute
    try:
        documents = []
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                documents.append(doc.page_content)
            else:
                documents.append(str(doc))
    except Exception as e:
        print(f"ERROR: Failed to extract document content: {e}")
        documents = []
    
    print(f"Retrieved documents (first 500 chars): {str(documents)[:500]}...")
    return {"documents": documents, "question": question, "chat_history": chat_history} # Pass chat_history

def generate_answer(state):
    print("---GENERATING ANSWER---")
    
    try:
        question = state.get("question", "")
        documents = state.get("documents", [])
        chat_history = state.get("chat_history", []) # Get chat history from state
        
        if not question:
            print("WARNING: No question provided in state")
            return {"documents": [], "question": "", "chat_history": chat_history, "generation": "No question provided."}

        # Convert documents list to a single context string
        try:
            if documents:
                context = "\n\n".join(str(doc) for doc in documents) 
            else:
                context = "No relevant context found." 
        except Exception as e:
            print(f"ERROR: Failed to create context from documents: {e}")
            context = "No relevant context found."

        print(f"Context (first 500 chars): {context[:500]}...")

        # Check if LLM is available
        if llm is None:
            print("ERROR: LLM not available")
            return {
                "documents": documents, 
                "question": question, 
                "chat_history": chat_history,
                "generation": "I apologize, but the language model is not available at the moment. Please try again later."
            }

        try:
            # Updated prompt to include chat history
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a friendly and intelligent AI assistant.\n "
                            "Your job is to help users by providing accurate, engaging, and well-rounded answers based on "
                            "either the provided context or trusted external sources like Google Gemini.\n"
                            "Follow this approach:\n"
                            "1. **If the context clearly contains enough relevant information to fully answer the user's question:**\n"
                            "- Answer directly using the context.\n"
                            "- Be clear and helpful, but keep a natural and conversational tone.\n"
                            "- Do not reference the context explicitly (e.g., avoid saying \"Based on the context...\").\n"
                            "- Feel free to elaborate with helpful examples, analogies, or additional insight "
                            "- as long as it's grounded in the context.\n"
                            "2. **If the context does NOT include enough to answer the question:**\n"
                            "- Recognize that the answer requires external knowledge.\n"
                            "- Use Google Gemini or other trusted external sources to find accurate and current information.\n"
                            "- Then, craft a rich, helpful, and human-readable answer. \n"
                            "Present information from **multiple angles** (e.g., definiton, function, classification, behavior, history, examples, etc.) \n"
                            "so that the user walks away feeling informed.\n"
                            "- Speak in a friendly and engaging tone - not overly formal. \n"
                            "Think of explaining something to a curious, intelligent friend.\n"
                            "- Never say \"this is not in the context.\" Instead, smoothly transition into the answer.\n"
                            "**Your tone should always be:**\n"
                            "- Friendly, clear, and informative\n"
                            "- Curious and eager to help\n"
                            "- Avoid robotic phrases or repetition\n"
                            "**Your goal is to make the user feel heard, respected, and fully informed - "
                            "whether using provided context or external knowledge.**\n"
                            "**IMPORTANT:** Consider the conversation history to provide contextually relevant and coherent responses. "
                            "If the user is asking follow-up questions or referring to previous parts of the conversation, "
                            "make sure your response builds upon and references that context appropriately.\n"
                ),
                MessagesPlaceholder(variable_name="chat_history"), # Placeholder for chat history
                ("human", "Context:\n{context}\n\nQuestion:\n{question}")
            ])
        except Exception as e:
            print(f"ERROR: Failed to create prompt template: {e}")
            return {
                "documents": documents, 
                "question": question, 
                "chat_history": chat_history,
                "generation": "I apologize, but I encountered an error while setting up the response generation."
            }
        
        try:
            rag_chain = prompt | llm | StrOutputParser()
        except Exception as e:
            print(f"ERROR: Failed to create RAG chain: {e}")
            return {
                "documents": documents, 
                "question": question, 
                "chat_history": chat_history,
                "generation": "I apologize, but I encountered an error while setting up the response generation."
            }

        try:
            # Pass chat_history to the invoke method
            generation = rag_chain.invoke({"context": context, "question": question, "chat_history": chat_history})
        except Exception as e:
            print(f"ERROR: Failed during LLM invocation: {e}")
            generation = "I apologize, but I encountered an error while trying to generate a response. Please try again."
        
        print(f"Generation: {generation}")

        return {"documents": documents, "question": question, "chat_history": chat_history, "generation": generation}
        
    except Exception as e:
        print(f"CRITICAL ERROR during answer generation: {e}")
        return {
            "documents": state.get("documents", []), 
            "question": state.get("question", ""), 
            "chat_history": state.get("chat_history", []),
            "generation": "I apologize, but I encountered a critical error while processing your request. Please try again."
        }

# --- Build the LangGraph (No changes here) ---
def build_rag_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

rag_graph = build_rag_graph()


# --- FastAPI App ---
app = FastAPI(
    title="RAG System with ChromaDB, LangGraph, and FastAPI",
    description="An API for uploading documents and chatting with a RAG pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch any unhandled exceptions."""
    try:
        print(f"UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")
        print(f"Request URL: {request.url}")
        print(f"Request method: {request.method}")
        
        # Log additional context if available
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    print(f"Request body: {body[:500]}...")  # First 500 chars
        except Exception as e:
            print(f"Could not read request body: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "type": type(exc).__name__
            }
        )
    except Exception as handler_error:
        # If the exception handler itself fails, return a simple error response
        print(f"CRITICAL: Exception handler failed: {handler_error}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred."
            }
        )

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("--- APPLICATION STARTUP ---")
    try:
        # Ensure the chroma_db directory exists
        os.makedirs(CHROMA_PATH, exist_ok=True)
        print(f"Ensured chroma_db directory exists: {CHROMA_PATH}")
        
        # Initialize vector store
        initialize_vector_store()
        
        print("--- APPLICATION STARTUP COMPLETE ---")
    except Exception as e:
        print(f"Error during startup: {e}")
        # Don't raise the exception, just log it

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    print("--- APPLICATION SHUTDOWN ---")
    try:
        # Clean up any resources if needed
        print("Cleaning up resources...")
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        print("--- APPLICATION SHUTDOWN COMPLETE ---")

@app.post("/files/delete")
async def delete_file(request: Request):
    try:
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        user_id = body.get("userId")
        filename = body.get("filename")

        # Validate inputs
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="userId is required and cannot be empty.")
        
        if not filename or not filename.strip():
            raise HTTPException(status_code=400, detail="filename is required and cannot be empty.")

        s3_key = f"document-uploaded/{user_id}/{filename}"

        try:
            print(f"Deleting the file from S3 bucket")
            s3_client.delete_object(Bucket=os.getenv("AWS_S3_BUCKET_NAME"), Key=s3_key)
            print(f"Delete file from S3 bucket successfully")
            return JSONResponse(
                status_code=200,
                content={"message": f"{filename} was deleted from s3 bucket successfully"}
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "NoSuchKey":
                print(f"File not found in S3: {s3_key}")
                raise HTTPException(status_code=404, detail=f"File {filename} not found in S3 bucket.")
            else:
                print(f"S3 ClientError during deletion: {error_code} - {e}")
                raise HTTPException(status_code=500, detail=f"S3 deletion error: {e}")
        except Exception as e:
            print(f"ERROR: Failed to delete file from S3: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during file deletion: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in delete_file: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file deletion.")


@app.post("/files/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    print(f"Received file: {file.filename}")
    """
    Handles file uploads. It appends the new document to the existing ChromaDB vector store if it exists,
    or creates a new one if not. Also uploads the file to S3.
    """
    global vectorstore, retriever

    try:
        # Validate file
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided or invalid file.")

        if not file.filename.endswith((".pdf", ".csv")):
            raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")

        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty.")

        # Check if embeddings are available
        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings not initialized. Please check your Google API key.")

        temp_dir = "temp_docs"
        file_path = os.path.join(temp_dir, file.filename)
        s3_key = f"document-uploaded/{user_id}/{file.filename}"

        try:
            # Create temporary directory
            os.makedirs(temp_dir, exist_ok=True)
            print("Creating temporary directory and saving uploaded file...")
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Upload to S3
            try:
                with open(file_path, "rb") as data:
                    print(f"Uploading file to S3... {data}")
                    s3_client.upload_fileobj(data, os.getenv("AWS_S3_BUCKET_NAME"), s3_key)
                print(f"File '{file.filename}' uploaded to S3 at '{s3_key}'.")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                print(f"S3 ClientError during upload: {error_code} - {e}")
                raise HTTPException(status_code=500, detail=f"S3 upload error (ClientError): {e}")
            except Exception as e:
                print(f"Generic S3 upload error: {e}")
                raise HTTPException(status_code=500, detail=f"S3 upload error: {e}")

            # Load the document
            try:
                if file.filename.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file.filename.lower().endswith(".csv"):
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                else:
                    raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")
            except Exception as e:
                print(f"ERROR: Failed to load document: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load document: {str(e)}")

            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the uploaded file.")

            # Split the document into chunks
            try:
                # Calculate total content length for optimal chunk sizing
                total_content_length = sum(len(doc.page_content) for doc in docs)
                chunk_size, chunk_overlap = get_optimal_chunk_size(total_content_length)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]  # More granular separators
                )
                splits = text_splitter.split_documents(docs)
                
                if not splits:
                    raise HTTPException(status_code=400, detail="No text chunks could be created from the document.")
            except Exception as e:
                print(f"ERROR: Failed to split document: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
            
            # Print chunks when uploaded
            print(f"--- CHUNKS CREATED FROM UPLOADED FILE: {file.filename} ---")
            print(f"Total chunks created: {len(splits)}")
            for i, chunk in enumerate(splits):
                try:
                    print(f"\n--- CHUNK {i+1}/{len(splits)} ---")
                    print(f"Content: {chunk.page_content[:200]}...")  # Show first 200 chars
                    print(f"Metadata: {chunk.metadata}")
                    print("-" * 50)
                except Exception as e:
                    print(f"ERROR: Failed to print chunk {i+1}: {e}")

            # Append to existing vector store or create a new one
            try:
                if vectorstore:
                    print("--- APPENDING TO EXISTING VECTOR STORE ---")
                    vectorstore.add_documents(splits)
                else:
                    print("--- CREATING NEW VECTOR STORE ---")
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory=CHROMA_PATH
                    )
            except Exception as e:
                print(f"ERROR: Failed to add documents to vector store: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to add documents to vector store: {str(e)}")

            # Update the global retriever
            try:
                retriever = create_advanced_retriever(vectorstore)
                if retriever is None:
                    print("WARNING: Failed to create advanced retriever")
            except Exception as e:
                print(f"ERROR: Failed to create retriever: {e}")
                # Continue without retriever update
            
            # Print updated vector store count
            try:
                collection_count = vectorstore._collection.count()
                print(f"--- VECTOR STORE UPDATED ---")
                print(f"Total chunks in vector store after upload: {collection_count}")
            except Exception as e:
                print(f"Could not get updated collection count: {e}")

            print(f"File '{file.filename}' processed. Vector store is ready in '{CHROMA_PATH}'.")

            return JSONResponse(
                status_code=200,
                content={"message": f"File '{file.filename}' uploaded, processed, and stored in S3 successfully.", "key": f"{s3_key}"}
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                # Ensure the uploaded file's handle is closed
                if not file.file.closed:
                    file.file.close()
            except Exception as e:
                print(f"ERROR: Failed to cleanup temporary files: {e}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in upload_file: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file upload.")



# --- NEW ENDPOINT FOR WEBSITE LINKS ---
@app.post("/ingest-website/")
async def ingest_website_link(request: Request):
    """
    Ingests content from a given website URL into the RAG system.
    It scrapes the content and adds it to the ChromaDB vector store.
    """
    global vectorstore, retriever

    try:
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        url = body.get("url")
        user_id = body.get("userId") # Assuming user_id is also provided for context/tracking

        # Validate inputs
        if not url or not url.strip():
            raise HTTPException(status_code=400, detail="Website URL not provided or empty.")
        
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty.")

        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL format. URL must start with http:// or https://")

        # Check if embeddings are available
        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings not initialized. Please check your Google API key.")

        print(f"Received website URL: {url} for user: {user_id}")

        # Load content from the website URL
        try:
            loader = WebBaseLoader(web_paths=[url])
            # For simplicity, we'll use load(). For very large sites or many links, consider lazy_load() or RecursiveUrlLoader
            docs = loader.load()
            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the provided URL.")
            print(f"Successfully loaded {len(docs)} documents from {url}")
        except Exception as e:
            print(f"ERROR: Failed to load content from URL: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to load content from URL: {str(e)}. Please check the URL and try again."
            )

        # Split the document into chunks
        try:
            # Calculate total content length for optimal chunk sizing
            total_content_length = sum(len(doc.page_content) for doc in docs)
            chunk_size, chunk_overlap = get_optimal_chunk_size(total_content_length)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # More granular separators
            )
            splits = text_splitter.split_documents(docs)
            
            if not splits:
                raise HTTPException(status_code=400, detail="No text chunks could be created from the website content.")
        except Exception as e:
            print(f"ERROR: Failed to split website content: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process website content: {str(e)}")
        
        # Print chunks when uploaded from website
        print(f"--- CHUNKS CREATED FROM WEBSITE: {url} ---")
        print(f"Total chunks created: {len(splits)}")
        for i, chunk in enumerate(splits):
            try:
                print(f"\n--- CHUNK {i+1}/{len(splits)} ---")
                print(f"Content: {chunk.page_content[:200]}...")  # Show first 200 chars
                print(f"Metadata: {chunk.metadata}")
                print("-" * 50)
            except Exception as e:
                print(f"ERROR: Failed to print chunk {i+1}: {e}")

        # Append to existing vector store or create a new one
        try:
            if vectorstore:
                print("--- APPENDING WEBSITE CONTENT TO EXISTING VECTOR STORE ---")
                vectorstore.add_documents(splits)
            else:
                print("--- CREATING NEW VECTOR STORE FROM WEBSITE CONTENT ---")
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=CHROMA_PATH
                )
        except Exception as e:
            print(f"ERROR: Failed to add website content to vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add website content to vector store: {str(e)}")

        # Update the global retriever
        try:
            retriever = create_advanced_retriever(vectorstore)
            if retriever is None:
                print("WARNING: Failed to create advanced retriever")
        except Exception as e:
            print(f"ERROR: Failed to create retriever: {e}")
            # Continue without retriever update
        
        # Print updated vector store count
        try:
            collection_count = vectorstore._collection.count()
            print(f"--- VECTOR STORE UPDATED ---")
            print(f"Total chunks in vector store after website ingestion: {collection_count}")
        except Exception as e:
            print(f"Could not get updated collection count: {e}")

        print(f"Website content from '{url}' processed and added to vector store.")

        return JSONResponse(
            status_code=200,
            content={"message": f"Content from '{url}' ingested successfully."}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in website ingestion: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred during website ingestion."
        )

@app.post("/chat/")
async def chat_with_rag(request_data: ChatRequest):
    """
    Receives a question and returns an answer from the RAG pipeline,
    incorporating chat history.
    Uses the retriever initialized from the persistent ChromaDB.
    """
    try:
        # Re-initialize vectorstore and retriever if they were reset by upload_file
        # or not initialized at startup (e.g., if chroma_db didn't exist)
        global vectorstore, retriever
        if not vectorstore or not retriever:
            # Attempt to load it again if a file was uploaded recently
            try:
                initialize_vector_store()
            except Exception as e:
                print(f"ERROR: Failed to re-initialize vector store: {e}")
            
            # Check again after re-initialization
            if not vectorstore or not retriever:
                raise HTTPException(
                    status_code=400, 
                    detail="Vector store not initialized. Please upload a document first."
                )

        question = request_data.question.strip()
        
        # Convert Pydantic Message objects to LangChain's BaseMessage format
        lc_chat_history = []
        for msg in request_data.chat_history:
            if msg.role == "user":
                lc_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_chat_history.append(AIMessage(content=msg.content))
            # You can add more roles if needed (e.g., "system" might also be useful)
        
        if not question:
            raise HTTPException(status_code=400, detail="Question not provided or empty.")

        # Validate question length
        if len(question) > 10000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Question too long. Please keep it under 10,000 characters.")

        try:
            # Pass both question and chat_history to the RAG graph
            inputs = {"question": question, "chat_history": lc_chat_history}
            result = rag_graph.invoke(inputs)
        except Exception as e:
            print(f"ERROR: Failed to invoke RAG graph: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to process your question. Please try again."
            )

        # Validate result
        if not result or "generation" not in result:
            print("ERROR: Invalid result from RAG graph")
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate a response. Please try again."
            )

        return JSONResponse(
            status_code=200,
            content={"answer": result["generation"]}
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in chat_with_rag: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing your request."
        )


@app.get("/")
def read_root():
    try:
        return {"message": "LangChain RAG API is running!"}
    except Exception as e:
        print(f"CRITICAL ERROR in read_root: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/configure-retrieval/")
async def configure_retrieval(request: Request):
    """Configure retrieval parameters dynamically."""
    global retriever, vectorstore
    
    try:
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            print(f"ERROR: Failed to parse request body: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        if not body:
            raise HTTPException(status_code=400, detail="Request body is empty.")
        
        # Validate and update configuration
        try:
            if 'default_k' in body:
                value = body['default_k']
                if not isinstance(value, int) or value <= 0 or value > 100:
                    raise HTTPException(status_code=400, detail="default_k must be a positive integer between 1 and 100.")
                RETRIEVAL_CONFIG['default_k'] = value
                
            if 'mmr_k' in body:
                value = body['mmr_k']
                if not isinstance(value, int) or value <= 0 or value > 100:
                    raise HTTPException(status_code=400, detail="mmr_k must be a positive integer between 1 and 100.")
                RETRIEVAL_CONFIG['mmr_k'] = value
                
            if 'mmr_lambda' in body:
                value = body['mmr_lambda']
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise HTTPException(status_code=400, detail="mmr_lambda must be a number between 0 and 1.")
                RETRIEVAL_CONFIG['mmr_lambda'] = value
                
            if 'similarity_threshold' in body:
                value = body['similarity_threshold']
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    raise HTTPException(status_code=400, detail="similarity_threshold must be a number between 0 and 1.")
                RETRIEVAL_CONFIG['similarity_threshold'] = value
                
            if 'max_chunk_size' in body:
                value = body['max_chunk_size']
                if not isinstance(value, int) or value <= 0 or value > 10000:
                    raise HTTPException(status_code=400, detail="max_chunk_size must be a positive integer between 1 and 10000.")
                RETRIEVAL_CONFIG['max_chunk_size'] = value
                
            if 'chunk_overlap' in body:
                value = body['chunk_overlap']
                if not isinstance(value, int) or value < 0 or value > 5000:
                    raise HTTPException(status_code=400, detail="chunk_overlap must be a non-negative integer between 0 and 5000.")
                RETRIEVAL_CONFIG['chunk_overlap'] = value
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"ERROR: Failed to validate configuration parameters: {e}")
            raise HTTPException(status_code=400, detail="Invalid configuration parameters.")
        
        # Recreate retriever with new configuration if vectorstore exists
        try:
            if vectorstore:
                retriever = create_advanced_retriever(vectorstore)
                if retriever is None:
                    print("WARNING: Failed to create retriever with new configuration")
                else:
                    print(f"Retriever reconfigured with new parameters: {RETRIEVAL_CONFIG}")
            else:
                print("WARNING: No vector store available, configuration saved but not applied")
        except Exception as e:
            print(f"ERROR: Failed to recreate retriever: {e}")
            # Continue without retriever update
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Retrieval configuration updated successfully",
                "current_config": RETRIEVAL_CONFIG
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in configure_retrieval: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating configuration.")

@app.get("/retrieval-config/")
async def get_retrieval_config():
    """Get current retrieval configuration."""
    try:
        # Get vector store status
        vectorstore_status = "loaded" if vectorstore else "not_loaded"
        total_chunks = 0
        
        try:
            if vectorstore:
                total_chunks = vectorstore._collection.count()
        except Exception as e:
            print(f"ERROR: Failed to get collection count: {e}")
            vectorstore_status = "error"
        
        return JSONResponse(
            status_code=200,
            content={
                "current_config": RETRIEVAL_CONFIG,
                "vectorstore_status": vectorstore_status,
                "total_chunks": total_chunks
            }
        )
    except Exception as e:
        print(f"CRITICAL ERROR in get_retrieval_config: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while retrieving configuration.")