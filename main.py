import os
from dotenv import load_dotenv
import shutil
import gc
from typing import List, TypedDict
import json
from botocore.exceptions import ClientError

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# LangChain components
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Replace with ...
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangGraph components
from langgraph.graph import END, StateGraph

# Supabase client and s3 client
from src.utils.supabase_client import supabase
from src.utils.s3_client import s3_client

# --- Global Variables & Setup ---
load_dotenv()

# Define the persistent path for the Chroma database
CHROMA_PATH = "chroma_db"

# LLM and Embeddings with Gemini API
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # You can also use "gemini-1.5-pro" or "gemini-2.0-flash" if available and suitable
    temperature=0,
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Recommended embedding model for Gemini
)

if os.getenv("GOOGLE_API_KEY") is None:
    print("WARNING: GOOGLE_API_KEY environment variable is not set. The application might not work correctly without it.")

# Initialize global variables for the vector store and retriever
vectorstore = None
retriever = None

# --- Startup Logic to Load Existing DB ---
# Try to load the vector store from the persistent directory on startup
def initialize_vector_store():
    global vectorstore, retriever
    print(f"------->>>>> Initialize_Vector Store")
    print(os.path.exists(CHROMA_PATH))
    print(os.listdir(CHROMA_PATH))
    try:
        if os.path.exists(CHROMA_PATH):
            # print(f"--- LOADING VECTOR STORE FROM {CHROMA_PATH} ---")
            # LOADING VECTOR STORE FROM {CHROMA_PATH}
            print(f"------->>>>> vectorstore")
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            print(f"..........>>>>>>>>>>>>>There are {vectorstore._collection.count()} chunks in the vector store and 6 chunks~~~~<<<<<<<<<-------")
            retriever = vectorstore.as_retriever(search_kwargs={'k': 6})
            
            # Print total chunks in vector store
            try:
                collection_count = vectorstore._collection.count()
                print(f"--- VECTOR STORE INITIALIZED ---")
                print(f"Total chunks in vector store: {collection_count}")
            except Exception as e:
                print(f"Could not get collection count: {e}")
        else:
            print(f"--- NO EXISTING VECTOR STORE FOUND AT {CHROMA_PATH} ---")
            vectorstore = None
            retriever = None
    except Exception as e:
        print(f"--- ERROR LOADING VECTOR STORE: {e} ---")
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
    documents: List[str]

# --- RAG Graph Nodes (No changes here) ---
def retrieve_documents(state):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]

    global vectorstore, retriever # Ensure we use the global instance

    if not vectorstore:
        # If vectorstore is None, it means no DB has been loaded/created yet
        raise HTTPException(status_code=400, detail="No document database found. Please upload a PDF first.")

    try:
        # Check if the collection actually exists and has data
        collection_count = vectorstore._collection.count()
        print(f"Collection Count: {collection_count}")

        if collection_count == 0:
            print("No document exists in the collection.")
            documents = []
        else:
            retrieved_docs = retriever.invoke(question)
            
            # Print chunks when retrieved
            print(f"--- CHUNKS RETRIEVED FOR QUESTION: '{question}' ---")
            print(f"Total chunks retrieved: {len(retrieved_docs)}")
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- RETRIEVED CHUNK {i+1}/{len(retrieved_docs)} ---")
                print(f"Content: {doc.page_content[:200]}...")  # Show first 200 chars
                print(f"Metadata: {doc.metadata}")
                print("-" * 50)
            
            # Ensure documents are strings or have a 'page_content' attribute
            documents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retrieved_docs]
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

    print(f"Retrieved documents (first 500 chars): {str(documents)[:500]}...")
    return {"documents": documents, "question": question}

def generate_answer(state):
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]

    # Convert documents list to a single context string
    if documents:
        context = "\n\n".join(str(doc) for doc in documents) # Already ensured they are strings in retrieve_documents
    else:
        context = "No relevant context found." # Provide a default for the prompt

    print(f"Context (first 500 chars): {context[:500]}...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and intelligent AI assistant.\n "
            "Your job is to help users by providing accurate, engaging, and well-rounded answers based on "
            "either the provided context or trusted external sources like Google Gemini.\n"
            "Follow this approach:\n"
            "1. **If the context clearly contains enough relevant information to fully answer the user's question:**\n"
            "- Answer directly using the context.\n"
            "- Be clear and helpful, but keep a natural and conversational tone.\n"
            "- Do not reference the context explicitly (e.g., avoid saying “Based on the context…”).\n"
            "- Feel free to elaborate with helpful examples, analogies, or additional insight "
            "— as long as it's grounded in the context.\n"
            "2. **If the context does NOT include enough to answer the question:**\n"
            "- Recognize that the answer requires external knowledge.\n"
            "- Use Google Gemini or other trusted external sources to find accurate and current information.\n"
            "- Then, craft a rich, helpful, and human-readable answer. \n"
            "Present information from **multiple angles** (e.g., definiton, function, classification, behavior, history, examples, etc.) \n"
            "so that the user walks away feeling informed.\n"
            "- Speak in a friendly and engaging tone — not overly formal. \n"
            "Think of explaining something to a curious, intelligent friend.\n"
            "- Never say “this is not in the context.” Instead, smoothly transition into the answer.\n"
            "**Your tone should always be:**\n"
            "- Friendly, clear, and informative\n"
            "- Curious and eager to help\n"
            "- Avoid robotic phrases or repetition\n"
            "**Your goal is to make the user feel heard, respected, and fully informed — "
            "whether using provided context or external knowledge.**\n"
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    rag_chain = prompt | llm | StrOutputParser()

    try:
        generation = rag_chain.invoke({"context": context, "question": question})
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        generation = "I apologize, but I encountered an error while trying to generate a response."
    
    print(f"Generation: {generation}")

    return {"documents": documents, "question": question, "generation": generation}


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

@app.post("/files/delete")
async def delete_file(request: Request):
    body = await request.json()
    user_id = body.get("userId")
    filename = body.get("filename")

    s3_key = f"document-uploaded/{user_id}/{filename}"

    try:
        print(f"Deleting the file from S3 bucket")
        s3_client.delete_object(Bucket=os.getenv("AWS_S3_BUCKET_NAME"), Key=s3_key)
        print(f"Delete file from S3 bucket successfully")
        return JSONResponse(
            status_code=200,
            content={"message": f"{filename} was deleted from s3 bucket successfully"}
        )
    except Exception as e:
        print(f"Delet Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/files/upload/")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Handles file uploads. It appends the new document to the existing ChromaDB vector store if it exists,
    or creates a new one if not. Also uploads the file to S3.
    """
    global vectorstore, retriever

    if not file.filename.endswith((".pdf", ".csv")):
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")

    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    s3_key = f"document-uploaded/{user_id}/{file.filename}"

    try:
        print(f"Received file: {file.filename}")
        print("Creating temporary directory and saving uploaded file...")
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            with open(file_path, "rb") as data: # Always open in 'rb' mode for S3 upload
                print(f"Uploading file to S3... {data}")
                s3_client.upload_fileobj(data, os.getenv("AWS_S3_BUCKET_NAME"), s3_key)
            print(f"File '{file.filename}' uploaded to S3 at '{s3_key}'.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            print(f"S3 ClientError during upload: {error_code} - {e}")
            raise HTTPException(status_code=500, detail=f"S3 upload error (ClientError): {e}")
        except Exception as e:
            print(f"Generic S3 upload error: {e}")
            raise HTTPException (status_code=500, detail=f"S3 upload error: {e}")

        # 1. Load the document
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file.filename.lower().endswith(".csv"):
            loader = CSVLoader(file_path)
            docs = loader.load()
        else:
            raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Print chunks when uploaded
        print(f"--- CHUNKS CREATED FROM UPLOADED FILE: {file.filename} ---")
        print(f"Total chunks created: {len(splits)}")
        for i, chunk in enumerate(splits):
            print(f"\n--- CHUNK {i+1}/{len(splits)} ---")
            print(f"Content: {chunk.page_content[:200]}...")  # Show first 200 chars
            print(f"Metadata: {chunk.metadata}")
            print("-" * 50)

        # 3. Append to existing vector store or create a new one
        if vectorstore:
            print("--- APPENDING TO EXISTING VECTOR STORE ---")
            vectorstore.add_documents(splits)
            # No need to call persist(); persistence is handled automatically
        else:
            print("--- CREATING NEW VECTOR STORE ---")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
            # No need to call persist(); persistence is handled automatically

        # 4. Update the global retriever
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        
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

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        # Ensure the uploaded file's handle is closed
        if not file.file.closed:
            file.file.close()



# --- NEW ENDPOINT FOR WEBSITE LINKS ---
@app.post("/ingest-website/")
async def ingest_website_link(request: Request):
    """
    Ingests content from a given website URL into the RAG system.
    It scrapes the content and adds it to the ChromaDB vector store.
    """
    global vectorstore, retriever

    try:
        body = await request.json()
        url = body.get("url")
        user_id = body.get("userId") # Assuming user_id is also provided for context/tracking

        if not url:
            raise HTTPException(status_code=400, detail="Website URL not provided.")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required.")

        print(f"Received website URL: {url} for user: {user_id}")

        # 1. Load content from the website URL
        try:
            loader = WebBaseLoader(web_paths=[url])
            # For simplicity, we'll use load(). For very large sites or many links, consider lazy_load() or RecursiveUrlLoader
            docs = loader.load()
            if not docs:
                raise HTTPException(status_code=400, detail="No content could be extracted from the provided URL.")
            print(f"Successfully loaded {len(docs)} documents from {url}")
        except Exception as e:
            print(f"Error loading content from URL: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load content from URL: {str(e)}. Please check the URL and try again.")

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Print chunks when uploaded from website
        print(f"--- CHUNKS CREATED FROM WEBSITE: {url} ---")
        print(f"Total chunks created: {len(splits)}")
        for i, chunk in enumerate(splits):
            print(f"\n--- CHUNK {i+1}/{len(splits)} ---")
            print(f"Content: {chunk.page_content[:200]}...")  # Show first 200 chars
            print(f"Metadata: {chunk.metadata}")
            print("-" * 50)

        # 3. Append to existing vector store or create a new one
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

        # 4. Update the global retriever
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        
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

    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        print(f"Website ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during website ingestion: {str(e)}")


@app.post("/chat/")
async def chat_with_rag(request: Request):
    """
    Receives a question and returns an answer from the RAG pipeline.
    Uses the retriever initialized from the persistent ChromaDB.
    """
    # Re-initialize vectorstore and retriever if they were reset by upload_file
    # or not initialized at startup (e.g., if chroma_db didn't exist)
    global vectorstore, retriever
    if not vectorstore or not retriever:
        # Attempt to load it again if a file was uploaded recently
        initialize_vector_store()
        if not vectorstore or not retriever:
            raise HTTPException(status_code=400, detail="Vector store not initialized. Please upload a document first.")

    try:
        body = await request.json()
        question = body.get("question")
        # userId = body.get("userId")
        # sessionId = body.get("sessionId")
        if not question:
            raise HTTPException(status_code=400, detail="Question not provided.")

        inputs = {"question": question}
        result = rag_graph.invoke(inputs)

        # data = [
        #     {"user_id": userId, "session_id": sessionId, "role": "user", "content": question},
        #     {"user_id": userId, "session_id": sessionId, "role": "ai", "content": result["generation"]}
        # ]
        # response = supabase.table('chat_history').insert(data).execute()
        # print(f"response: {response}")

        return JSONResponse(
            status_code=200,
            content={"answer": result["generation"]}
        )

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



# @app.get('/chat-history/')
# async def chathistory_from_supabase(userId: str, sessionId: str):
#     try:
#         response = supabase.table('chat_history').select('role, content').eq('user_id', userId).eq('session_id', sessionId).execute()
#         data = json.dumps(response.data, indent=2)

#         return JSONResponse(status_code=200, content={"history": data})
#     except Exception as e:
#         print(f"Error fetching chat history: {e}")
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "API is running"}