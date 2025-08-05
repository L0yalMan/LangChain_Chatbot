from fastapi import HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, AIMessage

from src.core.models import ChatRequest
from src.core.config_pinecone import get_global_state
from src.core.retriver_pinecone import initialize_vector_store
from src.core.rag_graph_pinecone import rag_graph
from src.utils.dependencies import TokenData


async def chat_with_rag(request_data: ChatRequest, current_user: TokenData):
    """
    Receives a question and returns an answer from the RAG pipeline,
    incorporating chat history. Uses the retriever initialized for the specific user.
    """
    user_id = current_user.user_id
    
    try:
        question = request_data.question.strip()
        print(f"Current user id is {user_id}")
        print(f"QUESTION IS {question}")
        
        from src.core.config_pinecone import vectorstores, retrievers
        # Check if vectorstore and retriever are initialized for the user
        if not vectorstores.get(user_id) or not retrievers.get(user_id):
            print(f"DEBUG: Global state before re-initialization for user {user_id}: {get_global_state(user_id)}")
            try:
                # Try to load existing vector store first
                from src.core.retriver_pinecone import load_existing_vector_store, check_vector_store_exists
                from src.core.config_pinecone import get_pinecone_client, get_embeddings, settings
                
                pinecone_client = get_pinecone_client()
                embeddings_instance = get_embeddings()
                index_name = settings.PINECONE_INDEX_NAME
                
                # Check if vector store exists for this user
                if check_vector_store_exists(user_id, index_name):
                    print(f"Found existing vector store for user {user_id}, loading it...")
                    existing_vectorstore = load_existing_vector_store(user_id, index_name, embeddings_instance)
                    if existing_vectorstore:
                        from src.core.retriver_pinecone import create_advanced_retriever
                        new_retriever = create_advanced_retriever(existing_vectorstore)
                        if new_retriever:
                            from src.core.config_pinecone import update_vector_store_globals
                            update_vector_store_globals(user_id, existing_vectorstore, new_retriever)
                            print(f"Successfully loaded existing vector store and retriever for user {user_id}")
                        else:
                            print(f"Failed to create retriever for existing vector store for user {user_id}")
                            initialize_vector_store(user_id)
                    else:
                        print(f"Failed to load existing vector store for user {user_id}, initializing new one...")
                        initialize_vector_store(user_id)
                else:
                    print(f"No existing vector store found for user {user_id}, initializing new one...")
                    initialize_vector_store(user_id)
                
                from src.core.config_pinecone import vectorstores, retrievers
            except Exception as e:
                print(f"ERROR: Failed to re-initialize vector store for user {user_id}: {e}")

            if not vectorstores.get(user_id) or not retrievers.get(user_id):
                print(f"DEBUG: Global state after re-initialization for user {user_id}: {get_global_state(user_id)}")
                raise HTTPException(
                    status_code=500,
                    detail="Vector store is not ready. Please try uploading a document or website first."
                )

        # Check if vector store has any documents
        try:
            # Improved: Use public API to get collection count, handle missing attribute gracefully
            vectorstore = vectorstores.get(user_id)
            collection_count = 0
            if hasattr(vectorstore, "index"):
                try:
                    stats = vectorstore.index.describe_index_stats()
                    collection_count = stats.get("total_vector_count", 0)
                except Exception as e:
                    print(f"WARNING: Could not retrieve index stats: {e}")
            if collection_count == 0:
                print(f"detail=Vector store is empty. Please upload a document first.")
                # raise HTTPException(
                #     status_code=400,
                #     detail="Vector store is empty. Please upload a document first."
                # )
        except Exception as e:
            print(f"ERROR: Failed to check collection count: {e}")
            raise HTTPException(
                status_code=400,
                detail="Vector store not properly initialized. Please upload a document first."
            )
        
        lc_chat_history = []
        for msg in request_data.chat_history:
            if msg.role == "user":
                lc_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                lc_chat_history.append(AIMessage(content=msg.content))

        if not question:
            raise HTTPException(status_code=400, detail="Question not provided or empty.")

        if len(question) > 10000:
            raise HTTPException(status_code=400, detail="Question too long. Please keep it under 10,000 characters.")

        try:
            # Pass user_id to the graph to be used for retrieval
            inputs = {"question": question, "chat_history": lc_chat_history, "user_id": user_id}
            result = rag_graph.invoke(inputs)
        except Exception as e:
            print(f"ERROR: Failed to invoke RAG graph: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process your question. Please try again."
            )

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
        print(f"CRITICAL ERROR in chat_with_rag_route: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request."
        )
