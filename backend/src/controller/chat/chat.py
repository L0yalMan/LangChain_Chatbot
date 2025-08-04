from fastapi import HTTPException
from fastapi.responses import JSONResponse

from langchain_core.messages import HumanMessage, AIMessage

from src.core.models import ChatRequest
from src.core.config import get_global_state
from src.core.retriever import initialize_vector_store
from src.core.rag_graph import rag_graph
from src.utils.dependencies import TokenData


async def chat_with_rag(request_data: ChatRequest, current_user: TokenData):
    """
    Receives a question and returns an answer from the RAG pipeline,
    incorporating chat history.
    Uses the retriever initialized from the persistent ChromaDB.
    """
    
    try:
        question = request_data.question.strip()
        print(f"Current user id is {current_user.user_id}")
        print(f"QUESTION IS {question} - first")
        print(f"QUESTION IS {question} - second")

        from src.core.config import vectorstore, retriever
        if not vectorstore or not retriever:
            print(f"DEBUG: Global state before re-initialization: {get_global_state()}")
            try:
                initialize_vector_store()
                # Re-import the updated global variables
                from src.core.config import vectorstore, retriever
            except Exception as e:
                print(f"ERROR: Failed to re-initialize vector store: {e}")

            if not vectorstore or not retriever:
                print(f"DEBUG: Global state after re-initialization: {get_global_state()}")
                print(f"detail=Vector store not initialized. Please upload a document first.")
                raise HTTPException(
                    status_code=400,
                    detail="Vector store not initialized. Please upload a document first."
                )

        # Check if vector store has any documents
        try:
            collection_count = vectorstore._collection.count()
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
            elif msg.role == "assistant":
                lc_chat_history.append(AIMessage(content=msg.content))

        if not question:
            raise HTTPException(status_code=400, detail="Question not provided or empty.")

        if len(question) > 10000:
            raise HTTPException(status_code=400, detail="Question too long. Please keep it under 10,000 characters.")

        try:
            inputs = {"question": question, "chat_history": lc_chat_history}
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
