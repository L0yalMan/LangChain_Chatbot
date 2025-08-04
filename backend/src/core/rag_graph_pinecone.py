from typing import List, TypedDict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from src.core.config_pinecone import RETRIEVAL_CONFIG
from src.core.retriver_pinecone import preprocess_query, evaluate_retrieval_quality

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: The user's question.
        generation: The LLM's generation.
        documents: A list of retrieved documents.
        chat_history: The history of the conversation.
        user_id: The ID of the current user.
    """
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]
    user_id: str

def retrieve_documents(state: GraphState):
    """
    Retrieves documents for the user's question using the user-specific retriever.
    """
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    user_id = state["user_id"]
    
    from src.core.config_pinecone import vectorstores, retrievers

    retriever = retrievers.get(user_id)
    vectorstore = vectorstores.get(user_id)

    if not vectorstore:
        print("ERROR: No vector store available for this user.")
        return {"documents": [], "question": question, "chat_history": chat_history}
    
    try:
        # Pinecone does not have a simple collection.count() for a namespace
        # We can perform a dummy fetch to check if the namespace has any vectors.
        # This is a bit of a hack, but better than nothing.
        index = vectorstore.index
        stats = index.describe_index_stats()
        namespace_stats = stats.namespaces.get(user_id)
        if not namespace_stats or namespace_stats.vector_count == 0:
             print("No documents exist in the collection for this user.")
             return {"documents": [], "question": question, "chat_history": chat_history}
    except Exception as e:
        print(f"ERROR: Failed to get index stats: {e}")
        return {"documents": [], "question": question, "chat_history": chat_history}

    try:
        processed_question = preprocess_query(question)
        print(f"Original question: '{question}'")
        print(f"Processed question: '{processed_question}'")
    except Exception as e:
        print(f"ERROR: Failed to preprocess question: {e}")
        processed_question = question

    retrieved_docs = []
    try:
        if retriever:
            retrieved_docs = retriever.invoke(processed_question)
            print(f"Advanced retrieval successful with {len(retrieved_docs)} documents.")
        else:
            print("WARNING: No retriever available, performing basic similarity search.")
            retrieved_docs = vectorstore.similarity_search(
                query=processed_question,
                k=RETRIEVAL_CONFIG['default_k']
            )

        print(f"Number of retrieved docs: {len(retrieved_docs)}")
        
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"Retrieved Document {i+1}: {doc.page_content[:200]}...")

    except Exception as e:
        print(f"CRITICAL ERROR during retrieval: {e}")
        retrieved_docs = []

    if retrieved_docs:
        try:
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                query=processed_question,
                k=len(retrieved_docs) * 2
            )
            filtered_docs = [
                (doc, score) for doc, score in docs_with_scores
                if score >= RETRIEVAL_CONFIG['similarity_threshold']
            ]
            filtered_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, score in filtered_docs[:RETRIEVAL_CONFIG['default_k']]]
            if top_docs:
                retrieved_docs = top_docs
                print(f"Filtered to {len(retrieved_docs)} high-quality documents.")
            else:
                print("No documents met similarity threshold, using all retrieved docs.")
        except Exception as e:
            print(f"ERROR: Failed to filter documents by relevance: {e}")

    print(f"--- CHUNKS RETRIEVED FOR QUESTION: '{question}' ---")
    print(f"Total chunks retrieved: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- RETRIEVED CHUNK {i+1}/{len(retrieved_docs)} ---")
        print(f"Relevance Score: {doc.metadata.get('score', 'N/A')}")
    
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

    documents = []
    for doc in retrieved_docs:
        if hasattr(doc, 'page_content'):
            documents.append(doc.page_content)
        else:
            documents.append(str(doc))

    print(f"Retrieved documents (first 500 chars): {str(documents)[:500]}...")
    return {"documents": documents, "question": question, "chat_history": chat_history, "user_id": user_id}

def generate_answer(state: GraphState):
    """
    Generates an answer using the retrieved documents as context.
    """
    print("---GENERATING ANSWER---")
    from src.core.config_pinecone import llm
    try:
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])
        
        if not question:
            print("WARNING: No question provided in state")
            return {"documents": [], "question": "", "chat_history": chat_history, "generation": "No question provided."}

        context = "\n\n".join(str(doc) for doc in documents) if documents else "No relevant context found."
        print(f"Context (first 500 chars): {context[:500]}...")

        if not llm:
            print("ERROR: LLM is not initialized.")
            return {
                "documents": documents, 
                "question": question, 
                "chat_history": chat_history, 
                "generation": "I'm sorry, I cannot generate an answer at this time. The language model is not available."
            }
        
        # Prepare the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
                "Here is an improved version of your prompt, specifically optimized to:\n"
"\n"
"* Handle **comparisons** naturally and intelligently\n"
"* Avoid robotic or redundant phrases like “In the context, I can’t...”\n"
"* Maintain a **friendly, smart, human-like tone**\n"
"* Adapt seamlessly whether or not the information is in the context\n"
"\n"
"---\n"
"\n"
"### ✅ Improved Prompt for Your AI Chatbot:\n"
"\n"
"You are a friendly, intelligent AI assistant.\n"
"Your job is to help users by providing accurate, engaging, and well-rounded answers using either the provided context or trusted external sources like Google Gemini.\n"
"\n"
"Follow this approach:\n"
"\n"
"---\n"
"\n"
"#### 🔹 If the provided context contains enough relevant information:\n"
"\n"
"* Answer clearly and directly using that information.\n"
"* Keep your tone natural, conversational, and helpful.\n"
"* Do **not** mention the existence or use of context.\n"
"* Feel free to expand with examples, analogies, or insights that clarify and enrich the answer, as long as they align with the context.\n"
"\n"
"---\n"
"\n"
"#### 🔹 If the context does **not** include enough to fully answer the question:\n"
"\n"
"* Do **not** say “the context doesn’t provide enough info.”\n"
"* Instead, seamlessly pull in accurate and current knowledge from trusted external sources (e.g., Google Gemini).\n"
"* Then give a helpful, well-rounded answer that feels natural and complete.\n"
"* When explaining, aim to explore the topic from multiple angles — such as definitions, differences, pros/cons, examples, applications, comparisons, etc.\n"
"\n"
"---\n"
"\n"
"#### 🔹 When answering **comparisons**:\n"
"\n"
"* Compare clearly, fairly, and insightfully — even if the context lacks full data.\n"
"* Use categories like functionality, performance, cost, popularity, pros/cons, etc.\n"
"* Always strive to make comparisons useful, balanced, and easy to understand.\n"
"\n"
"---\n"
"\n"
"### ✅ Tone & Style:\n"
"\n"
"* Friendly, clear, and informative — like a smart friend who enjoys helping.\n"
"* Avoid robotic or repetitive phrases.\n"
"* Be curious, insightful, and respectful.\n"
"* Build on previous parts of the conversation when relevant.\n"
"\n"
"---\n"
"\n"
"### 🎯 Your goal:\n"
"\n"
"Make every user feel heard, understood, and better informed — whether the answer comes from provided context or external trusted knowledge.\n"
"\n"
"---\n"
"\n"
"Let me know if you’d like a version of this tailored to a **specific use case** (e.g., e-commerce support bot, educational tutor, travel planner, etc.).\n"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question:\n{question}")
        ])
        
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": context, "question": question, "chat_history": chat_history})
        
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

def build_rag_graph():
    """Builds and compiles the RAG graph."""
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

rag_graph = build_rag_graph()
