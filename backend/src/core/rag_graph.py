from typing import List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from src.core.config import RETRIEVAL_CONFIG
from src.core.retriever import preprocess_query, evaluate_retrieval_quality

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generation.
        documents: A list of retrieved documents.
        chat_history: The history of the conversation.
    """
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]

def retrieve_documents(state):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    chat_history = state.get("chat_history", [])

    from src.core.config import vectorstore, retriever

    if not vectorstore:
        print("ERROR: No vector store available")
        return {"documents": [], "question": question, "chat_history": chat_history}

    try:
        collection_count = vectorstore._collection.count()
        print(f"Collection Count: {collection_count}")

        if collection_count == 0:
            print("No document exists in the collection.")
            return {"documents": [], "question": question, "chat_history": chat_history}
    except Exception as e:
        print(f"ERROR: Failed to get collection count: {e}")
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
            print(f"Advanced retrieval successful with {len(retrieved_docs)} documents")
        else:
            print("WARNING: No retriever available, using basic similarity search")
            retrieved_docs = vectorstore.similarity_search(
                processed_question,
                k=RETRIEVAL_CONFIG['default_k']
            )
    except Exception as e:
        print(f"Retrieval failed, falling back to basic retrieval: {e}")
        try: # Fallback to basic if advanced failed
            retrieved_docs = vectorstore.similarity_search(
                processed_question,
                k=RETRIEVAL_CONFIG['default_k']
            )
        except Exception as fallback_error:
            print(f"ERROR: Basic retrieval also failed: {fallback_error}")
            return {"documents": [], "question": question, "chat_history": chat_history}

    if retrieved_docs:
        try:
            docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
                processed_question,
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
                print(f"Filtered to {len(retrieved_docs)} high-quality documents")
            else:
                print("No documents met similarity threshold, using all retrieved docs")
        except Exception as e:
            print(f"ERROR: Failed to filter documents by relevance: {e}")

    print(f"--- CHUNKS RETRIEVED FOR QUESTION: '{question}' ---")
    print(f"Total chunks retrieved: {len(retrieved_docs)}")
    for i, doc in enumerate(retrieved_docs):
        try:
            print(f"\n--- RETRIEVED CHUNK {i+1}/{len(retrieved_docs)} ---")
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            print(f"Content: {content[:200]}...")
            print(f"Metadata: {getattr(doc, 'metadata', {})}")
            print("-" * 50)
        except Exception as e:
            print(f"ERROR: Failed to print chunk {i+1}: {e}")

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
    return {"documents": documents, "question": question, "chat_history": chat_history}

def generate_answer(state):
    print("---GENERATING ANSWER---")

    from src.core.config import llm
    try:
        question = state.get("question", "")
        documents = state.get("documents", [])
        chat_history = state.get("chat_history", [])

        if not question:
            print("WARNING: No question provided in state")
            return {"documents": [], "question": "", "chat_history": chat_history, "generation": "No question provided."}

        context = "\n\n".join(str(doc) for doc in documents) if documents else "No relevant context found."
        print(f"Context (first 500 chars): {context[:500]}...")

        if llm is None:
            print("ERROR: LLM not available")
            return {
                "documents": documents,
                "question": question,
                "chat_history": chat_history,
                "generation": "I apologize, but the language model is not available at the moment. Please try again later."
            }

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Here is an improved version of your prompt, specifically optimized to:\n"
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
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
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
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

rag_graph = build_rag_graph()