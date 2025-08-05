from pinecone import Pinecone
from typing import List

def get_collection_count(index: Pinecone.Index, user_id: str, question: str, chat_history: List[str]):
  stats = index.describe_index_stats()
  namespace_stats = stats.namespaces.get(user_id)
  if not namespace_stats or namespace_stats.vector_count == 0:
        print("No documents exist in the collection for this user.")
        return {"documents": [], "question": question, "chat_history": chat_history, "user_id": user_id}
  
  collection_count = namespace_stats.vector_count
  print(f"Collection Count: {collection_count}")

