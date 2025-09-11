# agents/communication_and_collaboration/12.context_cache_agent.py

import os
import time
import redis
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class HybridContextCacheAgent:
    """
    Hybrid memory store combining Redis (short-term cache with TTL)
    and ChromaDB (long-term vector storage) to cache conversation context embeddings.
    """

    REDIS_TTL_SECONDS = 3600  # Short-term cache TTL (1 hour)

    def __init__(self, redis_url: str, chroma_dir: str, openai_api_key: str, model_name: str = "all-MiniLM-L6-v2"):
        # Redis client for short-term context caching
        self.redis_client = redis.from_url(redis_url)
        # Chroma client for long-term vector similarity search
        self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chroma_dir))
        self.collection = self._get_or_create_collection("conversation_context")
        # Sentence embedding model for vector representation
        self.embedder = SentenceTransformer(model_name)
        # OpenAI client for responses (optional, if needed for downstream usage)
        self.chat_model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    def _get_or_create_collection(self, name: str):
        try:
            return self.chroma_client.get_collection(name)
        except Exception:
            return self.chroma_client.create_collection(name)

    def _make_redis_key(self, session_id: str) -> str:
        return f"context_cache:{session_id}"

    def store_short_term_context(self, session_id: str, messages: List[str]):
        """
        Store conversation context in Redis with TTL.
        Messages are concatenated before saving.
        """
        key = self._make_redis_key(session_id)
        concatenated = "\n".join(messages)
        self.redis_client.setex(key, self.REDIS_TTL_SECONDS, concatenated)

    def retrieve_short_term_context(self, session_id: str) -> Optional[str]:
        """
        Retrieve conversation context from Redis short-term cache.
        """
        key = self._make_redis_key(session_id)
        result = self.redis_client.get(key)
        if result:
            return result.decode("utf-8")
        return None

    def store_long_term_context(self, session_id: str, texts: List[str]):
        """
        Store conversation snippets with embeddings in Chroma vector DB.
        """
        embeddings = self.embedder.encode(texts).tolist()
        ids = [f"{session_id}_{int(time.time()*1000)}_{i}" for i in range(len(texts))]
        metadatas = [{"session_id": session_id, "timestamp": time.time()} for _ in texts]
        self.collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
        self.chroma_client.persist()

    def retrieve_long_term_context(self, session_id: str, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top_k relevant context messages for a query from long-term store.
        """
        if self.collection.count() == 0:
            return []
        query_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k, where={"session_id": session_id})
        return results["documents"][0] if results["documents"] else []

    def get_combined_context(self, session_id: str, query: str, top_k: int = 5) -> str:
        """
        Retrieve combined short-term and long-term context for coherent response generation.
        """
        short_term = self.retrieve_short_term_context(session_id) or ""
        long_term_list = self.retrieve_long_term_context(session_id, query, top_k)
        long_term = "\n".join(long_term_list) if long_term_list else ""
        combined = "\n\n-- Short-Term Context --\n" + short_term + "\n\n-- Long-Term Context --\n" + long_term
        return combined.strip()

    def update_context(self, session_id: str, new_messages: List[str]):
        """
        Update both short-term and long-term caches with new conversation messages.
        """
        # Update short term: keep last messages (can customize, here all)
        existing = self.retrieve_short_term_context(session_id)
        prev_msgs = existing.split("\n") if existing else []
        updated_msgs = prev_msgs + new_messages
        # Optionally trim to recent n messages; here keep all for example:
        self.store_short_term_context(session_id, updated_msgs)

        # Update long term vector db with new messages
        self.store_long_term_context(session_id, new_messages)


