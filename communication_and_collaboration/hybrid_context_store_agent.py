import os
import logging
import redis
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("HybridContextStore")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))  # 1 hour TTL for short term cache

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "conversation_context")

# Redis client for short-term context caching
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Chroma client for long term vector storage (in-memory by default)
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# Sentence transformer embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> List[float]:
    """Compute embedding and return as a Python list of floats."""
    return embedder.encode([text])[0].tolist()


class HybridContextStore:
    def __init__(self, redis_client, chroma_collection, ttl: int = REDIS_TTL):
        self.redis = redis_client
        self.chroma = chroma_collection
        self.ttl = ttl

    def store_short_term_context(self, session_id: str, text: str):
        """
        Store text and embedding in Redis with TTL for short-term recall
        """
        embedding = embed_text(text)
        key_text = f"context:text:{session_id}"
        key_embedding = f"context:embedding:{session_id}"

        pipe = self.redis.pipeline()
        pipe.set(key_text, text, ex=self.ttl)
        pipe.set(key_embedding, str(embedding), ex=self.ttl)
        pipe.execute()
        logger.debug(f"Stored short-term context for {session_id} with TTL {self.ttl}s")

    def fetch_short_term_context(self, session_id: str) -> Optional[str]:
        text = self.redis.get(f"context:text:{session_id}")
        if text:
            return text.decode("utf-8")
        return None

    def store_long_term_context(self, session_id: str, text: str):
        """
        Store text embedding in Chroma for long-term persistent similarity search
        """
        embedding = embed_text(text)
        # Upsert document
        self.chroma.upsert(
            ids=[session_id],
            documents=[text],
            embeddings=[embedding],
        )
        logger.debug(f"Stored long-term context for {session_id} in Chroma")

    def query_similar_contexts(self, text: str, top_k: int = 5):
        """
        Query similar past contexts in Chroma vector DB
        """
        embedding = embed_text(text)
        results = self.chroma.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        return results


# Singleton instance
hybrid_context_store = HybridContextStore(redis_client, chroma_collection)