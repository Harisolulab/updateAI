import hashlib
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time

logger = logging.getLogger("AlertDeduplicationAgent")

# Load sentence transformer model once
_MODEL = None
_MODEL_LOCK = threading.Lock()

def get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        return _MODEL

class DeduplicationService:
    def __init__(self, similarity_threshold=0.85, cache_ttl=3600):
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        # Store alerts as dict hash -> {embedding: np.array, timestamp: float}
        self.alert_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def _hash_message(self, message: str) -> str:
        return hashlib.sha256(message.encode('utf-8')).hexdigest()

    def _cleanup_cache(self):
        now = time.time()
        to_delete = []
        with self.lock:
            for h, val in self.alert_cache.items():
                if now - val['timestamp'] > self.cache_ttl:
                    to_delete.append(h)
            for h in to_delete:
                del self.alert_cache[h]

    def is_duplicate(self, message: str) -> bool:
        self._cleanup_cache()
        model = get_model()
        msg_hash = self._hash_message(message)
        embedding = model.encode([message])[0]

        with self.lock:
            # Exact hash duplicate?
            if msg_hash in self.alert_cache:
                logger.debug(f"Exact duplicate found by hash: {msg_hash}")
                return True

            # Check cosine similarity for near-duplicates
            for existing_hash, data in self.alert_cache.items():
                sim = cosine_similarity(
                    [embedding], [data['embedding']]
                )[0][0]
                if sim >= self.similarity_threshold:
                    logger.debug(f"Near-duplicate found: {sim:.3f} >= {self.similarity_threshold}")
                    return True

            # No duplicate, store new alert
            self.alert_cache[msg_hash] = {'embedding': embedding, 'timestamp': time.time()}
        return False

    def reset(self):
        with self.lock:
            self.alert_cache.clear()

# Instantiate singleton service
dedup_service = DeduplicationService()
