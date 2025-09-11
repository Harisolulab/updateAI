import hashlib
from typing import List
from sentence_transformers import SentenceTransformer, util
import torch

class AlertDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes = set()
        self.embeddings = []               # List of tensors
        self.alert_texts = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _hash_alert(self, alert_text: str) -> str:
        return hashlib.sha256(alert_text.encode('utf-8')).hexdigest()

    def is_duplicate(self, alert_text: str) -> bool:
        alert_hash = self._hash_alert(alert_text)
        if alert_hash in self.seen_hashes:
            return True

        query_emb = self.model.encode(alert_text, convert_to_tensor=True)
        if self.embeddings:
            embeddings_tensor = torch.stack(self.embeddings)
            similarities = util.cos_sim(query_emb, embeddings_tensor)[0]
            max_sim = similarities.max().item()
            if max_sim >= self.similarity_threshold:
                return True

        self.seen_hashes.add(alert_hash)
        self.embeddings.append(query_emb)
        self.alert_texts.append(alert_text)
        return False

    def filter_alerts(self, alerts: List[str]) -> List[str]:
        unique_alerts = []
        for alert in alerts:
            if not self.is_duplicate(alert):
                unique_alerts.append(alert)
        return unique_alerts

