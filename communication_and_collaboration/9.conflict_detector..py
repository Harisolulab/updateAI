from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
import warnings

# Suppress warnings about max_length if you want cleaner logs
warnings.filterwarnings("ignore")


class ConflictDetector:
    """
    Detect conflict, toxicity, or sentiment decay in conversation text.
    Supports lightweight fine-tuned RoBERTa classifier usage or GPT-4o context-based detection.
    """

    def __init__(
            self,
            use_gpt: bool = False,
            roberta_model_name: str = "cardiffnlp/twitter-roberta-base-offensive",
            device: str = None,
    ):
        """
        Args:
            use_gpt: If True, uses GPT-4o model for context-based conflict detection.
                     Else, uses fine-tuned RoBERTa model for text classification.
            roberta_model_name: Huggingface model name for RoBERTa-based classifier.
            device: "cuda" or "cpu", defaults to cuda if available.
        """
        self.use_gpt = use_gpt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not self.use_gpt:
            self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name).to(self.device)
            self.labels = ["no_conflict", "conflict"]  # Updated label names

        else:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY must be set in environment for GPT mode.")
            self.gpt = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

    def roberta_predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict conflict/toxicity probability on list of texts using RoBERTa classifier.
        Returns a list of dicts with {"text": ..., "conflict_prob": ..., "label": ...}.
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,  # Set max length to avoid warnings
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**encodings)
        probs = F.softmax(outputs.logits, dim=-1)
        results = []
        for text, prob in zip(texts, probs):
            conflict_prob = prob[1].item()  # conflict class probability
            label = self.labels[1] if conflict_prob >= 0.5 else self.labels[0]
            results.append({"text": text, "conflict_prob": conflict_prob, "label": label})
        return results

    def gpt_predict(self, texts: List[str]) -> List[Dict]:
        """
        Uses GPT-4o to detect conflict/toxicity in each text with contextual prompt.
        Returns same output schema as roberta_predict.
        """
        results = []
        prompt_template = (
            "You are a helpful assistant trained to detect conflict, toxicity, or "
            "negative sentiment decay in workplace conversations. Classify the text as "
            "'conflict' or 'no_conflict'. Also provide confidence as a score from 0 to 1.\n\n"
            "Text: \"{text}\"\n\n"
            "Respond with JSON: {{\"label\": \"conflict\" or \"no_conflict\", \"confidence\": float}}"
        )
        for text in texts:
            prompt = prompt_template.format(text=text)
            response = self.gpt.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
            try:
                import json
                parsed = json.loads(response.content)
            except Exception:
                parsed = {"label": "no_conflict", "confidence": 0.0}
            parsed["text"] = text
            results.append(parsed)
        return results

    def detect_conflicts(self, texts: List[str]) -> List[Dict]:
        """
        Main entry: detects conflicts in list of texts.
        """
        if self.use_gpt:
            return self.gpt_predict(texts)
        else:
            return self.roberta_predict(texts)


