import os
import logging
from typing import List
from langchain_openai import ChatOpenAI

logger = logging.getLogger("ConflictSentimentAgent")

def get_llm():
    return ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

# Analyze messages for conflict, toxicity, sentiment decay
def detect_conflict_and_sentiment_decay(messages: List[str]) -> List[dict]:
    """
    messages: list of message strings in conversation order.
    returns: list of dicts with message index, conflict flag, sentiment score.
    """
    llm = get_llm()
    conversation_text = "\n".join([f"Message {i+1}: {msg}" for i, msg in enumerate(messages)])

    prompt = (
        "You are a workplace communication assistant.\n"
        "Analyze the following conversation messages for conflict phrases, toxicity, and sentiment decay.\n"
        "For each message, respond with:\n"
        "- conflict: true/false\n"
        "- sentiment_score: float [-1.0 (negative) to 1.0 (positive)]\n"
        "Return JSON array with these fields per message.\n\n"
        f"{conversation_text}\n\nAnalysis:"
    )

    response = llm.invoke(prompt)
    content = getattr(response, "content", None) or str(response)

    import json
    try:
        results = json.loads(content)
        if not isinstance(results, list):
            raise ValueError("Expected JSON array in response")
        return results
    except Exception as e:
        logger.error(f"Error parsing conflict sentiment detection output: {e}")
        # Fallback empty analysis
        return [{"conflict": False, "sentiment_score": 0.0} for _ in messages]
