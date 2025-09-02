import os
import json
from transformers import pipeline
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Updated emotion classifier (no warning)
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion",
    top_k=None
)

# GPT initialization for summary and urgency detection
gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("openai_api_key")
)

def get_emotion_scores(text: str):
    """Return dashboard-ready emotion scores."""
    results = emotion_classifier(text)
    return {entry['label']: entry['score'] for entry in results[0]}

def gpt_sentiment_summary(text: str):
    """Summarize sentiment, flag urgency via GPT."""
    prompt = (
        "You are an AI assistant tasked with detecting sentiment in customer messages.\n"
        "Classify as Positive, Neutral, or Negative and flag urgent emotions as true/false.\n"
        f"Message: \"{text}\"\n"
        "Reply ONLY with JSON: {\"sentiment\": \"Positive\", \"urgent_emotion\": false}"
    )
    response = gpt.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
    try:
        return json.loads(response)
    except Exception:
        return {"sentiment": "Unknown", "urgent_emotion": False}

def live_sentiment_agent(state: dict) -> dict:
    """Workflow agent: annotates state with sentiment and emotion info."""
    message = state.get("raw_message", "").strip()
    if not message:
        return state

    state["emotion_scores"] = get_emotion_scores(message)
    sentiment = gpt_sentiment_summary(message)
    state["sentiment"] = sentiment.get("sentiment", "Unknown")
    state["urgent_emotion"] = sentiment.get("urgent_emotion", False)
    return state

# Example usage
if __name__ == "__main__":
    state = {"raw_message": "I'm upset! The product broke after one day and your support hasn't replied."}
    print("Before:", state)
    updated = live_sentiment_agent(state)
    print("After:", updated)
