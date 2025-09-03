import os
from dotenv import load_dotenv

from openai import OpenAI
from transformers import pipeline

# Load your API keys/secrets
load_dotenv()
openai_api_key = os.getenv("openai_api_key")
client = OpenAI(api_key=openai_api_key)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

CATEGORIES = [
    "technical support",
    "billing",
    "account management",
    "product inquiry",
    "returns & refunds",
    "escalations"
]

AGENT_MAPPING = {
    "technical support": "tech_support_agent",
    "billing": "billing_agent",
    "account management": "account_agent",
    "product inquiry": "product_agent",
    "returns & refunds": "returns_agent",
    "escalations": "escalation_queue"
}

def summarize_ticket(text):
    prompt = f"Summarize this support ticket in 2 sentences and extract main keywords:\n\n{text}\n\nSummary and keywords:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    summary = response.choices[0].message.content.strip()
    return summary

def classify_ticket(text):
    # Use Huggingface zero-shot classifier to tag category
    result = classifier(text, CATEGORIES)
    top_category = result['labels'][0]       # Top predicted category (string)
    top_score = result['scores'][0]          # Confidence score for top category (float)
    complexity = "complex" if len(text) > 400 or top_score < 0.6 else "simple"
    return top_category, complexity, result['labels'], result['scores']


def route_ticket(summary, top_category, complexity):
    if complexity == "complex":
        assigned = "escalation_queue"
    else:
        assigned = AGENT_MAPPING.get(top_category, "general_support_agent")
    print(f"Routing ticket: {summary} -> {assigned} ({top_category}, {complexity})")
    return assigned


