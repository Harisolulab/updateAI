import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger("ToneRephraseAgent")

# Initialize LLM instance
def get_llm():
    return ChatOpenAI(
        temperature=0.3,
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

# Prompt template for casual to professional tone adaptation
PROMPT_TEMPLATE = """
You are an expert communication assistant. You will be given an informal or casual message.
Rewrite it carefully into a professional, clear, polite, and contextually accurate message suitable for business communication.

Original message:
"{message}"

Rewritten professionally:
"""

def rephrase_to_professional(message: str) -> str:
    try:
        llm = get_llm()
        prompt = PROMPT_TEMPLATE.format(message=message)
        response = llm.invoke(prompt)
        content = getattr(response, "content", None) or str(response)
        return content.strip()
    except Exception as e:
        logger.error(f"Error during tone rephrasing: {e}")
        return f"Error: {str(e)}"
