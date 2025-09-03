import re
import pytesseract
from PIL import Image
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import os
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()
API_KEY = os.getenv("openai_api_key")

# Initialize GPT-4 chat model
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=API_KEY)

def ocr_extract_text(image_path: str) -> str:
    """
    Convert scanned survey image into searchable text using OCR (Tesseract).
    """
    text = pytesseract.image_to_string(Image.open(image_path))
    return text

def parse_nps_score(text: str) -> int:
    """
    Parse NPS score (0-10) from survey text using regex.
    """
    match = re.search(r'(\b10\b|\b[0-9]\b)', text)
    if match:
        return int(match.group(0))
    return -1  # Indicates no valid score found

def analyze_feedback_sentiment(feedback: str) -> Dict[str, Any]:
    """
    Use GPT-4o to classify sentiment (Positive, Neutral, Negative) 
    and identify key drivers for satisfaction/dissatisfaction from text feedback.
    Returns dictionary with sentiment and driver list.
    """
    prompt = (
        f"You are an AI assistant analyzing customer feedback.\n"
        f"Feedback: \"{feedback}\"\n"
        f"Classify overall sentiment as Positive, Neutral, or Negative.\n"
        f"List main satisfaction/dissatisfaction drivers as comma-separated keywords.\n"
        f"Return JSON like {{\"sentiment\": \"Positive\", \"drivers\": [\"fast service\", \"friendly staff\"]}}"
    )
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=feedback)])
    
    try:
        result = json.loads(response.content)
    except Exception:
        result = {"sentiment": "Unknown", "drivers": []}
    return result

def process_surveys(surveys: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes survey data with optional scanned images and feedback text.
    Returns list with NPS scores, sentiment analysis, and drivers.
    
    Each survey dict expected keys:
        - 'response_text' (optional for digital surveys)
        - 'image_path' (optional, for scanned images)
        - 'feedback' (optional, free-text feedback)
    """
    processed_results = []
    for survey in surveys:
        # Extract text from scanned image if provided
        if 'image_path' in survey and os.path.exists(survey['image_path']):
            text = ocr_extract_text(survey['image_path'])
        else:
            text = survey.get('response_text', '')
        
        # Parse NPS score from extracted text
        nps_score = parse_nps_score(text)
        
        # Analyze feedback sentiment and drivers
        feedback = survey.get('feedback', '')
        sentiment_info = analyze_feedback_sentiment(feedback)
        
        processed_results.append({
            "nps_score": nps_score,
            "sentiment": sentiment_info.get("sentiment"),
            "drivers": sentiment_info.get("drivers"),
            "raw_feedback": feedback
        })
    return processed_results

