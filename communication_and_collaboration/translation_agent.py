import os
import logging
import requests
from langchain_openai import ChatOpenAI

logger = logging.getLogger("TranslationAgent")

DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

def detect_language(text: str) -> str:
    # Simple detection can be done via DeepL or GPT fallback
    # Here we rely on DeepL auto-detect feature when translating
    return "auto"

def deepl_translate(text: str, target_lang: str) -> str:
    if not DEEPL_API_KEY:
        raise RuntimeError("DeepL API key is not configured")
    params = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "target_lang": target_lang.upper(),
        # Source_lang set to auto detect if you want
    }
    response = requests.post(DEEPL_API_URL, data=params)
    if response.status_code == 200:
        result = response.json()
        translations = result.get("translations", [])
        if translations:
            return translations[0].get("text", "")
    logger.warning(f"DeepL translation failed: {response.text}")
    raise RuntimeError("DeepL translation failed")

def gpt_translate(text: str, target_lang: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = (
        f"Translate the following text to {target_lang} maintaining the original tone and context.\n"
        f"Text:\n{text}\n\nTranslation:"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()

def translate_text(text: str, target_lang: str) -> str:
    try:
        return deepl_translate(text, target_lang)
    except Exception as e:
        logger.warning(f"DeepL failed, falling back to GPT translation: {e}")
        return gpt_translate(text, target_lang)
