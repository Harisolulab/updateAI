# agents/communication_and_collaboration/10.message_translator.py

import os
import requests
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
# If you use the newer LangChain, you can import from `langchain_openai` instead.
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


def _load_env_walkup(filename: str = ".env") -> list[str]:
    """
    Walk up from this file's directory to root, loading the first .env found.
    Returns the list of paths that were checked (for error messages).
    """
    tried = []
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / filename
        tried.append(str(candidate))
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=True)
            break
    return tried


class MessageTranslator:
    """
    Translates messages across English, French, German, and Spanish.
    Uses DeepL API for high-fidelity translation; falls back to GPT-4o.
    Supports automatic language detection and tone/context preservation.
    """

    SUPPORTED_LANGUAGES = {"en", "fr", "de", "es"}

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        deepl_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        # Load environment (searches for .env up the tree)
        tried_paths = _load_env_walkup()

        # Prefer explicit args, then env vars
        self.openai_api_key = (
            openai_api_key
            or os.getenv("openai_api_key")
            or os.environ.get("openai_api_key")
        )
        self.deepl_api_key = (
            deepl_api_key
            or os.getenv("DEEPL_API_KEY")
            or os.environ.get("DEEPL_API_KEY")
        )

        if not self.openai_api_key:
            details = [
                "OPENAI_API_KEY not found.",
                f"cwd: {Path.cwd()}",
                f"script: {Path(__file__).resolve()}",
                "searched for .env at (first existing is used):",
                *("  - " + p for p in tried_paths),
                "",
                "Fix by one of these:",
                "1) Put a .env file with:",
                "   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "   DEEPL_API_KEY=your-deepl-key   # optional",
                "2) Or pass the key into MessageTranslator(openai_api_key='sk-...').",
                "3) Or export OPENAI_API_KEY in your shell/VS Code env.",
            ]
            raise ValueError("\n".join(details))

        # Initialize LLM
        self.chat_model = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=self.openai_api_key,
        )

    # ---------- Translation backends ----------

    def translate_deepl(self, text: str, target_lang: str) -> Optional[str]:
        """
        Translate text using DeepL API.
        Returns translated text or None if failure or key missing.
        """
        if not self.deepl_api_key:
            return None

        url = "https://api-free.deepl.com/v2/translate"
        data = {
            "auth_key": self.deepl_api_key,
            "text": text,
            "target_lang": target_lang.upper(),  # EN, FR, DE, ES
        }

        try:
            resp = requests.post(url, data=data, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            translations = payload.get("translations", [])
            if translations:
                return translations[0].get("text")
        except Exception:
            pass
        return None

    def translate_gpt(self, text: str, target_lang: str) -> str:
        """
        Use GPT to translate, preserving tone/context.
        """
        system_prompt = (
            f"You are a precise translation assistant. Translate into {target_lang.upper()} "
            f"while preserving tone and context. Return only the translation."
        )
        user_prompt = f"{text}"
        response = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return response.content.strip()

    def detect_language(self, text: str) -> Optional[str]:
        """
        Language detection via LLM. Returns ISO 639-1 code if supported.
        """
        system_prompt = (
            "Identify the language of the user's message. "
            "Respond with ONLY the ISO 639-1 code (e.g., en, fr, de, es)."
        )
        response = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text),
            ]
        )
        code = response.content.strip().lower()
        return code if code in self.SUPPORTED_LANGUAGES else None

    # ---------- Public API ----------

    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate `text` into `target_lang` with auto detection and DeepL->GPT fallback.
        """
        target_lang = target_lang.lower()
        if target_lang not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported target language '{target_lang}'. "
                f"Supported: {sorted(self.SUPPORTED_LANGUAGES)}"
            )

        src_lang = self.detect_language(text)
        if src_lang == target_lang:
            return text  # No translation needed

        # Try DeepL first if available
        deepl_result = self.translate_deepl(text, target_lang)
        if deepl_result:
            return deepl_result

        # Fallback to GPT
        return self.translate_gpt(text, target_lang)


