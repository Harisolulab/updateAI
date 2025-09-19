import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger("PersonaToneAgent")

# Example persona matrix with tone styles
PERSONA_MATRIX = {
    "intern": {
        "role": "intern",
        "language": "en",
        "tone": "formal and simple",
        "description": "Use formal and clear language suitable for an intern."
    },
    "pm": {
        "role": "project_manager",
        "language": "en",
        "tone": "casual and concise",
        "description": "Use casual and concise language suitable for project managers."
    },
    "executive_fr": {
        "role": "executive",
        "language": "fr",
        "tone": "formal and respectful",
        "description": "Utilisez un ton formel et respectueux propre aux cadres."
    },
    # Add more personas as needed
}

def get_llm():
    return ChatOpenAI(
        temperature=0.2,
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def adjust_tone_by_persona(message: str, persona_key: str) -> str:
    persona = PERSONA_MATRIX.get(persona_key.lower(), None)
    if persona is None:
        logger.warning(f"Persona '{persona_key}' not found, using neutral tone.")
        persona = {
            "tone": "neutral",
            "description": "Use a neutral and clear tone appropriate for general audiences."
        }

    llm = get_llm()
    prompt = (
        f"Rephrase the following message according to this persona:\n"
        f"Role: {persona.get('role', 'general')}\n"
        f"Language: {persona.get('language', 'en')}\n"
        f"Tone: {persona.get('tone')}\n"
        f"Description: {persona.get('description')}\n\n"
        f"Message:\n{message}\n\n"
        f"Rephrased message:"
    )

    response = llm.invoke(prompt)
    content = getattr(response, "content", None) or str(response)
    return content.strip()
