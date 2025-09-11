# agents/communication_and_collaboration/11.persona_tone_adjuster.py

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("openai_api_key")

class PersonaToneAdjuster:
    """
    Adjust message tone dynamically based on recipient's role, seniority, language, and communication preferences.
    Uses prompt engineering with GPT-4o to reformulate messages appropriately.
    """

    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
        # Define baseline persona prompt components for role and language
        self.persona_prompts = {
            "intern": {
                "en": "You are an AI assistant reformulating messages for an intern. Use simple language, explain technical terms clearly, and maintain a polite and supportive tone.",
                "fr": "Vous êtes un assistant IA reformulant les messages pour un stagiaire. Utilisez un langage simple, expliquez clairement les termes techniques et maintenez un ton poli et encourageant.",
                "de": "Sie sind ein KI-Assistent, der Nachrichten für einen Praktikanten umformuliert. Verwenden Sie einfache Sprache, erklären Sie technische Begriffe klar und bewahren Sie einen höflichen und unterstützenden Ton.",
                "es": "Eres un asistente de IA que reformula mensajes para un becario. Usa un lenguaje simple, explica términos técnicos claramente y mantén un tono cortés y alentador."
            },
            "pm": {
                "en": "You are an AI assistant reformulating messages for a project manager. Use a casual but professional tone with clear and concise language tuned for busy professionals.",
                "fr": "Vous êtes un assistant IA reformulant les messages pour un chef de projet. Utilisez un ton décontracté mais professionnel avec un langage clair et concis adapté aux professionnels occupés.",
                "de": "Sie sind ein KI-Assistent, der Nachrichten für einen Projektleiter umformuliert. Verwenden Sie einen lockeren, aber professionellen Ton mit klarer und prägnanter Sprache für beschäftigte Fachleute.",
                "es": "Eres un asistente de IA que reformula mensajes para un gerente de proyecto. Usa un tono casual pero profesional con un lenguaje claro y conciso dirigido a profesionales ocupados."
            },
            "executive": {
                "en": "You are an AI assistant reformulating messages for an executive. Use formal, respectful, and polished language with emphasis on key points and strategic insights.",
                "fr": "Vous êtes un assistant IA reformulant les messages pour un cadre dirigeant. Utilisez un langage formel, respectueux et soigné en insistant sur les points clés et les perspectives stratégiques.",
                "de": "Sie sind ein KI-Assistent, der Nachrichten für einen Geschäftsführer umformuliert. Verwenden Sie formelle, respektvolle und ausgefeilte Sprache mit Schwerpunkt auf Schlüsselpunkten und strategischen Erkenntnissen.",
                "es": "Eres un asistente de IA que reformula mensajes para un ejecutivo. Usa un lenguaje formal, respetuoso y pulido con énfasis en los puntos clave e ideas estratégicas."
            }
            # Add more roles and languages as needed
        }

    def adjust_tone(self, message: str, role: str, language: str = "en") -> str:
        """
        Reformulate message by given role and language.
        """
        role = role.lower()
        language = language.lower()
        if role not in self.persona_prompts:
            raise ValueError(f"Unknown role '{role}'. Valid roles: {list(self.persona_prompts.keys())}")
        if language not in self.persona_prompts[role]:
            raise ValueError(f"Unsupported language '{language}'. Supported languages for role {role}: {list(self.persona_prompts[role].keys())}")

        persona_prompt = self.persona_prompts[role][language]
        system_message = SystemMessage(content=persona_prompt)
        user_prompt = (
            f"Original message:\n{message}\n\n"
            f"Rephrase the above message in a way that suits a {role} and is in {language}."
        )
        human_message = HumanMessage(content=user_prompt)

        response = self.llm.invoke([system_message, human_message])
        return response.content.strip()





