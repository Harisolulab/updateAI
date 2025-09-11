from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("openai_api_key")
class ToneRephraser:
    """
    Rephrases informal/casual messages to professional, clear, and polite tone
    using OpenAI GPT-4o and prompt engineering.
    """
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
        self.system_prompt = (
            "You are a professional assistant. Rephrase messages to be clear, polite, "
            "and suitable for formal workplace communication. Do not change the original meaning."
        )

    def rephrase(self, informal_text: str) -> str:
        prompt = (
            f"Rephrase this message in a formal, professional tone:\n"
            f"{informal_text}\n\nProfessional version:"
        )
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()



