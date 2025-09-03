import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables from .env file (make sure OPENAI_API_KEY is set)
load_dotenv()
api_key = os.getenv("openai_api_key")

# Initialize GPT-4o chat model
llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key, temperature=0)

def generate_ticket_summary(conversation_thread: str) -> str:
    """
    Takes a multi-turn ticket conversation as plain text,
    generates a concise markdown summary for dashboards.
    """
    system_prompt = (
        "You are a helpful AI assistant that summarizes customer support "
        "ticket conversations. Create a clear, concise markdown summary."
    )
    user_prompt = (
        f"Here is the full conversation thread:\n\n{conversation_thread}\n\n"
        "Provide a short markdown summary highlighting key points, "
        "resolution status, and any pending actions."
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content.strip()

 




