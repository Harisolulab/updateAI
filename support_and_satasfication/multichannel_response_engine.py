import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load your OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("openai_api_key")

# Initialize AI Model
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

# ----------- Multichannel Message Schema -----------

class Message:
    def __init__(self, platform, sender, content, metadata=None):
        self.platform = platform  # e.g., 'email', 'whatsapp', 'zendesk', ...
        self.sender = sender
        self.content = content
        self.metadata = metadata or {}

    def normalize(self):
        # Strip extraneous info, map platform-specific fields to universal schema
        return {
            "platform": self.platform,
            "sender": self.sender,
            "content": self.content.strip(),
            "metadata": self.metadata
        }

# ----------- Message Ingestion Pipeline -----------

def ingest_message(raw_msg):
    """
    Receives raw message input from any platform and wraps it as a normalized Message object.
    """
    # Example - convert raw input to Message object (real use: parse from webhook/event)
    msg = Message(
        platform=raw_msg.get("platform"),
        sender=raw_msg.get("sender"),
        content=raw_msg.get("content"),
        metadata=raw_msg.get("metadata", {})
    )
    return msg.normalize()

# ----------- Unified Response Generation -----------

def generate_response(message):
    """
    Takes a normalized message dict and generates a consistent reply using OpenAI LLM.
    """
    prompt = f"You are a customer support AI. Reply with a clear, professional response to this customer, suitable for any channel.\n\nCustomer message:\n{message['content']}\n\nReply:"
    response = llm.predict(prompt)
    return response.strip()

# ----------- Channel Integrations (Stubs) -----------

def send_email(to, content):
    print(f"[Email] To: {to}\nContent: {content}")

def send_whatsapp(to, content):
    print(f"[WhatsApp] To: {to}\nContent: {content}")

def send_chat(to, content, platform="generic"):
    print(f"[Chat-{platform}] To: {to}\nContent: {content}")

def send_zendesk(ticket_id, content):
    print(f"[Zendesk] Ticket ID: {ticket_id}\nContent: {content}")

def send_freshdesk(ticket_id, content):
    print(f"[Freshdesk] Ticket ID: {ticket_id}\nContent: {content}")

def send_salesforce(case_id, content):
    print(f"[Salesforce] Case ID: {case_id}\nContent: {content}")

# ----------- Dispatcher -----------

def dispatch_response(message, response):
    """
    Based on the original platform, send the reply through the correct channel.
    You can also configure to send through ALL channels if needed.
    """
    platform = message["platform"]
    sender = message["sender"]
    meta = message["metadata"]
    # Here, choose which function to call based on 'platform'
    if platform == "email":
        send_email(sender, response)
    elif platform == "whatsapp":
        send_whatsapp(sender, response)
    elif platform == "chat":
        send_chat(sender, response)
    elif platform == "zendesk":
        send_zendesk(meta.get("ticket_id", "N/A"), response)
    elif platform == "freshdesk":
        send_freshdesk(meta.get("ticket_id", "N/A"), response)
    elif platform == "salesforce":
        send_salesforce(meta.get("case_id", "N/A"), response)
    else:
        print(f"[Default] To: {sender}\nContent: {response}")


