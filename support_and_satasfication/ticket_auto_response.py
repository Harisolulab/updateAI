import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("your_openai_key")

# Initialize GPT-4o chat model with API key
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# Prepare FAQ and historical ticket data (replace with your real data or database)
faq_texts = "enter your real data"

# Prepare documents for vector store
docs = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faq_texts]

# Split documents into chunks for better indexing
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=20)
texts = []
for doc in docs:
    texts.extend(text_splitter.split_text(doc))

# Initialize embeddings with API key
embedding = OpenAIEmbeddings()

# Create FAISS vector store from embedded texts
vector_store = FAISS.from_texts(texts, embedding)

# Function to classify user intent
def classify_intent(user_query: str) -> str:
    prompt = f"""Classify the intent of this user query into one of these labels:
order_status, return_policy, shipping_info, account_details, product_info, unknown.

Query: "{user_query}"

Respond with only the intent label."""
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_query)])
    intent = response.content.strip().lower()
    if intent not in [
        "order_status",
        "return_policy",
        "shipping_info",
        "account_details",
        "product_info",
    ]:
        intent = "unknown"
    return intent

# Function to generate response based on intent and vector search
def generate_response(user_query: str) -> str:
    intent = classify_intent(user_query)
    if intent == "unknown":
        return "Sorry, I couldn't understand your query clearly. Please rephrase or contact support."

    # Retrieve relevant FAQ snippets using vector similarity search
    similar_docs = vector_store.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in similar_docs])

    # Compose prompt to GPT-4o to generate an accurate answer
    prompt = f"""
You are a helpful customer support AI. Use the context below to answer the question precisely.

Context:
{context}

Question:
{user_query}

Answer:
"""
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_query)])
    return response.content.strip()


