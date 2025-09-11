import os
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web import WebClient
import openai
import asyncio

# Load environment variables (SLACK_BOT_TOKEN, SLACK_APP_TOKEN, OPENAI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("openai_api_key")
if not (SLACK_BOT_TOKEN and SLACK_APP_TOKEN and OPENAI_API_KEY):
    raise ValueError("Please set SLACK_BOT_TOKEN, SLACK_APP_TOKEN, OPENAI_API_KEY in your environment.")

openai.api_key = OPENAI_API_KEY

app = App(token=SLACK_BOT_TOKEN)


# --- Helper: GPT prompt generation & streaming completion call ---

async def fetch_gpt_suggestions(user_message: str) -> str:
    """
    Asynchronously query GPT-4 for inline reply suggestions based on user_message.
    Returns formatted suggestions string separated by newlines.
    """
    prompt = (
        "You are an assistant suggesting concise helpful Slack reply options to a user's message.\n"
        "Provide 3 short, relevant reply suggestions. Each suggestion should be brief and actionable.\n"
        "User message: " + user_message + "\nSuggestions:\n"
    )
    # Use OpenAI Async API (need python >= 3.7)
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=100,
        temperature=0.5,
        n=1,
        stop=None,
    )
    text = response.choices[0].message.content.strip()
    # Expecting suggestions separated by lines or bullets, normalize:
    suggestions = [s.strip("- ").strip() for s in text.split("\n") if s.strip()]
    # Return as list to caller
    return suggestions


# --- Slack command to trigger modal with GPT inline suggestions ---

@app.command("/suggest-replies")
def open_suggestion_modal(ack, body, client: WebClient):
    ack()
    trigger_id = body["trigger_id"]
    user_message = body.get("text", "").strip()
    if not user_message:
        user_message = "Hello, can you help me with this?"

    # Open a modal with initial loading text, real suggestions fetched asynchronously later
    client.views_open(
        trigger_id=trigger_id,
        view={
            "type": "modal",
            "callback_id": "suggestions_modal",
            "title": {"type": "plain_text", "text": "GPT Reply Suggestions"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "user_message_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "user_message_input",
                        "initial_value": user_message,
                        "multiline": True,
                    },
                    "label": {"type": "plain_text", "text": "Message to respond to"},
                },
                {
                    "type": "section",
                    "block_id": "suggestions_block",
                    "text": {"type": "mrkdwn", "text": "_Loading suggestions..._"},
                },
                {
                    "type": "actions",
                    "block_id": "actions_block",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "refresh_suggestions",
                            "text": {"type": "plain_text", "text": "Refresh Suggestions"},
                        }
                    ],
                },
            ],
            "submit": {"type": "plain_text", "text": "Close"},
        },
    )


# --- Listen to refresh button click, fetch GPT suggestions and update modal ---

@app.action("refresh_suggestions")
def handle_refresh_suggestions(ack, body, client: WebClient):
    ack()

    # Extract user input from modal state
    state_values = body["view"]["state"]["values"]
    user_message = ""
    for block in state_values.values():
        # Plain text input for user_message_input
        if "user_message_input" in block:
            user_message = block["user_message_input"]["value"]
            break

    # Launch async task to fetch GPT suggestions and update modal
    asyncio.run(update_suggestions_modal(client, body["view"]["id"], user_message))


async def update_suggestions_modal(client: WebClient, view_id: str, user_message: str):
    suggestions = await fetch_gpt_suggestions(user_message)
    if not suggestions:
        suggestions_text = "_No suggestions generated._"
    else:
        suggestions_text = "\n".join(f"*{i+1}.* {s}" for i, s in enumerate(suggestions))

    # Update modal view with new suggestions
    client.views_update(
        view_id=view_id,
        hash="",  # Optional: can include to control concurrency
        view={
            "type": "modal",
            "callback_id": "suggestions_modal",
            "title": {"type": "plain_text", "text": "GPT Reply Suggestions"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "user_message_block",
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "user_message_input",
                        "initial_value": user_message,
                        "multiline": True,
                    },
                    "label": {"type": "plain_text", "text": "Message to respond to"},
                },
                {
                    "type": "section",
                    "block_id": "suggestions_block",
                    "text": {
                        "type": "mrkdwn",
                        "text": suggestions_text,
                    },
                },
                {
                    "type": "actions",
                    "block_id": "actions_block",
                    "elements": [
                        {
                            "type": "button",
                            "action_id": "refresh_suggestions",
                            "text": {"type": "plain_text", "text": "Refresh Suggestions"},
                        }
                    ],
                },
            ],
            "submit": {"type": "plain_text", "text": "Close"},
        },
    )


