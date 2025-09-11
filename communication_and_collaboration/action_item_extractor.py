import os
import requests
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables from .env
load_dotenv()

# Securely fetch API keys and config variables
openai_api_key = os.getenv("openai_api_key")
TRELLO_KEY = os.getenv("TRELLO_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_TOKEN")
TRELLO_LIST_ID = os.getenv("TRELLO_LIST_ID")
CLICKUP_TOKEN = os.getenv("CLICKUP_TOKEN")
CLICKUP_LIST_ID = os.getenv("CLICKUP_LIST_ID")

class ActionItemExtractor:
    """
    Extract actionable meeting items (tasks, follow-ups) from transcript text
    by leveraging GPT function calling.
    """
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided.")
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=openai_api_key,
            streaming=False,
        )
        self.system_prompt = (
            "You are an AI assistant that detects actionable items (tasks, follow-ups, requests) "
            "from meeting transcripts. Each action item should have a short summary, optional due date, "
            "assignee (if identified), and details."
        )
        self.fn_schema = [
            {
                "name": "create_action_item",
                "description": "Create an actionable meeting item.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "assignee": {"type": "string"},
                        "due_date": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["summary", "details"],
                },
            }
        ]

    def extract_action_items(self, meeting_text: str) -> List[Dict[str, Any]]:
        """
        Extract action items from meeting text using GPT function calling.

        Returns:
            List of dicts with keys: summary, assignee, due_date, details.
        """
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=meeting_text),
            ],
            functions=self.fn_schema,
        )

        items = []
        func_calls = getattr(response.additional_kwargs, "function_call", None)
        if func_calls is None:
            # fallback: no function calls, possibly raw text response
            return items

        func_calls = func_calls if isinstance(func_calls, list) else [func_calls]

        for call in func_calls:
            # Arguments are usually JSON-string in function calling, parse if needed
            raw_args = call.get("arguments", "{}")
            try:
                import json
                params = json.loads(raw_args)
            except Exception:
                params = {}

            items.append({
                "summary": params.get("summary", ""),
                "assignee": params.get("assignee", ""),
                "due_date": params.get("due_date", ""),
                "details": params.get("details", "")
            })
        return items


def create_trello_task(
    summary: str,
    details: str,
    assignee: Optional[str] = None,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not all([TRELLO_KEY, TRELLO_TOKEN, TRELLO_LIST_ID]):
        raise RuntimeError("Missing Trello API credentials or list ID.")

    url = "https://api.trello.com/1/cards"
    params = {
        'key': TRELLO_KEY,
        'token': TRELLO_TOKEN,
        'idList': TRELLO_LIST_ID,
        'name': summary,
        'desc': details,
        'due': due_date or None,
    }
    if assignee:
        params['idMembers'] = assignee  # Trello member ID expected

    resp = requests.post(url, params=params)
    resp.raise_for_status()
    return resp.json()


def create_clickup_task(
    summary: str,
    details: str,
    assignee: Optional[str] = None,
    due_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not all([CLICKUP_TOKEN, CLICKUP_LIST_ID]):
        raise RuntimeError("Missing ClickUp API token or list ID.")

    url = f"https://api.clickup.com/api/v2/list/{CLICKUP_LIST_ID}/task"
    headers = {
        "Authorization": CLICKUP_TOKEN,
        "Content-Type": "application/json",
    }
    data = {
        "name": summary,
        "description": details,
    }
    if assignee:
        data["assignees"] = [assignee]
    if due_date:
        from dateutil.parser import parse
        data["due_date"] = int(parse(due_date).timestamp() * 1000)

    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()


def process_and_create_tasks(meeting_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract action items from meeting text and create corresponding tasks in Trello and ClickUp.
    Returns dictionary with keys 'trello' and 'clickup' containing created task details.
    """
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    extractor = ActionItemExtractor(openai_api_key)
    action_items = extractor.extract_action_items(meeting_text)

    trello_results = []
    clickup_results = []

    for item in action_items:
        try:
            if TRELLO_KEY and TRELLO_TOKEN and TRELLO_LIST_ID:
                trello_results.append(create_trello_task(
                    summary=item["summary"],
                    details=item["details"],
                    assignee=item.get("assignee"),
                    due_date=item.get("due_date"),
                ))
        except Exception as e:
            print(f"Failed to create Trello task: {e}")

        try:
            if CLICKUP_TOKEN and CLICKUP_LIST_ID:
                clickup_results.append(create_clickup_task(
                    summary=item["summary"],
                    details=item["details"],
                    assignee=item.get("assignee"),
                    due_date=item.get("due_date"),
                ))
        except Exception as e:
            print(f"Failed to create ClickUp task: {e}")

    return {"trello": trello_results, "clickup": clickup_results}


