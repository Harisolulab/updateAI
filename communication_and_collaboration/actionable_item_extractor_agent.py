import os
import logging
import requests
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger("ActionableItemExtractorAgent")

# Initialize OpenAI GPT-4o for actionable item extraction
def get_llm():
    return ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

# Sample OpenAI function calling style to extract tasks from meeting text
def extract_actionable_items(meeting_text: str) -> List[Dict[str, Any]]:
    llm = get_llm()

    prompt = (
        "You are an AI assistant. Extract all actionable items from the following meeting notes.\n"
        "For each item, provide:\n"
        "- description\n"
        "- owner (if mentioned)\n"
        "- due date (if mentioned)\n"
        "- tags (if any)\n\n"
        f"Meeting notes:\n{meeting_text}\n\n"
        "Return JSON array of action items:"
    )

    response = llm.invoke(prompt)
    content = getattr(response, "content", None) or str(response)

    try:
        import json
        items = json.loads(content)
        if not isinstance(items, list):
            raise ValueError("Response is not a list")
        return items
    except Exception as e:
        logger.error(f"Failed to parse actionable items JSON: {e}")
        # Fallback: return empty list
        return []

# Simple Trello task creator (replace with full API integration)
def create_trello_task(api_key: str, token: str, list_id: str, task: Dict[str, Any]) -> bool:
    url = f"https://api.trello.com/1/cards"
    query = {
        "key": api_key,
        "token": token,
        "idList": list_id,
        "name": task.get("description", "New Task"),
        "desc": f"Owner: {task.get('owner', 'Unassigned')}\nDue: {task.get('due_date', 'N/A')}\nTags: {', '.join(task.get('tags', []))}"
    }
    response = requests.post(url, params=query)
    if response.status_code == 200:
        logger.info("Created Trello task successfully")
        return True
    else:
        logger.error(f"Failed to create Trello task: {response.text}")
        return False

# Simple ClickUp task creator (replace with full API integration)
def create_clickup_task(api_token: str, list_id: str, task: Dict[str, Any]) -> bool:
    url = f"https://api.clickup.com/api/v2/list/{list_id}/task"
    headers = {
        "Authorization": api_token,
        "Content-Type": "application/json"
    }
    payload = {
        "name": task.get("description", "New Task"),
        "assignees": [],  # Map owner if possible
        "due_date": None, # Convert due_date string to timestamp if possible
        "tags": task.get("tags", []),
        "description": f"Created from meeting AI extraction."
    }
    import json
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200 or response.status_code == 201:
        logger.info("Created ClickUp task successfully")
        return True
    else:
        logger.error(f"Failed to create ClickUp task: {response.text}")
        return False

# Integration layer to parse meeting notes and create tasks
def detect_and_create_tasks(
    meeting_text: str,
    trello_config: Optional[Dict[str, str]] = None,
    clickup_config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:

    tasks = extract_actionable_items(meeting_text)
    results = {"created_trello": 0, "created_clickup": 0, "tasks_detected": len(tasks)}

    # Create tasks in Trello
    if trello_config:
        for task in tasks:
            success = create_trello_task(
                api_key=trello_config.get("api_key"),
                token=trello_config.get("token"),
                list_id=trello_config.get("list_id"),
                task=task
            )
            if success:
                results["created_trello"] += 1

    # Create tasks in ClickUp
    if clickup_config:
        for task in tasks:
            success = create_clickup_task(
                api_token=clickup_config.get("api_token"),
                list_id=clickup_config.get("list_id"),
                task=task
            )
            if success:
                results["created_clickup"] += 1

    return results
