import os
import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

logger = logging.getLogger("SlackInlineSuggestionAgent")

class SlackInlineSuggestionAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            temperature=0.4,
            model="gpt-4o",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def build_prompt(self, conversation_context: List[Dict[str, Any]]) -> str:
        # Build a prompt from recent conversation context in Slack
        messages = "\n".join([f"{msg['user']}: {msg['text']}" for msg in conversation_context])
        prompt = (
            "You are a helpful assistant that suggests relevant, concise reply options and actionable prompts "
            "based on the following Slack conversation context. Provide 3 reply suggestions suitable for Slack buttons or modals.\n\n"
            f"Conversation:\n{messages}\n\nSuggestions:"
        )
        return prompt

    def get_suggestions(self, conversation_context: List[Dict[str, Any]]) -> List[str]:
        prompt = self.build_prompt(conversation_context)
        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", str(response))
            # Parse GPT response into suggestions, e.g. split by newlines or bullets
            suggestions = [s.strip("- ").strip() for s in content.split("\n") if s.strip()]
            # Return max 3 suggestions
            return suggestions[:3]
        except Exception as e:
            logger.error(f"Error generating Slack inline suggestions: {e}")
            return []

# Singleton agent instance
slack_inline_agent = SlackInlineSuggestionAgent()
