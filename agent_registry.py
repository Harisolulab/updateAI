from datetime import datetime
from langchain_core.runnables import RunnableLambda
from agents.recuritment.sourcing_agent import get_sourcing_agent

# Communication & Collaboration agents
from agents.communication_and_collaboration.transcription_summarization_agent import transcribe_and_summarize
from agents.communication_and_collaboration.translation_agent import translate_text as cc_translate_text
from agents.communication_and_collaboration.tone_rephrase_agent import rephrase_to_professional
from agents.communication_and_collaboration.message_normalizer_agent import normalize_message

from agents.communication_and_collaboration.connector_sdk_agent import send_message as cc_send_message, get_message_status as cc_get_message_status
from agents.communication_and_collaboration.oauth2_authentication import initiate_oauth_flow as cc_initiate_oauth_flow,complete_oauth_flow as cc_complete_oauth_flow,refresh_access_token as cc_refresh_access_token,revoke_access_token as cc_revoke_access_token
from agents.communication_and_collaboration.actionable_item_extractor_agent import extract_actionable_items,detect_and_create_tasks

from agents.communication_and_collaboration.alert_deduplication_agent import dedup_service
from agents.communication_and_collaboration.conflict_sentiment_agent import detect_conflict_and_sentiment_decay
from agents.communication_and_collaboration.conversation_tree_agent import thread_message_transform,build_conversation_tree,ThreadMessage
from agents.communication_and_collaboration.hybrid_context_store_agent import hybrid_context_store
from agents.communication_and_collaboration.persona_tone_agent import adjust_tone_by_persona
from agents.communication_and_collaboration.slack_inline_suggestion_agent import slack_inline_agent


def sourcing_agent_wrapper(x):
    result = get_sourcing_agent()(x)
    # Ensure resumes and filtered_resumes are always present
    if isinstance(result, dict):
        if "resumes" not in result:
            result["resumes"] = []
        if "filtered_resumes" not in result:
            result["filtered_resumes"] = result["resumes"]
        # Always populate shortlisting info from filtered_resumes
        filtered = result.get("filtered_resumes", [])
        shortlisted = []
        for candidate in filtered:
            candidate_copy = candidate.copy()
            candidate_copy["shortlisted"] = True
            shortlisted.append(candidate_copy)
        result["shortlisted_candidates"] = shortlisted
        print(f"[SHORTLISTING AGENT] Shortlisted candidates: {shortlisted}")
    return result


# ---- Existing Communication wrappers ----

def transcribe_and_summarize_wrapper(x):
    """Input: { audio_path: str }
    Output: { speaker_segments: [...], summary: str, accuracy_estimate: float }
    """
    audio_path = (x or {}).get("audio_path") or (x or {}).get("path")
    if not audio_path:
        return {"error": "audio_path missing"}
    try:
        return transcribe_and_summarize(audio_path)
    except Exception as e:
        return {"error": str(e)}


def translation_wrapper(x):
    """Input: { text: str, target_lang?: str }
    Output: { translation: str }
    """
    text = (x or {}).get("text") or ""
    target_lang = (x or {}).get("target_lang") or (x or {}).get("target_language") or "EN"
    if not text:
        return {"error": "text missing"}
    try:
        translated = cc_translate_text(text, target_lang)
        return {"translation": translated, "target_lang": target_lang}
    except Exception as e:
        return {"error": str(e)}


def tone_rephrase_wrapper(x):
    """Input: { message: str }
    Output: { rewritten: str }
    """
    message = (x or {}).get("message") or (x or {}).get("text") or ""
    if not message:
        return {"error": "message missing"}
    try:
        rewritten = rephrase_to_professional(message)
        return {"rewritten": rewritten}
    except Exception as e:
        return {"error": str(e)}


def normalize_message_wrapper(x):
    """Input: { platform: 'slack'|'microsoft'|'gmail'|'notion', payload: dict }
    Output: { message: dict }
    """
    platform = (x or {}).get("platform")
    payload = (x or {}).get("payload") or {}
    if not platform:
        return {"error": "platform missing"}
    try:
        msg = normalize_message(platform, payload)
        # normalize_message returns an InternalMessage object with to_dict()
        return {"message": msg.to_dict()}
    except Exception as e:
        return {"error": str(e)}


# ---- OAuth2 + Connector SDK wrappers ----

def oauth2_initiate_wrapper(x):
    """Input: { oauth_provider: str, oauth_user_id: str, oauth_state?: str }
    Output: { authorization_url: str }
    """
    provider = (x or {}).get("oauth_provider") or (x or {}).get("provider")
    user_id = (x or {}).get("oauth_user_id") or (x or {}).get("user_id")
    state = (x or {}).get("oauth_state") or (x or {}).get("state")
    if not provider or not user_id:
        return {"error": "oauth_provider or oauth_user_id missing"}
    try:
        url = cc_initiate_oauth_flow(provider, user_id, state=state)
        return {"authorization_url": url}
    except Exception as e:
        return {"error": str(e)}


def oauth2_auth_wrapper(x):
    """Input: { oauth_provider: str, oauth_user_id: str, oauth_code?: str, oauth_state?: str }
    Output: { access_token, refresh_token?, expires_at? }
    """
    provider = (x or {}).get("oauth_provider") or (x or {}).get("provider")
    user_id = (x or {}).get("oauth_user_id") or (x or {}).get("user_id")
    code = (x or {}).get("oauth_code")
    state = (x or {}).get("oauth_state")
    if not provider or not user_id:
        return {"error": "oauth_provider or oauth_user_id missing"}
    try:
        token = None
        if code:
            token = cc_complete_oauth_flow(provider, user_id, code, state=state)
        if not token:
            token = cc_refresh_access_token(provider, user_id)
        result = {
            "access_token": token.get("access_token") if token else None,
            "refresh_token": token.get("refresh_token") if token else None,
        }
        if token and token.get("expires_at"):
            result["expires_at"] = token.get("expires_at")
        return result
    except Exception as e:
        return {"error": str(e)}


def oauth2_revoke_wrapper(x):
    """Input: { oauth_provider: str, oauth_user_id: str }
    Output: { revoked: bool }
    """
    provider = (x or {}).get("oauth_provider") or (x or {}).get("provider")
    user_id = (x or {}).get("oauth_user_id") or (x or {}).get("user_id")
    if not provider or not user_id:
        return {"error": "oauth_provider or oauth_user_id missing"}
    try:
        ok = cc_revoke_access_token(provider, user_id)
        return {"revoked": bool(ok)}
    except Exception as e:
        return {"error": str(e)}


def send_message_wrapper(x):
    """Input: { platform: str, recipient_id: str, message: str, metadata?: dict }
    Output: { status: str }
    """
    platform = (x or {}).get("platform")
    recipient = (x or {}).get("recipient_id")
    message = (x or {}).get("message")
    metadata = (x or {}).get("metadata") or {}
    if not platform or not recipient or not message:
        return {"error": "platform, recipient_id, and message are required"}
    try:
        status = cc_send_message(platform, recipient, message, metadata=metadata)
        return {"status": status}
    except Exception as e:
        return {"error": str(e)}


def get_message_status_wrapper(x):
    """Input: { platform: str, recipient_id: str }
    Output: { delivery_status: str }
    """
    platform = (x or {}).get("platform")
    recipient = (x or {}).get("recipient_id")
    if not platform or not recipient:
        return {"error": "platform and recipient_id are required"}
    try:
        status = cc_get_message_status(platform, recipient)
        return {"delivery_status": status}
    except Exception as e:
        return {"error": str(e)}


# ---- Additional Communication agent wrappers (synchronizing with folder) ----

def actionable_extract_wrapper(x):
    """Input: { text|meeting_text: str } -> { items: List[dict] }"""
    text = (x or {}).get("text") or (x or {}).get("meeting_text") or ""
    if not text:
        return {"error": "text missing"}
    try:
        items = extract_actionable_items(text)
        return {"items": items}
    except Exception as e:
        return {"error": str(e)}


def tasks_detect_create_wrapper(x):
    """Input: { meeting_text: str, trello_config?: dict, clickup_config?: dict } -> task creation stats"""
    text = (x or {}).get("meeting_text") or (x or {}).get("text") or ""
    trello = (x or {}).get("trello_config")
    clickup = (x or {}).get("clickup_config")
    if not text:
        return {"error": "meeting_text missing"}
    try:
        res = detect_and_create_tasks(text, trello_config=trello, clickup_config=clickup)
        return res
    except Exception as e:
        return {"error": str(e)}


def alert_dedup_check_wrapper(x):
    """Input: { message|text: str } -> { duplicate: bool }"""
    message = (x or {}).get("message") or (x or {}).get("text") or ""
    if not message:
        return {"error": "message missing"}
    try:
        is_dup = dedup_service.is_duplicate(message)
        return {"duplicate": bool(is_dup)}
    except Exception as e:
        return {"error": str(e)}


def alert_dedup_reset_wrapper(x):
    """Input: {} -> { reset: True }"""
    try:
        dedup_service.reset()
        return {"reset": True}
    except Exception as e:
        return {"error": str(e)}


def conflict_sentiment_wrapper(x):
    """Input: { messages: List[str] } -> { analysis: List[dict] }"""
    messages = (x or {}).get("messages") or []
    if not isinstance(messages, list) or not messages:
        return {"error": "messages list missing"}
    try:
        analysis = detect_conflict_and_sentiment_decay(messages)
        return {"analysis": analysis}
    except Exception as e:
        return {"error": str(e)}


def conversation_thread_transform_wrapper(x):
    """Input: { platform: str, payload: dict } -> { message: dict }"""
    platform = (x or {}).get("platform")
    payload = (x or {}).get("payload") or {}
    if not platform:
        return {"error": "platform missing"}
    try:
        msg = thread_message_transform(platform, payload)
        return {"message": msg.to_dict()}
    except Exception as e:
        return {"error": str(e)}


def conversation_build_tree_wrapper(x):
    """Input: { messages: List[ThreadMessage|dict] } -> { tree: dict }"""
    raw_messages = (x or {}).get("messages") or []
    if not isinstance(raw_messages, list) or not raw_messages:
        return {"error": "messages list missing"}

    try:
        def to_thread_message(obj):
            if isinstance(obj, ThreadMessage):
                return obj
            if isinstance(obj, dict):
                ts = obj.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts_dt = datetime.fromisoformat(ts)
                    except Exception:
                        ts_dt = datetime.utcnow()
                elif isinstance(ts, (int, float)):
                    ts_dt = datetime.fromtimestamp(ts)
                elif isinstance(ts, datetime):
                    ts_dt = ts
                else:
                    ts_dt = datetime.utcnow()
                return ThreadMessage(
                    platform=obj.get("platform", ""),
                    message_id=obj.get("message_id", ""),
                    conversation_id=obj.get("conversation_id"),
                    parent_id=obj.get("parent_id"),
                    sender_id=obj.get("sender_id", ""),
                    sender_name=obj.get("sender_name"),
                    content=obj.get("content", ""),
                    timestamp=ts_dt,
                    attachments=obj.get("attachments"),
                    reactions=obj.get("reactions"),
                    raw_payload=obj.get("raw_payload"),
                )
            raise TypeError("Unsupported message type")

        msgs = [to_thread_message(m) for m in raw_messages]
        tree = build_conversation_tree(msgs)
        return {"tree": tree}
    except Exception as e:
        return {"error": str(e)}


def context_store_short_term_wrapper(x):
    """Input: { session_id: str, text: str } -> { stored: bool }"""
    session_id = (x or {}).get("session_id")
    text = (x or {}).get("text") or ""
    if not session_id or not text:
        return {"error": "session_id or text missing"}
    try:
        hybrid_context_store.store_short_term_context(session_id, text)
        return {"stored": True}
    except Exception as e:
        return {"error": str(e)}


def context_fetch_short_term_wrapper(x):
    """Input: { session_id: str } -> { text: Optional[str] }"""
    session_id = (x or {}).get("session_id")
    if not session_id:
        return {"error": "session_id missing"}
    try:
        text = hybrid_context_store.fetch_short_term_context(session_id)
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}


def context_store_long_term_wrapper(x):
    """Input: { session_id: str, text: str } -> { stored: bool }"""
    session_id = (x or {}).get("session_id")
    text = (x or {}).get("text") or ""
    if not session_id or not text:
        return {"error": "session_id or text missing"}
    try:
        hybrid_context_store.store_long_term_context(session_id, text)
        return {"stored": True}
    except Exception as e:
        return {"error": str(e)}


def context_query_similar_wrapper(x):
    """Input: { text: str, top_k?: int } -> { results: dict }"""
    text = (x or {}).get("text") or ""
    top_k = int((x or {}).get("top_k") or 5)
    if not text:
        return {"error": "text missing"}
    try:
        results = hybrid_context_store.query_similar_contexts(text, top_k=top_k)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


def persona_tone_wrapper(x):
    """Input: { message: str, persona|persona_key: str } -> { rewritten: str }"""
    message = (x or {}).get("message") or (x or {}).get("text") or ""
    persona = (x or {}).get("persona") or (x or {}).get("persona_key")
    if not message or not persona:
        return {"error": "message or persona missing"}
    try:
        rewritten = adjust_tone_by_persona(message, persona)
        return {"rewritten": rewritten}
    except Exception as e:
        return {"error": str(e)}


def slack_inline_suggestions_wrapper(x):
    """Input: { conversation_context: List[dict] } -> { suggestions: List[str] }"""
    ctx = (x or {}).get("conversation_context") or (x or {}).get("context") or []
    try:
        suggestions = slack_inline_agent.get_suggestions(ctx)
        return {"suggestions": suggestions}
    except Exception as e:
        return {"error": str(e)}


agent_nodes = {
    # Recruitment
    "SOURCING": RunnableLambda(sourcing_agent_wrapper),

    # Communication & Collaboration
    "COMM_TRANSCRIBE_SUMMARIZE": RunnableLambda(transcribe_and_summarize_wrapper),
    "COMM_TRANSLATE": RunnableLambda(translation_wrapper),
    "COMM_TONE_REPHRASE": RunnableLambda(tone_rephrase_wrapper),
    "COMM_NORMALIZE_MESSAGE": RunnableLambda(normalize_message_wrapper),

    # OAuth2 + Connector SDK
    "COMM_OAUTH2_INITIATE": RunnableLambda(oauth2_initiate_wrapper),
    "COMM_OAUTH2_AUTH": RunnableLambda(oauth2_auth_wrapper),
    "COMM_OAUTH2_REVOKE": RunnableLambda(oauth2_revoke_wrapper),
    "COMM_SEND_MESSAGE": RunnableLambda(send_message_wrapper),
    "COMM_GET_MESSAGE_STATUS": RunnableLambda(get_message_status_wrapper),

    # Additional Communication features
    "COMM_ACTIONABLE_EXTRACT": RunnableLambda(actionable_extract_wrapper),
    "COMM_TASKS_DETECT_CREATE": RunnableLambda(tasks_detect_create_wrapper),
    "COMM_ALERT_DEDUP_CHECK": RunnableLambda(alert_dedup_check_wrapper),
    "COMM_ALERT_DEDUP_RESET": RunnableLambda(alert_dedup_reset_wrapper),
    "COMM_CONFLICT_SENTIMENT": RunnableLambda(conflict_sentiment_wrapper),
    "COMM_CONV_THREAD_TRANSFORM": RunnableLambda(conversation_thread_transform_wrapper),
    "COMM_CONV_BUILD_TREE": RunnableLambda(conversation_build_tree_wrapper),
    "COMM_CONTEXT_STORE_SHORT_TERM": RunnableLambda(context_store_short_term_wrapper),
    "COMM_CONTEXT_FETCH_SHORT_TERM": RunnableLambda(context_fetch_short_term_wrapper),
    "COMM_CONTEXT_STORE_LONG_TERM": RunnableLambda(context_store_long_term_wrapper),
    "COMM_CONTEXT_QUERY_SIMILAR": RunnableLambda(context_query_similar_wrapper),
    "COMM_PERSONA_TONE": RunnableLambda(persona_tone_wrapper),
    "COMM_SLACK_INLINE_SUGGEST": RunnableLambda(slack_inline_suggestions_wrapper),
}

# Backward-compatibility aliases for module runners without introducing NameError in static analysis
_registry = globals().get("CONTENT_MODULE_RUNNERS")
if isinstance(_registry, dict):
    if "snippets" in _registry:
        _registry.setdefault("seo_content", _registry["snippets"])  # legacy alias
        _registry.setdefault("content_seo", _registry["snippets"])  # alternate alias
