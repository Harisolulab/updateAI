from langgraph.graph import StateGraph
from agents.recuritment.sourcing_agent import get_sourcing_agent
from agents.content_generation.content_Seo import generate_seo_content
from agents.content_generation.content_snippet_generator import generate_snippets
from agents.content_generation.keyword_injection_engine import keyword_injection_engine
from agents.content_generation.multi_variant_content_generator import generate_content_variants
from agents.content_generation.persona_content_rewriter import persona_content_rewriter
from agents.content_generation.multi_channel_formatter import multi_channel_format
from agents.content_generation.visual import content_to_slides
from agents.content_generation.visual_asset_generator import generate_visual_assets
from agents.content_generation.seo_scoring_engine import seo_score_content
from agents.content_generation.engagement_optimizer import optimize_engagement
from agents.content_generation.publishing_service import publish_canonical
from agents.content_generation.persona_rules_engine import persona_validate
from agents.content_generation.video_generator import generate_video

# Communication & Collaboration agents
from agents.communication_and_collaboration.transcription_summarization_agent import transcribe_and_summarize
from agents.communication_and_collaboration.translation_agent import translate_text
from agents.communication_and_collaboration.tone_rephrase_agent import rephrase_to_professional
from agents.communication_and_collaboration.message_normalizer_agent import normalize_message
from agents.communication_and_collaboration.persona_tone_agent import adjust_tone_by_persona
from agents.communication_and_collaboration.conflict_sentiment_agent import detect_conflict_and_sentiment_decay
from agents.communication_and_collaboration.conversation_tree_agent import thread_message_transform, build_conversation_tree
from agents.communication_and_collaboration.hybrid_context_store_agent import hybrid_context_store
from agents.communication_and_collaboration.slack_inline_suggestion_agent import slack_inline_agent
from agents.communication_and_collaboration.actionable_item_extractor_agent import detect_and_create_tasks
from agents.communication_and_collaboration.alert_deduplication_agent import dedup_service
from agents.communication_and_collaboration.connector_sdk_agent import send_message,  get_message_status
from agents.communication_and_collaboration.oauth2_authentication import complete_oauth_flow, refresh_access_token

from typing import TypedDict, Dict, Any, List, Optional
import logging
import time

logger = logging.getLogger("JARVIS")


class SourcingState(TypedDict):
    job_need: str
    requirements: str
    application: dict
    filters: list
    session_id: str
    # Add other fields as needed


class ContentState(TypedDict):
    topic: str
    keywords: str
    persona: str
    tone: str
    channel: str
    body: str
    audience: str
    cta_styles: list
    sentiment_tones: list
    num_variants: int
    headings: list
    subheadings: list
    key_points: list
    brand_guidelines: dict
    template: str
    images: list
    session_id: str
    aggregate: dict
    # Add other fields as needed


class CommunicationState(TypedDict, total=False):
    # Inputs
    message: str
    text: str
    target_lang: str
    target_language: str
    platform: str
    payload: dict
    audio_path: str
    comm_audio_path: str
    session_id: str

    # New/optional inputs for extended nodes
    persona_key: str
    conversation_messages: List[str]
    threaded_payloads: List[Dict[str, Any]]
    conversation_context: List[Dict[str, Any]]
    meeting_text: str
    trello_config: dict
    clickup_config: dict
    alert_message: str
    context_text: str
    context_query: str
    recipient_id: str
    outgoing_message: str
    oauth_provider: str
    oauth_user_id: str
    oauth_code: str
    oauth_state: str

    # Outputs
    rewritten_message: str
    translation: str
    translation_target_lang: str
    normalized_message: dict
    comm_speaker_segments: list
    comm_summary: str
    comm_accuracy: float

    # New/optional outputs for extended nodes
    persona_toned_message: str
    conflict_sentiment_analysis: list
    conversation_tree: dict
    inline_suggestions: list
    is_duplicate_alert: bool
    similar_contexts: dict
    send_status: str
    delivery_status: str
    action_items_result: dict
    oauth_access_token: str
    oauth_refresh_token: str
    oauth_expires_at: float


# Content node wrappers accepting state

def seo_content_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] SEO_CONTENT start")
    try:
        res = generate_seo_content(
            state.get("topic"),
            state.get("keywords", ""),
            state.get("persona", "general"),
            state.get("tone", "neutral"),
            state.get("channel", "blog"),
        )
    except Exception as e:
        logger.error(f"SEO_CONTENT error: {e}")
        res = {
            "content": f"[FALLBACK CONTENT] {state.get('topic','No topic')}",
            "error": str(e),
        }
    state["seo_content"] = res
    if not state.get("body"):
        state["body"] = res.get("content", "")
    logger.info("[CONTENT FLOW] SEO_CONTENT done")
    return state


def snippets_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] SNIPPETS start")
    try:
        res = generate_snippets(
            state.get("body", ""),
            state.get("persona", "general"),
            state.get("tone", "neutral"),
            state.get("audience", "general"),
            state.get("channel", "blog"),
            state.get("keywords", ""),
        )
    except Exception as e:
        logger.error(f"SNIPPETS error: {e}")
        res = {"error": str(e)}
    state["snippets"] = res
    logger.info("[CONTENT FLOW] SNIPPETS done")
    return state


def keyword_injection_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] KEYWORD_INJECTION start")
    try:
        res = keyword_injection_engine(
            state.get("body", ""),
            state.get("topic", ""),
            float(state.get("density", 0.02) or 0.02),
        )
    except Exception as e:
        logger.error(f"KEYWORD_INJECTION error: {e}")
        res = {"optimized_content": state.get("body", ""), "error": str(e)}
    state["keyword_injection"] = res
    if res.get("optimized_content"):
        state["body"] = res["optimized_content"]
    logger.info("[CONTENT FLOW] KEYWORD_INJECTION done")
    return state


def seo_score_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] SEO_SCORE start")
    try:
        scored = seo_score_content(state.get("body", ""), state.get("keywords", ""))
    except Exception as e:
        logger.error(f"SEO_SCORE error: {e}")
        scored = {"error": str(e)}
    state["seo_score"] = scored
    logger.info("[CONTENT FLOW] SEO_SCORE done")
    return state


def variants_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] VARIANTS start")
    try:
        res = generate_content_variants(
            state.get("body", ""),
            state.get("cta_styles", ["standard"]),
            state.get("sentiment_tones", ["neutral"]),
            int(state.get("num_variants", 3) or 3),
        )
    except Exception as e:
        logger.error(f"VARIANTS error: {e}")
        res = [{"variant_content": state.get("body", ""), "error": str(e)}]
    state["variants"] = res
    logger.info("[CONTENT FLOW] VARIANTS done")
    return state


def persona_rewrite_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] PERSONA_REWRITE start")
    try:
        res = persona_content_rewriter(
            state.get("body", ""),
            state.get("persona", "general"),
            state.get("channel", "blog"),
            state.get("keywords", ""),
        )
    except Exception as e:
        logger.error(f"PERSONA_REWRITE error: {e}")
        res = {"rewritten": state.get("body", ""), "error": str(e)}
    state["persona_rewrite"] = res
    rewritten = res.get("rewritten")
    if isinstance(rewritten, str):
        state["body"] = rewritten
    logger.info("[CONTENT FLOW] PERSONA_REWRITE done")
    return state


def persona_validation_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] PERSONA_VALIDATE start")
    try:
        val = persona_validate(
            state.get("body", ""),
            state.get("persona", "executive"),
            state.get("channel", "blog"),
            apply=True,
        )
        # update body with enforced form
        state["body"] = val.get("final_text", state.get("body", ""))
        state["persona_validation"] = val
    except Exception as e:
        logger.error(f"PERSONA_VALIDATE error: {e}")
        state["persona_validation"] = {"error": str(e)}
    logger.info("[CONTENT FLOW] PERSONA_VALIDATE done")
    return state


def engagement_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] ENGAGEMENT start")
    try:
        body = state.get("body", "")
        eng = optimize_engagement(state.get("content_id"), body)
    except Exception as e:
        logger.error(f"ENGAGEMENT error: {e}")
        eng = {"error": str(e)}
    state["engagement"] = eng
    logger.info("[CONTENT FLOW] ENGAGEMENT done")
    return state


def multi_channel_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] MULTI_CHANNEL start")
    content_payload = {
        "title": state.get("topic", ""),
        "body": state.get("body", ""),
        "persona": state.get("persona", "general"),
        "images": state.get("images", []),
    }
    channels = state.get("channels") or ["linkedin", "instagram", "email"]
    res = multi_channel_format(content_payload, channels)
    state["multi_channel"] = res
    logger.info("[CONTENT FLOW] MULTI_CHANNEL done")
    return state


def slides_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] SLIDES start")
    try:
        res = content_to_slides(
            state.get("body", ""),
            state.get("headings", []),
            state.get("subheadings", []),
            state.get("key_points", []),
            state.get("brand_guidelines", {}),
            state.get("template", "default"),
        )
    except Exception as e:
        logger.error(f"SLIDES error: {e}")
        res = {"error": str(e), "raw_body": state.get("body", "")}
    state["slides"] = res
    logger.info("[CONTENT FLOW] SLIDES done")
    return state


def visual_assets_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] VISUAL_ASSETS start")
    try:
        res = generate_visual_assets(
            prompt=state.get("visual_prompt")
            or state.get("topic")
            or state.get("body", ""),
            brand_guidelines=state.get("brand_guidelines", {}),
            platforms=state.get("asset_platforms"),
            variants=int(state.get("asset_variants", 1) or 1),
        )
    except Exception as e:
        logger.error(f"VISUAL_ASSETS error: {e}")
        res = {"error": str(e)}
    state["visual_assets"] = res
    logger.info("[CONTENT FLOW] VISUAL_ASSETS done")
    return state


def video_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] VIDEO start")
    try:
        res = generate_video(
            script=state.get("video_script") or state.get("body", ""),
            brand_guidelines=state.get("brand_guidelines", {}),
            providers=state.get("video_providers"),
            exports=state.get("video_exports"),
        )
    except Exception as e:
        logger.error(f"VIDEO error: {e}")
        res = {"error": str(e)}
    state["video"] = res
    logger.info("[CONTENT FLOW] VIDEO done")
    return state


def publish_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] PUBLISH start")
    try:
        publish_input = {
            "campaign_id": state.get("campaign_id"),
            "version": state.get("version", 1),
            "title": state.get("topic", "Untitled"),
            "body": state.get("body", ""),
            "persona": state.get("persona"),
            "keywords": (state.get("keywords") or "").split(",")
            if isinstance(state.get("keywords"), str)
            else state.get("keywords"),
            "media": state.get("images", []),
            "channels": state.get("publish_channels")
            or state.get("channels")
            or ["linkedin"],
            "metadata": {"source": "content_graph"},
        }
        res = publish_canonical(publish_input)
    except Exception as e:
        logger.error(f"PUBLISH error: {e}")
        res = {"error": str(e)}
    state["publish"] = res
    logger.info("[CONTENT FLOW] PUBLISH done")
    return state


# New aggregate node

def aggregate_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] AGGREGATE start")
    state["aggregate"] = {
        "topic": state.get("topic"),
        "final_body": state.get("body"),
        "snippets": state.get("snippets"),
        "variants": state.get("variants"),
        "slides_summary": state.get("slides"),
        "multi_channel": state.get("multi_channel"),
        "visual_assets": state.get("visual_assets"),
        "video": state.get("video"),
        "publish": state.get("publish"),
        "seo": state.get("seo_content"),
        "seo_score": state.get("seo_score"),
        "persona_validation": state.get("persona_validation"),
        "persona_rewrite": state.get("persona_rewrite"),
        "engagement": state.get("engagement"),
    }
    logger.info("[CONTENT FLOW] AGGREGATE done")
    return state


# Communication & Collaboration nodes

def comm_tone_rephrase_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] TONE_REPHRASE start")
    try:
        message = state.get("message") or state.get("text")
        if not message:
            logger.info("[COMM FLOW] TONE_REPHRASE skipped (no message)")
            return state
        rewritten = rephrase_to_professional(message)
        state["rewritten_message"] = rewritten
    except Exception as e:
        logger.error(f"TONE_REPHRASE error: {e}")
        state["rewritten_message"] = state.get("message") or state.get("text") or ""
        state["tone_rephrase_error"] = str(e)
    logger.info("[COMM FLOW] TONE_REPHRASE done")
    return state


def comm_persona_tone_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] PERSONA_TONE start")
    try:
        persona_key = state.get("persona_key")
        base_text = (
            state.get("rewritten_message")
            or state.get("message")
            or state.get("text")
        )
        if not persona_key or not base_text:
            logger.info("[COMM FLOW] PERSONA_TONE skipped (missing persona_key or text)")
            return state
        toned = adjust_tone_by_persona(base_text, persona_key)
        state["persona_toned_message"] = toned
        # Make it the current best message for downstream
        state["rewritten_message"] = toned
    except Exception as e:
        logger.error(f"PERSONA_TONE error: {e}")
        state["persona_tone_error"] = str(e)
    logger.info("[COMM FLOW] PERSONA_TONE done")
    return state


def comm_translate_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] TRANSLATE start")
    try:
        text = (
            state.get("text")
            or state.get("message")
            or state.get("rewritten_message")
        )
        if not text:
            logger.info("[COMM FLOW] TRANSLATE skipped (no text)")
            return state
        target = state.get("target_lang") or state.get("target_language") or "EN"
        translated = translate_text(text, target)
        state["translation"] = translated
        state["translation_target_lang"] = target
    except Exception as e:
        logger.error(f"TRANSLATE error: {e}")
        state["translation_error"] = str(e)
    logger.info("[COMM FLOW] TRANSLATE done")
    return state


def comm_normalize_message_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] NORMALIZE_MESSAGE start")
    try:
        platform = (state.get("platform") or "").lower()
        payload = state.get("payload") or {}
        if not platform or not payload:
            logger.info(
                "[COMM FLOW] NORMALIZE_MESSAGE skipped (missing platform or payload)"
            )
            return state
        msg = normalize_message(platform, payload)
        state["normalized_message"] = (
            msg.to_dict() if hasattr(msg, "to_dict") else getattr(msg, "__dict__", msg)
        )
    except Exception as e:
        logger.error(f"NORMALIZE_MESSAGE error: {e}")
        state["normalize_error"] = str(e)
    logger.info("[COMM FLOW] NORMALIZE_MESSAGE done")
    return state


def comm_conversation_tree_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] CONVERSATION_TREE start")
    try:
        items = state.get("threaded_payloads") or []
        if not items:
            logger.info("[COMM FLOW] CONVERSATION_TREE skipped (no threaded_payloads)")
            return state
        messages = []
        for item in items:
            platform = (item.get("platform") or state.get("platform") or "").lower()
            payload = item.get("payload") if isinstance(item, dict) else item
            if not platform or not payload:
                continue
            try:
                tm = thread_message_transform(platform, payload)
                messages.append(tm)
            except Exception as inner:
                logger.warning(f"CONVERSATION_TREE transform skip: {inner}")
        if messages:
            tree = build_conversation_tree(messages)
            state["conversation_tree"] = tree
    except Exception as e:
        logger.error(f"CONVERSATION_TREE error: {e}")
        state["conversation_tree_error"] = str(e)
    logger.info("[COMM FLOW] CONVERSATION_TREE done")
    return state


def comm_conflict_sentiment_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] CONFLICT_SENTIMENT start")
    try:
        msgs = state.get("conversation_messages") or []
        if not msgs:
            logger.info("[COMM FLOW] CONFLICT_SENTIMENT skipped (no conversation_messages)")
            return state
        analysis = detect_conflict_and_sentiment_decay(msgs)
        state["conflict_sentiment_analysis"] = analysis
    except Exception as e:
        logger.error(f"CONFLICT_SENTIMENT error: {e}")
        state["conflict_sentiment_error"] = str(e)
    logger.info("[COMM FLOW] CONFLICT_SENTIMENT done")
    return state


def comm_inline_suggestions_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] INLINE_SUGGESTIONS start")
    try:
        ctx = state.get("conversation_context") or []
        if not ctx:
            logger.info("[COMM FLOW] INLINE_SUGGESTIONS skipped (no conversation_context)")
            return state
        suggestions = slack_inline_agent.get_suggestions(ctx)
        state["inline_suggestions"] = suggestions
    except Exception as e:
        logger.error(f"INLINE_SUGGESTIONS error: {e}")
        state["inline_suggestions_error"] = str(e)
    logger.info("[COMM FLOW] INLINE_SUGGESTIONS done")
    return state


def comm_transcribe_summarize_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] TRANSCRIBE_SUMMARIZE start")
    try:
        audio_path = state.get("audio_path") or state.get("comm_audio_path")
        if not audio_path:
            logger.info("[COMM FLOW] TRANSCRIBE_SUMMARIZE skipped (no audio_path)")
            return state
        res = transcribe_and_summarize(audio_path)
        state["comm_speaker_segments"] = res.get("speaker_segments")
        state["comm_summary"] = res.get("summary")
        state["comm_accuracy"] = res.get("accuracy_estimate")
    except Exception as e:
        logger.error(f"TRANSCRIBE_SUMMARIZE error: {e}")
        state["transcribe_error"] = str(e)
    logger.info("[COMM FLOW] TRANSCRIBE_SUMMARIZE done")
    return state


def comm_action_items_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] ACTION_ITEMS start")
    try:
        text = state.get("meeting_text") or state.get("comm_summary")
        if not text:
            logger.info("[COMM FLOW] ACTION_ITEMS skipped (no meeting_text or comm_summary)")
            return state
        res = detect_and_create_tasks(
            text,
            trello_config=state.get("trello_config"),
            clickup_config=state.get("clickup_config"),
        )
        state["action_items_result"] = res
    except Exception as e:
        logger.error(f"ACTION_ITEMS error: {e}")
        state["action_items_error"] = str(e)
    logger.info("[COMM FLOW] ACTION_ITEMS done")
    return state


def comm_dedup_alert_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] ALERT_DEDUP start")
    try:
        message = state.get("alert_message") or state.get("text") or state.get("message")
        if not message:
            logger.info("[COMM FLOW] ALERT_DEDUP skipped (no alert message)")
            return state
        state["is_duplicate_alert"] = dedup_service.is_duplicate(message)
    except Exception as e:
        logger.error(f"ALERT_DEDUP error: {e}")
        state["alert_dedup_error"] = str(e)
    logger.info("[COMM FLOW] ALERT_DEDUP done")
    return state


def comm_context_store_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] CONTEXT_STORE start")
    try:
        session_id = state.get("session_id")
        text = state.get("context_text") or state.get("text") or state.get("message")
        if session_id and text:
            try:
                hybrid_context_store.store_short_term_context(session_id, text)
            except Exception as inner:
                logger.warning(f"Short-term store failed: {inner}")
            try:
                hybrid_context_store.store_long_term_context(session_id, text)
            except Exception as inner:
                logger.warning(f"Long-term store failed: {inner}")
        query = state.get("context_query")
        if query:
            try:
                sims = hybrid_context_store.query_similar_contexts(query)
                state["similar_contexts"] = sims
            except Exception as inner:
                logger.warning(f"Context query failed: {inner}")
    except Exception as e:
        logger.error(f"CONTEXT_STORE error: {e}")
        state["context_store_error"] = str(e)
    logger.info("[COMM FLOW] CONTEXT_STORE done")
    return state


def comm_oauth2_auth_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] OAUTH2_AUTH start")
    try:
        provider = (state.get("oauth_provider") or state.get("platform") or '').lower()
        user_id = state.get("oauth_user_id") or state.get("recipient_id")
        code = state.get("oauth_code")
        # If code is provided, complete flow; else try refresh; else skip
        if not provider or not user_id:
            logger.info("[COMM FLOW] OAUTH2_AUTH skipped (missing provider or user_id)")
            return state
        token = None
        if code:
            try:
                token = complete_oauth_flow(provider, user_id, code, state=state.get("oauth_state"))
            except Exception as inner:
                logger.warning(f"OAuth code exchange failed: {inner}")
        if not token:
            try:
                token = refresh_access_token(provider, user_id)
            except Exception as inner:
                logger.warning(f"OAuth refresh failed: {inner}")
        if token:
            state["oauth_access_token"] = token.get("access_token")
            state["oauth_refresh_token"] = token.get("refresh_token")
            # Prefer expires_at if present; else compute approximate from expires_in
            expires_at = token.get("expires_at")
            if not expires_at and token.get("expires_in"):
                try:
                    expires_at = time.time() + float(token.get("expires_in"))
                except Exception:
                    expires_at = None
            if expires_at:
                state["oauth_expires_at"] = expires_at
        else:
            logger.info("[COMM FLOW] OAUTH2_AUTH no token obtained")
    except Exception as e:
        logger.error(f"OAUTH2_AUTH error: {e}")
        state["oauth_error"] = str(e)
    logger.info("[COMM FLOW] OAUTH2_AUTH done")
    return state


def comm_send_message_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] SEND_MESSAGE start")
    try:
        platform = state.get("platform")
        recipient = state.get("recipient_id")
        msg = (
            state.get("outgoing_message")
            or state.get("translation")
            or state.get("rewritten_message")
            or state.get("message")
            or state.get("text")
        )
        if not platform or not recipient or not msg:
            logger.info(
                "[COMM FLOW] SEND_MESSAGE skipped (missing platform, recipient_id, or message)"
            )
            return state
        metadata = {}
        if state.get("oauth_access_token"):
            metadata["access_token"] = state["oauth_access_token"]
            if state.get("oauth_expires_at"):
                metadata["expires_at"] = state["oauth_expires_at"]
        status = send_message(platform, recipient, msg, metadata=metadata)
        state["send_status"] = status
    except Exception as e:
        logger.error(f"SEND_MESSAGE error: {e}")
        state["send_message_error"] = str(e)
    logger.info("[COMM FLOW] SEND_MESSAGE done")
    return state


def comm_get_message_status_node(state: Dict[str, Any]):
    logger.info("[COMM FLOW] GET_MESSAGE_STATUS start")
    try:
        platform = state.get("platform")
        recipient = state.get("recipient_id")
        if not platform or not recipient:
            logger.info(
                "[COMM FLOW] GET_MESSAGE_STATUS skipped (missing platform or recipient_id)"
            )
            return state
        status = get_message_status(platform, recipient)
        state["delivery_status"] = status
    except Exception as e:
        logger.error(f"GET_MESSAGE_STATUS error: {e}")
        state["get_message_status_error"] = str(e)
    logger.info("[COMM FLOW] GET_MESSAGE_STATUS done")
    return state


# Build and compile LangGraph for different workflows

def build_graph(agent_nodes: dict, workflow_type: str = "sourcing"):
    if workflow_type == "sourcing":
        builder = StateGraph(SourcingState)
        builder.add_node("SOURCING", get_sourcing_agent())
        builder.set_entry_point("SOURCING")
        builder.set_finish_point("SOURCING")
        return builder
    elif workflow_type == "content":
        builder = StateGraph(ContentState)
        builder.add_node("SEO_CONTENT", seo_content_node)
        builder.add_node("SNIPPETS", snippets_node)
        builder.add_node("KEYWORD_INJECTION", keyword_injection_node)
        builder.add_node("SEO_SCORE", seo_score_node)
        builder.add_node("VARIANTS", variants_node)
        builder.add_node("PERSONA_VALIDATE", persona_validation_node)
        builder.add_node("PERSONA_REWRITE", persona_rewrite_node)
        builder.add_node("ENGAGEMENT", engagement_node)
        builder.add_node("MULTI_CHANNEL", multi_channel_node)
        builder.add_node("VISUAL_ASSETS", visual_assets_node)
        builder.add_node("VIDEO", video_node)
        builder.add_node("PUBLISH", publish_node)
        builder.add_node("SLIDES", slides_node)
        builder.add_node("AGGREGATE", aggregate_node)
        builder.set_entry_point("SEO_CONTENT")
        builder.set_finish_point("AGGREGATE")
        builder.add_edge("SEO_CONTENT", "SNIPPETS")
        builder.add_edge("SNIPPETS", "KEYWORD_INJECTION")
        builder.add_edge("KEYWORD_INJECTION", "SEO_SCORE")
        builder.add_edge("SEO_SCORE", "VARIANTS")
        builder.add_edge("VARIANTS", "PERSONA_VALIDATE")
        builder.add_edge("PERSONA_VALIDATE", "PERSONA_REWRITE")
        builder.add_edge("PERSONA_REWRITE", "ENGAGEMENT")
        builder.add_edge("ENGAGEMENT", "MULTI_CHANNEL")
        builder.add_edge("MULTI_CHANNEL", "VISUAL_ASSETS")
        builder.add_edge("VISUAL_ASSETS", "VIDEO")
        builder.add_edge("VIDEO", "PUBLISH")
        builder.add_edge("PUBLISH", "SLIDES")
        builder.add_edge("SLIDES", "AGGREGATE")
        return builder
    elif workflow_type == "communication":
        builder = StateGraph(CommunicationState)
        # Core nodes
        builder.add_node("COMM_TONE_REPHRASE", comm_tone_rephrase_node)
        builder.add_node("COMM_PERSONA_TONE", comm_persona_tone_node)
        builder.add_node("COMM_TRANSLATE", comm_translate_node)
        builder.add_node("COMM_NORMALIZE_MESSAGE", comm_normalize_message_node)
        builder.add_node("COMM_CONVERSATION_TREE", comm_conversation_tree_node)
        builder.add_node("COMM_CONFLICT_SENTIMENT", comm_conflict_sentiment_node)
        builder.add_node("COMM_INLINE_SUGGESTIONS", comm_inline_suggestions_node)
        builder.add_node("COMM_TRANSCRIBE_SUMMARIZE", comm_transcribe_summarize_node)
        builder.add_node("COMM_ACTION_ITEMS", comm_action_items_node)
        builder.add_node("COMM_ALERT_DEDUP", comm_dedup_alert_node)
        builder.add_node("COMM_CONTEXT_STORE", comm_context_store_node)
        builder.add_node("COMM_OAUTH2_AUTH", comm_oauth2_auth_node)
        builder.add_node("COMM_SEND_MESSAGE", comm_send_message_node)
        builder.add_node("COMM_GET_MESSAGE_STATUS", comm_get_message_status_node)
        # Flow
        builder.set_entry_point("COMM_TONE_REPHRASE")
        builder.set_finish_point("COMM_GET_MESSAGE_STATUS")
        builder.add_edge("COMM_TONE_REPHRASE", "COMM_PERSONA_TONE")
        builder.add_edge("COMM_PERSONA_TONE", "COMM_TRANSLATE")
        builder.add_edge("COMM_TRANSLATE", "COMM_NORMALIZE_MESSAGE")
        builder.add_edge("COMM_NORMALIZE_MESSAGE", "COMM_CONVERSATION_TREE")
        builder.add_edge("COMM_CONVERSATION_TREE", "COMM_CONFLICT_SENTIMENT")
        builder.add_edge("COMM_CONFLICT_SENTIMENT", "COMM_INLINE_SUGGESTIONS")
        builder.add_edge("COMM_INLINE_SUGGESTIONS", "COMM_TRANSCRIBE_SUMMARIZE")
        builder.add_edge("COMM_TRANSCRIBE_SUMMARIZE", "COMM_ACTION_ITEMS")
        builder.add_edge("COMM_ACTION_ITEMS", "COMM_ALERT_DEDUP")
        builder.add_edge("COMM_ALERT_DEDUP", "COMM_CONTEXT_STORE")
        builder.add_edge("COMM_CONTEXT_STORE", "COMM_OAUTH2_AUTH")
        builder.add_edge("COMM_OAUTH2_AUTH", "COMM_SEND_MESSAGE")
        builder.add_edge("COMM_SEND_MESSAGE", "COMM_GET_MESSAGE_STATUS")
        return builder
    else:
        raise ValueError("Unknown workflow type")
