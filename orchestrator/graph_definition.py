from langgraph.graph import StateGraph
from agents.recuritment.sourcing_agent import get_sourcing_agent
from agents.content_generation.content_Seo import generate_seo_content
from agents.content_generation.content_snippet_generator import generate_snippets
from agents.content_generation.keyword_injection_engine import keyword_injection_engine
from agents.content_generation.multi_variant_content_generator import generate_content_variants
from agents.content_generation.persona_content_rewriter import persona_content_rewriter
from agents.content_generation.multi_channel_formatter import multi_channel_format
from agents.content_generation.visual import content_to_slides
from agents.support_and_satisfaction.ticket_auto_response import generate_response
from agents.support_and_satisfaction.smart_ticket_router import summarize_ticket, classify_ticket, route_ticket
from agents.support_and_satisfaction.multichannel_response_engine import ingest_message, generate_response, dispatch_response
from agents.support_and_satisfaction.ticket_conversation_summarizer import generate_ticket_summary
from agents.support_and_satisfaction.nps_extraction_and_analysis import process_surveys
from agents.support_and_satisfaction.real_time_sentiment_detection import live_sentiment_agent
from agents.support_and_satisfaction.emotional_pattern_churn_detection import calculate_churn_risk
from agents.support_and_satisfaction.sla_alerting import check_sla_and_alert
from agents.support_and_satisfaction.sensitive_data_filter import detect_sensitive_terms
from agents.support_and_satisfaction.emotion_tagging import tag_emotions_in_chat
from agents.support_and_satisfaction.audit_log_mongodb import log_failed_response
from typing import TypedDict, Dict, Any
import logging

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

class SupportAndSatisfactionState(TypedDict, total=False):
    ticket_text: str
    raw_message: str
    timestamps: list
    sentiment_scores: list
    messages: list
    compliance_text: str
    conversation_thread: str
    auto_response: str
    ticket_summary: str
    category: str
    complexity: str
    labels: list
    scores: list
    assigned_agent: str
    emotion_scores: dict
    churn_risk_score: float
    sla_alert_status: str
    sensitive_data_filter: dict
    session_id: str

# Content node wrappers accepting state

def seo_content_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] SEO_CONTENT start")
    try:
        res = generate_seo_content(
            state.get("topic"),
            state.get("keywords", ""),
            state.get("persona", "general"),
            state.get("tone", "neutral"),
            state.get("channel", "blog")
        )
    except Exception as e:
        logger.error(f"SEO_CONTENT error: {e}")
        res = {"content": f"[FALLBACK CONTENT] {state.get('topic','No topic')}", "error": str(e)}
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
            state.get("keywords", "")
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
            float(state.get("density", 0.02) or 0.02)
        )
    except Exception as e:
        logger.error(f"KEYWORD_INJECTION error: {e}")
        res = {"optimized_content": state.get("body", ""), "error": str(e)}
    state["keyword_injection"] = res
    if res.get("optimized_content"):
        state["body"] = res["optimized_content"]
    logger.info("[CONTENT FLOW] KEYWORD_INJECTION done")
    return state

def variants_node(state: Dict[str, Any]):
    logger.info("[CONTENT FLOW] VARIANTS start")
    try:
        res = generate_content_variants(
            state.get("body", ""),
            state.get("cta_styles", ["standard"]),
            state.get("sentiment_tones", ["neutral"]),
            int(state.get("num_variants", 3) or 3)
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
            state.get("keywords", "")
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
            state.get("template", "default")
        )
    except Exception as e:
        logger.error(f"SLIDES error: {e}")
        res = {"error": str(e), "raw_body": state.get("body", "")}
    state["slides"] = res
    logger.info("[CONTENT FLOW] SLIDES done")
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
        "seo": state.get("seo_content"),
        "persona_rewrite": state.get("persona_rewrite")
    }
    logger.info("[CONTENT FLOW] AGGREGATE done")
    return state

# Content node wrappers accepting state

def ticket_auto_response_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] TICKET_AUTO_RESPONSE start")
    try:
        res = generate_response(state.get("ticket_text", ""))
    except Exception as e:
        logger.error(f"TICKET_AUTO_RESPONSE error: {e}")
        res = {"auto_response": "", "error": str(e)}
    state["auto_response"] = res
    logger.info("[SUPPORT FLOW] TICKET_AUTO_RESPONSE done")
    return state

def smart_ticket_router_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] SMART_TICKET_ROUTER start")
    try:
        ticket_text = state.get("ticket_text", "")
        if ticket_text:
            summary = summarize_ticket(ticket_text)
            category, complexity, labels, scores = classify_ticket(summary)
            assigned_agent = route_ticket(summary, category, complexity)
            state.update({
                "ticket_summary": summary,
                "category": category,
                "complexity": complexity,
                "labels": labels,
                "scores": scores,
                "assigned_agent": assigned_agent
            })
        else:
            logger.warning("No ticket_text found for routing")
    except Exception as e:
        logger.error(f"SMART_TICKET_ROUTER error: {e}")
    logger.info("[SUPPORT FLOW] SMART_TICKET_ROUTER done")
    return state

def live_sentiment_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] LIVE_SENTIMENT start")
    try:
        raw_message = state.get("raw_message", "")
        if raw_message:
            sentiment_data = live_sentiment_agent({"raw_message": raw_message})
            state.update(sentiment_data)
        else:
            logger.warning("No raw_message found for sentiment analysis")
    except Exception as e:
        logger.error(f"LIVE_SENTIMENT error: {e}")
    logger.info("[SUPPORT FLOW] LIVE_SENTIMENT done")
    return state

def emotion_tagging_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] EMOTION_TAGGING start")
    try:
        message = state.get("raw_message", "")
        if message:
            emotions = tag_emotions_in_chat(message)
            state["emotion_scores"] = emotions
        else:
            logger.warning("No message found for emotion tagging")
    except Exception as e:
        logger.error(f"EMOTION_TAGGING error: {e}")
    logger.info("[SUPPORT FLOW] EMOTION_TAGGING done")
    return state

def churn_detection_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] CHURN_DETECTION start")
    try:
        timestamps_raw = state.get("timestamps", [])
        sentiment_scores = state.get("sentiment_scores", [])
        messages = state.get("messages", [])
        if timestamps_raw and sentiment_scores and messages:
            from datetime import datetime
            timestamps = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps_raw]
            churn_score = calculate_churn_risk(timestamps, sentiment_scores, messages)
            state["churn_risk_score"] = churn_score
        else:
            logger.warning("Insufficient data for churn detection")
    except Exception as e:
        logger.error(f"CHURN_DETECTION error: {e}")
    logger.info("[SUPPORT FLOW] CHURN_DETECTION done")
    return state

def sla_alert_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] SLA_ALERT start")
    try:
        check_sla_and_alert()
        state["sla_alert_status"] = "SLA alert check executed"
    except Exception as e:
        logger.error(f"SLA_ALERT error: {e}")
        state["sla_alert_status"] = f"Error: {e}"
    logger.info("[SUPPORT FLOW] SLA_ALERT done")
    return state

def sensitive_data_filter_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] SENSITIVE_DATA_FILTER start")
    try:
        text = state.get("compliance_text", "")
        if text:
            result = detect_sensitive_terms(text)
            state["sensitive_data_filter"] = result
        else:
            logger.warning("No compliance_text found for sensitive data filtering")
    except Exception as e:
        logger.error(f"SENSITIVE_DATA_FILTER error: {e}")
    logger.info("[SUPPORT FLOW] SENSITIVE_DATA_FILTER done")
    return state

def ticket_summary_node(state: Dict[str, Any]):
    logger.info("[SUPPORT FLOW] TICKET_SUMMARY start")
    try:
        conversation = state.get("conversation_thread", "")
        if conversation:
            summary = generate_ticket_summary(conversation)
            state["ticket_summary"] = summary
        else:
            logger.warning("No conversation_thread found for ticket summary")
    except Exception as e:
        logger.error(f"TICKET_SUMMARY error: {e}")
    logger.info("[SUPPORT FLOW] TICKET_SUMMARY done")
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
        builder.add_node("VARIANTS", variants_node)
        builder.add_node("PERSONA_REWRITE", persona_rewrite_node)
        builder.add_node("MULTI_CHANNEL", multi_channel_node)
        builder.add_node("SLIDES", slides_node)
        builder.add_node("AGGREGATE", aggregate_node)
        builder.set_entry_point("SEO_CONTENT")
        builder.set_finish_point("AGGREGATE")
        builder.add_edge("SEO_CONTENT", "SNIPPETS")
        builder.add_edge("SNIPPETS", "KEYWORD_INJECTION")
        builder.add_edge("KEYWORD_INJECTION", "VARIANTS")
        builder.add_edge("VARIANTS", "PERSONA_REWRITE")
        builder.add_edge("PERSONA_REWRITE", "MULTI_CHANNEL")
        builder.add_edge("MULTI_CHANNEL", "SLIDES")
        builder.add_edge("SLIDES", "AGGREGATE")
        return builder
    elif workflow_type == "support_and_satisfaction":
        builder = StateGraph(SupportAndSatisfactionState)
        builder.add_node("TICKET_AUTO_RESPONSE", ticket_auto_response_node)
        builder.add_node("SMART_TICKET_ROUTER", smart_ticket_router_node)
        builder.add_node("LIVE_SENTIMENT", live_sentiment_agent)
        builder.add_node("EMOTION_TAGGING", emotion_tagging_node)
        builder.add_node("CHURN_DETECTION", churn_detection_node)
        builder.add_node("SLA_ALERTING", sla_alert_node)
        builder.add_node("SENSITIVE_DATA_FILTER", sensitive_data_filter_node)
        builder.add_node("TICKET_SUMMARY", ticket_summary_node)
        # Define start and finish nodes as appropriate
        builder.set_entry_point("TICKET_AUTO_RESPONSE")
        builder.set_finish_point("TICKET_SUMMARY")
        # Define edges/order of execution
        builder.add_edge("TICKET_AUTO_RESPONSE", "SMART_TICKET_ROUTER")
        builder.add_edge("SMART_TICKET_ROUTER", "LIVE_SENTIMENT")
        builder.add_edge("LIVE_SENTIMENT", "EMOTION_TAGGING")
        builder.add_edge("EMOTION_TAGGING", "CHURN_DETECTION")
        builder.add_edge("CHURN_DETECTION", "SLA_ALERTING")
        builder.add_edge("SLA_ALERTING", "SENSITIVE_DATA_FILTER")
        builder.add_edge("SENSITIVE_DATA_FILTER", "TICKET_SUMMARY")
        return builder
    else:
        raise ValueError("Unknown workflow type")

