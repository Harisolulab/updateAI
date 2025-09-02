from langchain_core.runnables import RunnableLambda
from agents.recuritment.sourcing_agent import get_sourcing_agent
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

def ticket_auto_response_wrapper(state):
    if "ticket_text" in state and state["ticket_text"]:
        state["auto_response"] = auto_response(state["ticket_text"])
    return state


def smart_ticket_router_wrapper(state):
    if "ticket_text" in state and state["ticket_text"]:
        summary = summarize_ticket(state["ticket_text"])
        category, complexity, labels, scores = classify_ticket(summary)
        assigned_agent = route_ticket(summary, category, complexity)
        state.update(
            {
                "ticket_summary": summary,
                "category": category,
                "complexity": complexity,
                "labels": labels,
                "scores": scores,
                "assigned_agent": assigned_agent,
            }
        )
    return state


def live_sentiment_wrapper(state):
    if "raw_message" in state and state["raw_message"]:
        sentiment_state = live_sentiment_agent({"raw_message": state["raw_message"]})
        state.update(
            {
                "emotion_scores": sentiment_state.get("emotion_scores", {}),
                "sentiment": sentiment_state.get("sentiment", "Unknown"),
                "urgent_emotion": sentiment_state.get("urgent_emotion", False),
            }
        )
    return state


def churn_detection_wrapper(state):
    required_keys = ("timestamps", "sentiment_scores", "messages")
    if all(k in state for k in required_keys):
        churn_score = calculate_churn_risk(
            state["timestamps"], state["sentiment_scores"], state["messages"]
        )
        state["churn_risk_score"] = churn_score
    return state


def sla_alerting_wrapper(state):
    try:
        check_sla_and_alert()
        state["sla_alert_status"] = "SLA alert check completed"
    except Exception as e:
        state["sla_alert_status"] = f"SLA alert error: {str(e)}"
    return state


def sensitive_data_filter_wrapper(state):
    text = state.get("text", "")
    if text:
        state["sensitive_data_filter"] = detect_sensitive_terms(text)
    else:
        state["sensitive_data_filter"] = {}
    return state


def ticket_summary_wrapper(state):
    conversation = state.get("conversation_thread", "")
    if conversation:
        state["ticket_summary"] = generate_ticket_summary(conversation)
    else:
        state["ticket_summary"] = ""
    return state


def audit_log_failure_wrapper(state):
    # Typically used for logging only; state unmodified
    return state


def emotion_tagging_wrapper(state):
    message = state.get("message", "")
    if message:
        state["emotion_scores"] = tag_emotions_in_chat(message)
    else:
        state["emotion_scores"] = {}
    return state

agent_nodes = {
    "SOURCING": RunnableLambda(sourcing_agent_wrapper),
    "TICKET_AUTO_RESPONSE": RunnableLambda(ticket_auto_response_wrapper),
    "SMART_TICKET_ROUTER": RunnableLambda(smart_ticket_router_wrapper),
    "LIVE_SENTIMENT": RunnableLambda(live_sentiment_wrapper),
    "CHURN_DETECTION": RunnableLambda(churn_detection_wrapper),
    "SLA_ALERTING": RunnableLambda(sla_alerting_wrapper),
    "SENSITIVE_DATA_FILTER": RunnableLambda(sensitive_data_filter_wrapper),
    "TICKET_SUMMARY": RunnableLambda(ticket_summary_wrapper),
    "AUDIT_LOG_FAILURE": RunnableLambda(audit_log_failure_wrapper),
    "EMOTION_TAGGING": RunnableLambda(emotion_tagging_wrapper),
}
