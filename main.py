import json
import uuid
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from orchestrator.graph_definition import build_graph
from agents.agent_registry import agent_nodes
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
from agents.support_and_satisfaction.nps_csat_dashboard_api import nps_csat_metrics,SessionLocal,text
from agents.support_and_satisfaction.audit_log_mongodb import log_failed_response
from agents.content_generation.content_Seo import generate_seo_content
from agents.content_generation.multi_channel_formatter import multi_channel_format
from agents.content_generation.persona_content_rewriter import persona_content_rewriter
from agents.content_generation.content_snippet_generator import generate_snippets
from agents.content_generation.keyword_injection_engine import keyword_injection_engine
from agents.content_generation.multi_variant_content_generator import generate_content_variants
from agents.content_generation.visual import content_to_slides
from agents.recuritment.sourcing_agent import get_sourcing_agent
from utils.logger import log_event
from utils.compliance import critical_incident_notification



# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JARVIS")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="JARVIS Orchestrator", version="1.0.0")

# If your frontend is on a different origin, enable CORS here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers (fixed)
# -----------------------------------------------------------------------------
def json_ok(payload: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code)

def json_err(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(content={"response": message}, status_code=status_code)

def get_compiled_graph(workflow_type: str):
    graph = build_graph(agent_nodes, workflow_type)
    return graph.compile()

def detect_workflow_type(input_data: dict) -> str:
    if "topic" in input_data and "keywords" in input_data:
        return "content"
    if "job_need" in input_data and "requirements" in input_data:
        return "sourcing"
    return "default"

def cleanup_llm_output(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s

# -----------------------------------------------------------------------------
# Global exception handler (completed)
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    try:
        body = await request.body()
        if body:
            data = json.loads(body.decode("utf-8"))
            sid = data.get("session_id")
            if sid:
                log_event(sid, "error", {"error": str(exc)}, level="error")
    except Exception:
        pass
    try:
        critical_incident_notification(str(exc))
    except Exception:
        pass
    return json_err(f"Internal Server Error: {str(exc)}", status_code=500)

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "JARVIS Orchestrator Running"}

# -----------------------------------------------------------------------------
# Core Run Task (LangGraph streaming)
# -----------------------------------------------------------------------------
@app.post("/run-task")
async def run_task(request: Request):
    session_id = None
    try:
        body = await request.json()
        input_data = body.get("input", {}) or {}
        workflow_type = body.get("workflow_type") or detect_workflow_type(input_data)
        if workflow_type not in ("sourcing", "content"):
            raise HTTPException(status_code=400, detail="Unsupported or undetected workflow_type. Use 'sourcing' or 'content'.")
        session_id = body.get("session_id", str(uuid.uuid4()))
        logger.info(f"[{session_id}] run-task workflow={workflow_type}")
        log_event(session_id, "execution_context", {"input": input_data, "workflow_type": workflow_type})
        compiled_graph = get_compiled_graph(workflow_type)
        result_state = compiled_graph.invoke(input_data)
        if workflow_type == "content" and isinstance(result_state, dict):
            response_payload = result_state.get("aggregate", result_state)
        else:
            response_payload = result_state
        logger.info(f"[{session_id}] run-task finished keys={list(result_state.keys()) if isinstance(result_state, dict) else type(result_state)}")
        log_event(session_id, "decision_output", {"workflow": workflow_type, "output_keys": list(result_state.keys()) if isinstance(result_state, dict) else []})
        return json_ok({"session_id": session_id, "workflow_type": workflow_type, "result": response_payload})
    except HTTPException as he:
        if session_id:
            log_event(session_id, "error", {"error": he.detail}, level="error")
        return json_err(he.detail, status_code=he.status_code)
    except Exception as e:
        if session_id:
            log_event(session_id, "error", {"error": str(e)}, level="error")
        logger.exception("run-task error")
        return json_err(str(e), status_code=500)

# -----------------------------------------------------------------------------
# Support and Customer Satisfaction
# -----------------------------------------------------------------------------
@app.post("/ticket-auto-response")
async def ticket_auto_response_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    user_query = data.get("query", "")
    if not user_query:
        return json_err("Missing 'query' in request body", status_code=422)

    logger.info(f"[{session_id}] Ticket auto-response requested. Query: {user_query}")

    try:
        response_text = generate_response(user_query)
        logger.info(f"[{session_id}] Generated response for ticket query.")
        return json_ok({"session_id": session_id, "response": response_text})
    except Exception as e:
        logger.error(f"[{session_id}] Error generating ticket response: {e}")
        return json_err(str(e), status_code=500)


@app.post("/smart-ticket-router")
async def smart_ticket_router_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    ticket_text = data.get("ticket_text", "").strip()
    if not ticket_text:
        return json_err("Missing 'ticket_text' in request body", status_code=422)
    logger.info(f"[{session_id}] Smart Ticket Router processing ticket.")

    try:
        # Summarize the ticket
        summary = summarize_ticket(ticket_text)
        # Classify category and complexity
        category, complexity, labels, scores = classify_ticket(summary)
        # Decide routing agent
        assigned_agent = route_ticket(summary, category, complexity)

        result = {
            "summary": summary,
            "category": category,
            "complexity": complexity,
            "assigned_agent": assigned_agent,
            "labels": labels,
            "scores": scores
        }
        logger.info(
            f"[{session_id}] Routing decision: {assigned_agent} for category {category} with complexity {complexity}.")
        return json_ok({"session_id": session_id, "result": result})

    except Exception as e:
        logger.error(f"[{session_id}] Smart Ticket Router error: {e}")
        return json_err(str(e), status_code=500)


@app.post("/multichannel/respond")
async def multichannel_respond_api(request: Request):
    """
    Accepts a JSON payload with message details including platform, sender, content, etc.
    Normalizes input, generates AI response, and dispatches response to the correct channel.
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", str(uuid.uuid4()))

        # Normalize the incoming message
        message = ingest_message(data)

        logger.info(f"[{session_id}] Received message from platform: {message['platform']} sender: {message['sender']}")

        # Generate the AI response text
        ai_response = generate_response(message)

        # Dispatch the response to appropriate channel
        dispatch_response(message, ai_response)

        logger.info(f"[{session_id}] Response dispatched successfully.")
        return json_ok({"session_id": session_id, "response": ai_response})

    except Exception as e:
        logger.error(f"Error in multichannel responder: {str(e)}")
        return json_err(str(e), status_code=500)

@app.post("/ticket/summary")
async def ticket_summary_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    conversation_thread = data.get("conversation_thread", "").strip()
    if not conversation_thread:
        return json_err("Missing 'conversation_thread' in request body", status_code=422)

    logger.info(f"[{session_id}] Generating ticket summary.")
    try:
        summary = generate_ticket_summary(conversation_thread)
        logger.info(f"[{session_id}] Ticket summary generated.")
        return json_ok({"session_id": session_id, "summary": summary})
    except Exception as e:
        logger.error(f"[{session_id}] Error generating ticket summary: {e}")
        return json_err(str(e), status_code=500)

@app.post("/nps/process-surveys")
async def nps_process_surveys_api(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id", str(uuid.uuid4()))
        surveys = data.get("surveys", [])
        if not surveys:
            return json_err("Missing 'surveys' in request body", status_code=422)

        logging.info(f"[{session_id}] Processing NPS surveys.")
        results = process_surveys(surveys)
        logging.info(f"[{session_id}] Completed NPS processing.")
        return json_ok({"session_id": session_id, "results": results})
    except Exception as e:
        logging.error(f"Error processing surveys: {e}")
        return json_err(str(e), status_code=500)

@app.post("/sentiment/live")
async def live_sentiment_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    raw_message = data.get("raw_message", "").strip()
    if not raw_message:
        return json_err("Missing 'raw_message' in request body", status_code=422)

    logging.info(f"[{session_id}] Processing live sentiment for message.")
    try:
        # Pass in state dict with raw message
        state = {"raw_message": raw_message}
        updated_state = live_sentiment_agent(state)
        result = {
            "emotion_scores": updated_state.get("emotion_scores", {}),
            "sentiment": updated_state.get("sentiment", "Unknown"),
            "urgent_emotion": updated_state.get("urgent_emotion", False)
        }
        logging.info(f"[{session_id}] Sentiment detection completed.")
        return json_ok({"session_id": session_id, "result": result})
    except Exception as e:
        logging.error(f"[{session_id}] Error in live sentiment detection: {e}")
        return json_err(str(e), status_code=500)

@app.post("/churn/detect")
async def churn_detection_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))

    timestamps_raw = data.get("timestamps", [])
    sentiment_scores = data.get("sentiment_scores", [])
    messages = data.get("messages", [])

    # Basic input validation
    if not timestamps_raw or not sentiment_scores or not messages:
        return json_err("Missing one or more required fields: 'timestamps', 'sentiment_scores', 'messages'", status_code=422)

    try:
        # Parse timestamps strings to datetime objects
        from datetime import datetime
        timestamps = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps_raw]

        churn_score = calculate_churn_risk(timestamps, sentiment_scores, messages)

        logging.info(f"[{session_id}] Calculated churn risk: {churn_score:.2f}")
        return json_ok({"session_id": session_id, "churn_risk_score": churn_score})

    except Exception as e:
        logging.error(f"[{session_id}] Error in churn detection: {e}")
        return json_err(str(e), status_code=500)

@app.post("/sla/check-alerts")
async def sla_check_alerts_api(request: Request):
    session_id = str(uuid.uuid4())
    logging.info(f"[{session_id}] SLA alert check requested.")
    try:
        # Trigger SLA alert logic (sends alerts to Slack if needed)
        check_sla_and_alert()
        logging.info(f"[{session_id}] SLA alert check completed.")
        return json_ok({"session_id": session_id, "message": "SLA alert check executed."})
    except Exception as e:
        logging.error(f"[{session_id}] SLA alert check failed: {e}")
        return json_err(str(e), status_code=500)

@app.post("/compliance/sensitive-data")
async def sensitive_data_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    text = data.get("text", "").strip()
    if not text:
        return json_err("Missing 'text' field in request body", status_code=422)

    logging.info(f"[{session_id}] Detecting sensitive terms.")
    try:
        result = detect_sensitive_terms(text)
        logging.info(f"[{session_id}] Sensitive data detection completed.")
        return json_ok({"session_id": session_id, "result": result})
    except Exception as e:
        logging.error(f"[{session_id}] Error in sensitive data detection: {e}")
        return json_err(str(e), status_code=500)

@app.post("/emotion/tag")
async def emotion_tag_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    message = data.get("message", "").strip()
    if not message:
        return json_err("Missing 'message' in request body", status_code=422)

    logging.info(f"[{session_id}] Tagging emotions in chat message.")
    try:
        emotion_scores = tag_emotions_in_chat(message)
        logging.info(f"[{session_id}] Emotion tagging completed.")
        return json_ok({"session_id": session_id, "emotion_scores": emotion_scores})
    except Exception as e:
        logging.error(f"[{session_id}] Error tagging emotions: {e}")
        return json_err(str(e), status_code=500)

@app.post("/nps-csat-metrics")
async def nps_csat_metrics_api(request: Request):
    """
    Accepts optional filters like date ranges in request JSON,
    returns average NPS, CSAT, sentiment counts, and 30-day trends.
    """
    data = await request.json()
    start_date = data.get("start_date")  # e.g. "2025-08-01"
    end_date = data.get("end_date")      # e.g. "2025-08-28"

    filters = []
    params = {}

    if start_date:
        filters.append("created_at >= :start_date")
        params["start_date"] = start_date
    if end_date:
        filters.append("created_at <= :end_date")
        params["end_date"] = end_date

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    with SessionLocal() as db:
        overall_query = f"""
            SELECT
                ROUND(AVG(nps_score)::numeric, 2) AS avg_nps,
                ROUND(AVG(csat_score)::numeric, 2) AS avg_csat,
                COUNT(*) AS total_responses,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS positive_count,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral') AS neutral_count,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS negative_count
            FROM customer_feedback
            {where_clause}
        """

        trend_query = f"""
            SELECT
                TO_CHAR(created_at, 'YYYY-MM-DD') AS day,
                ROUND(AVG(nps_score)::numeric, 2) AS avg_nps,
                ROUND(AVG(csat_score)::numeric, 2) AS avg_csat,
                COUNT(*) AS response_count
            FROM customer_feedback
            {where_clause}
            GROUP BY day
            ORDER BY day ASC
            LIMIT 30
        """

        overall_metrics = db.execute(text(overall_query), params).first()
        trend_rows = db.execute(text(trend_query), params).fetchall()

        trend_data = [
            {
                "date": row.day,
                "avg_nps": float(row.avg_nps or 0),
                "avg_csat": float(row.avg_csat or 0),
                "response_count": row.response_count,
            }
            for row in trend_rows
        ]

        result = {
            "avg_nps": float(overall_metrics.avg_nps or 0),
            "avg_csat": float(overall_metrics.avg_csat or 0),
            "total_responses": overall_metrics.total_responses,
            "sentiment_counts": {
                "positive": overall_metrics.positive_count,
                "neutral": overall_metrics.neutral_count,
                "negative": overall_metrics.negative_count,
            },
            "trend": trend_data,
        }

        return json_ok(result)

@app.post("/audit/log-failure")
async def audit_log_failure_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    user_id = data.get("user_id")
    input_data = data.get("input_data", {})
    error_details = data.get("error_details", "")
    escalation = data.get("escalation", False)

    if not user_id or not error_details:
        return json_err("Missing required fields: 'user_id' and/or 'error_details'", status_code=422)

    logging.info(f"[{session_id}] Logging failed AI response for user: {user_id}")

    try:
        log_failed_response(user_id, input_data, error_details, escalation)
        logging.info(f"[{session_id}] Logged failed response.")
        return json_ok({"session_id": session_id, "message": "Failed response logged successfully"})
    except Exception as e:
        logging.error(f"[{session_id}] Error logging failed response: {e}")
        return json_err(str(e), status_code=500)


# -----------------------------------------------------------------------------
# Content Generation Endpoints
# -----------------------------------------------------------------------------
@app.post("/generate-seo-content")
async def generate_seo_content_api(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))
    topic = data.get("topic")
    keywords = data.get("keywords")
    persona = data.get("persona")
    tone = data.get("tone")
    channel = data.get("channel")

    logger.info(f"[{session_id}] SEO content generation requested.")
    log_event(session_id, "seo_content_request", {"topic": topic, "keywords": keywords, "persona": persona, "tone": tone, "channel": channel})

    result = generate_seo_content(topic, keywords, persona, tone, channel)

    logger.info(f"[{session_id}] SEO content generated.")
    log_event(session_id, "seo_content_result", result)

    return json_ok({"session_id": session_id, "result": result})

@app.post("/generate-snippets")
async def generate_snippets_api(request: Request):
    data = await request.json()
    long_content = data.get("long_content")
    persona = data.get("persona")
    tone = data.get("tone")
    audience = data.get("audience")
    channel = data.get("channel")
    keywords = data.get("keywords", "")
    result = generate_snippets(long_content, persona, tone, audience, channel, keywords)
    return json_ok(result)

@app.post("/keyword-injection")
async def keyword_injection_api(request: Request):
    data = await request.json()
    content = data.get("content")
    topic = data.get("topic")
    density = float(data.get("density", 0.02))
    result = keyword_injection_engine(content, topic, density)
    return json_ok(result)

@app.post("/generate-content-variants")
async def generate_content_variants_api(request: Request):
    data = await request.json()
    base_content = data.get("base_content")
    cta_styles = data.get("cta_styles", ["standard"])
    sentiment_tones = data.get("sentiment_tones", ["neutral"])
    num_variants = int(data.get("num_variants", 3))
    result = generate_content_variants(base_content, cta_styles, sentiment_tones, num_variants)
    return json_ok({"variants": result})

@app.post("/content-to-slides")
async def content_to_slides_api(request: Request):
    data = await request.json()
    content = data.get("content")
    headings = data.get("headings", [])
    subheadings = data.get("subheadings", [])
    key_points = data.get("key_points", [])
    brand_guidelines = data.get("brand_guidelines", {})
    template = data.get("template", "default")
    result = content_to_slides(content, headings, subheadings, key_points, brand_guidelines, template)
    return json_ok(result)

# -----------------------------------------------------------------------------
# Cross/Composite Endpoints
# -----------------------------------------------------------------------------
@app.post("/cross-agent-test")
async def cross_agent_test(request: Request):
    data = await request.json()
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Sourcing
    job_need = data.get("job_need")
    requirements = data.get("requirements")
    sourcing_agent = get_sourcing_agent()
    sourcing_state = {"job_need": job_need, "requirements": requirements, "session_id": session_id}
    sourcing_result = sourcing_agent(sourcing_state)

    # Content
    topic = data.get("topic")
    keywords = data.get("keywords")
    persona = data.get("persona")
    tone = data.get("tone")
    channel = data.get("channel")
    content_result = generate_seo_content(topic, keywords, persona, tone, channel)

    # Support and Satisfaction

    # Ticket Auto Response
    ticket_text = data.get("ticket_text", "").strip()
    auto_response = generate_response(ticket_text) if ticket_text else ""

    # Smart Ticket Routing
    if ticket_text:
        summary = summarize_ticket(ticket_text)
        category, complexity, labels, scores = classify_ticket(summary)
        assigned_agent = route_ticket(summary, category, complexity)
        routing_result = {
            "summary": summary,
            "category": category,
            "complexity": complexity,
            "labels": labels,
            "scores": scores,
            "assigned_agent": assigned_agent
        }
    else:
        routing_result = {}

    # Live Sentiment Analysis
    raw_message = data.get("raw_message", "")
    sentiment_result = live_sentiment_agent({"raw_message": raw_message}) if raw_message else {}

    # Emotion Tagging
    emotion_scores = tag_emotions_in_chat(raw_message) if raw_message else {}

    # Churn Risk Scoring
    timestamps_raw = data.get("timestamps", [])
    sentiment_scores = data.get("sentiment_scores", [])
    messages = data.get("messages", [])
    from datetime import datetime
    timestamps = [datetime.fromisoformat(ts) for ts in timestamps_raw] if timestamps_raw else []
    churn_score = calculate_churn_risk(timestamps, sentiment_scores, messages) if timestamps else None

    # SLA Alerting
    try:
        sla_alert_status = check_sla_and_alert()
    except Exception as e:
        sla_alert_status = f"Error running SLA alert: {str(e)}"

    # Sensitive Data Filtering
    compliance_text = data.get("compliance_text", "")
    sensitive_data_result = detect_sensitive_terms(compliance_text) if compliance_text else {}

    # Aggregate all support & satisfaction results
    support_result = {
        "auto_response": auto_response,
        "routing_result": routing_result,
        "sentiment_result": sentiment_result,
        "emotion_scores": emotion_scores,
        "churn_risk_score": churn_score,
        "sla_alert_status": sla_alert_status,
        "sensitive_data_filter": sensitive_data_result
    }

    return json_ok({
        "session_id": session_id,
        "sourcing_result": sourcing_result,
        "content_result": content_result,
        "support_result": support_result
    })


# Multi-agent workflow orchestrator (simple, non-graph)
WORKFLOWS = {
    "onboarding": [
        "onboarding_agent",
        "billing_agent",
        "legal_agent"
    ],
    "content": [
        "content_agent",
        "generate_snippets",
        "keyword_injection_engine",
        "multi_variant_content_generator",
        "persona_content_rewriter",
        "multi_channel_formatter",
        "content_to_slides",
    ],
    "Support": [
        "ticket_auto_response",
        "smart_ticket_router",
        "live_sentiment_agent",
        "emotion_tagging",
        "churn_detection",
        "sla_alerting",
        "sensitive_data_filter",
        "ticket_conversation_summarizer",
        "audit_log_failure",
    ],
}

def get_agent(agent_name):
    if agent_name == "onboarding_agent":
        from agents.crm.crm import onboarding_agent
        return onboarding_agent
    if agent_name == "billing_agent":
        from tools.db_tools import update_lead_status
        return update_lead_status
    if agent_name == "legal_agent":
        return lambda x: {"legal": "checked", **x}
    if agent_name == "content_agent":
        from agents.content_generation.content_Seo import generate_seo_content as _seo
        return lambda x: _seo(
            x.get("topic"),
            x.get("keywords"),
            x.get("persona"),
            x.get("tone"),
            x.get("channel")
        )
    if agent_name == "visual_agent":
        return lambda x: {"visual": "not implemented", **x}
    if agent_name == "multi_channel_formatter":
        from agents.content_generation.multi_channel_formatter import multi_channel_format as _fmt
        return lambda x: _fmt(x, x.get("channels", ["instagram", "linkedin", "email", "cms"]))
    if agent_name == "persona_content_rewriter":
        from agents.content_generation.persona_content_rewriter import persona_content_rewriter as _rew
        return lambda x: _rew(
            x.get("body", ""),
            x.get("persona", ""),
            x.get("channel", ""),
            x.get("keywords", "")
        )
    if agent_name == "generate_snippets":
        from agents.content_generation.content_snippet_generator import generate_snippets as _snip
        return lambda x: _snip(
            x.get("body", ""),
            x.get("persona", ""),
            x.get("tone", ""),
            x.get("audience", ""),
            x.get("channel", ""),
            x.get("keywords", "")
        )
    if agent_name == "keyword_injection_engine":
        from agents.content_generation.keyword_injection_engine import keyword_injection_engine as _inj
        return lambda x: _inj(
            x.get("body", ""),
            x.get("topic", ""),
            float(x.get("density", 0.02))
        )
    if agent_name == "multi_variant_content_generator":
        from agents.content_generation.multi_variant_content_generator import generate_content_variants as _var
        return lambda x: _var(
            x.get("body", ""),
            x.get("cta_styles", ["standard"]),
            x.get("sentiment_tones", ["neutral"]),
            int(x.get("num_variants", 3))
        )
    if agent_name == "content_to_slides":
        from agents.content_generation.visual import content_to_slides as _slides
        return lambda x: _slides(
            x.get("body", ""),
            x.get("headings", []),
            x.get("subheadings", []),
            x.get("key_points", []),
            x.get("brand_guidelines", {}),
            x.get("template", "default")
        )

    # Support and Satisfaction Agents

    if agent_name == "ticket_auto_response":
        from agents.support_and_satisfaction.ticket_auto_response import generate_response
        return lambda x: {"auto_response": generate_response(x.get("ticket_text", ""))}

    if agent_name == "smart_ticket_router":
        from agents.support_and_satisfaction.smart_ticket_router import summarize_ticket, classify_ticket, route_ticket
        def router(state):
            text = state.get("ticket_text", "")
            if not text:
                return state
            summary = summarize_ticket(text)
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
            return state
        return router

    if agent_name == "live_sentiment_agent":
        from agents.support_and_satisfaction.real_time_sentiment_detection import live_sentiment_agent
        return live_sentiment_agent

    if agent_name == "emotion_tagging":
        from agents.support_and_satisfaction.emotion_tagging import tag_emotions_in_chat
        def tagger(state):
            message = state.get("message", "")
            state["emotion_scores"] = tag_emotions_in_chat(message) if message else {}
            return state
        return tagger

    if agent_name == "churn_detection":
        from agents.support_and_satisfaction.emotional_pattern_churn_detection import calculate_churn_risk
        def churn(state):
            timestamps = state.get("timestamps", [])
            sentiment_scores = state.get("sentiment_scores", [])
            messages = state.get("messages", [])
            from datetime import datetime
            parsed_timestamps = [datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps] if timestamps else []
            churn_score = calculate_churn_risk(parsed_timestamps, sentiment_scores, messages) if parsed_timestamps else None
            state["churn_risk_score"] = churn_score
            return state
        return churn

    if agent_name == "sla_alerting":
        from agents.support_and_satisfaction.sla_alerting import check_sla_and_alert
        def sla(state):
            try:
                check_sla_and_alert()
                state["sla_alert_status"] = "SLA alert check completed."
            except Exception as e:
                state["sla_alert_status"] = f"Error in SLA alert: {str(e)}"
            return state
        return sla

    if agent_name == "sensitive_data_filter":
        from agents.support_and_satisfaction.sensitive_data_filter import detect_sensitive_terms
        def filter_sensitive(state):
            text = state.get("text", "")
            state["sensitive_data_filter"] = detect_sensitive_terms(text) if text else {}
            return state
        return filter_sensitive

    if agent_name == "ticket_conversation_summarizer":
        from agents.support_and_satisfaction.ticket_conversation_summarizer import generate_ticket_summary
        def summarizer(state):
            conversation = state.get("conversation_thread", "")
            state["ticket_summary"] = generate_ticket_summary(conversation) if conversation else ""
            return state
        return summarizer

    if agent_name == "audit_log_failure":
        from agents.support_and_satisfaction.audit_log_mongodb import log_failed_response
        def audit(state):
            # Typically for logging on failure, no state change
            return state
        return audit

    # Default fallback agent
    return lambda x: x


@app.post("/workflow")
async def workflow_api(request: Request):
    data = await request.json()
    input_data = data.get("input", {}) or {}
    workflow_type = data.get("workflow_type") or detect_workflow_type(input_data)
    session_id = input_data.get("session_id", str(uuid.uuid4()))
    steps = WORKFLOWS.get(workflow_type, [])
    state = input_data
    for agent_name in steps:
        agent = get_agent(agent_name)
        state = agent(state)
    return json_ok({"session_id": session_id, "workflow_type": workflow_type, "result": state})

# -----------------------------------------------------------------------------
# Chat Orchestrator (LLM classification -> route to graph)
# -----------------------------------------------------------------------------
# Sanitize quoted API key early
_raw_key = os.getenv("OPENAI_API_KEY", "").strip()
if (_raw_key.startswith('"') and _raw_key.endswith('"')) or (_raw_key.startswith("'") and _raw_key.endswith("'")):
    os.environ["OPENAI_API_KEY"] = _raw_key[1:-1]


@app.post("/chat-orchestrate")
async def chat_orchestrate(request: Request):
    session_id = str(uuid.uuid4())
    try:
        data = await request.json()
        message = (data.get("message") or "").strip()
        if not message:
            return json_ok({"response": "Please enter your requirement.", "session_id": session_id})

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

        prompt = (
                "You are a strict router. Respond ONLY with JSON. Allowed schemas:\n"
                '{"workflow_type":"sourcing","parameters":{"job_need":"...","requirements":"..."}},\n'
                '{"workflow_type":"content","parameters":{"topic":"...","keywords":"..."}},\n'
                '{"workflow_type":"support_and_satisfaction","parameters":{"ticket_text":"...","raw_message":"...","timestamps":[],"sentiment_scores":[],"messages":[],"compliance_text":"...","conversation_thread":"..."}}\n'
                'If the user intent is unclear respond with {"error":"Could not understand"}. No code fences. User message: ' + message
        )

        try:
            llm_result = llm.invoke(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return json_ok({"response": "Model unavailable. Please retry later.", "session_id": session_id})

        raw = getattr(llm_result, "content", str(llm_result))
        raw = cleanup_llm_output(raw)
        logger.info(f"[chat:{session_id}] raw={raw!r}")

        try:
            parsed = json.loads(raw)
        except Exception as e:
            logger.error(f"[chat:{session_id}] parse error: {e}")
            return json_ok({"response": "Sorry, I could not understand your requirement. Please rephrase.",
                            "session_id": session_id})

        if isinstance(parsed, dict) and parsed.get("error"):
            return json_ok({"response": "Sorry, I could not understand your requirement. Please rephrase.",
                            "session_id": session_id})

        workflow_type = parsed.get("workflow_type")
        params = parsed.get("parameters", {})

        synonym_map = {"onboarding": "sourcing", "recruitment": "sourcing", "hiring": "sourcing"}
        workflow_type = synonym_map.get(workflow_type, workflow_type)

        if workflow_type not in ("sourcing", "content", "support_and_satisfaction"):
            return json_ok({"response": "Unsupported workflow. Please clarify hiring, content, or support.",
                            "session_id": session_id})

        if workflow_type == "content":
            params.setdefault("persona", "executive")
            params.setdefault("tone", "professional")
            params.setdefault("channel", "blog")
            params.setdefault("keywords", params.get("keywords", ""))
            params.setdefault("topic", params.get("topic", message))

        compiled = None
        try:
            compiled = get_compiled_graph(workflow_type)
        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            return json_ok({"response": "Internal routing error.", "session_id": session_id})

        try:
            output_state = compiled.invoke(params)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return json_ok({"response": f"Workflow execution failed: {e.__class__.__name__}", "session_id": session_id})

        if workflow_type == "content" and isinstance(output_state, dict):
            response_payload = output_state.get("aggregate", output_state)
        else:
            response_payload = output_state

        return json_ok({"response": response_payload, "workflow_type": workflow_type, "session_id": session_id})

    except Exception as e:
        logger.exception("chat-orchestrate fatal")
        return json_err(f"Error: {str(e)}", status_code=500)

