import os
import datetime
from sqlalchemy import create_engine, select, Table, MetaData, Column, String, DateTime, Float,text
from pydantic import BaseModel, Field
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

# Load .env configuration
load_dotenv()

# Required environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")
MS_TEAMS_WEBHOOK = os.getenv("MS_TEAMS_WEBHOOK")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

# Validate essential variables
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")
if not SLACK_TOKEN:
    raise RuntimeError("SLACK_TOKEN not set")
if not SLACK_CHANNEL:
    raise RuntimeError("SLACK_CHANNEL not set")

# Setup DB connection and metadata
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define ticket table; adjust schema as needed
tickets = Table(
    "tickets", metadata,
    Column("ticket_id", String, primary_key=True),
    Column("last_agent_response", DateTime),
    Column("created_at", DateTime),
    Column("sla_hours", Float),
    Column("assigned_team", String),
    Column("customer_phone", String, nullable=True)  # For SMS alerts
)

# Initialize Slack client
slack_client = WebClient(token=SLACK_TOKEN)

# Initialize Twilio client if credentials exist
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

WARNING_THRESHOLD_HOURS = 0.5  # Threshold for early alert (30 mins before breach)

def fetch_open_tickets():
    with engine.connect() as conn:
        stmt = select(
            tickets.c.ticket_id,
            tickets.c.last_agent_response,
            tickets.c.created_at,
            tickets.c.sla_hours,
            tickets.c.assigned_team,
            tickets.c.customer_phone
        )
        return conn.execute(stmt).fetchall()

def send_slack_alert(message, team):
    try:
        slack_client.chat_postMessage(
            channel=SLACK_CHANNEL,
            text=f"{message}\nTeam: {team}"
        )
        print("Slack alert sent.")
    except SlackApiError as e:
        print(f"Slack API error: {e.response['error']}")

def send_teams_alert(message):
    if not MS_TEAMS_WEBHOOK:
        print("MS Teams webhook not configured.")
        return
    try:
        payload = {"text": message}
        r = requests.post(MS_TEAMS_WEBHOOK, json=payload)
        if r.status_code == 200:
            print("MS Teams alert sent.")
        else:
            print(f"MS Teams alert failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"MS Teams error: {e}")

def send_sms_alert(message, to_number):
    if not twilio_client or not to_number:
        print("Twilio client not configured or no phone number provided.")
        return
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=to_number
        )
        print(f"SMS alert sent to {to_number}.")
    except Exception as e:
        print(f"Twilio SMS error: {e}")

def check_sla_and_alert():
    now = datetime.datetime.utcnow()
    tickets_list = fetch_open_tickets()
    for ticket in tickets_list:
        last_resp = ticket.last_agent_response or ticket.created_at
        elapsed_hours = (now - last_resp).total_seconds() / 3600.0
        remaining = ticket.sla_hours - elapsed_hours

        if remaining <= 0:
            alert = (
                f":rotating_light: *SLA BREACHED!* Ticket {ticket.ticket_id} "
                f"exceeded SLA of {ticket.sla_hours} hours."
            )
            send_slack_alert(alert, ticket.assigned_team)
            send_teams_alert(alert)
            send_sms_alert(alert, ticket.customer_phone)
        elif remaining <= WARNING_THRESHOLD_HOURS:
            alert = (
                f":warning: *SLA WARNING:* Ticket {ticket.ticket_id} nearing breach. "
                f"Only {remaining*60:.0f} minutes left."
            )
            send_slack_alert(alert, ticket.assigned_team)
            send_teams_alert(alert)
            send_sms_alert(alert, ticket.customer_phone)



if __name__ == "__main__":
    check_sla_and_alert()
