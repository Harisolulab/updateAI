from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os

# Load your DB URL from environment or config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/yourdb")

app = FastAPI(title="NPS & CSAT Dashboard API")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

@app.get("/metrics/nps-csat")
def nps_csat_metrics():
    with SessionLocal() as db:
        overall_metrics = db.execute(text("""
            SELECT
                ROUND(AVG(nps_score)::numeric, 2) AS avg_nps,
                ROUND(AVG(csat_score)::numeric, 2) AS avg_csat,
                COUNT(*) AS total_responses,
                COUNT(*) FILTER (WHERE sentiment = 'Positive') AS positive_count,
                COUNT(*) FILTER (WHERE sentiment = 'Neutral') AS neutral_count,
                COUNT(*) FILTER (WHERE sentiment = 'Negative') AS negative_count
            FROM customer_feedback
            WHERE nps_score IS NOT NULL OR csat_score IS NOT NULL
        """)).first()

        trend_rows = db.execute(text("""
            SELECT
                TO_CHAR(created_at, 'YYYY-MM-DD') AS day,
                ROUND(AVG(nps_score)::numeric, 2) AS avg_nps,
                ROUND(AVG(csat_score)::numeric, 2) AS avg_csat,
                COUNT(*) AS response_count
            FROM customer_feedback
            GROUP BY day
            ORDER BY day ASC
            LIMIT 30
        """)).fetchall()

        trend_data = [
            {
                "date": row.day,
                "avg_nps": float(row.avg_nps or 0),
                "avg_csat": float(row.avg_csat or 0),
                "response_count": row.response_count,
            }
            for row in trend_rows
        ]

        return {
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
