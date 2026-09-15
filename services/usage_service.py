import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def track_usage(endpoint: str) -> None:
    if not DATABASE_URL:
        print("DATABASE_URL is not configured")
        return

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO api_usage (endpoint)
                    VALUES (%s)
                    """,
                    (endpoint,)
                )

    except Exception as error:
        # Don't let usage tracking break the actual API
        print(f"Usage tracking failed: {error}")