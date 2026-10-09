"""
gemini_client.py — Wrapper for Google Gemini AI (google-genai SDK).

Provides:
  - generate_sql()  : Generates a SQL query from a natural-language question.
  - explain_query() : Returns a short plain-English explanation of a SQL query.
"""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
import os
import re
import textwrap

from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Configuration (read from environment; no defaults for the API key)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Lazy client — instantiated once on first use so import never crashes
# ---------------------------------------------------------------------------
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Create backend/.env with GEMINI_API_KEY=<your_key>."
            )
        _client = genai.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------
_SQL_SYSTEM = textwrap.dedent("""
    You are an expert SQLite query generator.

    Rules — follow them without exception:
    1. Return ONLY a single raw SQL SELECT statement. No markdown, no code fences,
       no explanations, no trailing semicolons, no comments, no extra text.
    2. Use only tables and columns that appear in the schema provided.
    3. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA,
       or any statement other than SELECT.
    4. Do not use multiple statements separated by semicolons.
    5. Use standard SQLite syntax (e.g. strftime, LIMIT, GROUP BY).
    6. Limit results to at most 100 rows using LIMIT where sensible.
    7. Use table aliases to keep queries readable.
    8. If the question cannot be answered from the available schema, return exactly:
       CANNOT_ANSWER
""").strip()

_EXPLAIN_SYSTEM = textwrap.dedent("""
    You are a concise technical communicator.
    Given a SQL query and the original question, write 1–3 plain-English sentences
    explaining what the query does — no bullet points, no markdown, no code fences.
    Speak directly to a non-technical audience.
""").strip()


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from the model's response."""
    text = raw.strip()
    # Remove ```sql ... ``` or ``` ... ``` blocks
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().rstrip(";")


def _call_model(system_instruction: str, user_content: str) -> str:
    """Call Gemini and return the text response."""
    client = _get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=1024,
        ),
        contents=user_content,
    )
    # Handle both streaming and non-streaming responses
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    # Fallback: iterate parts
    parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text"):
                parts.append(part.text)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sql(question: str, schema_ddl: str) -> str:
    """
    Generate a SQLite SELECT statement for the given natural-language question.

    Returns the SQL string, or raises RuntimeError on API/model failures.
    Raises ValueError if the model indicates it cannot answer.
    """
    user_content = (
        f"Database schema:\n{schema_ddl}\n\n"
        f"Question: {question}"
    )
    raw = _call_model(_SQL_SYSTEM, user_content)
    sql = _clean_sql(raw)

    if sql.upper().startswith("CANNOT_ANSWER"):
        raise ValueError(
            "The AI model could not generate a query for this question "
            "based on the available schema."
        )

    return sql


def explain_query(question: str, sql: str) -> str:
    """
    Generate a plain-English explanation of what the SQL query does.

    Returns the explanation string. Never raises; returns a fallback on error.
    """
    try:
        user_content = (
            f"Original question: {question}\n\n"
            f"SQL query:\n{sql}"
        )
        return _call_model(_EXPLAIN_SYSTEM, user_content)
    except Exception:
        return "This query retrieves data from the database based on your question."
