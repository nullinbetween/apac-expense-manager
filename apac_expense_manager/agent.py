"""
APAC Expense Manager — Multi-Agent System (v9.3) · PUBLIC SHOWCASE EDITION

Copyright (c) 2026 Meredith Cheung (nullinbetween). All rights reserved.
Source-visible for reference only — see LICENSE. Not open source.

This file shows the real agent architecture and wiring of the hackathon build.
System prompts and the correction-guard / modify implementations are withheld
(see WITHHELD.md), so this edition is intentionally not runnable as-is.

Architecture:
  Primary Agent (router + set_language + modify_expense tools)
  ├── expense_categorizer  → categorize free text / receipt photos + save
  │     ├── before_model_callback → Python-layer correction guard
  │     └── after_tool_callback   → last-saved-expense state + confirmation
  ├── expense_query        → query / delete via MCP Toolbox (BigQuery)
  └── expense_reporter     → spending analysis, reimbursement, what-if
"""

import logging
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
from google.cloud import bigquery
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loader — system prompts live in ./prompts/ (withheld in this edition)
# ---------------------------------------------------------------------------
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a system prompt by name. Prompts are not part of the public showcase."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt '{name}' is withheld from the public showcase. See WITHHELD.md."
        )
    return path.read_text(encoding="utf-8")


STATE_USER_LANGUAGE = "user_language"
STATE_LAST_SAVED_EXPENSE = "last_saved_expense"
STATE_CORRECTION_INTERCEPTED = "_correction_intercepted"

# ---------------------------------------------------------------------------
# Dynamic date context — so Gemini doesn't try to write Python code
# ---------------------------------------------------------------------------
_tz = ZoneInfo("Asia/Tokyo")
_now = datetime.now(_tz)
TODAY_DATE = _now.strftime("%Y-%m-%d")
YESTERDAY_DATE = (_now - timedelta(days=1)).strftime("%Y-%m-%d")
THIS_MONTH_START = _now.strftime("%Y-%m-01")
_last_month = _now.replace(day=1) - timedelta(days=1)
LAST_MONTH_START = _last_month.replace(day=1).strftime("%Y-%m-%d")
LAST_MONTH_END = _last_month.strftime("%Y-%m-%d")

DATE_CONTEXT = f"""
**Date reference (use these directly, do NOT write code to calculate dates):**
- Today: {TODAY_DATE}
- Yesterday: {YESTERDAY_DATE}
- This month starts: {THIS_MONTH_START}
- Last month: {LAST_MONTH_START} to {LAST_MONTH_END}
- IMPORTANT: When the user says "today", "this month", etc., use the dates above directly as string parameters. NEVER generate Python code. Just pass the date string to the tool.
"""

# ---------------------------------------------------------------------------
# Cross-currency exchange rates (demo / approximate)
# ---------------------------------------------------------------------------
# Base currency: JPY = 1.0
# These are approximate rates for demonstration purposes only.
# In production, these would come from a live exchange rate API.
EXCHANGE_RATES_TO_JPY = {
    "JPY": 1.0,
    "HKD": 19.5,      # 1 HKD ≈ 19.5 JPY
    "SGD": 112.0,      # 1 SGD ≈ 112 JPY
    "TWD": 4.7,        # 1 TWD ≈ 4.7 JPY
    "THB": 4.3,        # 1 THB ≈ 4.3 JPY
    "KRW": 0.11,       # 1 KRW ≈ 0.11 JPY
    "MYR": 33.0,       # 1 MYR ≈ 33 JPY
    "AUD": 97.0,       # 1 AUD ≈ 97 JPY
    "INR": 1.75,       # 1 INR ≈ 1.75 JPY
    "PHP": 2.6,        # 1 PHP ≈ 2.6 JPY
    "USD": 150.0,      # 1 USD ≈ 150 JPY
    "VND": 0.006,      # 1 VND ≈ 0.006 JPY
    "KHR": 0.037,      # 1 KHR ≈ 0.037 JPY
    "MMK": 0.071,      # 1 MMK ≈ 0.071 JPY
}

FX_CONTEXT = load_prompt("currency_display_rules")

# ---------------------------------------------------------------------------
# OpenCC setup — for Traditional/Simplified Chinese conversion
# ---------------------------------------------------------------------------
try:
    import opencc
    s2t_converter = opencc.OpenCC('s2t')  # Simplified → Traditional
    t2s_converter = opencc.OpenCC('t2s')  # Traditional → Simplified
    OPENCC_AVAILABLE = True
except ImportError:
    OPENCC_AVAILABLE = False
    s2t_converter = None
    t2s_converter = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
model_name = "gemini-2.5-flash"
TOOLBOX_URL = os.environ.get("TOOLBOX_URL")
if not TOOLBOX_URL:
    raise ValueError("TOOLBOX_URL environment variable is required")

# ---------------------------------------------------------------------------
# Language callback — intercepts model output and converts Chinese script
# ---------------------------------------------------------------------------
def language_callback(callback_context, llm_response):
    """After-model callback: enforce correct Chinese script based on user's language choice.

    Reads 'user_language' from session state (set during onboarding).
    - If user chose 繁體中文: convert any Simplified Chinese → Traditional
    - If user chose 简体中文: convert any Traditional Chinese → Simplified
    - For other languages (EN, JP, KR): no conversion needed
    """
    if not OPENCC_AVAILABLE:
        return llm_response

    # Get user's language preference from session state
    user_lang = callback_context.state.get(STATE_USER_LANGUAGE, "")

    if not user_lang or user_lang not in ("繁體中文", "简体中文"):
        return llm_response

    # Process each part of the response
    if llm_response and llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                if user_lang == "繁體中文":
                    part.text = s2t_converter.convert(part.text)
                elif user_lang == "简体中文":
                    part.text = t2s_converter.convert(part.text)

    return llm_response

# ---------------------------------------------------------------------------
# Language tool — programmatic guarantee for user_language in session state
# ---------------------------------------------------------------------------
VALID_LANGUAGES = {"繁體中文", "简体中文", "English", "日本語", "한국어"}
LANGUAGE_NUMBER_MAP = {
    "1": "繁體中文",
    "2": "简体中文",
    "3": "English",
    "4": "日本語",
    "5": "한국어",
}

def set_language(language: str, tool_context: ToolContext) -> str:
    """Set the user's preferred language for this session.

    Call this tool whenever the user selects a language (from the language picker
    or detected from their input). This guarantees the language preference is
    stored in session state for OpenCC and all sub-agents to use.

    Args:
        language: One of: 繁體中文, 简体中文, English, 日本語, 한국어
    """
    # Normalize: accept number shortcuts (1-5) from language picker
    if language in LANGUAGE_NUMBER_MAP:
        language = LANGUAGE_NUMBER_MAP[language]

    if language not in VALID_LANGUAGES:
        return f"Invalid language: {language}. Valid options: {', '.join(VALID_LANGUAGES)}"

    tool_context.state[STATE_USER_LANGUAGE] = language
    return f"Language set to {language}"

set_language_tool = FunctionTool(set_language)

# ---------------------------------------------------------------------------
# modify_expense tool — direct BigQuery UPDATE, bypasses Gemini reasoning
# ---------------------------------------------------------------------------
# Why this exists: Gemini 2.5 Flash used to autonomously combine query+delete+save
# to handle modify requests (emergent behavior, worked 4/8). The model regressed
# by 4/28 — same prompt, same code, modify stopped working. Four rounds of prompt
# engineering (v8.1.1-v8.1.4) all failed because the root cause was model-side.
# Solution: hardcode the modify logic in Python. LLM only needs to extract params.
# ---------------------------------------------------------------------------
BQ_PROJECT = os.environ.get("PROJECT_ID", "your-gcp-project-id")
BQ_TABLE = f"{BQ_PROJECT}.expense_data.expenses"


def modify_expense(
    old_store: str,
    old_amount: str,
    old_currency: str,
    new_amount: str,
    new_currency: str,
    new_store: str,
    new_country: str,
    new_category: str,
    tool_context: ToolContext,
) -> str:
    """Modify an existing expense record in BigQuery (deterministic path).

    Finds the record by its OLD values, then applies a parameterized UPDATE for
    any NEW values provided. The LLM only extracts parameters; the write itself
    is plain Python, so it keeps working even if model behaviour regresses.

    Args:
        old_store / old_amount / old_currency: values used to locate the record.
        new_amount / new_currency / new_store / new_country / new_category:
            values to change ("" = keep unchanged).

    Implementation withheld from the public showcase.
    """
    raise NotImplementedError("Withheld from the public showcase. See WITHHELD.md.")


modify_expense_tool = FunctionTool(modify_expense)

# ---------------------------------------------------------------------------
# MCP Toolset — connects to MCP Toolbox for BigQuery
# ---------------------------------------------------------------------------
def create_mcp_toolset():
    """Create MCP Toolset for BigQuery access via MCP Toolbox on Cloud Run."""
    return MCPToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=f"{TOOLBOX_URL}/mcp"
        )
    )

# Each sub-agent that needs BigQuery gets its own toolset instance
query_toolset = create_mcp_toolset()
reporter_toolset = create_mcp_toolset()
categorizer_toolset = create_mcp_toolset()

# ---------------------------------------------------------------------------
# Shared constants — single source of truth for all agents
# ---------------------------------------------------------------------------
APAC_COUNTRIES = "JP, HK, SG, TW, TH, KR, MY, AU, IN, PH, VN, KH, MM, or other APAC country code"
APAC_CURRENCIES = "JPY, HKD, SGD, TWD, THB, KRW, MYR, AUD, INR, PHP, VND, KHR, MMK, USD"
APAC_CATEGORIES = "食費, 交通, 日用品, 子供関連, サブスク, 住居・光熱費, 医療, 娯楽, 衣服・美容, その他"

# ---------------------------------------------------------------------------
# Shared language instruction block (used by all agents)
# ---------------------------------------------------------------------------
LANGUAGE_INSTRUCTION = load_prompt("language_rules")

# ---------------------------------------------------------------------------
# Categorizer guard: before_model_callback — intercept corrections BEFORE Gemini
# ---------------------------------------------------------------------------
# Why this exists: ADK sub-agent "stickiness" means that after categorizer handles
# a message, the NEXT user message may go directly to categorizer without root
# routing. If that message is a correction ("sorry it should be 500"), categorizer
# saves it as a NEW expense. Prompt-based guards (v9.1.1) were completely ignored
# by Gemini. This callback runs in Python BEFORE Gemini sees the message, so it
# cannot be overridden by the model.
# ---------------------------------------------------------------------------
# CORRECTION_SIGNALS — multilingual correction lexicon (EN / 繁 / 简 / 粵 / 日 / 한).
# Withheld from the public showcase.


def categorizer_before_model(callback_context, llm_request):
    """Python-layer correction guard (runs BEFORE Gemini).

    ADK sub-agents are "sticky": after the categorizer saves an expense, the
    next user message may reach it directly. A correction such as
    "sorry, it should be 500" would then be saved as a NEW expense.

    This callback detects correction turns with a multilingual lexicon,
    returns a localized acknowledgement and hands control back to the root
    agent (which owns modify_expense) — so the model never gets the chance
    to misread the correction.

    Implementation (lexicon, loop-guard and hand-off mechanics) withheld.
    """
    raise NotImplementedError("Withheld from the public showcase. See WITHHELD.md.")


# ---------------------------------------------------------------------------
# Categorizer after_tool_callback — save last expense to state for modify
# ---------------------------------------------------------------------------
def categorizer_after_tool(tool, args, tool_context, tool_response):
    """State tracking after save_expense succeeds.

    Records the last saved expense in session state so the root agent can
    correct it without relying on the model's conversational memory, and makes
    sure a user-facing confirmation is always produced (including after
    receipt-image turns, where the model occasionally returns empty text).

    Implementation withheld from the public showcase.
    """
    raise NotImplementedError("Withheld from the public showcase. See WITHHELD.md.")


# ---------------------------------------------------------------------------
# Sub-Agent 1: Expense Categorizer
# ---------------------------------------------------------------------------
expense_categorizer = Agent(
    name="expense_categorizer",
    model=model_name,
    description="Categorizes expenses into structured data with APAC country/currency detection, and saves them to the database. Handles both free-text descriptions AND receipt/image inputs (photos of receipts, credit card slips, etc.).",
    instruction=load_prompt("categorizer"),
    tools=[categorizer_toolset],
    before_model_callback=categorizer_before_model,
    after_model_callback=language_callback,
    after_tool_callback=categorizer_after_tool,
)

# ---------------------------------------------------------------------------
# Sub-Agent 2: Expense Query (with date range support)
# ---------------------------------------------------------------------------
expense_query = Agent(
    name="expense_query",
    model=model_name,
    description="Queries and retrieves expense data from BigQuery. Use for questions about past spending, finding specific expenses, or getting raw data. Supports date range filtering.",
    instruction=load_prompt("query"),
    tools=[query_toolset],
    after_model_callback=language_callback,
)

# ---------------------------------------------------------------------------
# Sub-Agent 3: Expense Reporter (analysis & insights + cross-currency)
# ---------------------------------------------------------------------------
expense_reporter = Agent(
    name="expense_reporter",
    model=model_name,
    description="Generates spending analysis reports, trend insights, budget comparisons, and financial summaries across APAC countries. Supports date-range-based analysis, cross-currency summaries, and conversion when the user requests a target currency.",
    instruction=load_prompt("reporter"),
    tools=[reporter_toolset],
    after_model_callback=language_callback,
)

# ---------------------------------------------------------------------------
# Primary Agent (Router / Coordinator)
# ---------------------------------------------------------------------------
root_agent = Agent(
    name="apac_expense_manager",
    model=model_name,
    description="Primary agent that coordinates APAC household expense management — categorizing, querying, and analyzing expenses across Asia-Pacific countries.",
    instruction=load_prompt("root"),
    tools=[set_language_tool, modify_expense_tool],
    sub_agents=[expense_categorizer, expense_query, expense_reporter],
    after_model_callback=language_callback,
)
