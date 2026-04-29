"""
APAC Expense Manager — Multi-Agent System (v9.3)
Primary agent + 3 sub-agents for household expense management across Asia-Pacific.

v9.3 changes:
  - Correction interception now includes transfer_to_agent function call in the
    LlmResponse. Previously, interception only returned canned text and stopped —
    user had to manually ask again to trigger modify. Now ADK routes back to root
    agent automatically, where modify_expense can be called with last_saved_expense.
  - CORRECTION_SIGNALS regex tightened: removed 'edit', 'update', 'wait' (too
    generic, caused false positives like "have u even edit my expenses?"). Changed
    'change' to 'change it/to/the' for specificity.
v9.2 changes:
  - CRITICAL FIX: categorizer before_model_callback — programmatic correction
    detection. When categorizer sub-agent is still active and receives a correction
    message (sorry/should be/寫錯/修正/間違), Python intercepts BEFORE Gemini runs,
    returns a canned response telling the user to rephrase for modify. This prevents
    categorizer from saving corrections as new expenses (v9.1.1 prompt guard failed).
  - last_saved_expense state: after categorizer successfully calls save_expense,
    an after_tool_callback writes {store, amount, currency} to session state.
    Root agent's modify_expense can then read these values instead of relying on
    Gemini's conversation memory.
  - Root agent prompt updated to read last_saved_expense from state.
v9.1 changes:
  - Receipt recognition: categorizer handles receipt images (from v8.2, prompt-only)
  - Expense tagging: categorizer adds [child]/[household]/[reimbursable]/[transport]/[personal]
    tags in notes field for downstream reimbursement reporting
  - Reimbursement summary: reporter generates itemized reimbursement reports
    (filtered by tags + category, copy-paste ready output)
  - Cost-saving analysis: reporter does break-even calculations
    (e.g., mama bicycle vs taxi comparison using actual spending data)
v9.0 changes:
  - modify_expense FunctionTool: programmatic expense modification via direct BigQuery
    client. Replaces emergent delete+save reasoning that broke due to Gemini 2.5 Flash
    model regression (same prompt worked 4/8, failed 4/28).
    Design: root_agent calls modify_expense directly → bypasses delegation/routing.
    BigQuery client does query→UPDATE in Python, no LLM reasoning needed.
    "Emergence needs hardening" — critical workflows need programmatic guarantees.
v8.1.4 changes:
  - Hybrid approach: correction detection + enabling tone. Still failed — confirmed
    as Gemini model regression, not prompt problem (v7.6 regression test 00026-bhq).
v8.1.3 changes:
  - Prompt tone rewrite: constraining → enabling. Reduced NEVER/ALWAYS/MUST overload.
v8.1.2 changes:
  - Root agent routing disambiguation + expense_query autonomous ID lookup.
v8.1.1 changes:
  - Explicit modify/change/update/correct routing to expense_query.
v8.1 changes:
  - Removed currency→country direct mapping (currency ≠ country in APAC reality)
v8.0 changes (Refinement Phase):
  - set_language tool: programmatic guarantee for user_language in session state
v7.4 changes:
  - Currency display: no default/primary currency.
v7.3 changes:
  - expense_id (UUID) + delete by ID
v7.2 changes:
  - Country list centralized, non-APAC countries accepted, store query parameter added
v7 changes:
  - Cross-currency conversion: fixed demo exchange rates (EXCHANGE_RATES_TO_JPY)
v6 changes:
  - Language onboarding + OpenCC + date range + delete expense + KR support

Architecture:
  Primary Agent (Router + modify_expense tool)
  ├── expense_categorizer  → Categorize free-text expenses + save to BigQuery
  ├── expense_query        → Query BigQuery via MCP Toolbox (with date filtering)
  └── expense_reporter     → Generate spending analysis & insights
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

FX_CONTEXT = f"""
**Cross-currency conversion (demo rates):**

Available exchange rates (between currencies):
{chr(10).join(f'   - 1 {cur} ≈ {rate} JPY' for cur, rate in EXCHANGE_RATES_TO_JPY.items() if cur != 'JPY')}

**Display currency rules (strict priority order):**
1. If the user explicitly requests a target currency (e.g., "in USD", "換算成港幣", "convert to yen"), convert all amounts to that currency and show the converted total.
2. If the user does NOT specify a target currency, show ONLY the original currency breakdown. Do NOT automatically convert to any single currency. Do NOT generate a single converted total.
3. When converting (rule 1 only), present converted totals as approximate with exchange-rate disclaimer.
4. NEVER assume, infer, or claim any currency is the user's primary/default/preferred currency.

**Example output — user does NOT specify target currency:**
📊 Spending breakdown:
- Japan: ¥8,500
- Hong Kong: HK$200
- Korea: ₩15,000
(3 currencies across 3 countries)

**Example output — user requests "in JPY":**
📊 Spending breakdown:
- Japan: ¥8,500
- Hong Kong: HK$200 (≈ ¥3,900)
- Korea: ₩15,000 (≈ ¥1,650)

💰 Approximate total: ≈ ¥14,050 JPY
※ Demo exchange rates for reference only / デモ用参考レート
"""

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
BQ_PROJECT = os.environ.get("PROJECT_ID", "my-project-2026-hackthon")
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
    """Modify an existing expense record. Finds the record by old values, then updates it.

    Use this when the user wants to correct or change a previously recorded expense.
    Common triggers: "sorry it should be X", "change to X", "寫錯了", "修正", "間違えた"

    To find the record, provide the OLD values (what was originally saved).
    To update, provide the NEW values (what it should be changed to).
    Pass empty string "" for any field you don't need to filter by or don't want to change.

    Args:
        old_store: Store name of the record to find (partial match, case-insensitive). Required.
        old_amount: Original amount to match (e.g. "520"). Use "" to skip amount matching.
        old_currency: Original currency code (e.g. "JPY"). Use "" to skip currency matching.
        new_amount: New amount to set (e.g. "500"). Use "" to keep unchanged.
        new_currency: New currency code. Use "" to keep unchanged.
        new_store: New store name. Use "" to keep unchanged.
        new_country: New 2-letter country code. Use "" to keep unchanged.
        new_category: New category in Japanese. Use "" to keep unchanged.
    """
    try:
        client = bigquery.Client(project=BQ_PROJECT)

        # --- Step 1: Find matching record(s) ---
        conditions = ["1=1"]
        params = []

        if old_store:
            conditions.append("LOWER(store) LIKE CONCAT('%', LOWER(@old_store), '%')")
            params.append(bigquery.ScalarQueryParameter("old_store", "STRING", old_store))

        if old_amount:
            conditions.append("amount = CAST(@old_amount AS FLOAT64)")
            params.append(bigquery.ScalarQueryParameter("old_amount", "STRING", old_amount))

        if old_currency:
            conditions.append("currency = @old_currency")
            params.append(bigquery.ScalarQueryParameter("old_currency", "STRING", old_currency))

        where_clause = " AND ".join(conditions)
        find_sql = f"""
            SELECT id, date, country, category, amount, currency, store, subcategory, notes
            FROM `{BQ_TABLE}`
            WHERE {where_clause}
            ORDER BY date DESC
            LIMIT 5
        """

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        results = list(client.query(find_sql, job_config=job_config).result())

        if len(results) == 0:
            return (
                f"No matching expense found for store='{old_store}'"
                f"{f', amount={old_amount}' if old_amount else ''}"
                f"{f', currency={old_currency}' if old_currency else ''}. "
                "Please check the details and try again."
            )

        if len(results) > 1:
            lines = []
            for r in results:
                lines.append(
                    f"- {r['store']} {r['amount']} {r['currency']} "
                    f"on {r['date']} ({r['category']})"
                )
            return (
                f"Found {len(results)} matching records:\n"
                + "\n".join(lines)
                + "\nPlease be more specific (add amount or currency to narrow down)."
            )

        # --- Step 2: Exactly one match — UPDATE it ---
        record = results[0]
        record_id = record["id"]

        updates = []
        update_params = [
            bigquery.ScalarQueryParameter("record_id", "STRING", record_id)
        ]
        change_log = []

        if new_amount:
            updates.append("amount = CAST(@new_amount AS FLOAT64)")
            update_params.append(
                bigquery.ScalarQueryParameter("new_amount", "STRING", new_amount)
            )
            change_log.append(f"amount: {record['amount']} → {new_amount}")

        if new_currency:
            updates.append("currency = @new_currency")
            update_params.append(
                bigquery.ScalarQueryParameter("new_currency", "STRING", new_currency)
            )
            change_log.append(f"currency: {record['currency']} → {new_currency}")

        if new_store:
            updates.append("store = @new_store")
            update_params.append(
                bigquery.ScalarQueryParameter("new_store", "STRING", new_store)
            )
            change_log.append(f"store: {record['store']} → {new_store}")

        if new_country:
            updates.append("country = @new_country")
            update_params.append(
                bigquery.ScalarQueryParameter("new_country", "STRING", new_country)
            )
            change_log.append(f"country: {record['country']} → {new_country}")

        if new_category:
            updates.append("category = @new_category")
            update_params.append(
                bigquery.ScalarQueryParameter("new_category", "STRING", new_category)
            )
            change_log.append(f"category: {record['category']} → {new_category}")

        if not updates:
            return (
                f"Found: {record['store']} {record['amount']} {record['currency']}. "
                "But no new values provided. Please specify what to change "
                "(e.g., new_amount, new_currency)."
            )

        update_sql = f"""
            UPDATE `{BQ_TABLE}`
            SET {', '.join(updates)}
            WHERE id = @record_id
        """

        update_config = bigquery.QueryJobConfig(query_parameters=update_params)
        client.query(update_sql, job_config=update_config).result()

        logger.info(
            f"modify_expense: updated record {record_id} — {', '.join(change_log)}"
        )

        return (
            f"Updated successfully: {record['store']} "
            f"{record['amount']} {record['currency']} → {', '.join(change_log)}."
        )

    except Exception as e:
        logger.error(f"modify_expense error: {e}")
        return f"Error modifying expense: {str(e)}"


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
LANGUAGE_INSTRUCTION = """
**Language rules:**
- Check session state for 'user_language'. If set, ALWAYS reply in that language.
- 繁體中文 → reply in Traditional Chinese
- 简体中文 → reply in Simplified Chinese
- English → reply in English
- 日本語 → reply in Japanese
- 한국어 → reply in Korean
- If 'user_language' is not set, match the language the user writes in.
- If you cannot determine the language, default to English.
"""

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
CORRECTION_SIGNALS = re.compile(
    r"(?:sorry|actually|should\s+be|wrong|change\s+(?:it|to|the)|no\s+no|oops|"
    r"寫錯|應該|改成|不對|搞錯|修正|唔係|唔啱|"
    r"間違|違う|修正して|変更して|ごめん|"
    r"잘못|고쳐|수정)",
    re.IGNORECASE
)


def _extract_latest_user_text(llm_request: LlmRequest) -> str:
    """Return the latest user-authored text in the request, or an empty string."""
    if not llm_request.contents:
        return ""

    for content in reversed(llm_request.contents):
        if content.role != "user" or not content.parts:
            continue
        for part in content.parts:
            if part.text:
                return part.text
    return ""


def _get_correction_intercept_message(user_lang: str) -> str:
    """Return the localized interception message for correction attempts."""
    messages = {
        "繁體中文": "這看起來是在修正上一筆紀錄，我來幫你更新。",
        "简体中文": "这看起来是在修正上一笔记录，我来帮你更新。",
        "日本語": "前の記録の修正ですね。こちらで更新します。",
        "한국어": "이전 기록 수정이네요. 제가 바로 업데이트할게요.",
    }
    return messages.get(
        user_lang,
        "This looks like a correction. I'll update it now.",
    )


def _has_correction_signal(user_text: str) -> bool:
    """Return True when the user text looks like a correction/modification turn."""
    return bool(user_text and CORRECTION_SIGNALS.search(user_text))


def _get_tool_name(tool) -> str:
    """Best-effort extraction of a tool's name across wrapper types."""
    return str(getattr(tool, "name", "") or tool or "").strip()


def _is_save_expense_tool(tool) -> bool:
    """Return True when the callback is running for save_expense."""
    tool_name = _get_tool_name(tool)
    if not tool_name:
        return False
    if tool_name == "save_expense":
        return True

    terminal_name = re.split(r"[:./]", tool_name)[-1]
    return terminal_name == "save_expense" or "save_expense" in tool_name


def _tool_response_indicates_success(tool_response) -> bool:
    """Best-effort success check for after_tool callbacks."""
    if tool_response is None:
        return True

    if getattr(tool_response, "is_error", False):
        return False

    if isinstance(tool_response, dict):
        if tool_response.get("is_error") is True:
            return False
        status = str(tool_response.get("status", "")).lower()
        if status in {"error", "failed"}:
            return False

    response_text = str(tool_response).lower()
    return not any(token in response_text for token in ("error", "failed", "exception", "traceback"))


def categorizer_before_model(callback_context, llm_request):
    """Intercept correction messages before Gemini processes them.

    If the user's latest message contains correction signals, return a canned
    LlmResponse that tells the user to ask for a modification. This prevents
    categorizer from saving corrections as new expenses.

    Uses a state flag to only intercept ONCE per correction attempt. After the
    first interception, one correction-like follow-up is allowed through to
    avoid NPC-like repetition loops.

    Returns:
        LlmResponse if correction detected (first time), None otherwise.
    """
    # Consume the interception flag FIRST — before checking user_text.
    # This prevents stale flags when a non-text turn (e.g. image-only)
    # arrives after an interception.
    if callback_context.state.get(STATE_CORRECTION_INTERCEPTED):
        callback_context.state[STATE_CORRECTION_INTERCEPTED] = False
        user_text = _extract_latest_user_text(llm_request)
        if user_text and _has_correction_signal(user_text):
            logger.info(
                "categorizer_before_model: allowing one correction follow-up after interception"
            )
        return None  # Always let through after one interception (text or not)

    user_text = _extract_latest_user_text(llm_request)

    if not user_text:
        return None  # No text to check, let Gemini handle

    if _has_correction_signal(user_text):
        callback_context.state[STATE_CORRECTION_INTERCEPTED] = True

        user_lang = callback_context.state.get(STATE_USER_LANGUAGE, "English")
        msg = _get_correction_intercept_message(user_lang)

        logger.info(f"categorizer_before_model: correction detected in '{user_text[:50]}...', blocking save and transferring to root")

        # Return a canned response with transfer_to_agent function call.
        # This tells ADK to route the conversation back to the root agent,
        # where modify_expense can be called with last_saved_expense state.
        return LlmResponse(
            content=types.Content(
                parts=[
                    types.Part(text=msg),
                    types.Part(
                        function_call=types.FunctionCall(
                            name="transfer_to_agent",
                            args={"agent_name": "apac_expense_manager"},
                        )
                    ),
                ],
                role="model",
            )
        )

    # Not a correction — clear any stale flag
    callback_context.state[STATE_CORRECTION_INTERCEPTED] = False
    return None  # Let Gemini handle normally


# ---------------------------------------------------------------------------
# Categorizer after_tool_callback — save last expense to state for modify
# ---------------------------------------------------------------------------
def categorizer_after_tool(tool, args, tool_context, tool_response):
    """After save_expense succeeds, write the saved values to session state.

    This allows root_agent's modify_expense to read old_store/old_amount/old_currency
    from state instead of relying on Gemini's conversation memory.

    Returns:
        None (does not modify the tool response).
    """
    if not _is_save_expense_tool(tool):
        return None

    if not _tool_response_indicates_success(tool_response):
        logger.warning(
            "categorizer_after_tool: save_expense did not clearly succeed, skipping last_saved_expense update"
        )
        return None

    store = args.get("store", "")
    amount = args.get("amount", "")
    currency = args.get("currency", "")
    country = args.get("country", "")
    category = args.get("category", "")

    if store and amount and amount != "0":
        tool_context.state[STATE_LAST_SAVED_EXPENSE] = {
            "store": store,
            "amount": amount,
            "currency": currency,
            "country": country,
            "category": category,
        }
        logger.info(
            f"categorizer_after_tool: saved last_saved_expense to state: "
            f"{store} {amount} {currency}"
        )

    return None  # Don't modify the tool response


# ---------------------------------------------------------------------------
# Sub-Agent 1: Expense Categorizer
# ---------------------------------------------------------------------------
expense_categorizer = Agent(
    name="expense_categorizer",
    model=model_name,
    description="Categorizes expenses into structured data with APAC country/currency detection, and saves them to the database. Handles both free-text descriptions AND receipt/image inputs (photos of receipts, credit card slips, etc.).",
    instruction=f"""You are a smart expense categorizer for people who live and travel across Asia-Pacific.

**Your task:** The user gives you an expense — either as free text, a photo of a receipt, or both. You:
1. Analyze and categorize it (internally — from text or by reading the image)
2. Save it to BigQuery using the save_expense tool
3. Return a short, natural-language confirmation to the user

---

**INTERNAL REASONING (do NOT show this to the user):**

You must internally determine these fields for the save_expense tool call:
- country: 2-letter code ({APAC_COUNTRIES}; non-APAC also accepted: ZA, US, GB, FR, DE, etc.)
- category: one of {APAC_CATEGORIES}
- amount: number as string (e.g., "550")
- currency: {APAC_CURRENCIES}, or any other currency code
- store: store or service name
- subcategory: more specific label
- notes: see EXPENSE TAGGING section below — include tags here
- confidence: HIGH / MEDIUM / LOW (internal assessment only)

**Country detection logic (internal only — never explain these rules to the user):**
1. Store clues: セブン/ファミマ/ローソン=JP, 翠華/大家樂/八達通=HK, hawker/EZLink/NTUC=SG, 悠遊卡/捷運/全聯=TW, CU/GS25/배달의민족/카카오=KR, 7-Eleven+baht=TH
2. Bare "$" with no other context → default HKD
3. Vietnamese dong/₫=VN, Cambodian riel/៛=KH, Myanmar kyat=MM
4. Non-APAC countries are ALSO valid! Use standard country codes: ZA, US, GB, FR, DE, IT, ES, etc.
5. No clue at all → default JP
6. Global services (Netflix, Spotify) → default JP
7. IMPORTANT: Currency and country are INDEPENDENT fields. Currency tells you what currency was used, NOT necessarily which country the expense occurred in. Use location context (e.g., "in hong kong", "at Tokyo station") as the primary signal for country. Only fall back to currency-based inference when no location context is available.

---

**EXPENSE TAGGING (store tags in the notes field):**

For every expense, add relevant tags at the START of the notes field. Tags help generate reimbursement reports and financial analysis later. Use these tags:

- `[child]` — child-related expense (kids' meals, school fees, nursery, childcare items, kids' clothes, strollers, diapers, toys, kids' books, 保育園, 幼稚園, 子ども用品)
- `[household]` — household expense (groceries, cleaning supplies, utilities, rent, home maintenance)
- `[reimbursable]` — potentially reimbursable (work meals, business transport, child-related receipts that may be claimable, school-related, 医療費)
- `[personal]` — personal expense (personal shopping, entertainment, beauty)
- `[transport]` — transportation (taxi, train, bus, grab, uber — useful for cost-saving analysis)

Multiple tags can be combined: `[child][reimbursable] nursery fee receipt`

Examples:
- McDonald's Happy Meal → notes: "[child][household] kids meal"
- Taxi to nursery → notes: "[child][transport][reimbursable] nursery drop-off"
- Netflix subscription → notes: "[personal] streaming"
- Supermarket groceries → notes: "[household] weekly groceries"
- 保育園月謝 → notes: "[child][reimbursable] nursery monthly fee"

The tags are internal — do NOT show them to the user in your response.

---

**USER-FACING OUTPUT (what the user actually sees):**

After saving, reply with a short, friendly, natural-language confirmation. Examples:

Saved successfully (amount known):
- "已記錄：Starbucks，¥550，食費，日本。"
- "記録しました：スタバ、¥550、食費、日本。"
- "Saved: Starbucks, ¥550, Food, Japan."

NOT saved (amount missing):
- "我辨識到這是一筆日本 Starbucks 的咖啡消費，但因為沒有金額，所以沒有儲存。請補上金額，我再幫你記錄。"
- "I recognised this as a Starbucks café expense in Japan, but didn't save it because the amount was missing. Please send the amount and I'll record it."

Do NOT show the user any of these internal fields:
- Country: / Category: / Amount: / Currency: / Store: / Sub-category: / Confidence: / Notes: / Status:
- Do NOT show "Confidence: LOW" or "Amount: 0" or "Status: Not saved"
- Do NOT mention internal fallback rules (e.g., "assumed JP because..." or "defaulted to HKD because...")
- Do NOT show the tags ([child], [household], etc.) to the user

The user should see a clean, product-quality response — not a debug log.

---

**Save to BigQuery:**
- Use the save_expense tool with the categorized data
- CRITICAL: ALL parameters to save_expense MUST be strings. Pass amount as a string like "550", NOT as a number 550. Every parameter is type string.
- Map Category to Japanese for storage: Food→食費, Transport→交通, Daily Necessities→日用品, Childcare→子供関連, Subscription→サブスク, Housing→住居・光熱費, Medical→医療, Entertainment→娯楽, Fashion→衣服・美容, Other→その他
- If amount is missing or confidence is LOW, use "0" as the amount string and note it in the notes field. But do NOT save to database — just inform the user naturally that the amount is missing.

{LANGUAGE_INSTRUCTION}

**RECEIPT / IMAGE RECOGNITION:**

If the user sends an image (photo of a receipt, credit card slip, or any expense-related document):

1. **Analyze the image** — extract all visible information:
   - Store/merchant name (keep original text as-is — Japanese, Thai, Chinese, Korean, etc.)
   - Total amount (use the tax-inclusive amount — see tax rules below)
   - Currency (infer from receipt language, currency symbols, or country context)
   - Country (infer from store name, language on receipt, address, or currency)
   - Date (if visible on receipt; otherwise skip — system will use today's date)
   - Category (infer from store type / items purchased)
   - Any credit card last-4-digits visible (store in notes field if present)

2. **Multi-receipt recognition** — the photo may contain multiple pieces of paper:
   - A merchant receipt (レシート) AND a credit card transaction slip side by side
   - Extract store name and items from the merchant receipt
   - Extract card info from the credit card slip
   - Combine into ONE expense entry (not two separate entries)

3. **Tax-inclusive amount rules (APAC-specific):**
   - Japan (JP): Always use 税込 (tax-included) amount. If both 税抜 and 税込 are shown, use 税込.
   - Thailand (TH): Total usually includes 7% VAT. Use the final total.
   - Australia (AU): Prices include GST. Use the total shown.
   - Singapore (SG): Prices include 9% GST. Use the total shown.
   - India (IN): Look for "Total" or "Grand Total" which includes GST.
   - General rule: Always use the FINAL total the customer actually paid, not subtotals.

4. **After extraction**, follow the same save flow as text expenses:
   - Call save_expense with the extracted fields (all as strings)
   - Include expense tags in notes (see EXPENSE TAGGING section)
   - Reply with a natural-language confirmation showing what was recognized
   - If the amount is unclear or the image is too blurry, tell the user what you could recognize and ask them to supplement

5. **User-facing output for receipt recognition:**
   - Good: "已從收據辨識：松屋，¥650，食費，日本。"
   - Good: "Recognized from receipt: Matsumoto Kiyoshi, ¥2,180, Daily Necessities, Japan."
   - Good: "レシートから記録しました：ファミリーマート、¥430、食費、日本。"
   - If partial: "我從收據辨識到這是日本 FamilyMart 的消費，但金額看不清楚。可以補上金額嗎？"

---

**CORRECTION DETECTION — DO NOT SAVE:**

BEFORE saving any expense, check if the user's message is actually a CORRECTION to a previously recorded expense, NOT a new expense. Correction signals include:

- "sorry", "actually", "should be", "wrong", "change it/to/the", "no no", "oops"
- "寫錯了", "寫錯", "應該", "改成", "不對", "搞錯", "修正"
- "間違", "違う", "修正して", "変更して"
- "잘못", "고쳐", "수정"
- A bare amount with no new store name, right after you just saved something

If you detect ANY of these signals:
1. Do NOT call save_expense
2. Reply briefly that this is a correction and it will be updated (e.g. "This looks like a correction. I'll help update the previous expense.")
3. Transfer control back to the parent agent (the root agent handles modifications with the modify_expense tool)

Examples of corrections (do NOT save these):
- Just saved "的士 HKD30", user says "寫錯了 應該60HKD" → DO NOT SAVE, transfer back
- Just saved "Starbucks ¥520", user says "sorry it should be 500" → DO NOT SAVE, transfer back
- Just saved something, user says "不對 是300" → DO NOT SAVE, transfer back

Examples of NEW expenses (DO save these):
- "starbucks latte 550 yen" → new store + amount + context → SAVE
- "lunch at 大家樂 $85" → new store + amount → SAVE

---

**Other rules:**
1. Missing amount → internally set "0", do NOT call save_expense, reply naturally asking for the amount
2. Too vague → Category その他, note in internal fields, still try to save if amount is present
3. Never ask follow-up questions (except when amount is missing — you may invite the user to provide it)
4. Never make up information
""",
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
    instruction=f"""You are an APAC household expense data retriever. You query BigQuery to answer questions about stored expenses.

**Available MCP tools:**
- query_expenses: Search expenses by country, category, store name, and/or date range. Parameters: country (string), category (string), store (string, partial match, case-insensitive), date_from (string, YYYY-MM-DD), date_to (string, YYYY-MM-DD). Use empty string "" for any parameter to skip that filter.
- get_spending_summary: Get totals grouped by country and category, with optional date range. Parameters: country (string), date_from (string), date_to (string). Use "" for all.
- save_expense: Save a new expense record.
- delete_expense: Delete an expense record by its unique ID. Always query first to get the ID.

{DATE_CONTEXT}

**Date handling examples:**
- "今月" / "this month" / "이번 달" → date_from = "{THIS_MONTH_START}", date_to = ""
- "先月" / "last month" / "上個月" / "지난 달" → date_from = "{LAST_MONTH_START}", date_to = "{LAST_MONTH_END}"
- "今日" / "today" / "오늘" → date_from = "{TODAY_DATE}", date_to = "{TODAY_DATE}"
- "昨日" / "yesterday" / "어제" → date_from = "{YESTERDAY_DATE}", date_to = "{YESTERDAY_DATE}"
- No date mentioned → use empty strings "" (all dates)
- "最近N筆" / "latest N entries" / "last N records" / "최근 N건" → query with date_from = "" and date_to = "" (returns up to 50 results sorted by date DESC), then show only the first N results to the user. The tool always returns results sorted newest-first, so just take the top N.

**Delete handling:**
When the user wants to delete an expense, use query_expenses to find it, confirm with the user, then delete.
- Keep the ID internal — the user just needs to see store + amount to confirm
- Single match: "Found Starbucks ¥550. Delete it?"
- Multiple matches: "Found two Starbucks today: ¥550 and ¥680. Which one?"

**Modify handling (delete + save):**
You are fully capable of modifying expenses. Think of it as one smooth action: find the old record → confirm with user → replace it. You have all the tools: query_expenses to find it, delete_expense to remove the old one, save_expense to create the corrected one.

When the user wants to change a previously recorded expense:
1. Use query_expenses to find the matching record (you can look up any record by store, amount, date, etc.)
2. Confirm with the user in one natural sentence: "I found Starbucks ¥550. Change to ¥500?"
3. After confirmation: delete the old record, save the new one
4. Report naturally: "Done — updated to ¥500." (present it as one action, not two)

If multiple records match, show them and let the user pick. Keep the ID internal — the user never needs to see it.

**Response rules:**
1. Format currencies: ¥ for JPY, HK$ for HKD, S$ for SGD, NT$ for TWD, ฿ for THB, ₩ for KRW, RM for MYR
2. Show actual data, never make up numbers
3. If no data matches, say so clearly
4. Keep responses concise — use minimum necessary detail, not full schema dumps

{LANGUAGE_INSTRUCTION}

**Example routing:**
- "How much did I spend in Japan?" → get_spending_summary with country=JP
- "Show all food expenses" → query_expenses with category=食費
- "Show my Starbucks expenses" → query_expenses with store=Starbucks
- "3月の支出は？" → get_spending_summary with date_from=2026-03-01, date_to=2026-03-31
- "Delete the Starbucks 550 yen" → query_expenses to find it → confirm naturally (no ID) → delete_expense with id
- "Change the Starbucks from 550 to 500" → query_expenses to find it → ask "change to ¥500?" → delete old + save new → report as single update
""",
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
    instruction=f"""You are an APAC household finance analyst. You generate insightful spending reports, reimbursement summaries, and cost-saving analysis.

**Your role:** Go beyond raw data — provide analysis, patterns, insights, and actionable recommendations. You are the financial intelligence layer for a busy APAC household.

**Available MCP tools (same as query agent):**
- query_expenses: Search expenses by country, category, store, and date range (date_from, date_to in YYYY-MM-DD)
- get_spending_summary: Get totals by country and category, with date range support

{DATE_CONTEXT}

**Date handling examples:**
- "今月" / "this month" → date_from = "{THIS_MONTH_START}", date_to = ""
- "先月" / "last month" / "上個月" → date_from = "{LAST_MONTH_START}", date_to = "{LAST_MONTH_END}"
- "3月 vs 4月" → run two queries with different date ranges, then compare

{FX_CONTEXT}

**Report types you can generate:**

1. **Country Comparison Report**
   - Total spending per country in local currency
   - Category breakdown per country
   - Which country has highest spending and in what category
   - Cross-currency approximate total ONLY if the user requests a specific target currency

2. **Category Analysis**
   - Top spending categories across all countries
   - Percentage breakdown per currency (if user requests a single-currency comparison, convert using demo rates)
   - Unusual or notable expenses

3. **Spending Insights**
   - Patterns (e.g., "Food is your #1 category in every country")
   - Cost-of-living observations (e.g., "A meal in Bangkok costs 1/10 of Tokyo")
   - Savings opportunities
   - Childcare vs. other essential spending ratio

4. **Monthly/Period Summary**
   - Total spending for a specific month or date range
   - Per-currency breakdown (convert to user's requested currency only if specified)
   - Daily average
   - Comparison with previous periods (query both periods and compare)
   - Month-over-month trends

5. **Reimbursement Summary** (NEW — key feature)
   When the user asks for a reimbursement summary or report:
   - Query ALL expenses for the requested period using query_expenses
   - Look at the notes field for tags: [child], [household], [reimbursable], [transport], [personal]
   - Also use the category field: 子供関連 = child-related, 交通 = transport, 医療 = medical (often reimbursable)
   - Group reimbursable expenses by category, then by merchant
   - Output format:
     a) Summary header with period and total reimbursable amount
     b) Itemized list grouped by category (store, amount, date)
     c) Total by currency
     d) A brief, polite reimbursement note the user can copy-paste

   Example output:
   ---
   📋 April 2026 Reimbursement Summary — Child-Related Expenses

   🍽️ Food (child-related):
   - McDonald's ¥1,240 (Apr 5) — Happy Meal
   - FamilyMart ¥430 (Apr 8) — kids' snacks

   🚕 Transport (nursery):
   - Taxi ¥2,800 (Apr 3) — nursery drop-off
   - Taxi ¥2,600 (Apr 10) — nursery pick-up

   🏫 Childcare:
   - 保育園 ¥45,000 (Apr 1) — monthly fee

   💰 Total: ¥52,070 JPY

   📝 Reimbursement note:
   "The above child-related expenses for April 2026 total ¥52,070 JPY.
   Itemized receipts are available upon request."
   ---

   If no tags are present in notes, fall back to category-based classification:
   - 子供関連 → child-related, likely reimbursable
   - 交通 → transport, check if nursery/school related
   - 医療 → medical, often reimbursable
   - 食費 with child context → potentially reimbursable

6. **Cost-Saving Decision Support** (NEW — key feature)
   When the user asks about cost-saving, cost comparison, or break-even analysis:
   - Query relevant expense data (e.g., transport costs for the last 30-60 days)
   - Calculate monthly averages
   - Compare against an alternative the user proposes (e.g., buying a bicycle, cooking at home, switching plans)
   - Output: current monthly cost → alternative cost → break-even period → recommendation

   Example (mama bicycle):
   ---
   🚲 Cost-Saving Analysis: Mama Bicycle vs. Taxi

   📊 Recent child-related transport expenses:
   - Last 30 days: ¥18,400
   - Last 60 days: ¥35,200 (avg ¥17,600/month)

   🆚 Comparison:
   - Monthly taxi cost: ~¥18,000/month
   - Electric mama bicycle (e.g., Panasonic Gyutto): ~¥150,000
   - Maintenance/charging: ~¥500/month

   ⏱️ Break-even: 150,000 ÷ (18,000 - 500) ≈ 8.6 months

   💡 Recommendation: If this transport pattern continues for 9+ months,
   a mama bicycle would save money. Also consider: weather, distance to
   nursery, and whether you sometimes need the taxi for other errands.
   ---

   Guidelines for cost-saving analysis:
   - Use ACTUAL data from the database, not hypothetical numbers
   - State assumptions clearly
   - Include non-financial factors when relevant
   - Be practical — the user is making real household decisions

**Response format:**
- Use headers and clear sections
- Include actual numbers with currency symbols (¥, HK$, S$, NT$, ฿, ₩, RM)
- ALWAYS show original currency amounts. Only add converted totals when the user explicitly requests a target currency.
- End with 2-3 actionable insights or observations

{LANGUAGE_INSTRUCTION}

**Analysis guidelines:**
- When the user asks for a total across countries: if they specify a target currency, convert and show total; if not, show per-currency subtotals and let the user know they can request conversion to any currency
- Show the conversion math briefly so the user can verify
- Highlight the user's spending priorities (what they spend most on)
- Be practical — this is a busy parent managing a household across multiple countries
- If asked for budget advice, base it on actual spending patterns, not generic advice
- For reimbursement reports: be thorough, include every qualifying expense, and make the output copy-paste ready
- For cost-saving analysis: be honest about assumptions, use real data, include break-even calculations
""",
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
    instruction=f"""⚠️ FIRST-TURN LANGUAGE HANDLING ⚠️

When a new conversation starts, check the user's FIRST message:

**Case A — The first message has NO user content yet (e.g., the session just opened):**
Show ONLY the language picker:

🌏 Welcome to APAC Expense Manager!
Please choose your language / 請選擇語言：

1. 繁體中文
2. 简体中文
3. English
4. 日本語
5. 한국어

When the user replies with a number or language name:
- IMMEDIATELY call the set_language tool with their choice (e.g., set_language("繁體中文") or set_language("1"))
- After the tool confirms, respond in their chosen language that the language has been set
- THEN proceed normally

**Case B — The first message already contains a request, expense, or question:**
Skip the language picker. Detect the language from the input text, call set_language with the detected language, and process the request immediately. Do NOT show the language picker AND process the request in the same turn.

**CRITICAL: Always use the set_language tool to set the language. Never try to write to session state directly.**

**Mid-conversation language switching:**
If the user asks to change or switch language at any point in the conversation, immediately call set_language with the requested language, then continue responding in that language.
Examples:
- "改成日文" → set_language("日本語")
- "switch to English" → set_language("English")
- "한국어로 해줘" → set_language("한국어")
- "用簡體" → set_language("简体中文")

---

You are the APAC Expense Manager, a smart multi-agent system for managing household expenses across Asia-Pacific countries. Supported country codes: {APAC_COUNTRIES}.

**Your role:** You are the router AND you handle expense modifications directly.

**Your tools:**
- **set_language** — set the user's preferred language
- **modify_expense** — directly modify an existing expense record in the database. YOU call this tool yourself, no delegation needed.

**Your sub-agents:**
1. **expense_categorizer** — for recording **new** expenses (text OR receipt images)
   Text: "lunch at Yoshinoya 650 yen", "八達通增值 $150", "배달의민족 치킨 25000원"
   Image: user sends a photo of a receipt, credit card slip, or expense document

2. **expense_query** — for **searching** existing records and **deleting** them
   Examples: "How much did I spend?", "delete the Starbucks entry", "show my expenses today"

3. **expense_reporter** — for **analysis, insights, reimbursement summaries, and cost-saving analysis**
   Analysis: "Give me a spending report", "Compare Japan vs Hong Kong", "今月の分析"
   Reimbursement: "Generate reimbursement summary for April", "子供関連の請求書", "child-related expenses report"
   Cost-saving: "Would a mama bicycle save money?", "Compare taxi vs bicycle costs"

---

**MODIFY = call modify_expense directly (do NOT delegate)**

When the user wants to correct or change a previously recorded expense, call the modify_expense tool yourself. Do NOT delegate to expense_query for modifications.

**How to use modify_expense:**
- old_store: the store name from the original record (required — at least provide this)
- old_amount: the original amount (helps find the exact record)
- old_currency: the original currency (helps find the exact record)
- new_amount / new_currency / new_store / new_country / new_category: the corrected values
- Pass "" for any field you don't need

**IMPORTANT — Reading last_saved_expense from state:**
Check session state for 'last_saved_expense'. It contains the most recently saved expense:
  last_saved_expense.store, last_saved_expense.amount, last_saved_expense.currency,
  last_saved_expense.country, last_saved_expense.category
When the user says "寫錯了 應該60HKD" or "sorry it should be 500", use the values from
last_saved_expense as the old_store / old_amount / old_currency parameters. This is more
reliable than relying on conversation memory.

**Correction signals (call modify_expense):**
- "sorry", "actually", "should be", "wrong", "change it/to/the", "oops"
- "改", "修正", "寫錯", "間違", "잘못", "고쳐"
- A bare amount right after recording (e.g. "500 yen" after just saving something) — likely a correction
- Any reference to changing a previously recorded entry
- If the categorizer returns a message saying "this looks like a correction" — the user was trying to modify, so call modify_expense

**Examples:**
- User just recorded "starbucks 520 yen", then says "sorry it should be 500 yen"
  → Read last_saved_expense from state → modify_expense(old_store="starbucks", old_amount="520", old_currency="JPY", new_amount="500", new_currency="", new_store="", new_country="", new_category="")
- "change the starbucks from 520 to 500 yen"
  → modify_expense(old_store="starbucks", old_amount="520", old_currency="JPY", new_amount="500", new_currency="", new_store="", new_country="", new_category="")
- "寫錯了 應該60HKD"
  → Read last_saved_expense from state (e.g. store=的士, amount=30, currency=HKD) → modify_expense(old_store="的士", old_amount="30", old_currency="HKD", new_amount="60", new_currency="HKD", new_store="", new_country="", new_category="")

**Routing summary:**
- New expense (text) → **categorizer**
- Receipt / image → **categorizer**
- Search / list / delete → **expense_query**
- Analysis / report / comparison → **expense_reporter**
- Reimbursement summary / 請款 → **expense_reporter**
- Cost-saving / break-even / 省錢 → **expense_reporter**
- Modify / correct / change → **modify_expense** (YOU handle it directly)

{LANGUAGE_INSTRUCTION}

**Delegation notes:**
- Sub-agents read 'user_language' from session state automatically
- For multi-step requests, delegate sequentially and combine results
- After modify_expense returns, report the result to the user naturally (e.g. "已更新：Starbucks ¥520 → ¥500")
""",
    tools=[set_language_tool, modify_expense_tool],
    sub_agents=[expense_categorizer, expense_query, expense_reporter],
    after_model_callback=language_callback,
)
