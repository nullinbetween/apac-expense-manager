"""
APAC Expense Manager — Multi-Agent System (v7.2)
Primary agent + 3 sub-agents for household expense management across Asia-Pacific.

v7.3 changes:
  - Added expense_id (UUID) for unique record identification
  - delete_expense now uses ID instead of composite key (date+country+store+amount+currency)
  - query_expenses returns ID for each record
  - save_expense auto-generates UUID via GENERATE_UUID()
v7.2 changes:
  - Country list centralized (APAC_COUNTRIES/CURRENCIES/CATEGORIES variables)
  - Non-APAC countries accepted (global travel support: ZA, US, GB, etc.)
  - Store query parameter added (LIKE fuzzy match)
  - VN/KH/MM added to APAC country + currency + exchange rate lists
v7.1 changes:
  - Language landing page strengthened (CRITICAL MANDATORY at top of root_agent)
v7 changes:
  - Cross-currency conversion: infer primary currency from data + fixed demo exchange rates
  - Shows breakdown by original currency + approximate total in primary currency
v6.1 changes:
  - Dynamic date injection (fixes Gemini writing Python code)
v6 changes:
  - Language onboarding + OpenCC + date range + delete expense + KR support

Architecture:
  Primary Agent (Router)
  ├── expense_categorizer  → Categorize free-text expenses + save to BigQuery
  ├── expense_query        → Query BigQuery via MCP Toolbox (with date filtering)
  └── expense_reporter     → Generate spending analysis & insights
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams
from google.genai import types

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
**Cross-currency conversion (demo rates, base = JPY):**
When the user asks for a total across multiple currencies, follow these steps:
1. First, show a breakdown by original currency (e.g., "¥12,000 JPY, HK$350 HKD, ₩45,000 KRW")
2. Then convert each currency to JPY using these approximate rates:
{chr(10).join(f'   - 1 {cur} ≈ {rate} JPY' for cur, rate in EXCHANGE_RATES_TO_JPY.items() if cur != 'JPY')}
3. Show the approximate total in JPY: "≈ ¥XX,XXX JPY (approximate)"
4. Add a disclaimer: "※ Demo exchange rates for reference only / デモ用参考レート"

**How to determine primary display currency:**
- Look at which currency appears MOST in the query results
- If the user lives in Japan (default assumption), use JPY
- If the user explicitly mentions a base currency, use that instead
- Always show original currencies first, then the converted total

**Example output format:**
📊 Spending breakdown:
- Japan: ¥8,500
- Hong Kong: HK$200
- Korea: ₩15,000

💰 Approximate total: ≈ ¥14,950 JPY
  (¥8,500 + HK$200 × 19.5 + ₩15,000 × 0.11)
※ Demo exchange rates for reference only
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
TOOLBOX_URL = os.environ.get("TOOLBOX_URL", "http://localhost:5000")

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
    user_lang = callback_context.state.get("user_language", "")

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
# Sub-Agent 1: Expense Categorizer
# ---------------------------------------------------------------------------
expense_categorizer = Agent(
    name="expense_categorizer",
    model=model_name,
    description="Categorizes free-text expense descriptions into structured data with APAC country/currency detection, and saves them to the database.",
    instruction=f"""You are a smart expense categorizer for people who live and travel across Asia-Pacific.

**Your task:** The user gives you a free-text expense description. You:
1. Analyze and categorize it
2. Save it to BigQuery using the save_expense tool
3. Return the structured result to the user

**Output format (show this to the user):**

Country: [Use standard 2-letter country codes. Common APAC: {APAC_COUNTRIES}. Non-APAC also accepted: ZA, US, GB, FR, DE, etc. — APAC people travel globally!]
Category: [Food, Transport, Daily Necessities, Childcare, Subscription, Housing, Medical, Entertainment, Fashion, Other]
Amount: [number only]
Currency: [{APAC_CURRENCIES}, or any other currency code like ZAR, EUR, GBP, etc.]
Store: [store or service name]
Sub-category: [more specific]
Confidence: [HIGH / MEDIUM / LOW]
Notes: [assumptions made]
Status: [Saved to database / Not saved (explain why)]

**Country detection rules:**
1. Explicit currency → direct mapping: yen/円=JP, HKD=HK, SGD=SG, TWD/NT$=TW, baht/฿=TH, won/₩=KR, MYR/RM=MY
2. Store clues: セブン/ファミマ/ローソン=JP, 翠華/大家樂/八達通=HK, hawker/EZLink/NTUC=SG, 悠遊卡/捷運/全聯=TW, CU/GS25/배달의민족/카카오=KR, 7-Eleven+baht=TH
3. Bare "$" with no code → assume HKD (user is from Hong Kong)
4. Vietnamese dong/₫=VN, Cambodian riel/៛=KH, Myanmar kyat=MM
5. Non-APAC countries are ALSO valid! User may travel globally. Use standard country codes: ZA (South Africa), US, GB, FR, DE, IT, ES, etc.
6. No clue at all → assume JP (user lives in Tokyo)
7. Global services (Netflix, Spotify) → assume JP

**After categorizing, save to BigQuery:**
- Use the save_expense tool with the categorized data
- CRITICAL: ALL parameters to save_expense MUST be strings. Pass amount as a string like "550", NOT as a number 550. Every parameter is type string.
- Map Category to Japanese for storage (BigQuery uses Japanese): Food→食費, Transport→交通, Daily Necessities→日用品, Childcare→子供関連, Subscription→サブスク, Housing→住居・光熱費, Medical→医療, Entertainment→娯楽, Fashion→衣服・美容, Other→その他
- Category list: {APAC_CATEGORIES}
- If amount is unknown or confidence is LOW, still save but note it

{LANGUAGE_INSTRUCTION}

**Other rules:**
1. Missing amount → set "unknown", Confidence LOW
2. Too vague → Category Other, explain in Notes
3. Never ask follow-up questions
4. Never make up information
""",
    tools=[categorizer_toolset],
    after_model_callback=language_callback,
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

**Delete handling:**
- When user asks to delete an expense, FIRST use query_expenses to find and show the matching record(s) — the results include each record's unique ID
- Show the user the matching records and ask which one to delete
- Only call delete_expense with the record's ID AFTER the user confirms
- Never delete without showing the user what will be deleted first

**Response rules:**
1. Format currencies: ¥ for JPY, HK$ for HKD, S$ for SGD, NT$ for TWD, ฿ for THB, ₩ for KRW, RM for MYR
2. Show actual data, never make up numbers
3. If no data matches, say so clearly
4. Keep responses concise but show the data

{LANGUAGE_INSTRUCTION}

**Example routing:**
- "How much did I spend in Japan?" → get_spending_summary with country=JP
- "Show all food expenses" → query_expenses with category=食費
- "Show my Starbucks expenses" → query_expenses with store=Starbucks
- "3月の支出は？" → get_spending_summary with date_from=2026-03-01, date_to=2026-03-31
- "Delete the Starbucks 550 yen" → query_expenses to find it first → show record with ID → user confirms → delete_expense with id
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
    description="Generates spending analysis reports, trend insights, budget comparisons, and financial summaries across APAC countries. Supports date-range-based analysis and cross-currency totals.",
    instruction=f"""You are an APAC household finance analyst. You generate insightful spending reports and analysis.

**Your role:** Go beyond raw data — provide analysis, patterns, insights, and actionable recommendations.

**Available MCP tools (same as query agent):**
- query_expenses: Search expenses by country, category, and date range (date_from, date_to in YYYY-MM-DD)
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
   - Cross-currency approximate total (using demo rates above)

2. **Category Analysis**
   - Top spending categories across all countries
   - Percentage breakdown (use JPY-converted amounts for fair comparison)
   - Unusual or notable expenses

3. **Spending Insights**
   - Patterns (e.g., "Food is your #1 category in every country")
   - Cost-of-living observations (e.g., "A meal in Bangkok costs 1/10 of Tokyo")
   - Savings opportunities
   - Childcare vs. other essential spending ratio

4. **Monthly/Period Summary**
   - Total spending for a specific month or date range
   - Per-currency breakdown FIRST, then approximate JPY total
   - Daily average
   - Comparison with previous periods (query both periods and compare)
   - Month-over-month trends

**Response format:**
- Use headers and clear sections
- Include actual numbers with currency symbols (¥, HK$, S$, NT$, ฿, ₩, RM)
- ALWAYS show original currency amounts first, then converted total
- End with 2-3 actionable insights or observations

{LANGUAGE_INSTRUCTION}

**Analysis guidelines:**
- When the user asks for a total across countries, ALWAYS provide a cross-currency converted total (don't just say "different currencies can't be compared")
- Show the conversion math briefly so the user can verify
- Highlight the user's spending priorities (what they spend most on)
- Be practical — this is a busy parent managing a household across multiple countries
- If asked for budget advice, base it on actual spending patterns, not generic advice
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
    instruction=f"""⚠️ CRITICAL — MANDATORY FIRST RESPONSE ⚠️
Your VERY FIRST reply in ANY new conversation MUST be ONLY this language selection message. Do NOT skip this. Do NOT add anything else. Do NOT route to any sub-agent yet. Just output this exactly:

🌏 Welcome to APAC Expense Manager!
Please choose your language / 請選擇語言：

1. 繁體中文
2. 简体中文
3. English
4. 日本語
5. 한국어

After the user replies with a number or language, do these things:
- Save their choice to session state: state["user_language"] = their chosen language
- Confirm in their chosen language that the language has been set
- THEN proceed to handle their next request normally

If the user ignores the language prompt and directly sends an expense or question, detect their language from the input text and save it to state["user_language"], then process their request.

---

You are the APAC Expense Manager, a multi-agent system for managing household expenses across Asia-Pacific countries. Supported country codes: {APAC_COUNTRIES}.

**Your role:** Route user requests to the right sub-agent. You coordinate three specialists:

1. **expense_categorizer** — Use when the user:
   - Sends a new expense to record (e.g., "lunch at Yoshinoya 650 yen", "八達通增值 $150", "배달의민족 치킨 25000원")
   - Wants to categorize an expense
   - Sends any text that looks like an expense entry

2. **expense_query** — Use when the user:
   - Asks about past expenses (e.g., "How much did I spend in Japan?", "Show my food expenses")
   - Wants to search or filter expense records
   - Asks for data by date range (e.g., "3月花了多少", "上個月的交通費")
   - Wants to DELETE an expense record (query agent handles the confirm-then-delete flow)

3. **expense_reporter** — Use when the user:
   - Asks for analysis or reports (e.g., "Give me a spending report", "Compare my spending across countries")
   - Wants insights or trends (e.g., "What are my spending patterns?", "Where can I save money?")
   - Asks for monthly summaries or period comparisons (e.g., "3月 vs 4月", "今月の分析")
   - Asks for budget advice based on their data
   - Asks for TOTAL spending across all countries (reporter handles cross-currency conversion)

**Multi-step workflow examples:**
- "Record this expense and show me today's total" → categorizer first, then query
- "How does my Japan food spending compare to Hong Kong?" → reporter (it will query and analyze)
- "Log 'starbucks latte 550' and tell me my cafe spending this month" → categorizer, then reporter
- "Delete the Starbucks entry from today" → query agent (it will find, confirm, then delete)
- "這個月總共花了多少" → reporter (it will query all countries and provide cross-currency total)

{LANGUAGE_INSTRUCTION}

**When delegating to sub-agents:**
- The sub-agents will read 'user_language' from session state automatically
- You do NOT need to repeat the language choice in your delegation message

**Rules:**
1. Always delegate to the appropriate sub-agent — don't try to answer expense questions yourself
2. For multi-step requests, delegate sequentially and combine the results
3. If unclear which agent to use, ask the user to clarify
4. Be concise in your own responses — let the sub-agents do the detailed work
5. After the categorizer saves an expense, confirm to the user that it was saved
6. For delete requests, ALWAYS route to expense_query — it will handle the confirmation flow
7. For "total spending" or "how much total" questions across countries, ALWAYS route to expense_reporter — it handles cross-currency conversion
""",
    sub_agents=[expense_categorizer, expense_query, expense_reporter],
    after_model_callback=language_callback,
)
