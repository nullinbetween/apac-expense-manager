# APAC Expense Manager

> An AI-powered multilingual, multi-currency household expense manager for APAC families living across borders.

Built for the **Google Cloud x Hack2skill Gen AI Academy APAC Edition** hackathon.

## The Problem

APAC families living across borders face a unique challenge: expenses span multiple countries, currencies, and languages — all within the same household. Existing expense trackers assume a single-country, single-currency lifestyle. For a working parent who commutes between Tokyo and Hong Kong, sends their child to school in Japan, and visits family in Taiwan, no simple tool exists to manage it all in one place.

## The Solution

APAC Expense Manager is a conversational expense tracker that understands the multilingual, multi-currency reality of APAC families.

**Talk to it in Traditional Chinese, Simplified Chinese, English, Japanese, or Korean. It handles the rest.**

### Key Capabilities

- **Natural language expense tracking**: Say "スタバ ラテ 550円" — the agent auto-detects Japan, categorizes under Food, and saves to BigQuery. Responses are natural language confirmations, not database output.
- **5-language support**: Traditional Chinese, Simplified Chinese, English, Japanese, Korean — with OpenCC for accurate Traditional/Simplified Chinese conversion. Language detected automatically from input if user skips onboarding.
- **Multi-currency intelligence**: Records in local currency. Cross-country summaries show original currency breakdown per country. Users can request conversion to any supported currency for approximate totals. 14 APAC currencies + global travel support.
- **Conversation context**: The agent maintains context across turns. Say "Starbucks latte" (missing amount) → agent recognizes but doesn't save → reply just "100yen" → agent recalls the prior context and saves with full details. No need to repeat store, country, or category.
- **Context-aware financial guidance**: Ask "if I cancel Netflix, how much do I save per year?" — the agent finds the monthly charge in BigQuery, calculates ¥23,760/year, and gives actionable savings suggestions based on real spending data.
- **Emergent multi-tool reasoning**: No "modify" tool exists. When asked to change an expense, the agent reasons through: query → confirm with user → delete old → save new → report as single update. This behavior emerged from Gemini's reasoning, not hardcoded logic.

## Architecture

```
User (ADK Web UI)
  ↓
Primary Agent (apac_expense_manager) — Gemini 2.5 Flash on Vertex AI
  ├── expense_categorizer  → Categorize + save to BigQuery via MCP Toolbox
  ├── expense_query        → Query / delete / modify records (date, category, country, store)
  └── expense_reporter     → Spending analysis with cross-currency conversion + financial guidance
```

### Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Framework | Google ADK (Agent Development Kit) | Multi-agent orchestration with sub-100ms tool routing |
| Model | Gemini 2.5 Flash (Vertex AI) | Best cost-performance ratio for multilingual + reasoning tasks |
| Database | BigQuery (via MCP Toolbox for Databases) | Scales from demo to production; SQL tools defined in YAML, no code changes needed |
| Deployment | Cloud Run (europe-west1) | Serverless, zero cold-start config, auto-scaling |
| Config | Secret Manager (tools.yaml, 11 versions) | Safe rollback capability for SQL tool definitions |
| Language Processing | OpenCC | Accurate Traditional ↔ Simplified Chinese conversion in `after_model_callback` |

## Project Structure

```
apac_expense_manager/
├── apac_expense_manager/
│   ├── __init__.py
│   └── agent.py          # Main agent code (v7.6)
├── tools.yaml             # MCP Toolbox SQL configuration (4 tools: save, query, delete, summary)
├── Dockerfile
├── pyproject.toml
├── seed_demo_data.sh      # Demo data loader (66 records, 6 countries)
├── seed_demo_data.sql      # Same data in SQL format
├── DEPLOY_GUIDE.md        # Step-by-step deployment instructions
└── README.md
```

## Deployment

### Prerequisites

- Google Cloud project with billing enabled
- BigQuery dataset (`expense_data.expenses`)
- Secret Manager secret for `tools.yaml`
- Cloud Run services for both the agent and MCP Toolbox

### Quick Deploy

```bash
# Set environment
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export TOOLBOX_URL="https://your-toolbox-url.run.app"

# Deploy MCP Toolbox (BigQuery connector)
gcloud run deploy expense-toolbox \
  --image us-central1-docker.pkg.dev/database-toolbox/toolbox/toolbox:latest \
  --region=$REGION --project=$PROJECT_ID \
  --set-env-vars="TOOLBOX_CONFIG_SOURCE=secret:expense-tools" \
  --allow-unauthenticated --port=8080

# Deploy Agent
cd apac_expense_manager
gcloud run deploy apac-expense-manager --source . \
  --project=$PROJECT_ID --region=$REGION \
  --port=8000 --set-env-vars="TOOLBOX_URL=$TOOLBOX_URL" \
  --allow-unauthenticated
```

See [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) for detailed instructions.

## Demo Data

Load 66 realistic expense records across 6 APAC countries (JP, HK, TW, SG, KR, TH) with a "Tokyo-based working single mom" persona:

```bash
bash seed_demo_data.sh
```

## Engineering Highlights

- **Emergent modify = query + confirm + delete + save**: The agent decomposes "change Starbucks from ¥550 to ¥500" into four steps using only three base tools — demonstrating LLM reasoning beyond CRUD
- **Product-grade UX from prompt engineering alone**: User-facing output uses natural language ("已記錄 Starbucks ¥550（日本／食費）"), internal reasoning (country detection, confidence scoring) stays hidden. No ID/UUID ever shown to users.
- **Conversation context across turns**: Missing information (e.g., amount) triggers a clarify-first flow; user's follow-up with just the amount is understood in context without re-stating store/country/category
- **UUID-based record identification**: Each expense has a unique `id` via `GENERATE_UUID()`. Enables reliable single-record operations instead of fragile composite key matching
- **Dynamic date injection**: Dates injected at container startup to prevent Gemini from generating Python code when asked about "today" or "this month"
- **Graceful language handling**: Language selection shown at session start; if skipped, agent auto-detects from input — never blocks the user
- **APAC-first, globally aware**: Core support for 14 APAC countries/currencies, but accepts any ISO 3166 country code for travel scenarios

## Version History

| Version | Changes |
|---------|---------|
| v7.6 | Modify/delete UX polish — product voice, hide ID/UUID, minimum necessary info for confirmations |
| v7.5 | User-facing output rewrite — natural language confirmations, no debug fields, missing amount clarify-first |
| v7.4 | Currency display overhaul — breakdown-only by default, conversion only on request. First-turn language flow fix |
| v7.3 | UUID primary key for all records. Reliable single-record delete |
| v7.2 | Global country code support. Store name search (fuzzy match) |
| v7.0 | Cross-currency conversion with fixed APAC exchange rates |
| v6.0 | Language onboarding + OpenCC + date range query + delete function |

## Lessons Learned

- **LLM as reasoning engine, not CRUD wrapper**: The most impressive demo moment — emergent modify — was never coded. Good agent architecture gives the model room to reason.
- **Separate internal reasoning from user output**: Early versions showed debug fields (Country: JP, Category: Food, Confidence: HIGH) to users. Splitting prompt into "internal reasoning" vs "user-facing output" sections solved this cleanly.
- **Pin your infrastructure versions**: The `toolbox:latest` image silently changed its default port from 5000 to 8080, causing startup failures. Lesson learned the hard way.
- **Prompt engineering is product design**: Every v7.x release was a prompt-only change. No architecture changes, no new tools — just better instructions producing better user experience.

## License

This project was built for the Google Cloud x Hack2skill Gen AI Academy APAC Edition hackathon.
