# APAC Expense Manager

A multilingual, multi-currency AI financial intake layer for cross-border APAC households — turning messy daily expenses, receipts, corrections, and reimbursements into structured BigQuery data.

Built for the **Google Cloud x Hack2skill Gen AI Academy APAC Edition** hackathon. Selected for **Top 100 Refinement Phase**.

## Why This Matters

APAC families living across borders deal with expenses in multiple countries, currencies, and languages — all within the same household. A working parent commuting between Tokyo and Hong Kong, sending their child to school in Japan, visiting family in Taiwan.

The real problem is not calculation — it is intake friction. Busy parents don't have time to open spreadsheets, normalize currencies, translate receipts, or fix mistakes manually.

APAC Expense Manager eliminates that friction. Talk to it. Send it a receipt photo. Correct a mistake in natural language. It handles the rest — in five languages, across 14+ APAC currencies, backed by BigQuery.

## Demo & Links

- **Live deployment**: [Cloud Run (europe-west1)](https://apac-expense-manager-175546014414.europe-west1.run.app)
- **Demo video**: *(link TBD)*
- **GitHub**: [nullinbetween/apac-expense-manager](https://github.com/nullinbetween/apac-expense-manager)

## Key Capabilities

- **Natural language expense tracking**: Say "スタバ ラテ 550円" — the agent auto-detects Japan, categorizes under Food, and saves to BigQuery. Works in Traditional Chinese, Simplified Chinese, English, Japanese, and Korean.
- **Receipt recognition**: Snap a photo of any APAC receipt — Thai, Japanese, Hong Kong — the agent extracts store, amount, currency, and country automatically. Supports multi-slip photos and tax-inclusive amount detection.
- **Intelligent correction**: Say "sorry it should be 500 yen" right after recording. The system detects the correction at the Python layer, intercepts before the model can misinterpret it, and routes to modify automatically. One step, no follow-up needed.
- **Multi-currency intelligence**: Records in local currency. Summaries show original currency breakdown per country. Conversion to any supported currency on request. 14 APAC currencies + global travel support.
- **Conversation context**: Say "Starbucks latte" (missing amount) → agent recognizes but doesn't save → reply "100yen" → agent recalls context and saves with full details. No need to repeat anything.
- **Context-aware financial guidance**: Ask "if I cancel ベネッセ, how much do I save per year?" — the agent finds the ¥6,500 monthly charge, calculates ¥78,000/year, and gives actionable advice based on real spending data.

## Architecture

```
User (ADK Web UI — text or receipt photo)
  ↓
Primary Agent (apac_expense_manager) — Gemini 2.5 Flash on Vertex AI
  ├── [FunctionTool] set_language        → Programmatic user_language guarantee
  ├── [FunctionTool] modify_expense      → Direct BigQuery UPDATE (parameterized)
  ├── expense_categorizer                → Categorize + save (with correction guard callbacks)
  │     ├── before_model_callback        → Python-layer correction interception + auto-transfer
  │     └── after_tool_callback          → State tracking + receipt confirmation hint
  ├── expense_query                      → Query / delete records via MCP Toolbox
  └── expense_reporter                   → Analysis + financial guidance
```

## Refinement Story: From Agentic Demo to Production-Ready System

During our first submission, the most impressive behavior was emergent: Gemini independently handled expense corrections by reasoning through query → confirm → delete → save. No "modify" tool existed — the agent figured it out.

During refinement, the same behavior regressed. The identical prompt that worked on April 8 stopped working on April 28. **Model behavior is not a constant.** This became our core engineering challenge.

Our response was a **hybrid architecture**:

- **Keep Gemini for what it's best at**: multilingual understanding, receipt image parsing, intelligent classification, cross-currency financial analysis
- **Harden with deterministic code where reliability is critical**: `modify_expense` FunctionTool for direct BigQuery UPDATE, `before_model_callback` for Python-layer correction interception, `after_tool_callback` for guaranteed state tracking

We also discovered that prompt-based guards don't survive model regression — four iterations of prompt engineering all failed. The correction guard only became reliable when we moved it to Python code where the model has no opportunity to override.

The result: a system that preserves agentic intelligence where it adds value, and guarantees correctness where it matters most.

## Engineering Highlights

- **Programmatic correction guard**: ADK's sub-agent stickiness means the categorizer stays active after saving. A user correction would be misinterpreted as a new expense. Our `before_model_callback` intercepts with regex detection, returns a canned response with `transfer_to_agent` embedded in the LlmResponse, and ADK routes back to root for `modify_expense`. Entire flow completes in one step.

- **Receipt silence fix**: Gemini sometimes generates empty text after multimodal tool calls (receipt image → save → silence). The `after_tool_callback` appends a confirmation hint to the tool response, ensuring a user-facing confirmation is always produced.

- **Emergent reasoning preserved**: The original emergent modify path (delete + save) is preserved alongside the deterministic `modify_expense` tool. Both paths coexist — demonstrating evolution from discovery to production hardening, not abandonment of agentic design.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Google ADK (Agent Development Kit) |
| Model | Gemini 2.5 Flash (Vertex AI) |
| Database | BigQuery (MCP Toolbox + direct client) |
| Deployment | Cloud Run (europe-west1) |
| Config | Secret Manager (tools.yaml) |
| Language Processing | OpenCC (Traditional ↔ Simplified Chinese) |

## Project Structure

```
apac_expense_manager/
├── apac_expense_manager/
│   ├── __init__.py
│   └── agent.py          # Main agent code (v9.3, ~1155 lines)
├── tools.yaml             # MCP Toolbox SQL config (4 tools)
├── Dockerfile
├── pyproject.toml
├── seed_demo_data.sh      # Demo data (66 records, 6 countries)
└── README.md
```

## Deployment

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export TOOLBOX_URL="https://your-toolbox-url.run.app"

# Deploy Agent
gcloud run deploy apac-expense-manager --source . \
  --project=$PROJECT_ID --region=$REGION \
  --port=8000 --set-env-vars="TOOLBOX_URL=$TOOLBOX_URL,PROJECT_ID=$PROJECT_ID" \
  --allow-unauthenticated
```

Requires: Google Cloud project with BigQuery dataset, Secret Manager for tools.yaml, Cloud Run for both agent and MCP Toolbox. See `DEPLOY_GUIDE.md` for full instructions.

## Lessons Learned

- **Model behavior is not a constant**: The same Gemini 2.5 Flash prompt that worked reliably on April 8 stopped working on April 28. Production agent design means designing for model regression, not just model capability.

- **Prompt-based guards don't survive regression**: Four iterations of prompt engineering failed. The solution was Python-layer interception where the model has no opportunity to override.

- **Hybrid > pure agentic for production**: The most reliable architecture preserves model autonomy for classification and analysis, and hardens with deterministic code for financial writes. Neither pure agentic nor pure deterministic is optimal.

- **ADK sub-agent stickiness is a design constraint**: After a sub-agent handles a message, it stays active for the next — bypassing root routing. Building reliable multi-agent systems on ADK requires callbacks, state flags, and explicit transfer mechanisms.

## Version History

| Version | Date | Milestone |
|---------|------|-----------|
| v9.3 | 2026-04-29 | Correction auto-transfer + receipt silence fix. Production ready. |
| v9.0 | 2026-04-28 | modify_expense FunctionTool — direct BigQuery UPDATE |
| v8.0 | 2026-04-23 | set_language FunctionTool — programmatic language guarantee |
| v7.6 | 2026-04-08 | First submission version. Emergent modify, product-grade UX |
| v6.0 | 2026-04-05 | Language onboarding + OpenCC + delete function |
| v1.0 | 2026-04-03 | Initial multi-agent system |

## License

This project was built for the Google Cloud x Hack2skill Gen AI Academy APAC Edition hackathon.
