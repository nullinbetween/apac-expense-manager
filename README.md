# APAC Expense Manager

A multilingual, multi-currency AI financial intake layer for cross-border APAC households — turning messy daily expenses, receipts, corrections, and reimbursements into structured BigQuery data.

Built for the **Google Cloud x Hack2skill Gen AI Academy APAC Edition** hackathon (Cohort 1). Selected for the **Top 100 Refinement Phase**.

> **About this repository — public showcase edition.**
> This is a source-visible showcase of the v9.3 build: architecture, agent wiring, tool contracts, schema and deployment setup are real. System prompts and the core correction / modify logic are withheld (see [`WITHHELD.md`](WITHHELD.md)), so it is not runnable as-is.
> **All rights reserved — not open source.** See [`LICENSE`](LICENSE).

## Why This Matters

APAC families living across borders deal with expenses in multiple countries, currencies, and languages — all within the same household. A working parent might commute between Tokyo and Hong Kong, send their child to school in Japan, and visit family in Taiwan — all while tracking expenses across languages and currencies.

The real problem is not calculation — it is intake friction. Busy parents don't have time to open spreadsheets, normalize currencies, translate receipts, or fix mistakes manually.

APAC Expense Manager eliminates that friction. Talk to it. Send it a receipt photo. Correct a mistake in natural language. It handles the rest — in five languages, across 14+ APAC currencies, backed by BigQuery.

## Links

- **Write-up**: [I built an AI expense manager for life across borders (Medium)](https://medium.com/@meredithcheungmmc/i-built-an-ai-expense-manager-for-life-across-borders-82a8a4e55c8c)
- **Hackathon deployment**: Cloud Run (europe-west1) — may be offline after the event

## Key Capabilities

- **Natural language expense tracking**: Say "スタバ ラテ 550円" — the agent auto-detects Japan, categorizes under Food, and saves to BigQuery. Works in Traditional Chinese, Simplified Chinese, English, Japanese, and Korean.
- **Receipt recognition**: Snap a photo of an APAC receipt — Thai, Japanese, Hong Kong — the agent extracts store, amount, currency, and country automatically.
- **Intelligent correction**: Say "sorry it should be 500 yen" right after recording. The correction is caught at the Python layer before the model can misinterpret it, and routed to modify automatically. One step, no follow-up needed.
- **Multi-currency intelligence**: Records in local currency. Summaries show original currency breakdown per country. Conversion on request. 14 APAC currencies + global travel support.
- **Conversation context**: Say "Starbucks latte" (missing amount) → the agent asks → reply "100yen" → it recalls the context and saves with full details.
- **Context-aware financial guidance**: Ask "if I cancel ベネッセ, how much do I save per year?" — the agent finds the monthly charge, calculates the annual saving, and gives advice based on real spending data.

## Architecture

```
User (ADK Web UI — text or receipt photo)
  ↓
Primary Agent (apac_expense_manager) — Gemini 2.5 Flash on Vertex AI
  ├── [FunctionTool] set_language        → Programmatic user_language guarantee
  ├── [FunctionTool] modify_expense      → Direct BigQuery UPDATE (parameterized)
  ├── expense_categorizer                → Categorize + save (with correction guard callbacks)
  │     ├── before_model_callback        → Python-layer correction interception + hand-off
  │     └── after_tool_callback          → State tracking + receipt confirmation
  ├── expense_query                      → Query / delete records via MCP Toolbox
  └── expense_reporter                   → Analysis + financial guidance
```

## Refinement Story: From Agentic Demo to Production-Ready System

In the first submission, the most impressive behavior was emergent: Gemini independently handled expense corrections by reasoning through query → confirm → delete → save. No "modify" tool existed — the agent figured it out.

During refinement, the same behavior regressed. The identical prompt that worked on April 8 stopped working on April 28. **Model behavior is not a constant.** This became the core engineering challenge.

The response was a **hybrid architecture**:

- **Keep Gemini for what it's best at**: multilingual understanding, receipt image parsing, classification, cross-currency financial analysis.
- **Harden with deterministic code where reliability is critical**: a `modify_expense` FunctionTool for direct BigQuery updates, a `before_model_callback` for Python-layer correction interception, and an `after_tool_callback` for guaranteed state tracking.

Prompt-based guards did not survive model regression — four iterations of prompt engineering failed. The correction guard only became reliable once it moved to Python, where the model has no opportunity to override it.

## Engineering Highlights

- **Programmatic correction guard**: ADK's sub-agent stickiness keeps the categorizer active after saving, so a correction could be saved as a new expense. A `before_model_callback` catches correction turns before the model runs and hands control back to the root agent, which owns `modify_expense`.
- **Receipt silence fix**: Gemini sometimes produced empty text after multimodal tool calls (receipt → save → silence). The `after_tool_callback` guarantees a user-facing confirmation.
- **Emergent reasoning preserved**: the original emergent modify path coexists with the deterministic tool — evolution from discovery to production hardening, not abandonment of agentic design.
- **Script-correct Chinese**: an `after_model_callback` with OpenCC enforces Traditional or Simplified Chinese according to the user's choice (see `language_callback` in `agent.py`).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Google ADK (Agent Development Kit) |
| Model | Gemini 2.5 Flash (Vertex AI) |
| Database | BigQuery (MCP Toolbox + direct client) |
| Deployment | Cloud Run (europe-west1) |
| Config | Secret Manager (tools.yaml) |
| Language Processing | OpenCC (Traditional ↔ Simplified Chinese) |

## Repository Structure

```
.
├── apac_expense_manager/
│   ├── __init__.py
│   ├── agent.py          # Agent graph, tools, callbacks (core bodies withheld)
│   └── prompts/          # System prompts — withheld (see README inside)
├── tools.yaml            # MCP Toolbox SQL tool definitions (save / query / delete / summary)
├── seed_demo_data.sh     # Fictional demo data (66 records, 6 countries)
├── seed_demo_data.sql
├── Dockerfile
├── pyproject.toml
├── WITHHELD.md           # What is not included, and why
└── LICENSE               # All rights reserved (source-visible)
```

## Lessons Learned

- **Model behavior is not a constant**: the same prompt that worked on April 8 stopped working on April 28. Production agent design means designing for model regression, not just model capability.
- **Prompt-based guards don't survive regression**: the fix was Python-layer interception where the model cannot override.
- **Hybrid > pure agentic for production**: keep model autonomy for classification and analysis; harden financial writes with deterministic code.
- **ADK sub-agent stickiness is a design constraint**: reliable multi-agent systems on ADK need callbacks, state, and explicit hand-off.

## Version History

| Version | Date | Milestone |
|---------|------|-----------|
| v9.3 | 2026-04-29 | Correction auto-transfer + receipt silence fix |
| v9.0 | 2026-04-28 | modify_expense FunctionTool — direct BigQuery UPDATE |
| v8.0 | 2026-04-23 | set_language FunctionTool — programmatic language guarantee |
| v7.6 | 2026-04-08 | First submission. Emergent modify, product-grade UX |
| v6.0 | 2026-04-05 | Language onboarding + OpenCC + delete function |
| v1.0 | 2026-04-03 | Initial multi-agent system |

## License

Copyright (c) 2026 Meredith Cheung (nullinbetween). **All rights reserved.**
This repository is source-visible for reference only and is **not open source**. No permission is granted to copy, modify, redistribute, deploy, or use it (including for AI training) without prior written permission. See [`LICENSE`](LICENSE).
