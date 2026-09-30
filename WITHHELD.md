# What is withheld from this showcase — and why

This repository shows the **real architecture, agent wiring, tool contracts,
data schema and deployment setup** of APAC Expense Manager (v9.3, the
Refinement-phase build). The parts that carry the product's know-how are
withheld. The code is therefore intentionally **not runnable as-is**.

| Withheld | What it does (what / why — not how) |
|---|---|
| `prompts/root.md` | Primary agent: first-turn language onboarding, routing between sub-agents, when to call `modify_expense`, product voice for confirmations. |
| `prompts/categorizer.md` | Turns free text and receipt photos into structured records: APAC country / currency detection, category taxonomy, tagging for reimbursement, clarify-first flow when the amount is missing. |
| `prompts/query.md` | Retrieval and deletion flows, date-range handling, user-safe confirmations (no IDs shown). |
| `prompts/reporter.md` | Spending analysis, reimbursement summaries, what-if / break-even savings analysis. |
| `prompts/language_rules.md`, `prompts/currency_display_rules.md` | Shared language policy and multi-currency display policy. |
| `categorizer_before_model` | Python-layer correction guard — intercepts "sorry, it should be 500" before the model can save it as a new expense, then hands control back to the root agent. |
| Correction lexicon | Multilingual correction signals (EN / 繁 / 简 / 粵 / 日 / 한). |
| `categorizer_after_tool` | Last-saved-expense state + guaranteed confirmation after receipt turns. |
| `modify_expense` body | Deterministic, parameterized BigQuery update. |

**Included and complete:** agent graph and callbacks wiring, OpenCC script
enforcement (`language_callback`), `set_language` tool, MCP Toolbox setup,
`tools.yaml` SQL tool definitions, BigQuery schema, Dockerfile, demo seed data
(fictional).

Interested in the full implementation (collaboration, licensing, hiring)?
Contact the author via https://github.com/nullinbetween.
