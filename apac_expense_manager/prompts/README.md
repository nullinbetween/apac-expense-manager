# prompts/ — withheld

The production system prompts for the four agents (`root`, `categorizer`,
`query`, `reporter`) and the shared blocks (`language_rules`,
`currency_display_rules`) are not included in this public showcase.

`agent.py` loads them from this folder by name via `load_prompt()`. Without
them the agents will not start — this is intentional.

See [`../../WITHHELD.md`](../../WITHHELD.md) for what each prompt is responsible for.
