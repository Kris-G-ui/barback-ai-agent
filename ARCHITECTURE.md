## How BarBack works

```
 ┌──────────────┐  natural language   ┌────────────────────────┐
 │  Streamlit   │ ──────────────────▶ │       Agent loop       │
 │  GUI / CLI   │ ◀── results, trace ─│      (agent/core.py)   │
 └──────┬───────┘                     └───┬───────────────▲────┘
        │ approve / decline               │ system prompt │ JSON plan
        ▼                                 ▼ + history     │
 ┌──────────────┐                     ┌────────────────────────┐
 │ Confirmation │                     │  LLM client (llm.py)   │
 │    gate      │                     │ Claude / GPT / Gemini  │
 └──────┬───────┘                     │ retries, backoff, errs │
        │                             └────────────────────────┘
        ▼
 ┌────────────────────┐  valid?  ┌──────────────────────────────────────┐
 │ Validator          │ ───────▶ │ Executor / Toolbox (actions.py)      │
 │ JSON + schema +    │          │  • DB: SQLite inventory, read-only SQL│
 │ bounds + enums     │          │  • Files: CSV/MD exports, order drafts│
 │ (invalid → sent    │          │  • System: backup, reset              │
 │  back to LLM)      │          │  • API: Discord/Slack webhook         │
 └────────────────────┘          └──────────────┬───────────────────────┘
                                                │ every step
                                                ▼
                                     ┌───────────────────────┐
                                     │ Audit log (SQLite +   │
                                     │ JSONL)                │
                                     └───────────────────────┘
```

**One turn:**

1. You type something like *"poured 11 shots of Jameson and we got 2 kegs of Bud Light"*.
2. The agent builds a system prompt containing the action catalog, rules, few-shot
   examples and a **live inventory snapshot**, and sends it with the conversation.
3. The model replies with JSON: a short `thought`, a list of `actions`, and a `response`.
4. The **validator** checks the JSON, action names, required params, types, ranges and
   enums. If anything is wrong, nothing runs; the exact error goes back to the model,
   which fixes its own output (up to 2 retries).
5. Safe actions run immediately. **Destructive** (delete, reset) and **external**
   (webhook) actions pause until you click **Approve**.
6. Results go back to the model as an `OBSERVATION`; it can take more actions
   (ReAct loop, max 6 steps) or write the final reply.
7. Every input, LLM call, validation result, approval and action is written to the
   **audit log**.

**Prompt-engineering techniques:** role prompting · strict JSON output contract ·
tool catalog generated from code · dynamic context injection (live inventory, date) ·
few-shot examples (incl. multi-action, clarification and refusal) · ReAct
think→act→observe loop · short reasoning field · validator-driven self-correction ·
domain rules (bar unit conversions, verb→action mapping) · prompt-injection guard ·
temperature 0.
