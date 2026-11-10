# 🍺 BarBack: AI Inventory Agent for Bars & Restaurants

**CS4680 Prompt Engineering: AI Agent Project**

BarBack lets pub staff manage inventory in plain English mid-shift. Say *"we got 2
kegs of Bud Light, poured 11 shots of Jameson, and dropped a bottle of the house
cab"*. An LLM turns that into validated operations, BarBack runs them against the
inventory database, and it reports back. It also finds what's below par, writes
supplier order emails, exports reports and pings the team chat.

- **Live demo:** `<your Streamlit URL>`
- **Demo video:** `<your YouTube URL>`

---

## 1. Problem

Small bars and restaurants usually track inventory on clipboards or spreadsheets,
and nobody keeps them up to date during a busy shift. The results are surprise
stock-outs ("we're out of limes on a Friday"), over-ordering, and no idea where
the pour cost is going. Bartenders and kitchen staff already *say* what
happened ("86 the wings", "comped two Guinness"). What's missing is something
that turns those sentences into accurate records without making them open a
spreadsheet.

BarBack is that layer. It understands bar shorthand and does the unit math
(shots → bottles, pints → kegs). It updates stock, flags items that drop below
par, and drafts the order. A human still approves anything destructive or
anything that leaves the app.

## 2. Requirements → implementation

| Requirement | Where | What |
|---|---|---|
| **LLM integration** | `agent/llm.py` | One client for **Anthropic, OpenAI and Gemini** over plain HTTPS. Timeouts, **exponential backoff with jitter** on 429/5xx/network errors, honors `Retry-After`, typed errors (`LLMAuthError`, `LLMRateLimitError`, `LLMResponseError`) that the UI shows as friendly messages. |
| **Action interpreter** | `agent/validator.py`, `agent/core.py` | Pulls the JSON out of model text (handles code fences and chatter) and maps it to registered actions. A ReAct loop (think → act → observe, max 6 steps) chains multi-step work. |
| **Executor across domains** | `agent/actions.py` | 20 actions. **Database:** add/receive/use/count/update/delete items, low-stock report, valuation, activity, read-only SQL. **Files:** CSV/Markdown exports, supplier order drafts, list/read/delete files. **System:** DB backup, reset. **API:** Discord/Slack webhook. |
| **User interface** | `app.py` (Streamlit GUI), `cli.py` (terminal) | Chat with example prompts, an expandable **agent trace** per reply (thoughts, each action ✅/❌, results), approve/decline panel, live inventory table with below-par rows highlighted, file downloads, audit log viewer. |
| **Validation** | `agent/validator.py` | Checks action names, required/unknown params, types, numeric ranges, enums and step size before anything runs. Invalid output goes **back to the model with the exact error** so it can fix it (self-correction, 2 retries). |
| **Graceful errors** | everywhere | Handler failures (item not found, overdrawing stock, missing file) come back as observations, so the model can recover or explain. Fuzzy name matching suggests "Did you mean Jameson…?". LLM failures roll back the turn cleanly. |
| **Safety checks** | `actions.py`, `core.py` | Every action has a risk level. `destructive` and `external` actions **pause for human approval**. In a mixed step, the safe actions run and only the risky ones wait. SQL is SELECT-only, single-statement, keyword-filtered **and** opened read-only. File access is sandboxed with a filename allowlist and a path-traversal check. Quantities are capped. Each session gets its own database. |
| **Audit log** | `agent/audit.py` | Every user input, LLM call (provider, model, latency, tokens, retries), raw LLM output, validation pass/fail, confirmation decision and action result is written to SQLite **and** an append-only JSONL file. You can view and download it in the UI. |

## 3. Prompt engineering

The prompt lives in `agent/prompts.py`. Techniques used:

1. **Role prompting.** The model is "BarBack", an inventory agent for a neighborhood pub whose staff talk to it in a hurry mid-shift. That sets the tone and the domain.
2. **Strict output contract.** Every reply must be a single JSON object `{thought, actions, response}`. OpenAI and Gemini are also put in native JSON mode.
3. **Tool catalog generated from code.** The action list in the prompt is built from the same registry the validator enforces, so the prompt and the code can't drift apart. Risky actions are tagged `[REQUIRES USER CONFIRMATION]`.
4. **Dynamic context injection (grounding).** Every call includes the current inventory snapshot and today's date, so the model uses exact item names and real numbers and doesn't hallucinate.
5. **Few-shot examples.** Six worked examples: multiple independent actions, the observe → summarize step, unit conversion, asking a clarifying question instead of guessing, a parallel report + order draft, and refusing an out-of-scope request.
6. **ReAct loop.** The model acts, gets an `OBSERVATION`, and decides what to do next. This lets it recover from errors, e.g. retrying with a suggested item name.
7. **Short reasoning field.** A one- or two-sentence `thought` before acting. It improves action choice and is logged for auditing, but it isn't shown as the answer.
8. **Self-correction.** Validator errors are fed back verbatim ("`'quantity'=-4 is below the minimum`").
9. **Domain rules and constraints.** Bar unit conversions (1 L = ~22 shots; ½ bbl = 124 pints), a verb → action mapping ("86'd/comped/spilled" → `record_usage` with a reason), "ask, don't guess", and "never invent numbers".
10. **Prompt-injection guard.** Text in observations, item names or files is treated as data, not instructions.
11. **Temperature 0.** Tool calls are deterministic and repeatable.

## 4. Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add one API key
streamlit run app.py
# or the terminal version:
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GEMINI_API_KEY
python cli.py --provider anthropic
```

Tests (no API key needed, they use a scripted fake LLM):

```bash
pip install pytest && python -m pytest -q
```

## 5. Deploy (Streamlit Community Cloud, free)

1. Push this folder to a **public GitHub repo**.
2. Go to **share.streamlit.io**, click **Create app**, pick the repo and branch, and set the main file to `app.py`.
3. Under **Advanced settings → Secrets**, paste one key, e.g. `ANTHROPIC_API_KEY = "sk-ant-..."`. A Gemini key from aistudio.google.com has a free tier. You can also add `WEBHOOK_URL`.
4. Click **Deploy** and submit the `*.streamlit.app` URL.

Every visitor gets their own fresh demo database, so graders can't collide with each
other. Storage on Streamlit Cloud is temporary (data resets when the app restarts),
which is fine for a demo.

## 6. Project layout

```
app.py              Streamlit GUI
cli.py              Terminal UI
agent/
  llm.py            LLM integration (3 providers, retries, errors)
  prompts.py        System prompt, few-shots, repair/observation templates
  validator.py      JSON extraction + schema validation
  actions.py        Action registry, risk levels, handlers (DB/file/system/API)
  core.py           ReAct loop, confirmation gate, self-repair
  store.py          SQLite inventory + seed data + fuzzy lookup
  audit.py          Audit logger (SQLite + JSONL)
tests/test_agent.py 16 tests: loop, confirmation, repair, safety, retries
```
