"""BarBack — Streamlit GUI.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import LLMClient, LLMError, build_agent
from agent.llm import DEFAULT_MODELS, KEY_ENV

st.set_page_config(page_title="BarBack · AI Inventory Agent", page_icon="🍺", layout="wide")

BASE_DIR = Path(__file__).parent / "data"
PROVIDER_LABELS = {"anthropic": "Anthropic (Claude)", "openai": "OpenAI (GPT)", "gemini": "Google (Gemini)"}
EXAMPLES = [
    "What's running low?",
    "We just got 2 kegs of Bud Light and 3 bottles of Jack",
    "Poured 14 shots of Jameson and 6 of Patron tonight",
    "Dropped a bottle of house cab, and we're out of limes",
    "Draft orders for everything below par and post a heads-up to the team",
    "Export the inventory as CSV",
    "Delete Fireball from inventory",
]
ICONS = {"thought": "💭", "action": "⚙️", "pending": "⏸️", "rejected": "🚫", "repair": "🔧"}

# Load Streamlit secrets (if any) into env so the LLM client finds keys.
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass


# --------------------------------------------------------------------------- state
def init_state():
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex[:12])
    ss.setdefault("display", [])      # chat transcript for rendering
    ss.setdefault("agent", None)
    ss.setdefault("result", None)     # last TurnResult (for pending confirmation)


def get_llm(provider, model, user_key):
    return LLMClient(provider=provider, model=model or None, api_key=user_key or None)


def get_agent(llm):
    ss = st.session_state
    if ss.agent is None:
        ss.agent = build_agent(ss.session_id, llm, BASE_DIR)
    else:
        ss.agent.llm = llm
    return ss.agent


init_state()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🍺 BarBack")
    st.caption("Natural-language inventory agent for a pub. Say what happened on shift; "
               "BarBack updates the books.")

    available = [p for p in DEFAULT_MODELS if os.getenv(KEY_ENV[p])]
    options = available or list(DEFAULT_MODELS)
    provider = st.selectbox("LLM provider", options, format_func=PROVIDER_LABELS.get)
    model = st.text_input("Model", value=DEFAULT_MODELS[provider])
    user_key = ""
    if provider not in available:
        user_key = st.text_input(f"{PROVIDER_LABELS[provider]} API key", type="password",
                                 help="Used only for this browser session; never stored.")

    st.divider()
    st.subheader("Try one")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex-{ex}"):
            st.session_state.queued = ex

    st.divider()
    if st.button("🔄 New session (fresh demo data)", use_container_width=True):
        for k in ["session_id", "display", "agent", "result"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.caption(f"Session `{st.session_state.session_id}`")
    with st.expander("🛡️ Safety model"):
        st.markdown(
            "- Every LLM reply is **schema-validated** before anything runs\n"
            "- Unknown actions / bad params are rejected and sent back to the model to fix\n"
            "- **Destructive** (delete, reset) and **external** (webhook) actions pause for your approval\n"
            "- SQL is **read-only**; files are sandboxed to this session's folder\n"
            "- Every step is written to an **audit log**"
        )

# --------------------------------------------------------------------------- build agent
try:
    llm = get_llm(provider, model, user_key)
    agent = get_agent(llm)
    llm_ready = True
except LLMError as exc:
    llm_ready = False
    agent = st.session_state.agent
    if agent is None:  # still build the store so the Inventory tab works
        agent = build_agent(st.session_state.session_id, None, BASE_DIR)
        st.session_state.agent = agent
    key_msg = str(exc)


def render_steps(steps):
    if not steps:
        return
    with st.expander(f"Agent trace · {len(steps)} step(s)", expanded=False):
        for s in steps:
            icon = ICONS.get(s.kind, "•")
            if s.kind == "action":
                icon = "✅" if s.ok else "❌"
            st.markdown(f"{icon} **{s.kind}** — `{s.text}`" if s.kind != "thought" else f"{icon} _{s.text}_")
            if s.data is not None and s.kind == "action":
                st.json(s.data if not isinstance(s.data, str) else {"error": s.data}, expanded=False)


def push_result(res):
    st.session_state.result = res
    st.session_state.display.append({"role": "assistant", "content": res.response, "steps": res.steps,
                                     "status": res.status})


# --------------------------------------------------------------------------- tabs
tab_chat, tab_inv, tab_files, tab_audit, tab_about = st.tabs(
    ["💬 Agent", "📦 Inventory", "📁 Files", "🧾 Audit log", "ℹ️ How it works"]
)

with tab_chat:
    if not llm_ready:
        st.info(f"Add an API key in the sidebar to start. ({key_msg})")

    for msg in st.session_state.display:
        with st.chat_message(msg["role"], avatar="🧑‍🍳" if msg["role"] == "user" else "🍺"):
            if msg.get("status") == "error":
                st.error(msg["content"])
            else:
                st.markdown(msg["content"] or "_(no message)_")
            render_steps(msg.get("steps"))

    res = st.session_state.result
    awaiting = bool(res and res.status == "awaiting_confirmation" and agent.pending)
    if awaiting:
        with st.container(border=True):
            st.warning("**Confirmation required** — these actions can't be undone or leave the app:")
            for c in agent.pending:
                st.markdown(f"- `{c.describe()}` · _{c.risk}_")
            col1, col2, _ = st.columns([1, 1, 4])
            approve = col1.button("✅ Approve", type="primary", use_container_width=True)
            decline = col2.button("✖️ Decline", use_container_width=True)
            if approve or decline:
                with st.spinner("Running…" if approve else "Cancelling…"):
                    push_result(agent.confirm(bool(approve)))
                st.rerun()

    prompt = st.chat_input("e.g. 'sold 3 cases of Modelo and 86 the wings'", disabled=not llm_ready)
    prompt = prompt or st.session_state.pop("queued", None)
    if prompt and llm_ready:
        st.session_state.display.append({"role": "user", "content": prompt})
        with st.spinner("BarBack is thinking…"):
            push_result(agent.send(prompt))
        st.rerun()

with tab_inv:
    items = agent.toolbox.store.all_items()
    df = pd.DataFrame(items)
    if df.empty:
        st.info("No items.")
    else:
        df["value"] = (df["quantity"] * df["unit_cost"]).round(2)
        below = df[df["quantity"] < df["par_level"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Items tracked", len(df))
        c2.metric("Below par", len(below))
        c3.metric("Stock value", f"${df['value'].sum():,.2f}")
        view = df[["name", "category", "quantity", "unit", "par_level", "unit_cost", "value", "supplier"]]

        def highlight(row):
            low = row["quantity"] < row["par_level"]
            return ["background-color: rgba(255,80,80,0.18)" if low else "" for _ in row]

        st.dataframe(view.style.apply(highlight, axis=1).format({"unit_cost": "${:.2f}", "value": "${:.2f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Rows in red are below par.")
        st.subheader("Recent activity")
        st.dataframe(pd.DataFrame(agent.toolbox.recent_activity(25)), use_container_width=True, hide_index=True)

with tab_files:
    files = agent.toolbox.list_files()
    if not files:
        st.info("No files yet. Ask BarBack to export a report or draft an order.")
    for f in files:
        path = agent.toolbox.exports / f["file"]
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{f['file']}** · {f['bytes']} bytes")
        c2.download_button("Download", path.read_bytes(), file_name=f["file"], key=f"dl-{f['file']}")
        with st.expander("Preview"):
            st.code(path.read_text()[:3000])

with tab_audit:
    st.caption("Every user input, LLM call, validation result, confirmation and action for this session.")
    rows = agent.audit.recent(300)
    if rows:
        adf = pd.DataFrame(rows)
        events = sorted(adf["event"].unique())
        pick = st.multiselect("Filter events", events, default=events)
        st.dataframe(adf[adf["event"].isin(pick)], use_container_width=True, hide_index=True)
        st.download_button("Download audit log (JSON)", json.dumps(rows, indent=2), "audit_log.json")
    else:
        st.info("Nothing logged yet.")

with tab_about:
    st.markdown(Path(__file__).with_name("ARCHITECTURE.md").read_text())
