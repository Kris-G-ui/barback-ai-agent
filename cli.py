"""BarBack — terminal interface.

    python cli.py --provider anthropic
    python cli.py --provider gemini --model gemini-2.0-flash

Commands:  /inv  show inventory   /log  recent audit events   /reset  new session   /quit
"""
from __future__ import annotations

import argparse
import uuid

from agent import LLMClient, LLMError, build_agent

C = {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "c": "\033[36m", "d": "\033[2m", "b": "\033[1m", "x": "\033[0m"}


def show(res):
    for s in res.steps:
        if s.kind == "thought":
            print(f"{C['d']}  💭 {s.text}{C['x']}")
        elif s.kind == "action":
            mark = f"{C['g']}✔" if s.ok else f"{C['r']}✘"
            print(f"  {mark} {s.text}{C['x']}")
            if not s.ok:
                print(f"{C['r']}    {s.data}{C['x']}")
        elif s.kind == "repair":
            print(f"{C['y']}  🔧 {s.text}{C['x']}")
        elif s.kind == "rejected":
            print(f"{C['y']}  🚫 {s.text}{C['x']}")
    color = C["r"] if res.status == "error" else C["c"]
    if res.response:
        print(f"\n{color}{C['b']}BarBack:{C['x']} {res.response}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "gemini"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    try:
        llm = LLMClient(provider=args.provider, model=args.model)
    except LLMError as exc:
        raise SystemExit(f"Setup error: {exc}")
    agent = build_agent(f"cli-{uuid.uuid4().hex[:8]}", llm)
    print(f"{C['b']}🍺 BarBack{C['x']} ({llm.provider}/{llm.model}) — type a request, /inv, /log, /reset or /quit\n")

    while True:
        try:
            text = input(f"{C['b']}you>{C['x']} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/inv":
            for i in agent.toolbox.store.all_items():
                flag = f"{C['r']} LOW{C['x']}" if i["quantity"] < i["par_level"] else ""
                print(f"  {i['name']:<28} {i['quantity']:>7} {i['unit']:<7} par {i['par_level']}{flag}")
            continue
        if text == "/log":
            for r in reversed(agent.audit.recent(15)):
                print(f"{C['d']}  {r['ts']} {r['event']:<22} {r['detail'][:90]}{C['x']}")
            continue
        if text == "/reset":
            agent = build_agent(f"cli-{uuid.uuid4().hex[:8]}", llm)
            print("  New session with fresh demo data.\n")
            continue

        res = agent.send(text)
        show(res)
        while res.status == "awaiting_confirmation":
            print(f"{C['y']}{C['b']}⚠ Confirmation required:{C['x']}")
            for c in res.pending:
                print(f"{C['y']}   - {c.describe()}  [{c.risk}]{C['x']}")
            ans = input(f"{C['b']}Approve? [y/N]{C['x']} ").strip().lower()
            res = agent.confirm(ans in ("y", "yes"))
            show(res)


if __name__ == "__main__":
    main()
