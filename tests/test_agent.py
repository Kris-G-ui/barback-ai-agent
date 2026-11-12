import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import build_agent  # noqa: E402
from agent.llm import LLMResponse, LLMRateLimitError  # noqa: E402
from agent.validator import ValidationError, extract_json, parse_plan  # noqa: E402


class FakeLLM:
    """Returns scripted replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages):
        self.calls.append((system, [dict(m) for m in messages]))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        text = r if isinstance(r, str) else json.dumps(r)
        return LLMResponse(text=text, provider="fake", model="fake", latency_s=0)


def J(thought="", actions=None, response=""):
    return {"thought": thought, "actions": actions or [], "response": response}


def A(_action, **params):
    return {"action": _action, "params": params}


@pytest.fixture
def make(tmp_path):
    def _make(replies):
        llm = FakeLLM(replies)
        return build_agent("t1", llm, base_dir=tmp_path), llm
    return _make


def qty(agent, name):
    return agent.toolbox.store.resolve(name)["quantity"]


def test_receive_then_summarize(make):
    agent, llm = make([
        J("delivery", [A("receive_stock", name="Bud Light 1/2 bbl keg", quantity=2)]),
        J("done", [], "Logged 2 kegs."),
    ])
    res = agent.send("got 2 kegs of bud light")
    assert res.status == "done" and res.response == "Logged 2 kegs."
    assert qty(agent, "Bud Light 1/2 bbl keg") == 5
    # second LLM call must contain the observation
    assert llm.calls[1][1][-1]["content"].startswith("OBSERVATION")
    # snapshot is injected into the system prompt
    assert "Jameson Irish Whiskey 1L" in llm.calls[0][0]


def test_destructive_requires_confirmation_and_approve(make):
    agent, _ = make([
        J("delete", [A("delete_item", name="Fireball 1L")]),
        J("ok", [], "Deleted Fireball."),
    ])
    res = agent.send("remove fireball from inventory")
    assert res.status == "awaiting_confirmation"
    assert qty(agent, "Fireball 1L") == 5  # not yet executed
    res2 = agent.confirm(True)
    assert res2.status == "done"
    with pytest.raises(LookupError):
        agent.toolbox.store.resolve("Fireball 1L")


def test_reject_confirmation_keeps_data(make):
    agent, llm = make([
        J("reset", [A("reset_demo_data")]),
        J("ack", [], "OK, left it alone."),
    ])
    agent.toolbox.receive_stock("Limes", 3)
    assert agent.send("reset everything").status == "awaiting_confirmation"
    res = agent.confirm(False)
    assert res.status == "done"
    assert qty(agent, "Limes") == 3.5
    assert "DECLINED" in llm.calls[1][1][-1]["content"]


def test_mixed_step_runs_safe_actions_first(make):
    agent, _ = make([
        J("x", [A("receive_stock", name="Limes", quantity=1), A("delete_file", filename="nope.csv")]),
        J("y", [], "done"),
    ])
    res = agent.send("got limes and delete nope.csv")
    assert res.status == "awaiting_confirmation"
    assert qty(agent, "Limes") == 1.5  # safe one ran already
    res = agent.confirm(True)
    assert any(s.kind == "action" and not s.ok for s in res.steps)  # file didn't exist -> graceful error


def test_self_repair_on_invalid_output(make):
    agent, llm = make([
        "sure! I'll do that",                                   # not JSON
        J("x", [A("record_usage", name="Limes", quantity=-4)]),  # violates minimum
        J("x", [A("record_usage", name="Limes", quantity=0.25, reason="waste")]),
        J("y", [], "Logged."),
    ])
    res = agent.send("tossed a quarter case of limes")
    assert res.status == "done"
    assert qty(agent, "Limes") == 0.25
    repairs = [s for s in res.steps if s.kind == "repair"]
    assert len(repairs) == 2
    assert "SYSTEM VALIDATOR" in llm.calls[1][1][-1]["content"]


def test_gives_up_after_max_repairs_and_rolls_back(make):
    agent, _ = make(["nope", "still nope", "nah"])
    res = agent.send("hi")
    assert res.status == "error"
    assert agent.history == []


def test_unknown_action_rejected():
    from agent.actions import Toolbox  # noqa
    with pytest.raises(ValidationError):
        parse_plan(json.dumps(J("", [A("rm_rf", path="/")])), {})


def test_overdraw_is_reported_not_crashed(make):
    agent, _ = make([
        J("x", [A("record_usage", name="Patron Silver 750ml", quantity=5)]),
        J("y", [], "Only 1 on hand."),
    ])
    res = agent.send("sold 5 bottles of patron")
    act = [s for s in res.steps if s.kind == "action"][0]
    assert not act.ok and "only 1" in act.data
    assert qty(agent, "Patron Silver 750ml") == 1


def test_fuzzy_name_suggestion(make):
    agent, _ = make([])
    with pytest.raises(LookupError, match="Jameson"):
        agent.toolbox.store.resolve("jamison")
    assert agent.toolbox.store.resolve("guinness")["name"] == "Guinness 1/6 bbl keg"


def test_sql_is_read_only(make):
    agent, _ = make([])
    tb = agent.toolbox
    assert tb.run_sql_query("SELECT COUNT(*) AS n FROM items")[0]["n"] == 14
    for bad in ["DELETE FROM items", "SELECT 1; DROP TABLE items", "select * from items where name='x' or 1=1 union select sqlite_version(),1,1,1,1,1,1,1,1; update items set quantity=0"]:
        with pytest.raises(Exception):
            tb.execute("run_sql_query", {"sql": bad})


def test_file_sandbox(make):
    agent, _ = make([])
    tb = agent.toolbox
    out = tb.execute("export_report", {"report": "low_stock", "fmt": "csv"})
    assert (tb.exports / out["file"]).exists()
    for bad in ["../inventory.db", "/etc/passwd", "..", ".hidden"]:
        with pytest.raises(Exception):
            tb.execute("read_file", {"filename": bad})


def test_draft_order_and_notification_outbox(make):
    agent, _ = make([])
    tb = agent.toolbox
    out = tb.execute("draft_order", {})
    text = (tb.exports / out["file"]).read_text()
    assert "Southern Glazer's" in text and "Jameson" in text
    tb.webhook_url = None
    r = tb.execute("send_notification", {"message": "86 Guinness"}); assert r["posted_to_team_chat"] is False and "outbox" in r["saved_to"]


def test_llm_error_surfaces_cleanly(make):
    agent, _ = make([LLMRateLimitError("rate limited")])
    res = agent.send("what's low?")
    assert res.status == "error" and "rate limited" in res.response


def test_audit_log_written(make):
    agent, _ = make([J("x", [], "hi")])
    agent.send("hello")
    events = [r["event"] for r in agent.audit.recent()]
    assert {"user_input", "llm_call", "llm_output", "plan_validated"} <= set(events)


def test_extract_json_from_fenced_text():
    assert extract_json('Here you go:\n```json\n{"a": {"b": 1}}\n```')["a"]["b"] == 1
    assert extract_json('blah {"x": "has } brace"} trailing')["x"] == "has } brace"


def test_llm_client_retries_on_429(monkeypatch):
    from agent.llm import LLMClient

    class R:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.headers, self.text = code, body, {}, ""
        def json(self):
            return self._b

    seq = [R(429), R(503), R(200, {"content": [{"type": "text", "text": "{}"}], "usage": {}})]
    c = LLMClient("anthropic", api_key="k", base_backoff_s=0)
    monkeypatch.setattr(c._session, "post", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = c.chat("sys", [{"role": "user", "content": "hi"}])
    assert r.retries == 2 and r.text == "{}"


def test_llm_client_falls_back_when_overloaded(monkeypatch):
    from agent.llm import LLMClient

    class R:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.headers, self.text = code, body, {}, "high demand"
        def json(self):
            return self._b

    ok = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}}
    seq = [R(503), R(503), R(200, ok)]
    seen = []
    c = LLMClient("gemini", model="gemini-3.8-flash", api_key="k", max_retries=1, base_backoff_s=0)
    def post(url, **k):
        seen.append(url)
        return seq.pop(0)
    monkeypatch.setattr(c._session, "post", post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = c.chat("sys", [{"role": "user", "content": "hi"}])
    assert r.fallback_from == "gemini-3.8-flash" and r.model == "gemini-flash-latest"
    assert c.model == "gemini-3.8-flash"  # primary restored


def test_llm_client_respects_time_budget(monkeypatch):
    import time as _t
    from agent.llm import LLMClient, LLMRateLimitError

    clock = [0.0]
    monkeypatch.setattr(_t, "time", lambda: clock[0])
    monkeypatch.setattr(_t, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    class R:
        status_code, headers, text = 503, {"retry-after": "30"}, "busy"

    c = LLMClient("gemini", api_key="k", budget_s=60)
    def post(url, **k):
        clock[0] += 2  # each request takes 2s
        return R()
    monkeypatch.setattr(c._session, "post", post)
    with pytest.raises(LLMRateLimitError, match="overloaded"):
        c.chat("sys", [{"role": "user", "content": "hi"}])
    assert clock[0] <= 62


def test_results_kept_when_summary_call_fails(make):
    agent, _ = make([
        J("x", [A("low_stock_report")]),
        LLMRateLimitError("overloaded"),
    ])
    res = agent.send("what's low?")
    assert res.status == "done" and "| name |" in res.response and "Jameson" in res.response and "estimated_total" in res.response
    assert agent.history[-1]["role"] == "assistant"


def test_sticky_fallback(monkeypatch):
    import agent.llm as L
    L._STICKY.clear()
    class R:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.headers, self.text = code, body, {}, "busy"
        def json(self):
            return self._b
    ok = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}}
    urls = []
    seq = [R(503), R(503), R(200, ok), R(200, ok)]
    c = L.LLMClient("gemini", model="gemini-3.8-flash", api_key="k", max_retries=1, base_backoff_s=0)
    def post(url, **k):
        urls.append(url)
        return seq.pop(0)
    monkeypatch.setattr(c._session, "post", post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    c.chat("s", [{"role": "user", "content": "a"}])
    c.chat("s", [{"role": "user", "content": "b"}])
    assert "gemini-flash-latest" in urls[-1]   # second call went straight to the backup
    L._STICKY.clear()
