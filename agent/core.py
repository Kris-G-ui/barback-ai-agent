"""The agent loop: LLM output -> validated plan -> (confirmation) -> execution -> observation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .actions import ActionError, Toolbox
from .audit import AuditLogger
from .llm import LLMError
from .prompts import OBSERVATION_TEMPLATE, REJECTED_TEMPLATE, REPAIR_TEMPLATE, build_system_prompt
from .validator import ActionCall, Plan, ValidationError, parse_plan

HISTORY_LIMIT = 40  # messages kept for context


@dataclass
class Step:
    kind: str            # thought | action | pending | rejected | repair
    text: str
    ok: bool = True
    data: object = None


@dataclass
class TurnResult:
    status: str                      # done | awaiting_confirmation | error
    response: str = ""
    steps: list[Step] = field(default_factory=list)
    pending: list[ActionCall] = field(default_factory=list)


class Agent:
    def __init__(self, llm, toolbox: Toolbox, audit: AuditLogger, max_steps: int = 6, max_repairs: int = 2):
        self.llm = llm
        self.toolbox = toolbox
        self.audit = audit
        self.max_steps = max_steps
        self.max_repairs = max_repairs
        self.history: list[dict] = []
        self.pending: list[ActionCall] = []
        self._pending_results: list[dict] = []
        self._steps_used = 0
        self._turn_start = 0

    # ------------------------------------------------------------- public API
    def send(self, user_text: str) -> TurnResult:
        user_text = (user_text or "").strip()
        if not user_text:
            return TurnResult("error", "Please type a request.")
        if len(user_text) > 4000:
            return TurnResult("error", "That message is too long (4000 character max).")
        self.audit.log("user_input", text=user_text)
        prefix = ""
        if self.pending:  # user moved on without answering -> treat as declined
            prefix = REJECTED_TEMPLATE.format(actions=[c.describe() for c in self.pending]) + "\n\nNew user message: "
            self.audit.log("confirmation_rejected", actions=[c.describe() for c in self.pending], implicit=True)
            self.pending, self._pending_results = [], []
        self._turn_start = len(self.history)
        self._steps_used = 0
        self.history.append({"role": "user", "content": prefix + user_text})
        return self._loop([])

    def confirm(self, approved: bool) -> TurnResult:
        if not self.pending:
            return TurnResult("error", "There is nothing waiting for confirmation.")
        calls, results = self.pending, self._pending_results
        self.pending, self._pending_results = [], []
        steps: list[Step] = []
        if approved:
            self.audit.log("confirmation_approved", actions=[c.describe() for c in calls])
            for c in calls:
                results.append(self._run(c, steps))
            content = OBSERVATION_TEMPLATE.format(results=json.dumps(results, default=str))
        else:
            self.audit.log("confirmation_rejected", actions=[c.describe() for c in calls])
            for c in calls:
                steps.append(Step("rejected", f"Declined: {c.describe()}", ok=False))
            content = REJECTED_TEMPLATE.format(actions=[c.describe() for c in calls])
            if results:
                content += "\nOther actions from that step DID run: " + json.dumps(results, default=str)
        self.history.append({"role": "user", "content": content})
        return self._loop(steps)

    def reset(self) -> None:
        self.history, self.pending, self._pending_results = [], [], []

    # ---------------------------------------------------------------- internals
    def _context(self) -> list[dict]:
        msgs = self.history[-HISTORY_LIMIT:]
        while msgs and msgs[0]["role"] != "user":
            msgs = msgs[1:]
        return msgs

    def _think(self, steps: list[Step]) -> Plan:
        """Call the LLM and validate; on bad output feed the error back (self-repair)."""
        for attempt in range(self.max_repairs + 1):
            system = build_system_prompt(self.toolbox.catalog(), self.toolbox.store.all_items())
            resp = self.llm.chat(system, self._context())
            self.audit.log("llm_call", provider=resp.provider, model=resp.model, latency_s=resp.latency_s,
                           input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, retries=resp.retries,
                           fallback_from=resp.fallback_from)
            if resp.fallback_from:
                steps.append(Step("repair", f"{resp.fallback_from} was overloaded; used backup model {resp.model}"))
            self.audit.log("llm_output", text=resp.text)
            self.history.append({"role": "assistant", "content": resp.text})
            try:
                plan = parse_plan(resp.text, self.toolbox.actions)
                self.audit.log("plan_validated", thought=plan.thought, actions=[a.describe() for a in plan.actions])
                return plan
            except ValidationError as exc:
                self.audit.log("validation_failed", error=str(exc), attempt=attempt + 1)
                steps.append(Step("repair", f"Model output rejected by validator: {exc}", ok=False))
                self.history.append({"role": "user", "content": REPAIR_TEMPLATE.format(error=exc)})
        raise ValidationError("The model kept producing invalid output. Nothing was executed.")

    def _run(self, call: ActionCall, steps: list[Step]) -> dict:
        try:
            result = self.toolbox.execute(call.action, call.params)
            self.audit.log("action_executed", action=call.action, params=call.params, result=result)
            steps.append(Step("action", call.describe(), ok=True, data=result))
            return {"action": call.action, "ok": True, "result": result}
        except ActionError as exc:
            self.audit.log("action_failed", action=call.action, params=call.params, error=str(exc))
            steps.append(Step("action", call.describe(), ok=False, data=str(exc)))
            return {"action": call.action, "ok": False, "error": str(exc)}

    def _loop(self, steps: list[Step]) -> TurnResult:
        try:
            while self._steps_used < self.max_steps:
                self._steps_used += 1
                plan = self._think(steps)
                if plan.thought:
                    steps.append(Step("thought", plan.thought))
                if not plan.actions:
                    return TurnResult("done", plan.response, steps)

                safe = [c for c in plan.actions if not c.needs_confirmation]
                risky = [c for c in plan.actions if c.needs_confirmation]
                results = [self._run(c, steps) for c in safe]

                if risky:
                    self.pending, self._pending_results = risky, results
                    for c in risky:
                        steps.append(Step("pending", c.describe(), data=c.risk))
                    self.audit.log("confirmation_requested", actions=[c.describe() for c in risky])
                    msg = plan.response or "These actions need your OK before I run them."
                    return TurnResult("awaiting_confirmation", msg, steps, list(risky))

                self.history.append(
                    {"role": "user", "content": OBSERVATION_TEMPLATE.format(results=json.dumps(results, default=str))}
                )
            self.audit.log("step_limit_reached", max_steps=self.max_steps)
            return TurnResult("done", "I hit my step limit for this request. Here's what I did so far — "
                              "try asking again more specifically if something's missing.", steps)
        except (LLMError, ValidationError) as exc:
            self.audit.log("turn_error", error=str(exc))
            # roll back this turn so the conversation stays well-formed
            del self.history[self._turn_start:]
            self.pending, self._pending_results = [], []
            return TurnResult("error", f"⚠️ {exc}", steps)
