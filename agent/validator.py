"""Validate LLM output before anything is executed.

The model is asked to reply with JSON like::

    {"thought": "...", "actions": [{"action": "record_usage", "params": {...}}], "response": "..."}

This module turns raw model text into a checked ``Plan`` or raises
``ValidationError`` with a message precise enough to send back to the model
so it can repair its own output (self-correction loop).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .actions import NEEDS_CONFIRMATION, Action

MAX_ACTIONS_PER_STEP = 6


class ValidationError(Exception):
    pass


@dataclass
class ActionCall:
    action: str
    params: dict
    risk: str

    @property
    def needs_confirmation(self) -> bool:
        return self.risk in NEEDS_CONFIRMATION

    def describe(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.action}({args})"


@dataclass
class Plan:
    thought: str
    actions: list[ActionCall] = field(default_factory=list)
    response: str = ""


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of model text (handles ```json fences / chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise ValidationError("Response contained no JSON object.")
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        break
                    except json.JSONDecodeError as exc:
                        raise ValidationError(f"Malformed JSON: {exc}") from exc
        else:
            raise ValidationError("Unterminated JSON object.")
    if not isinstance(obj, dict):
        raise ValidationError("Top-level JSON must be an object.")
    return obj


def _coerce(pname: str, value, spec):
    t = spec.type
    if t == "number":
        if isinstance(value, bool):
            raise ValidationError(f"'{pname}' must be a number, got boolean.")
        if isinstance(value, str):
            try:
                value = float(value.replace(",", ""))
            except ValueError:
                raise ValidationError(f"'{pname}' must be a number, got {value!r}.")
        if not isinstance(value, (int, float)):
            raise ValidationError(f"'{pname}' must be a number.")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValidationError(f"'{pname}' must be finite.")
        if spec.minimum is not None and value < spec.minimum:
            raise ValidationError(f"'{pname}'={value} is below the minimum {spec.minimum}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValidationError(f"'{pname}'={value} is above the maximum {spec.maximum}.")
        return value
    if t == "boolean":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        if not isinstance(value, bool):
            raise ValidationError(f"'{pname}' must be true or false.")
        return value
    # string
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        raise ValidationError(f"'{pname}' must not be empty.")
    if len(value) > 2000:
        raise ValidationError(f"'{pname}' is too long.")
    if spec.enum and value not in spec.enum:
        raise ValidationError(f"'{pname}' must be one of {spec.enum}, got {value!r}.")
    return value


def validate_call(raw: dict, registry: dict[str, Action]) -> ActionCall:
    if not isinstance(raw, dict):
        raise ValidationError("Each entry in 'actions' must be an object.")
    name = raw.get("action")
    if name not in registry:
        raise ValidationError(f"Unknown action {name!r}. Valid actions: {', '.join(registry)}.")
    action = registry[name]
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ValidationError(f"'params' for {name} must be an object.")
    unknown = set(params) - set(action.params)
    if unknown:
        raise ValidationError(f"{name} got unexpected params {sorted(unknown)}. Allowed: {list(action.params)}.")
    clean = {}
    for pname, spec in action.params.items():
        if pname not in params or params[pname] is None:
            if spec.required:
                raise ValidationError(f"{name} is missing required param '{pname}'.")
            continue
        clean[pname] = _coerce(pname, params[pname], spec)
    return ActionCall(name, clean, action.risk)


def parse_plan(text: str, registry: dict[str, Action]) -> Plan:
    obj = extract_json(text)
    actions_raw = obj.get("actions", []) or []
    if not isinstance(actions_raw, list):
        raise ValidationError("'actions' must be a list.")
    if len(actions_raw) > MAX_ACTIONS_PER_STEP:
        raise ValidationError(f"Too many actions in one step (max {MAX_ACTIONS_PER_STEP}).")
    calls = [validate_call(a, registry) for a in actions_raw]
    response = obj.get("response") or ""
    if not isinstance(response, str):
        response = json.dumps(response)
    if not calls and not response.strip():
        raise ValidationError("Provide either at least one action or a non-empty 'response'.")
    return Plan(thought=str(obj.get("thought", "")), actions=calls, response=response.strip())
