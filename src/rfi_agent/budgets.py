from __future__ import annotations

import time
from typing import Any


def estimated_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    usd_per_million_input: float,
    usd_per_million_output: float,
) -> float:
    return (max(input_tokens, 0) / 1_000_000) * usd_per_million_input + (
        max(output_tokens, 0) / 1_000_000
    ) * usd_per_million_output


def over_budget(state: dict[str, Any]) -> str | None:
    started = state.get("started_at")
    wall = state.get("wall_seconds")
    if started and wall and (time.monotonic() - float(started)) > float(wall):
        return "wall time"
    steps = int(state.get("model_steps") or 0)
    max_steps = state.get("max_steps")
    if max_steps is not None and steps >= int(max_steps):
        return "model steps"
    tokens = state.get("tokens") or {}
    used_in = int(tokens.get("input") or 0)
    used_out = int(tokens.get("output") or 0)
    max_tokens = state.get("max_input_tokens")
    if max_tokens and used_in >= int(max_tokens):
        return "input tokens"
    ceiling = float(state.get("max_cost_usd") or 0)
    if ceiling > 0:
        cost = estimated_cost_usd(
            used_in,
            used_out,
            usd_per_million_input=float(state.get("usd_per_million_input") or 0),
            usd_per_million_output=float(state.get("usd_per_million_output") or 0),
        )
        if cost >= ceiling:
            return "cost"
    return None
