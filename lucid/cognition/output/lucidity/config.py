"""Thresholds and limits for the lucidity gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LucidityConfig:
    margin_threshold_answer: float = 0.08
    margin_threshold_chat: float = 0.0
    margin_threshold_solve_grid: float = 0.10
    coverage_threshold: float = 0.55
    coherence_threshold: float = 0.65
    binding_stability_threshold: float = 0.45
    projection_fit_threshold: float = 0.85
    contradiction_severity_threshold: float = 0.6
    maturity_hot_fraction: float = 0.5
    max_iterations: int = 3
    require_projection_on_grid_pre_check: bool = True
    salience_cutoff: float = 0.25

    def margin_threshold(self, task_intent: str, pass_kind: str) -> float:
        task = normalize_task_intent(task_intent)
        if task == "solve_grid" and pass_kind == "final_check":
            return self.margin_threshold_solve_grid
        if task == "solve_grid":
            return 1.0
        if task == "chat":
            return self.margin_threshold_chat
        return self.margin_threshold_answer


def normalize_task_intent(raw: str) -> str:
    text = (raw or "answer").strip()
    if text.startswith("TaskIntent."):
        text = text.split(".", 1)[-1]
    return text.lower()


def normalize_pass_kind(raw: str) -> str:
    text = (raw or "pre_check").strip().lower()
    if text in {"pre_check", "final_check", "recheck"}:
        return text
    return "pre_check"


def normalize_risk_level(raw: str) -> str:
    text = (raw or "medium").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    return "medium"
