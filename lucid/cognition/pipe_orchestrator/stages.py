"""Stage protocol and a tiny registry.

The orchestrator runs stages sequentially and stores their inputs/outputs into
`lucid.ir.pipeline.PipelineRun` for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class Stage(Protocol):
    stage_name: str

    def run(self, stage_input: Any, *, context: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class FunctionStage:
    stage_name: str
    fn: Callable[[Any, Any], Any]

    def run(self, stage_input: Any, *, context: Any) -> Any:
        return self.fn(stage_input, context)

