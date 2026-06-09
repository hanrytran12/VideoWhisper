"""Stage base class with output-file caching."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pipeline.context import PipelineContext


class Stage(ABC):
    name: str = "stage"

    def outputs(self, ctx: PipelineContext) -> list[Path]:
        """Files this stage produces. If all exist, the stage is skipped."""
        return []

    def is_cached(self, ctx: PipelineContext) -> bool:
        outs = self.outputs(ctx)
        return bool(outs) and all(p.exists() for p in outs)

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> None:
        ...

    def run(self, ctx: PipelineContext, force: bool = False) -> None:
        if not force and self.is_cached(ctx):
            print(f"[{self.name}] cached — skip")
            return
        print(f"[{self.name}] running...")
        self.execute(ctx)
        print(f"[{self.name}] done")
