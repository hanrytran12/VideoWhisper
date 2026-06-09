"""S7 — VLM context: describe each segment's representative frame via Ollama."""
from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.stage_base import Stage
from utils.ollama_client import OllamaVLM


class VLMContextStage(Stage):
    name = "vlm_context"

    def outputs(self, ctx):
        return [ctx.context_cache_path]

    def execute(self, ctx: PipelineContext) -> None:
        cfg = ctx.config["vlm"]
        vlm = OllamaVLM(
            host=cfg.get("ollama_host", "http://127.0.0.1:11434"),
            model=cfg.get("model", "minicpm-v"),
            prompt=cfg.get("prompt", "Describe this image."),
            timeout=cfg.get("timeout", 120),
            num_gpu=cfg.get("num_gpu"),
            cache_path=ctx.context_cache_path,
        )
        if not vlm.ping():
            raise RuntimeError(
                f"Ollama not reachable at {cfg.get('ollama_host')}. "
                "Start it with `ollama serve` and pull the model."
            )

        index = ctx.read_json(ctx.frames_index_path)
        descriptions: dict[str, str] = {}
        for entry in index:
            rel = entry.get("frame_path")
            seg_id = str(entry["segment"])
            if not rel:
                descriptions[seg_id] = ""
                continue
            fpath = ctx.out_dir / rel
            try:
                descriptions[seg_id] = vlm.describe(fpath)
            except Exception as e:
                print(f"[vlm_context] segment {seg_id} failed: {e}")
                descriptions[seg_id] = ""

        # context_cache.json holds two layers:
        #   - image-hash cache (managed inside OllamaVLM)
        #   - per-segment descriptions (written here, keyed "seg:<id>")
        seg_map = {f"seg:{k}": v for k, v in descriptions.items()}
        merged = dict(vlm._cache)
        merged.update(seg_map)
        ctx.write_json(ctx.context_cache_path, merged)
