#!/usr/bin/env python3
"""BVB Stage-1: run an agent over scenes and collect result.blend files.

Each scene gets a fresh sandbox container. The agent (host-side) drives it and
must leave /workspace/output/result.blend; we copy that out to
results/<run>/blends/<scene>.blend. Stage-2 evaluation renders the scene camera
and scores Dual VQA and Latent Similarity separately; see eval/README.md.

Example:
    export OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY / GEMINI_API_KEY
    python run_agent.py --model gpt-6-astra --reasoning high --output results/run_001 --limit 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
from pathlib import Path

from bvb_sandbox.agents.bash_agent import run_bash_agent
from bvb_sandbox.models import Model
from bvb_sandbox.sandbox import Sandbox
from bvb_sandbox.tasks import load_tasks


def resolve_reasoning(model: str, explicit: str | None) -> str:
    """Return the concrete reasoning label to record for this run.

    Result metadata should be directly table-ready.
    If a model is not in the known default map and no explicit reasoning was
    provided, record "none".
    """
    if explicit:
        return explicit
    name = model.split("/")[-1].lower()
    defaults = {
        "gpt-6-astra": "medium",
        "gpt-5.5": "medium",
        "gpt-5.4": "none",
        "gpt-5.4-mini": "none",
        "claude-opus-4-8": "none",
        "claude-opus-4-7": "none",
        "claude-opus-4-6": "none",
        "claude-sonnet-4-6": "none",
        "gemini-3.1-pro-preview": "high",
        "gemini-3.1-pro": "high",
    }
    return defaults.get(name, "none")


def main() -> None:
    parser = argparse.ArgumentParser(description="BVB Stage-1 agent runner.")
    parser.add_argument("--model", required=True,
                        help="litellm model name, e.g. gpt-6-astra, anthropic/claude-sonnet-4.6, gemini/gemini-2.5-pro")
    parser.add_argument("--reasoning",
                        help="Optional reasoning/thinking effort passed to litellm, e.g. none, low, medium, high, max. "
                        "Omit to use the provider/model API default.")
    parser.add_argument("--output", type=Path, required=True, help="Run directory, e.g. results/run_001")
    parser.add_argument("--image", default="bvb-sandbox:latest", help="Docker image tag.")
    parser.add_argument("--scenes", nargs="*", help="Specific scene_name(s); default = all with tests+video.")
    parser.add_argument("--skip-scenes", nargs="*", default=[],
                        help="Scene names to exclude before optional sharding.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", help="Run a subset 'i/N' (1-based), e.g. 1/4, for N parallel processes.")
    parser.add_argument("--cost-limit", type=float, default=3.0,
                        help="Per-scene USD budget (the stop lever; cost via litellm). Model decides turns/frames.")
    parser.add_argument("--model-timeout", type=float, default=600.0,
                        help="Timeout in seconds for one model API call.")
    parser.add_argument("--max-turns", type=int, default=0,
                        help="Optional safety cap on turns (0 = unlimited; cost-limit is the real stop).")
    parser.add_argument("--exec-timeout", type=float, default=300.0, help="Timeout per bash command (s).")
    parser.add_argument("--scene-timeout", type=float, default=0.0,
                        help="Optional per-scene wall-clock backstop (s); 0 disables.")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="4g")
    parser.add_argument("--allow-network", action="store_true", help="Give the sandbox container network access.")
    parser.add_argument("--resume", action="store_true", help="Skip scenes whose .blend already exists.")
    args = parser.parse_args()

    tasks = load_tasks(scenes=args.scenes, limit=args.limit, require_video=True)
    if args.skip_scenes:
        skipped = set(args.skip_scenes)
        tasks = [task for task in tasks if task.scene_name not in skipped]
    shard_i: int | None = None
    shard_n: int | None = None
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit("--shard must look like i/N, e.g. 1/4")
        if not (1 <= shard_i <= shard_n):
            raise SystemExit("--shard must be i/N with 1 <= i <= N")
        tasks = tasks[shard_i - 1 :: shard_n]
    if not tasks:
        raise SystemExit("No tasks with both unit tests and a video were found.")

    blends_dir = args.output / "blends"
    meta_dir = args.output / "agent_meta"
    blends_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    reasoning = resolve_reasoning(args.model, args.reasoning)
    client = Model(args.model, reasoning=args.reasoning, timeout=args.model_timeout)

    config = {
        "model": args.model,
        "reasoning": reasoning,
        "harness": "mini-BVB-harness",
        "image": args.image,
        "cost_limit": args.cost_limit,
        "model_timeout": args.model_timeout,
        "max_turns": args.max_turns,
        "scene_timeout": args.scene_timeout,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "num_tasks": len(tasks),
        "shard": args.shard,
        "skip_scenes": args.skip_scenes,
    }
    config_name = (
        f"config.shard-{shard_i}-of-{shard_n}.json"
        if shard_i is not None and shard_n is not None
        else "config.json"
    )
    (args.output / config_name).write_text(json.dumps(config, indent=2), encoding="utf-8")

    for index, task in enumerate(tasks, start=1):
        blend_out = blends_dir / f"{task.scene_name}.blend"
        if args.resume and blend_out.exists():
            print(f"[{index}/{len(tasks)}] skip {task.scene_name} (exists)", flush=True)
            continue
        print(f"[{index}/{len(tasks)}] {task.scene_name} ({task.dataset}) ...", flush=True)

        meta: dict = {"scene_name": task.scene_name}
        try:
            with Sandbox(
                image=args.image,
                video_path=task.video_path,
                cpus=args.cpus,
                memory=args.memory,
                network=args.allow_network,
            ) as sandbox:
                with tempfile.TemporaryDirectory(prefix=f"bvb_frames_{task.scene_name}_") as tmp:
                    result = run_bash_agent(
                        task, sandbox, client,
                        frames_dir=Path(tmp),
                        cost_limit=args.cost_limit,
                        max_turns=args.max_turns,
                        exec_timeout=args.exec_timeout,
                        wall_clock_s=(args.scene_timeout or None),
                    )
                meta.update(result)
                if sandbox.output_exists():
                    copied = sandbox.copy_out("/workspace/output/result.blend", blend_out)
                    meta["blend_file"] = str(blend_out) if copied else None
                else:
                    meta["blend_file"] = None
        except Exception as exc:  # noqa: BLE001
            meta["error"] = f"{exc.__class__.__name__}: {exc}"
            meta["blend_file"] = None
        meta["model"] = args.model
        meta["harness"] = "mini-BVB-harness"
        meta["reasoning"] = reasoning

        (meta_dir / f"{task.scene_name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"[{index}/{len(tasks)}] {task.scene_name}\n"
            f"      result : blend={'yes' if meta.get('blend_file') else 'no'}  "
            f"done={meta.get('done')}  stop={meta.get('stop_reason')}\n"
            f"      effort : turns={meta.get('turns')}  commands={meta.get('commands_run')}  "
            f"frames_seen={meta.get('frames_seen')}  frames_extracted={meta.get('frames_extracted')}\n"
            f"      cost   : ${meta.get('cost_usd', 0)}  "
            f"in_tok={meta.get('input_tokens')}  out_tok={meta.get('output_tokens')}  "
            f"dur={meta.get('duration_s')}s\n"
            f"      error  : {meta.get('error')}",
            flush=True,
        )

    print(f"Stage-1 complete. Blends in {blends_dir}", flush=True)


if __name__ == "__main__":
    main()
