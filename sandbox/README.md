# BVB Sandbox — Stage-1 Agent Runs

Two-stage, SWE-bench-style harness. This folder is **Stage 1**: run an agent over
scenes and collect one `result.blend` per scene. **Stage 2** (scoring) is fully
decoupled and lives in `../eval` (`unit_test_metric.py --mode execute`).

```
Stage 1 (here, expensive: calls model API)     Stage 2 (../eval, cheap, re-runnable)
  video frames ─▶ agent ─▶ result.blend   ─────▶  introspect .blend in Blender
                  (in Docker sandbox)              ─▶ run unit_tests.jsonl ─▶ scores
```

The container is a **generic Blender sandbox**; the agent runs on the host and
drives it via `docker exec`. Swapping the model client while keeping the harness
fixed gives a fair model-vs-model comparison. The same sandbox can later host
other harnesses (e.g. Claude Code) without changing the image.

## 0. One-time setup

Prerequisites on the host: Python 3.10+, Docker, and `ffmpeg`/`ffprobe`
(used for frame extraction before the sandbox runs).

```bash
docker build -t bvb-sandbox:latest .          # Ubuntu 22.04 + Blender 4.2 + xvfb + ffmpeg
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # litellm
```

Run Blender inside the container as `blender_run --python script.py`
(wraps `xvfb-run -a blender --background`).

## 1. Run an agent (Stage 1)

```bash
export OPENAI_API_KEY=...           # or ANTHROPIC_API_KEY / GEMINI_API_KEY
.venv/bin/python run_agent.py \
    --model gpt-5.5 \               # any litellm name (anthropic/claude-..., gemini/...)
    --output results/run_001 \
    --cost-limit 3.0 \             # the only budget; model decides turns/frames
    --limit 5                      # smoke test first; drop --limit for all 288
```

Outputs:

```
results/run_001/
├── config.json
├── blends/<scene>.blend      # the artifact, persisted
└── agent_meta/<scene>.json   # turns, frames_seen, cost_usd, done, stop_reason, ...
```

Useful flags: `--scenes <id...>`, `--resume`, `--cost-limit`, `--max-turns`
(0 = unlimited safety cap), `--allow-network` (off by default).

Tasks come from `../eval/unit_tests.jsonl` (scenes with tests) joined with
`../eval/test.jsonl` (dataset) and `../VSI-Bench/<dataset>/<scene>.mp4`. All 288
test scenes currently have a video.

## 3. Score the run (Stage 2)

```bash
python make_pairs.py --run results/run_001
export BLENDER_BIN="/path/to/blender" OPENAI_API_KEY=... BVB_JUDGE_MODEL=...
python ../eval/unit_test_metric.py \
    --pairs results/run_001/code_pairs.jsonl \
    --mode execute \
    --output results/run_001/unit_tests.jsonl \
    --summary-output results/run_001/summary.json
```

`--mode execute` loads each `.blend` in Blender and reads real geometry
(`matrix_world @ bound_box` on the evaluated depsgraph), so any scene is scored
against the same `unit_tests.jsonl` regardless of how the agent authored it.

## Philosophy: the model decides effort, cost is the only budget

mini-swe-agent style. The harness is a thin substrate and does not cap how many
turns or frames the model uses — the model decides, and the single hard budget is
`--cost-limit` (USD, computed by litellm's price DB, no hand-maintained prices).

Two primitives, both model-driven via fenced blocks:

- ```` ```bash ```` — run a command in the sandbox (stdout/stderr returned).
- ```` ```frames ```` — request frames to look at; the harness extracts them and
  injects them as images on the next turn. Content is explicit timestamps
  (`0 5 12 30`) or `count=N` (optionally `start=` `end=`).

The model finishes with `DONE`, but only after it has actually seen ≥1 frame —
`agent_meta` reports `frames_seen` (frames the model truly consumed) vs
`frames_extracted`. A run with `frames_seen=0` built the scene blind and is
invalid.

## Notes / current limits

- Simple fenced-block I/O (no formal tool calling), so any litellm chat model works.
- litellm handles provider routing, retries, and rate limits.
- Object counting still grounds object→category via the Stage-2 judge; multi-part
  counting by Collection is a known follow-up.
