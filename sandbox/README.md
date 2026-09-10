# BVB Sandbox — Stage-1 Agent Runs

Mini-BVB is the shared harness for programmatic video reconstruction. Stage 1
collects an animated `result.blend` for each source video. Stage 2 renders the
submission and evaluates **Dual VQA (DV)** and **Latent Similarity (LS)**;
their square-root mean is **Overall**. Scoring is decoupled from the agent loop.

```text
Source video → Mini-BVB agent → animated result.blend
                 ↕                       ↓
          Blender sandbox          camera render
                                  ↙             ↘
                              Dual VQA     Latent Similarity
                                  ↘             ↙
                                      Overall
```

The agent runs on the host and drives a fresh Blender container via
`docker exec`. Every configuration uses the same prompt, tools, sandbox, and
per-scene cost ceiling. External asset libraries are disallowed.

## 1. One-time setup

Run these commands from the `sandbox/` directory. Host prerequisites are
Python 3.10+, Docker, and `ffmpeg`/`ffprobe`. The Docker image includes Ubuntu
22.04, Blender 4.2, Python, Xvfb, and FFmpeg.

```bash
docker build -t bvb-sandbox:latest .
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Inside the container, `blender_run --python script.py` wraps
`xvfb-run -a blender --background`.

Prepare the source videos as `../VSI-Bench/<dataset>/<scene>.mp4`; see the
[repository README](../README.md#1-prepare-the-source-videos). The task loader
enumerates the 288 scenes directly from `../eval/test.jsonl`, using scene IDs
and dataset names. It does not expose questions or answers to the agent.

## 2. Reconstruct videos

Set the provider key used by the chosen model, such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`.

```bash
export OPENAI_API_KEY="..."
.venv/bin/python run_agent.py \
  --model gpt-6-astra \
  --reasoning high \
  --output results/run_001 \
  --cost-limit 3.0 \
  --limit 5
```

This is a five-scene smoke run. Remove `--limit 5` for the full 288-scene pool.
The example explicitly selects the paper's Astra `high` configuration; reasoning
settings are passed through LiteLLM to the selected provider.

```text
results/run_001/
  config.json
  blends/<scene>.blend
  agent_meta/<scene>.json
```

Metadata records turns, frames seen, cost, runtime, completion, and stop reason.
Useful options:

- `--scenes <id ...>`: select exact scenes.
- `--resume`: skip scenes whose `.blend` already exists.
- `--shard i/N`: distribute scenes across independent processes.
- `--cost-limit`: Stage-1 spend ceiling per scene; the paper uses $3.
- `--max-turns`: optional safety cap; `0` leaves turns unlimited.
- `--allow-network`: enables container networking; leave it off for the paper protocol.

## 3. Render and score

Render the saved scene cameras on a host with Blender and FFmpeg:

```bash
.venv/bin/python ../eval/batch_render_camera.py \
  --run results/run_001 \
  --num-frames 64 \
  --resume
```

Use `--blender /path/to/blender` or `BLENDER_BIN` if Blender is not on `PATH`.
The host renderer is separate from the Blender installation inside the Stage-1
container. This step writes `camera_renders/<scene>.mp4` and reuses them on
resume; it does not compute scores.

Continue with the [evaluation guide](../eval/README.md) for:

- **DV:** the same spatiotemporal questions on source and render, answered by
  `gpt-5.4-mini` from 16 sampled frames.
- **LS:** frozen V-JEPA 2.1 ViT-G features of 64 sampled frames, measuring
  layout and motion.
- **Overall:** `((sqrt(DV) + sqrt(LS)) / 2) ** 2` on the 0–100 scale.

## The model decides effort

The harness exposes two primitives through fenced blocks:

- `bash`: execute a command inside the Blender sandbox.
- `frames`: request explicit video timestamps, or `count=N` with optional
  `start=` and `end=` bounds.

There is no fixed step or frame budget in the paper protocol. The model decides
how much of the video to inspect and how long to work, under the spend cap.
The model finishes with `DONE` after observing at least one source frame.
`frames_seen` counts images consumed by the model, separately from
`frames_extracted`; a run with `frames_seen=0` is invalid.

Provider routing, retries, and cost accounting use LiteLLM. Record the model,
reasoning setting, harness configuration, and actual cost with each run.
