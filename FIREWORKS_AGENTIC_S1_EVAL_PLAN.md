# Agentic S1 evaluation plan

Status: read-only plan prepared; no deployment or paid inference created.

## Objective

Evaluate the completed Agentic S1 LoRA against the frozen 98-clip Flinter
evidence-policy replay before any production serving decision. The evaluation
must measure next-evidence action quality, not merely training loss.

The promotion gate remains:

- at least 95/98 detection;
- at least 94/98 typing;
- zero clean false positives;
- fewer than 437 judge calls;
- valid actions and stable candidate-order behavior;
- no material degradation under timeline translation or label permutation.

## Verified artifact and account state

- Account: `dave-z-d5jskgf9ohx3`
- Tuned model: `accounts/dave-z-d5jskgf9ohx3/models/jev-agentic-s1-bbe6341d88b3`
- Tuned model state: `READY`
- Training job: `jev-agentic-s1-bbe6341d88b3`, `JOB_STATE_COMPLETED`, 100%, `OK`
- Reported training cost: `$1.954854012`
- Existing deployments at preflight: zero
- Matched tuned-model serving shapes: five
- Matched multi-LoRA/addon shapes: zero

Because no addon-compatible shape was returned, the evaluation will use
sequential live-merge deployments: one base-model evaluation deployment and
one tuned-model evaluation deployment. They will never be active together.

## Evaluation path

Use Fireworks preemptible on-demand capacity when the account accepts it:

- one replica;
- one `NVIDIA_B200_180GB` GPU;
- BF16, one-GPU validated shape;
- `minReplicaCount=1`, `maxReplicaCount=1`;
- preemptible evaluation-only deployment;
- explicit expiry or immediate deletion after each arm;
- real HTTP 200 smoke request before replay.

The read-only shape match returned the validated BF16 one-B200 shape version:

`accounts/fireworks/deploymentShapes/rft-qwen3p6-35b-a3b-rl-b200-bf16-w1-p1/versions/a35zwd97`

The same validated shape is compatible with the base model and the live-merge
tuned model. The current `firectl` is 1.8.8, which supports the documented
preemptible flag.

## Cost estimate

Current published B200 on-demand price: `$0.217/minute` (`$13/hour`). This is
the conservative active-GPU equivalent used for the evaluation envelope; the
Fireworks preemptible documentation says preemptible capacity does not charge
to hold dedicated capacity, but it can be reclaimed without warning. Actual
preemptible billing must be confirmed by the account's usage report.

Planned evaluation envelope:

| Line | Maximum active time | Conservative equivalent |
|---|---:|---:|
| Base-model eval deployment | 30 minutes | $6.51 |
| Tuned-model eval deployment | 30 minutes | $6.51 |
| Total evaluation envelope | 60 GPU-minutes | **$13.02** |

The current package contains 532 exported decision states (460 train-side
states plus 72 held-out states). The actual replay will count every inference
request and report it; the GPU-minute envelope, not token count, is the cost
driver for this on-demand path. Smoke requests and one bounded retry are
included in the 30-minute arm windows. A standard non-preemptible deployment,
longer runtime, a different shape, or a second replay requires a revised cost
plan and approval.

## Exact sequence after approval

1. GET-only collision check for two deployment IDs:
   - `jev-agentic-s1-bbe6341d88b3-base-eval`;
   - `jev-agentic-s1-bbe6341d88b3-tuned-eval`.
2. Create the base-model preemptible deployment with the validated BF16 shape.
3. Wait for READY and send a real request; stop if the response is not HTTP 200.
4. Run the 98-clip replay against the base model and save raw/normalized
   distributions, request counts, latency, and errors.
5. Delete the base deployment and verify it is gone/inactive.
6. Create the tuned live-merge preemptible deployment with the same shape.
7. Wait for READY and send a real request; confirm the response is from the
   tuned model path, not the base path.
8. Run the identical replay against the tuned model.
9. Delete the tuned deployment and verify no active replicas remain.
10. Compare base, tuned, and deterministic-policy results on the same states.

## No-go conditions

Stop without fallback deployment if:

- no validated shape can serve the exact model;
- the account does not accept preemptible evaluation capacity;
- serving returns the base model or an unrecognized model path;
- logprob/answer-token behavior is unavailable or unstable;
- the deployment exceeds the 30-minute arm window;
- a second deployment would be required concurrently;
- the candidate-order or physical-frame-ID contract is violated.

Do not automatically switch to standard on-demand capacity. That requires a
new estimate and explicit approval.

## Sources

- https://docs.fireworks.ai/fine-tuning/evaluating-fine-tuned-models
- https://docs.fireworks.ai/fine-tuning/deploying-loras
- https://docs.fireworks.ai/api-reference/match-deployment-shape-versions
- https://docs.fireworks.ai/api-reference/create-deployment
- https://fireworks.ai/pricing
