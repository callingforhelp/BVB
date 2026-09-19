"""SystemOne-shaped adapter over a Tinker sampler.

Reimplements the OpenJev serving contract (prompts.py + scoring.py) on top of
Tinker's sampling API so jev_loop can run against a base model or a
fine-tuned checkpoint with zero code changes:

    import s1_serve, jev_loop
    jev_loop.jev = s1_serve.S1Jev("tinker://.../sampler_weights/final").answers

Each question compiles to an independent branch; one token is sampled at
temperature 0 with topk logprobs; the label-token distribution is
renormalized exactly like openjev scoring.normalize.

Run under SYSTEM python3 (needs tinker).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import tinker
from tinker.types import SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from s1_compile import S1Compiler, option_pairs  # noqa: E402

MODEL = "Qwen/Qwen3.6-35B-A3B"


def _api_key() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    import yaml
    creds = yaml.safe_load(open(Path.home() / ".dsh" / ".credentials.yaml"))
    os.environ["TINKER_API_KEY"] = creds["refs"]["TINKER_API_KEY"]


def _softmax(logprobs: list[float]) -> list[float]:
    peak = max(logprobs)
    ws = [math.exp(v - peak) for v in logprobs]
    tot = math.fsum(ws)
    return [w / tot for w in ws]


def _confidence(probs: list[float]) -> float:
    ent = -math.fsum(p * math.log(p) for p in probs if p > 0)
    return min(1.0, max(0.0, 1 - ent / math.log(len(probs))))


class S1Jev:
    """jev(state_dict, questions_dict) -> answers dict — same shape as the
    TypeSafe/OpenJev /v1/systemone response's `answers` object."""

    def __init__(self, model_path: str | None = None,
                 base_model: str = MODEL):
        _api_key()
        service = tinker.ServiceClient()
        if model_path:
            self.client = service.create_sampling_client(model_path=model_path)
        else:
            self.client = service.create_sampling_client(base_model=base_model)
        self.tok = self.client.get_tokenizer()
        self.compiler = S1Compiler(self.tok)
        self.n_calls = 0

    def answers(self, state: dict, questions: dict) -> dict:
        state_str = json.dumps(state)
        self.n_calls += 1
        out = {}
        # one branch per question — mirrors OpenJev's independent evaluation
        for qkey, q in questions.items():
            ids, label_ids, keys = self.compiler.compile(state_str, q)
            resp = self.client.sample(
                prompt=tinker.ModelInput.from_ints(ids),
                num_samples=1,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                topk_sample_logprobs=min(len(label_ids), 64),
            ).result()
            topk = resp.sequences[0].topk_logprobs
            pairs = topk[0] if topk and topk[0] else []
            lp = {t: v for t, v in pairs}
            label_lp = [lp.get(t, -30.0) for t in label_ids]
            probs = _softmax(label_lp)
            dist = dict(zip(keys, probs))
            qtype = q["type"]
            if qtype == "noul":
                out[qkey] = {"type": "noul", "noul": dist["true"]}
            elif qtype == "choice":
                out[qkey] = {
                    "type": "choice",
                    "choice": max(dist, key=dist.__getitem__),
                    "probabilities": dist,
                    "confidence": _confidence(probs),
                }
            else:  # score
                out[qkey] = {
                    "type": "score",
                    "score": math.fsum(i * p for i, p in enumerate(probs)),
                    "legend": {str(i): d for i, d in
                               enumerate(q["criteria"])},
                    "probabilities": {str(i): p for i, p in
                                      enumerate(probs)},
                    "confidence": _confidence(probs),
                }
        return out
