"""SystemOne-shaped adapter over a Fireworks deployment — same answers()
contract as s1_serve.S1Jev, backed by the OpenAI-compatible chat endpoint.

    import s1_fw_serve, jev_loop
    jev_loop.jev = s1_fw_serve.FWJev("accounts/<acct>/models/<id>").answers

Each question is sent as the same two user messages used in training
(ft_fw_dataset.py); the model emits one label token. Label distribution
is read off per-position top_logprobs. Because deployments may cap
top_logprobs at 5, a UNIFORM logit_bias is applied to all 64 label
token-ids — a constant shift preserves their relative probabilities, so
the normalized label distribution is exact even when only the top-k
labels are returned.

Needs only stdlib + label_ids.json (cached tokenizer output, in
results/ft_fw/). Set FIREWORKS_API_KEY or rely on ~/.dsh/.credentials.yaml.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from pathlib import Path

from s1_compile import EVAL_INSTR, option_pairs
from ft_fw_dataset import label_letters, question_suffix

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
LABEL_BIAS = 50.0


def _api_key() -> str:
    if os.environ.get("FIREWORKS_API_KEY"):
        return os.environ["FIREWORKS_API_KEY"]
    import yaml
    creds = yaml.safe_load(open(Path.home() / ".dsh" / ".credentials.yaml"))
    return creds["refs"]["FIREWORKS_API_KEY"]


def _softmax(logprobs: list[float]) -> list[float]:
    peak = max(logprobs)
    ws = [math.exp(v - peak) for v in logprobs]
    tot = math.fsum(ws)
    return [w / tot for w in ws]


def _confidence(probs: list[float]) -> float:
    ent = -math.fsum(p * math.log(p) for p in probs if p > 0)
    return min(1.0, max(0.0, 1 - ent / math.log(len(probs))))


class FWJev:
    """jev(state_dict, questions_dict) -> answers dict — drop-in for
    S1Jev / the OpenJev endpoint."""

    def __init__(self, model_id: str, bias: float = LABEL_BIAS):
        self.model = model_id
        self.key = _api_key()
        ids_path = HERE / "results" / "ft_fw" / "label_ids.json"
        label_ids = json.loads(ids_path.read_text())
        # uniform bias on every possible label id -> order preserved
        self.logit_bias = ({str(t): bias for t in label_ids.values()}
                           if bias else None)
        self.n_calls = 0

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def _label_dist(self, state_str: str, q: dict,
                    keys: list[str]) -> list[float]:
        messages = [
            {"role": "user", "content": state_str},
            {"role": "user", "content": question_suffix(q)},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 64,
            "reasoning_effort": "none",
        }
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias
        resp = self._post(payload)
        choice = resp["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        top = content[0].get("top_logprobs") or [] if content else []
        labels = label_letters(len(keys))
        lp = {t.get("token"): t.get("logprob") for t in top}
        label_lp = [lp.get(l, -30.0) for l in labels]
        return _softmax(label_lp)

    def answers(self, state: dict, questions: dict) -> dict:
        state_str = json.dumps(state)
        self.n_calls += 1
        out = {}
        for qkey, q in questions.items():
            opts = option_pairs(q)
            keys = [k for k, _ in opts]
            probs = self._label_dist(state_str, q, keys)
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
            else:
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
