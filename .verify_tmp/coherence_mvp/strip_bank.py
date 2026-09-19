"""Strip-episode reasoning bank for the Jev loop (Phase 2).

Follows the dsh-pes-reasoning.v1 conventions: bounded projections only
(no frames/pixels/prompts/CoT — signal scalars + verdict fields only),
idempotent hash ids, FIFO cap, schema version, cue/lesson split,
acceptance gating (only clips with a verdict enter as 'candidate',
manifest truth upgrades them to 'verified').

The embedding backend is a self-anchored perceptual descriptor (cv2):
per-frame HSV/edge features + cross-seam discontinuity features +
signal context. Swappable for Octen/CLIP later — the bank schema and
retrieval contract don't change.

LOSO discipline: banks are built per held-out source so retrieval never
sees same-source episodes at eval time.

Files (per bank dir):
  episodes.jsonl   one entry per line (bounded projection)
  embeddings.npy   L2-normalized descriptors, aligned with index.json rows
  index.json       {"ids": [...], "dim": int}
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus_v3"

SCHEMA_VERSION = "jev-strip-bank.v1"
MAX_ENTRIES = 4096
MAX_CUE = 240
MAX_LESSON = 320
MAX_EVENT_IDS = 32
TOPK_DEFAULT = 5

manifest = json.loads((CORPUS / "manifest.json").read_text())
CLIPS = {c["id"]: c for c in manifest["clips"]}


def _bounded(text, n):
    t = str(text or "")
    return t if len(t) <= n else t[: n - 3] + "..."


def stable_id(clip_id: str, frame: int, gate: str) -> str:
    h = hashlib.sha256(f"{clip_id}:{frame}:{gate}".encode()).hexdigest()
    return f"rb_{h[:24]}"


# ---------------------------------------------------------------- descriptor

def extract_strip_frames(mp4: Path, i: int) -> list[np.ndarray]:
    """4 raw frames around seam i|i+1 via ffmpeg (BGR np arrays, 480w)."""
    idxs = [max(0, min(899, t)) for t in (i - 1, i, i + 1, i + 2)]
    sel = "+".join(f"eq(n,{t})" for t in idxs)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", f"select='{sel}',scale=480:-1",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-"],
        capture_output=True, check=True)
    out, buf = [], p.stdout
    while True:
        s = buf.find(b"\xff\xd8")
        if s < 0:
            break
        e = buf.find(b"\xff\xd9", s)
        if e < 0:
            break
        arr = np.frombuffer(buf[s:e + 2], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
        buf = buf[e + 2:]
    return out


def _frame_feat(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    h = h.flatten() / max(h.sum(), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return np.concatenate([
        h, [edges.mean() / 255.0, gray.mean() / 255.0, gray.std() / 128.0]])


FRAME_DIMS = 4 * 67          # _frame_feat: 64 hist + 3 scalars, ×4 frames
SEAM_DIMS = 4
CTX_DIMS = 8
DESC_DIM = FRAME_DIMS + SEAM_DIMS + CTX_DIMS
# comparison weights: seam-diff channels dominate (they carry the
# discontinuity signal), signal-ctx second, appearance last — else the
# 268 appearance dims flood top-k with benign look-alikes.
DESC_WEIGHTS = np.concatenate([
    np.ones(FRAME_DIMS), np.full(SEAM_DIMS, 3.0), np.full(CTX_DIMS, 2.0)
]).astype(np.float32)


def strip_descriptor(frames: list[np.ndarray],
                     signal_ctx: dict | None = None) -> np.ndarray:
    """~280-dim raw descriptor (weighted cosine applied at retrieval)."""
    if len(frames) < 4:
        raise ValueError(f"strip needs 4 frames, got {len(frames)}")
    feats = [_frame_feat(f) for f in frames[:4]]
    a2, b1 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY).astype(np.float32), \
             cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    a1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    b2 = cv2.cvtColor(frames[3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    seam = np.array([
        np.abs(a2 - b1).mean() / 255.0,          # across-seam diff
        np.abs(a1 - a2).mean() / 255.0,          # within-A motion
        np.abs(b1 - b2).mean() / 255.0,          # within-B motion
        np.abs(a1 - b2).mean() / 255.0,          # outer-pair diff
    ])
    ctx = np.zeros(CTX_DIMS, dtype=np.float32)
    if signal_ctx:
        ctx = np.array([
            min(signal_ctx.get("recur", 0), 1.5) / 1.5,
            1.0 if signal_ctx.get("dup") else 0.0,
            np.log1p(min(signal_ctx.get("echo", 0), 1e6)) / 14.0,
            min(signal_ctx.get("pj_z", 0), 60.0) / 60.0,
            *(1.0 if signal_ctx.get("gate") == g else 0.0
              for g in ("pixel_gate", "scenecut", "wide_gate")),
            min(signal_ctx.get("frame", 0), 900) / 900.0,
        ], dtype=np.float32)
    return np.concatenate([*feats, seam, ctx]).astype(np.float32)


# ---------------------------------------------------------------- entries

def make_entry(clip_id: str, frame: int, gate: str, verdict: dict | None,
               clip_op: str | None, is_true_seam: bool | None,
               signal_ctx: dict, run_ordinal: int = 0,
               producer: str = "") -> dict:
    """Bounded projection — mirrors reasoningEntryFor(record, result).

    clip_op = the CLIP's corruption op (context); is_true_seam = this
    boundary sits within +-8 frames of a planted break. Both matter:
    'similar boundaries in swap clips' answers typing, 'similar seams
    were confirmed/discontinuous' answers the continuity prior."""
    v = verdict or {}
    status = "verified" if clip_op else "candidate"
    disc = v.get("continuous") is False
    cue = _bounded(
        f"{gate} seam@{frame} in {clip_id.split('__')[0]} "
        f"{'DISCONTINUOUS' if disc else 'continuous'} "
        f"break_kind={v.get('break_kind')} same_scene={v.get('same_scene')}",
        MAX_CUE)
    lesson = _bounded(
        f"verdict={v.get('break_kind') or ('discontinuous' if disc else 'continuous')} "
        f"conf={v.get('confidence')} clip_op={clip_op or 'unknown'} "
        f"true_seam={is_true_seam} — "
        f"{'planted-break boundary' if is_true_seam else 'context boundary'}",
        MAX_LESSON)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "id": stable_id(clip_id, frame, gate),
        "traceKind": "strip_judgment",
        "producerSha": producer,
        "runOrdinal": run_ordinal,
        "clipId": clip_id,
        "source": CLIPS.get(clip_id, {}).get("source", clip_id.split("__")[0]),
        "frame": int(frame),
        "gate": gate,
        "signalContext": {k: (round(float(x), 4) if isinstance(x, (int, float))
                              else x) for k, x in signal_ctx.items()},
        "verdict": {k: v.get(k) for k in
                    ("continuous", "same_scene", "break_kind", "confidence")},
        "clip_op": clip_op,
        "is_true_seam": is_true_seam,
        "status": status,
        "cue": cue,
        "lesson": lesson,
        "evidenceIds": [f"{clip_id}:{frame}"],
    }


# ---------------------------------------------------------------- bank

class StripBank:
    """Insertion-ordered episode store with cosine retrieval."""

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.entries: dict[str, dict] = {}
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []

    def ingest(self, entry: dict, vec: np.ndarray) -> dict:
        eid = entry["id"]
        if eid in self.entries:
            return self.entries[eid]            # idempotent
        if len(self._ids) >= self.max_entries:
            old = self._ids.pop(0)
            self._vecs.pop(0)
            self.entries.pop(old, None)
        self.entries[eid] = entry
        self._ids.append(eid)
        self._vecs.append(vec.astype(np.float32))
        return entry

    def retrieve(self, vec: np.ndarray, k: int = TOPK_DEFAULT,
                 exclude_source: str | None = None) -> list[tuple[dict, float]]:
        if not self._vecs:
            return []
        M = np.stack(self._vecs)
        w = DESC_WEIGHTS[: vec.shape[0]]
        num = (M * w) @ vec
        den = np.sqrt((M * M * w).sum(1) * (vec * vec * w).sum()) + 1e-9
        sims = num / den
        order = np.argsort(-sims)
        out = []
        for i in order:
            e = self.entries[self._ids[int(i)]]
            if exclude_source and e["source"] == exclude_source:
                continue
            out.append((e, float(sims[int(i)])))
            if len(out) >= k:
                break
        return out

    def precedent_summary(self, vec: np.ndarray, k: int = TOPK_DEFAULT,
                          exclude_source: str | None = None) -> dict:
        hits = self.retrieve(vec, k=k, exclude_source=exclude_source)
        if not hits:
            return {"n": 0, "summary": "no similar past boundaries"}
        disc = [e for e, s in hits if e["verdict"].get("continuous") is False]
        seams = [e for e, s in hits if e.get("is_true_seam")]
        ops = {}
        for e, s in hits:
            if e.get("clip_op"):
                ops[e["clip_op"]] = ops.get(e["clip_op"], 0) + 1
        top = ", ".join(f"{t}×{n}" for t, n in
                        sorted(ops.items(), key=lambda x: -x[1])) or "unlabeled"
        seam_ops = {}
        for e, s in hits:
            if e.get("is_true_seam") and e.get("clip_op"):
                seam_ops[e["clip_op"]] = seam_ops.get(e["clip_op"], 0) + 1
        seam_txt = (", ".join(f"{t}×{n}" for t, n in
                              sorted(seam_ops.items(), key=lambda x: -x[1]))
                    or "none")
        summary = (f"boundaries like this appeared mostly in: {top} "
                   f"(planted-seam hits: {seam_txt}); "
                   f"{len(disc)}/{len(hits)} judged discontinuous")
        return {"n": len(hits), "n_discontinuous": len(disc),
                "n_true_seams": len(seams), "clip_ops": ops,
                "seam_ops": seam_ops, "summary": summary}

    # ---- persistence ----
    def save(self, bank_dir) -> None:
        d = Path(bank_dir)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "episodes.jsonl").open("w") as f:
            for eid in self._ids:
                f.write(json.dumps(self.entries[eid]) + "\n")
        np.save(d / "embeddings.npy",
                np.stack(self._vecs) if self._vecs
                else np.zeros((0, 169), dtype=np.float32))
        (d / "index.json").write_text(json.dumps({"ids": self._ids}))

    @classmethod
    def load(cls, bank_dir) -> "StripBank":
        d = Path(bank_dir)
        bank = cls()
        idx = json.loads((d / "index.json").read_text())
        vecs = np.load(d / "embeddings.npy")
        entries = {}
        with (d / "episodes.jsonl").open() as f:
            for line in f:
                e = json.loads(line)
                entries[e["id"]] = e
        for i, eid in enumerate(idx["ids"]):
            if eid in entries and i < len(vecs):
                bank.entries[eid] = entries[eid]
                bank._ids.append(eid)
                bank._vecs.append(vecs[i])
        return bank


# ---------------------------------------------------------------- populate

def signal_context_for(clip_id: str, frame: int, gate: str) -> dict:
    st_dir = ROOT / "results"
    sig = np.load(st_dir / "signals_v3" / f"{clip_id}.npz")
    pj = sig["photo_jump"].astype(np.float64)
    recur = sig["recur_frac"].astype(np.float64)
    dup = np.load(st_dir / "dupfrac" / f"{clip_id}.npz")["dup"]
    echo = json.loads((st_dir / "echo" / f"{clip_id}.json").read_text())
    return {"recur": float(recur[max(0, frame - 5):frame + 5].max()),
            "dup": bool(dup[min(frame, len(dup) - 1)] > 0.2),
            "echo": float(echo["best_score"]),
            "pj_z": float(pj[min(frame, len(pj) - 1)]),
            "gate": gate, "frame": frame}


def build_bank(sources: set[str], bank_dir) -> tuple[StripBank, int]:
    """Ingest every cached judged candidate whose clip source is in `sources`."""
    bank = StripBank()
    n_skipped = 0
    for cid, c in CLIPS.items():
        if c["source"] not in sources:
            continue
        mp4 = CORPUS / c["path"]
        seam_frames = {int(round(s * 30)) for s in c.get("breaks_s", [])}
        for dname, gate in (("pairjudge", "pixel_gate"),
                            ("pairjudge_scenecut", "scenecut"),
                            ("pairjudge_wide", "wide_gate")):
            f = ROOT / "results" / dname / f"{cid}.json"
            if not f.exists():
                continue
            for cand in json.loads(f.read_text()).get("candidates", []):
                if cand.get("verdict") is None:
                    continue
                frame = int(cand["frame"])
                is_seam = any(abs(frame - sf) <= 8 for sf in seam_frames) \
                    if seam_frames else False
                try:
                    frames = extract_strip_frames(mp4, frame)
                    ctx = signal_context_for(cid, frame, gate)
                    vec = strip_descriptor(frames, ctx)
                except Exception:
                    n_skipped += 1
                    continue
                entry = make_entry(cid, frame, gate, cand["verdict"],
                                   clip_op=c["operator"],
                                   is_true_seam=is_seam, signal_ctx=ctx)
                bank.ingest(entry, vec)
    bank.save(bank_dir)
    return bank, n_skipped
