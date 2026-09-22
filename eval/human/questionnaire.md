# BVB ranking questionnaire (`bvb-human-rank-v1`)

9 questions. One scene each. Watch the original, then rank n anonymous
reconstructions from most to least similar to the original.

## Instruction

One room per question. Watch the reference, then rank candidates A/B/C… from
most to least similar. Do not guess the model. Rank spatial match, not
photorealism: objects (presence and counts), layout (left/right, near/far,
adjacency, facing), and camera motion (path, gaze, and the order objects
enter the frame).

The HTML form ships the same text in English and Chinese for raters.

## What the rater returns

One JSON file, no comments required:

```json
{
  "instrument_id": "bvb-human-rank-v1",
  "pack_id": "wave1",
  "rater_id": "r01",
  "rankings": [
    {"scene_id": "41069025", "source": "arkitscenes", "rank": ["C", "A", "B", "D", "E"]}
  ]
}
```

`rank[0]` is the best match. Letter codes are unblinded only by the
experimenter with `BLIND_MAP.json`.
