# BVB ranking questionnaire (`bvb-human-rank-v1`)

9 questions. One scene each. Watch the original, then rank n anonymous
reconstructions from most to least similar to the original.

## Instruction

**中文.** 每题一个房间。先看原视频，再看标成 A/B/C… 的重建。从最像原视频的排到最不像的。不要猜模型，也不要按好不好看排。主要看三件事：物体（有什么、
数量对不对），布局（左右远近、谁挨着谁、东西朝哪），镜头走位（相机怎么走、往哪看，以及物体进入画面的先后）。

**English.** One room per question. Watch the reference, then rank candidates
A/B/C… from most to least similar. Do not guess the model. Rank spatial match,
not photorealism.

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
