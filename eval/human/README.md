# BVB 盲评：16 份问卷

**24 题池**（每源 8 个），每人随机 **9 题**（每源 3 个）。生成 16 份独立 HTML。

```bash
python eval/human/make_pack.py \
  --out eval/human/packs/wave1 \
  --force \
  --forms 16 \
  --pool-per-source 8 \
  --per-source 3 \
  --seed 20260822
```

## 发什么

每人发 **一个** 文件，不要混：

| 人 | 文件 | ID（已锁定） |
| --- | --- | --- |
| 1 | `survey-r01.html` | r01 |
| … | … | … |
| 16 | `survey-r16.html` | r16 |

**不要发：** `BLIND_MAP.json`、`_clips_cache/`、`assignments.json`

## 收什么

每人保存 JSON 后点提交发邮件，或把 JSON 发回。`pack_id` 会是 `wave1-r01` 等。

## 收齐后

```bash
python eval/human/score_human.py \
  --pack-dir eval/human/packs/wave1 \
  --responses eval/human/packs/wave1/responses
```

`assignments.json` 记录每人抽到了哪 9 题。
