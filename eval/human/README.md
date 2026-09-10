# BVB 盲评

论文分析了 **15 位评审者**的回答。每人对 **9 个场景**中的 **5 个匿名重建**
排序，场景来自 **24 题池**（ARKitScenes、ScanNet、ScanNet++ 各 8 个），
每人从每个来源抽取 3 题。此前生成了 16 份问卷；生成数量不等于有效评审人数。

论文中的五个配置是 GPT-5.6 Sol xhigh、Grok-4.5 high、Gemini 3.1 Pro high、
Claude Opus 4.6 high 和 Qwen3.5-397B-A17B high。Astra 和 Opus 5 未参加该轮盲评。
具体设计见 [PROTOCOL.md](PROTOCOL.md)。以下命令均从仓库根目录运行。

## 生成问卷

需要 FFmpeg，以及所选模型完整的相机渲染和原始视频。生成器使用 Python 标准库。
使用新的输出目录保存一轮问卷；只有明确需要重建已有问卷时才加 `--force`。

```bash
python eval/human/make_pack.py \
  --out eval/human/packs/wave2 \
  --forms 16 \
  --pool-per-source 8 \
  --per-source 3 \
  --seed 20260822
```

生成 `survey-r01.html` 至 `survey-r16.html`。每份都是包含视频的独立 HTML，
每人只发对应的一份。`assignments.json` 记录抽题结果。
`BLIND_MAP.json`、`assignments.json` 和 `_clips_cache/` 留在实验端，不发给评审者。

每人完成排序后保存并返回 JSON；`pack_id` 用于对应问卷。
原始回答必须与生成它们的 `BLIND_MAP.json` 配套保存。新增模型需要生成新问卷。

## 汇总返回结果

把收到的 JSON 放入对应轮次的 `responses/`，再运行：

```bash
python eval/human/score_human.py \
  --pack-dir eval/human/packs/wave2 \
  --responses eval/human/packs/wave2/responses \
  --out eval/human/packs/wave2/scored-dv-ls
```

对已有轮次分析时替换为它自己的目录。脚本不会修改原始回答。
计算自动指标关联时，还需要在 `sandbox/results/<run>/` 下准备该轮模型的
`dual_vqa.jsonl` 和 `vision_sim.jsonl`。

| 输出 | 内容 |
| --- | --- |
| `human_mean_rank.csv` | 平均名次（越低越好）、第一名比例 |
| `human_pairwise.csv` | 从排序推导的两两胜率 |
| `human_by_scene_model.csv` | 每个场景、模型的平均人类排名及自动指标 |
| `human_metric_correlation.csv` | 自动指标与负平均名次的场景—模型级 Spearman 相关 |
| `summary.json` | 人数、覆盖场景、均值、相关系数及 Overall 口径 |

当前 **Overall 仅由 DV 和 LS 计算**：`((sqrt(DV) + sqrt(LS)) / 2) ** 2`。
脚本保存 0–1 分数，LS 的负余弦值在开平方前截为零。
旧 Scene Test 列仍可用于诊断，但不参与 Overall；缺少 Scene Test 文件不会阻止计算。
缺少 DV 或 LS 时 Overall 留空。某场景没有 source-correct 问题时，其 DV 也留空。

该脚本的相关系数以**场景—模型**为单位。论文另行比较了五个配置的整体排序；
那一项 Overall 的 ρ = 1.00 不能与这里的场景—模型相关系数混用。
论文的 LS 场景—模型相关系数为 ρ = 0.83。旧版三轴 Overall 输出如需更新，
请保存到新的分析目录，保留原始回答和历史结果。
