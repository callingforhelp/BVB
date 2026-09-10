# Repository figures and results

The main README uses tracked copies of manuscript figures so they render in a
fresh GitHub checkout without the local, ignored `misc/` manuscript directory.

| File | Content |
| --- | --- |
| [bvb-logo.png](bvb-logo.png) | BVB logo |
| [bvb-cost-frontier.png](bvb-cost-frontier.png) | Current two-axis Overall versus Stage-1 cost, including Astra and Opus 5 |
| [bvb-pipeline.png](bvb-pipeline.png) | Reconstruction pipeline with Dual VQA and Latent Similarity |
| [bvb-results.csv](bvb-results.csv) | Full-precision snapshot of all 46 manuscript configurations |

The figure snapshots were refreshed from the manuscript on September 10, 2026.
The results CSV matches the project page's download. Scores use a 0–100 scale;
cost is mean Stage-1 USD per scene. Overall combines DV and LS as their
square-root mean. See [the evaluation guide](../eval/README.md) for definitions
and coverage requirements, and the [interactive leaderboard](https://yoloytang.me/BVB/#leaderboard)
for filtering and plots.

`bvb-teaser-frontier.png`, `bvb-diagnostics.png`, `teaser.png`, and `metrics.png`
are older assets retained for historical references. They are not the current
leaderboard or metric specification.
