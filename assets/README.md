# Repository figures and results

The main README uses tracked copies of manuscript figures so they render in a
fresh GitHub checkout without the local, ignored `misc/` manuscript directory.

| File | Content |
| --- | --- |
| [bvb-logo.png](bvb-logo.png) | BVB logo |
| [bvb-figure1.png](bvb-figure1.png) | Complete Figure 1: benchmark overview and Overall cost frontier across 51 configurations |
| [bvb-cost-frontier.png](bvb-cost-frontier.png) | Standalone Overall versus Stage-1 cost plot |
| [bvb-pipeline.png](bvb-pipeline.png) | Reconstruction pipeline with Dual VQA and Latent Similarity |
| [bvb-results.csv](bvb-results.csv) | Full-precision snapshot of all 51 manuscript configurations |

The complete Figure 1 was exported from the arXiv manuscript figure on September 14, 2026,
including both the benchmark overview and the cost frontier.
The results CSV matches the project page's download. Scores use a 0–100 scale;
cost is mean Stage-1 USD per scene. Overall combines DV and LS as their
square-root mean. See [the evaluation guide](../eval/README.md) for definitions
and coverage requirements, and the [interactive leaderboard](https://yoloytang.me/BVB/#leaderboard)
for filtering and plots.

Earlier teaser, diagnostics, metrics, and curation illustrations have been
retired from the active directory. They remain available in Git history or a
local `_archive/` copy. Use the figures listed above for current results.
