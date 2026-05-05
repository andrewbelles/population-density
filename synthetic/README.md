# Synthetic Experiments

This module contains standalone synthetic experiments for graph-spectral residual correction.

## Rare-Instance Bags

Run:

```bash
python -m synthetic.rare_instance --config configs/synthetic/rare_instance.yaml
```

The experiment generates bag-level labels with a strong baseline and a residual that is smooth on a hidden latent graph. Each bag contains mostly noisy background instances, rare positive/negative signal instances, and rare high-magnitude distractors. The goal is to test whether pooled bag representations can recover enough latent structure to construct a graph whose MEM basis improves residual correction.

## Fair-Comparison Contract

- Graphs are constructed transductively from unlabeled bag features.
- Supervised residual correction is fit only on training labels inside each split.
- The learned graph is single-modality: `learned_graph_pool` selects one pooled bag representation, defaulting to `meanmax`, and trains a subsampled Barlow encoder before graph construction.
- All methods use the same outer splits, ridge grid, graph `k`, and MEM dimension unless explicitly changed by CLI flags.
- Oracle latent and oracle-signal methods are upper bounds, not valid deployable methods.

## Outputs

Default artifacts are written to `synthetic/artifacts/rare_instance`.

Default figures are written to `synthetic/images/rare_instance`; these images are intentionally not ignored by git.

Important outputs:

- `summary.csv`: method-level residual correction summary.
- `fold_metrics.csv`: per-fold metrics for each method.
- `graph_diagnostics.csv`: graph recovery and residual-energy diagnostics.
- `learned_graph_diagnostics.csv`: learned single-modality Barlow graph metadata.
- stdout method summary: compact performance table for the most useful comparisons.
- stdout graph diagnostics: compact graph recovery and MEM residual-energy table.
- `latent_signal.png`: latent residual and rare-signal surface.
- `trial_method_error.png`: per-trial residual error by method.
- `mem_graph_error.png`: per-trial residual error for MEM-only models by graph basis.
- `graph_edges_*.png`: latent-space edge maps for selected graphs.
