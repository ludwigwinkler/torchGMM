# CLAUDE.md — notebooks/

Demo scripts illustrating torchGMM diffusion sampling. Run as plain Python files (not `.ipynb`).

## Conventions

- Import `plt_show` from `_utils` rather than calling `plt.show()` directly — it auto-detects Jupyter vs CLI and closes figures silently when run as a script.
- Set `torch.manual_seed(0)` and `torch.set_default_device(...)` at the top for reproducibility.
- Build GMMs via the public API: `from torchGMM import GMM`, schedules from `torchGMM.schedule`, samplers from `torchGMM.sampling`.
- Tensor shapes follow the repo convention `[*B, K, D]` for GMM params and `[*N, *B, D]` for inputs.
- GIFs and rendered outputs live alongside the scripts (e.g. `steered_diffusion.gif`).

## Files

- `_utils.py` — shared helpers (`plt_show`).
- `steered_sampling.py`, `ve_steering.py` — reward-steered reverse sampling demos.
- `compare_sampling.py` — forward/reverse sampling comparison.
- `create_*_gif.py` — animation generators.
