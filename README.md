# SOC estimation — controlled evaluation of EKF internal diagnostic inputs

Code and derived results for the manuscript

> *Evaluating EKF Internal Diagnostic Inputs for Residual-Learning-Based Battery
> SOC Estimation under Distribution Shift*

This is an **evaluation / audit-style empirical study**. The physical skeleton is a
1RC-ECM + EKF (Joseph form); a compact GRU learns the residual. The study is a
controlled ablation and baseline audit of which EKF-derived internal signals actually
help a residual learner under composite distribution shift.

## Layout

```
06_实验代码/     experiment scripts (Python 3.12) + src/ shared modules
04_实验/数据/
    results/     derived CSV/MD referenced by the manuscript
    splits/      train / validation / test split manifests
```

Script stages: `p2_*` ECM identification & EKF baseline · `p3_*` residual-net
training, ablation, quantization · `p5_*` result tables & robustness · `p7_*`/`p8_*`/`p9_*`
reverse-training, channel-matched and OCV-kernel controls · `p_r1m*` stronger-baseline
(fair-tuned / adaptive) comparisons · `cb2*` / `l1_*` reference-convention analysis ·
`_recalc_*` statistics (paired deltas, permutation tests).

## Data (not included)

All datasets are public; raw data is not committed (size / licence).

- **Panasonic 18650PF** (Kollmeyer et al.), Mendeley Data,
  DOI [10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1) — main dataset.
- **NASA PCoE** battery set (Saha & Goebel, 2007) — zero-shot scope check.

Place the raw files under `04_实验/数据/raw/`; paths resolve relative to the repo root.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
# put the datasets in 04_实验/数据/raw/, then run the p2 → p3 → p5 … stages in order
```

The evaluation protocol (reference-SOC construction, statistics & the 10-seed protocol,
training protocol, symbol conventions) is documented in the manuscript Appendix A;
per-file derived values are under `04_实验/数据/results/`.

## License

MIT — see `LICENSE`.
