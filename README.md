# Radio Map Simulator

Synthetic **spectrum cartography** scenarios: spatial loss fields (SLFs), emitter maps, and frequency-domain coefficients over a 3D grid **I × J × K** (space × space × frequency). Use the outputs to benchmark tensor methods, deep priors, or quantized observation pipelines.

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10335642"><b>IEEE TSP 2023</b></a>
  &nbsp;·&nbsp;
  <a href="https://doi.org/10.1109/LSP.2025.3599714"><b>IEEE SPL 2025</b></a>
</p>

---

## Table of contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Output format](#output-format)
- [Related repositories](#related-repositories)
- [Citation](#citation)

---

## Overview

The simulator builds **multi-emitter** radio maps with **log-normal shadowing** controlled by variance and **decorrelation distance**. For each run you get:

- **`T`** / **`T_true`**: aggregated tensor at the sensors (with optional additive noise via SNR).
- **`S_true`**: per-emitter SLF slices stacked along the emitter dimension.
- **`C_true`**: normalized **spectral** mixing coefficients (columns) across frequency.

Generation is implemented in [`radio_map_utils.py`](radio_map_utils.py) and exposed through:

- **Command line:** [`generate_radio_map.py`](generate_radio_map.py)
- **Web platform:** a small **[Flask](https://flask.palletsprojects.com/)** app in [`main.py`](main.py) with HTML templates under [`templates/`](templates/) for browser-based parameter entry and visualization.

---

## Repository layout

| File / folder | Role |
|---------------|------|
| [`generate_radio_map.py`](generate_radio_map.py) | CLI: single scenario → `.npz` |
| [`run.sh`](run.sh) | Example defaults (one run) |
| [`generate_bulk_radiomap.sh`](generate_bulk_radiomap.sh) | Grid over emitters, shadow variance, decorrelation distance |
| [`radio_map_utils.py`](radio_map_utils.py) | Core `generate_map`, normalization |
| [`test_radio_maps.ipynb`](test_radio_maps.ipynb) | Notebook exploration |
| [`main.py`](main.py), [`templates/`](templates/) | **Flask** web app: interactive forms and map preview in the browser |

---

## Installation

Developed with **Python 3.9+**. Create a virtual environment and install dependencies:

```bash
cd Radio-map-simulator
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Quick example (shell)

```bash
bash run.sh
```

### Parameter grid (27 `.npz` files by default)

Three values each for **emitters**, **shadow-variance**, and **decorrelation-distance** (edit arrays in the script to customize):

```bash
bash generate_bulk_radiomap.sh
```

### Command line (custom scenario)

```bash
python generate_radio_map.py \
  --emitters 6 \
  --shadow-variance 4 \
  --decorrelation-distance 100 \
  --snr 0 \
  --space-x 100 \
  --space-y 100 \
  --bandwidth-length 64 \
  --psd-basis 'g' \
  --save-file 'test.npz' \
  --visualize 'True'
```

### CLI arguments

| Argument | Description |
|----------|-------------|
| `--emitters` | Number of emitters |
| `--shadow-variance` | Shadow variance (log-normal shadowing strength) |
| `--decorrelation-distance` | Decorrelation distance for spatial shadowing |
| `--snr` | SNR in dB for additive Gaussian noise on **`T`** (use `0` for noise-free **T** vs **T_true**) |
| `--space-x`, `--space-y` | Grid size **I**, **J** |
| `--bandwidth-length` | Number of frequency bins **K** |
| `--psd-basis` | PSD shape: **`g`** Gaussian, **`s`** sinc |
| `--save-file` | Output `.npz` path (empty string skips save) |
| `--visualize` | CLI flag (currently the save path does not depend on it; see `generate_radio_map.py`) |

### Web platform (Flask)

The browser UI is served by **[Flask](https://flask.palletsprojects.com/)**. Start the app locally:

```bash
python main.py
```

Open the URL shown in the terminal (default is usually `http://127.0.0.1:5000/`) and adjust parameters in the form. Defaults may use smaller **I**, **J** than the CLI example above.

---

## Output format

Saved **`.npz`** files (from `generate_radio_map.py`) typically contain:

| Key | Shape (conceptually) | Role |
|-----|----------------------|------|
| `T` | I × J × K | Observed / noisy tensor |
| `T_true` | I × J × K | Ground-truth aggregate map |
| `S_true` | I × J × R | Emitter SLFs |
| `C_true` | R × K | Normalized spectral coefficients |

Load in Python with `numpy.load(..., allow_pickle=True)` as needed.

---

## Related repositories

- [**Quantized radio map estimation (BTD + DGM)**](https://github.com/XiaoFuLab/Quantized-Radio-Map-Estimation-BTD-and-DGM) — PyTorch code for quantized spectrum cartography using tensor and deep generative priors ([IEEE TSP 2023](https://ieeexplore.ieee.org/document/10335642)).

---

## Citation

If you use this simulator or the associated methods, please cite the relevant paper(s):

**Quantized radio map estimation (tensor & deep generative models)** — *IEEE Transactions on Signal Processing*, 2023:

```bibtex
@article{timilsina2023quantized,
  title={Quantized radio map estimation using tensor and deep generative models},
  author={Timilsina, Subash and Shrestha, Sagar and Fu, Xiao},
  journal={IEEE Transactions on Signal Processing},
  volume={72},
  pages={173--189},
  year={2023},
  publisher={IEEE}
}
```

**Domain-factored untrained deep prior for spectrum cartography** — *IEEE Signal Processing Letters*, 2025:

```bibtex
@article{timilsina2025domain,
  author={Timilsina, Subash and Shrestha, Sagar and Cheng, Lei and Fu, Xiao},
  journal={IEEE Signal Processing Letters},
  title={Domain-Factored Untrained Deep Prior for Spectrum Cartography},
  year={2025},
  volume={32},
  pages={3440--3444},
  doi={10.1109/LSP.2025.3599714}
}
```

IEEE Xplore: [TSP paper](https://ieeexplore.ieee.org/document/10335642) · [SPL paper (DOI)](https://doi.org/10.1109/LSP.2025.3599714)
