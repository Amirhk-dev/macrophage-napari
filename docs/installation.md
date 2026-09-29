# Installation

## Requirements

- Python ≥ 3.10
- A working Qt binding (napari installs one via `napari[all]`)

## From source

The project can be installed with any of the standard tools.

### With `uv` (recommended)

```bash
git clone https://github.com/Amirhk-dev/napari-macrophage.git
cd napari-macrophage
uv sync                        # core dependencies
uv sync --extra detection      # + torch + onnxruntime for ONNX detection
```

### With `conda`

```bash
conda create -n mic_napari python=3.11 -y
conda activate mic_napari
pip install -e .
```

### With `venv`

```bash
python -m venv .mic_napari
source .mic_napari/bin/activate
pip install -e .
```

## Optional extras

| Extra | Contents | When you need it |
|-------|----------|------------------|
| `detection` | `torch`, `onnxruntime` | Running the ONNX detector on CD206 + DAPI |
| `test` | `pytest`, `pytest-qt` | Running the unit-test suite |
| `dev` | `ruff` | Linting the codebase |
| `docs` | `sphinx`, `furo`, `myst-parser` | Building this documentation |

Install any extra with `pip install -e ".[extra]"`, or combine them:

```bash
pip install -e ".[detection,test,dev,docs]"
```

## Launching napari with the plugin

```bash
napari
```

The plugin's commands appear under **Plugins → napari-macrophage** in the
napari menu bar.

### Launcher command (skip the menu clicks)

Installing the package also registers a `napari-macrophage` console script
that starts napari with the two most-used docks — **Load Image & Mask** and
**Macrophage Tools** — already docked side-by-side on the right:

```bash
napari-macrophage
```

The plain `napari` command is unaffected — use it when you want a clean
viewer without the plugin's docks pre-mounted.
