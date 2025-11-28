<div align="center">
<img src="./docs/OrionVis_Minimal_Logo.png" alt="OrionVis" width="20%">

# OrionVis

A PyTorch‑friendly Computer Vision toolkit for unified backbone loading, feature extraction, and registered preprocessing.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#)
[![Issues](https://img.shields.io/github/issues/bluemoon-o2/orionvis?label=issues&logo=github)](https://github.com/bluemoon-o2/orionvis/issues)
</div>

## Overview
OrionVis streamlines common CV workflows in PyTorch: load backbones with a single call, manage pretrained weights via YAML with lazy indexing, and apply registered transforms for classification inference. It consolidates model, weight, and preprocessing logic while staying lightweight and production‑ready.

## Features
- Unified backbone loader: `resnet*`, `dino*`, `mobilemamba_*`.
- YAML‑based weights with cached index and lazy loading.
- Transform registry (e.g., `ImageClassification`) for ready‑to‑use preprocessing.
- Feature extraction helpers and utilities.

<img src="./docs/OrionVis_Banner.png" alt="OrionVis_Banner">

## Installation
```bash
pip install orionvis
```
For source code installation:
```bash
git clone https://github.com/bluemoon-o2/orionvis
cd orionvis
pip install -e .
```

## Quickstart
```python
import orionvis

# Load a backbone with pretrained weights
model = orionvis.load("mobilemamba_b1s")

# Retrieve weight metadata and preprocessing transform
weight = orionvis.get_weights("mobilemamba_b1s")
transform = orionvis.get_transform("mobilemamba_b1s")

# Example: preprocess an input PIL image
# img = transform(img)
# logits = model(img.unsqueeze(0))
```

## OrionVis Hub

OrionVis Hub allows you to load models and entrypoints directly from GitHub repositories or local directories.

### Load Model

Load a model or entrypoint from a GitHub repository:

```python
import orionvis.hub as hub

# Load from GitHub
model = hub.load("pytorch/vision:v0.10.0", "resnet18")

# Load from a local directory
# model = hub.load("/path/to/local/repo", "custom_model", source="local")
```

### List Entrypoints

List all available entrypoints in a repository:

```python
import orionvis.hub as hub

entrypoints = hub.entrypoints("pytorch/vision:v0.10.0")
print(entrypoints)
```

### View Documentation

View the docstring of a specific entrypoint:

```python
import orionvis.hub as hub

doc = hub.docs("pytorch/vision:v0.10.0", "resnet18")
print(doc)
```

### Load State Dict

Download and load a state dictionary from a URL:

```python
import orionvis.hub as hub

state_dict = hub.load_state_dict_from_url(
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    check_hash=True
)
```

## Models
- Programmatic list: `orionvis.list_models()`
- Included backbones: `resnext*`, `wide_resnet*`, `dinov2_*`, `dinov3_*`, `mobilemamba_*`.

## Weights
- Config path: `orionvis/configs/weights/mobilemamba.yaml` (multi‑model YAML; one class per document).
- Download path cache: `~/.cache/orionvis` (configurable via `ORIONVIS_CACHE`).

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Citation
```
@article{mobilemamba,
  title={MobileMamba: Lightweight Multi-Receptive Visual Mamba Network},
  author={Haoyang He and Jiangning Zhang and Yuxuan Cai and Hongxu Chen and Xiaobin Hu and Zhenye Gan and Yabiao Wang and Chengjie Wang and Yunsheng Wu and Lei Xie},
  journal={arXiv preprint arXiv:2411.15941},
  year={2024}
}
```

## References
- [torchvision](https://github.com/pytorch/vision)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [MobileMamba](https://github.com/lewandofskee/MobileMamba)
