# umap-interactive
A clear, PyTorch-based reimplementation of UMAP with interactive visualizations and hyperparameter controls for learning and research.

Can run on both cuda and Apple M chips

## Demo

![Demo](asset/demo.gif)

## Installation

```bash
pip install -r requirements.txt
```
## Usage

This repository contains several demo files:
- `demo1_official.py`: The official UMAP implementation for comparison
- `demo1_ours.py`: Our interactive PyTorch implementation
- `demo7_ours.py`: Our latest demo with enhanced interactive controls

Our implementation allows you to interactively control key UMAP parameters in real-time:
- `min_dist`: Controls how tightly points cluster together
- `negative_sampling_rate`: Controls the balance between attractive and repulsive forces
- `n_neighbors`: Controls how many neighbors are considered when building the graph
- Other parameters like learning rate can also be adjusted

Run any demo with:

```bash
python demo7_ours.py  # For the latest version with all features
# or
python demo1_ours.py  # For the basic interactive version
```



