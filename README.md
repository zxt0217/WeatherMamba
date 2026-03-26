# WeatherMamba_Pro
[![DOI](https://zenodo.org/badge/1192359755.svg)](https://doi.org/10.5281/zenodo.19231094)
Note: This repository contains the official implementation of our manuscript titled "WeatherMamba: Robust LiDAR Point Cloud Segmentation for Autonomous Driving under Adverse Weather Conditions", which is currently submitted to The Visual Computer. If you find this code or our research helpful, we kindly urge you to cite the associated manuscript (see Citation below).
Official PyTorch implementation for weather-aware LiDAR point cloud segmentation.

## Table of Contents

- [Highlights](#highlights)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Outputs](#outputs)

## Highlights

- Mamba-based architecture for point cloud segmentation.
- Weather-aware workflow with configurable training and augmentation.
- Config-driven pipeline with CLI overrides for fast experiments.

## Project Structure

```text
.
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── scripts/
│   └── train.py
├── weathermamba/
│   ├── __init__.py
│   ├── cli/
│   │   ├── train.py
│   │   └── __init__.py
│   ├── configs/
│   │   ├── data.yaml
│   │   ├── model.yaml
│   │   ├── train.yaml
│   │   └── __init__.py
│   ├── data/
│   │   ├── augmentation.py
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── __init__.py
│   ├── models/
│   │   ├── mamba_mock.py
│   │   ├── weather_mamba.py
│   │   └── __init__.py
│   └── utils/
│       ├── config.py
│       ├── runtime.py
│       └── __init__.py
└── requirements.txt
```

## Installation

1. Clone the repository.

```bash
git clone https://github.com/<your-org>/WeatherMamba_Pro.git
cd WeatherMamba_Pro
```

2. Create and activate environment.

```bash
conda create -n weathermamba python=3.8 -y
conda activate weathermamba
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Set your dataset root in `configs/data.yaml`:

```yaml
dataset_path: ""
train_split: train
val_split: val
```

Expected structure:

```text
<dataset_root>/
├── train/
│   └── ...
└── val/
    └── ...
```

Supported input formats:

- `.bin` point clouds.
- `.txt` point clouds.

## Training

Basic run:

```bash
python scripts/train.py --dataset-path ./your_dataset_root
```

Dry run:

```bash
python scripts/train.py --dataset-path ./your_dataset_root --dry-run
```

Override common settings:

```bash
python scripts/train.py --dataset-path ./your_dataset_root --epochs 100 --batch-size 4 --lr 5e-4 --hidden-dim 512 --stage-depths 3,3,4
```

## Hyperparameter Tuning

Main tuning files:

- `configs/train.yaml`: `epochs`, `lr`, `weight_decay`, `amp`, `grad_clip`.
- `configs/data.yaml`: `num_points`, `loading.batch_size`, `loading.num_workers`, `augmentation`.
- `configs/model.yaml`: `hidden_dim`, `stage_depths`, `dropout`, neighborhood sizes.

## Outputs

Training outputs are saved to:

```text
outputs/weathermamba_pro/<run_name>/
├── checkpoints/
├── model_resolved.yaml
├── data_resolved.yaml
└── train_resolved.yaml
```
##  Citation

If you find WeatherMamba useful for your research, please cite our work:

```bibtex
@article{zxt2026weathermamba,
  title={WeatherMamba: Robust LiDAR Point Cloud Segmentation for Autonomous Driving under Adverse Weather Conditions},
  author={Your Name and Others},
  journal={The Visual Computer},
  year={2026},
  note={Under Review},
  url={[https://github.com/zxt0217/WeatherMamba](https://github.com/zxt0217/WeatherMamba)}
}

### Data Acknowledgements

We would like to thank the authors of the following datasets and codebases:

@inproceedings{behley2019semantickitti,
  author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

## Method Overview

WeatherMamba first encodes raw point features and applies MANF to aggregate multi-scale neighborhood cues and geometric statistics.  
RADM then estimates noise confidence and refines point features, which improves robustness under adverse-weather corruption.  
The refined tokens are processed by a hierarchical bidirectional Mamba backbone to capture long-range dependencies with efficient sequence modeling.  
Finally, WGRG uses weather-conditioned embeddings and gated fusion to adapt features before the segmentation head predicts per-point semantic labels.
