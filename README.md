# WeatherMamba

**DOI:** [10.5281/zenodo.19231095](https://doi.org/10.5281/zenodo.19231095)

This repository contains the official implementation of our manuscript:

**WeatherMamba: Robust LiDAR Point Cloud Segmentation for Autonomous Driving under Adverse Weather Conditions**

The code in this repository is associated with the manuscript above. If you find this repository helpful, please consider citing the corresponding work.

---

## Overview

WeatherMamba is a domain-generalized LiDAR point cloud semantic segmentation framework for autonomous driving under adverse weather conditions. The framework consists of:

- **MANF**: Multi-scale Adaptive Neighborhood Fusion for local geometric recovery
- **RADM**: Reliability-Aware Denoising Module for suppressing unstable noisy responses
- **Hierarchical WeatherMamba Backbone**: efficient long-range context modeling
- **WGRG**: Weather-Conditioned Geometry-Reflectance Gating for adaptive high-level feature recalibration

The method is evaluated on adverse-weather domain generalization benchmarks including:

- **SemanticKITTI → SemanticSTF**
- **SynLiDAR → SemanticSTF**

---

## Environment

The experiments in this repository were developed and tested under the following environment:

- **OS**: Ubuntu 20.04 
- **Python**: 3.8
- **PyTorch**: 2.0.1 
- **CUDA**: 11.8 
- **GPU**: NVIDIA RTX 4090

Create the environment:

```bash
conda create -n weathermamba python=3.8 -y
conda activate weathermamba
pip install -r requirements.txt
```

## Repository Structure

```text
.
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── scripts/
│   ├── train.py
│   └── test.py
├── weathermamba/
│   ├── data/
│   ├── engine/
│   ├── models/
│   ├── utils/
│   └── ...
├── outputs/
└── requirements.txt
```

## Dataset Preparation

### 1. Datasets

The experiments in the paper use the following datasets:

- SemanticKITTI
- SemanticSTF
- SynLiDAR

Please download the datasets from their official sources.

### 2. Expected directory structure

Set the dataset root in `configs/data.yaml`:

```yaml
dataset_path: /path/to/dataset_root
train_split: train
val_split: val
test_split: test
```

Expected structure:

```text
<dataset_root>/
├── SemanticKITTI/
│   ├── train/
│   ├── val/
│   └── ...
├── SemanticSTF/
│   ├── dense_fog/
│   ├── light_fog/
│   ├── rain/
│   ├── snow/
│   └── ...
└── SynLiDAR/
    ├── train/
    ├── val/
    └── ...
```

### 3. Input format

Supported point cloud input formats:

- `.bin`
- `.txt`

Each point should contain:

- `x, y, z`
- `intensity`

### 4. Label mapping / preprocessing

Please ensure that the label mapping and preprocessing settings are consistent with the paper configuration. The corresponding settings are defined in:

- `configs/data.yaml`
- Dataset loader implementation in `weathermamba/data/dataset.py`

## Training

### Main experiment: SemanticKITTI -> SemanticSTF

```bash
python scripts/train.py \
    --dataset-path /path/to/dataset_root \
    --source-dataset SemanticKITTI \
    --target-dataset SemanticSTF \
    --config configs/train.yaml
```

### Main experiment: SynLiDAR -> SemanticSTF

```bash
python scripts/train.py \
    --dataset-path /path/to/dataset_root \
    --source-dataset SynLiDAR \
    --target-dataset SemanticSTF \
    --config configs/train.yaml
```

### Common CLI overrides

```bash
python scripts/train.py \
    --dataset-path /path/to/dataset_root \
    --epochs 100 \
    --batch-size 4 \
    --lr 5e-4 \
    --hidden-dim 384 \
    --stage-depths 2,2,2
```

## Evaluation

### Evaluate a trained checkpoint

```bash
python scripts/test.py \
    --dataset-path /path/to/dataset_root \
    --checkpoint /path/to/checkpoint.pth \
    --source-dataset SemanticKITTI \
    --target-dataset SemanticSTF
```

### Evaluate on weather-specific subsets

#### Dense fog

```bash
python scripts/test.py \
    --dataset-path /path/to/dataset_root \
    --checkpoint /path/to/checkpoint.pth \
    --subset dense_fog
```

#### Light fog

```bash
python scripts/test.py \
    --dataset-path /path/to/dataset_root \
    --checkpoint /path/to/checkpoint.pth \
    --subset light_fog
```

#### Rain

```bash
python scripts/test.py \
    --dataset-path /path/to/dataset_root \
    --checkpoint /path/to/checkpoint.pth \
    --subset rain
```

#### Snow

```bash
python scripts/test.py \
    --dataset-path /path/to/dataset_root \
    --checkpoint /path/to/checkpoint.pth \
    --subset snow
```

## Reproducing the Paper Results

The following table provides a minimal guide for reproducing the main results reported in the paper.

| Paper Item | Setting | Command / Script | Output |
| --- | --- | --- | --- |
| Main result | SemanticKITTI -> SemanticSTF | `scripts/train.py + scripts/test.py` | Main comparison table |
| Auxiliary result | SynLiDAR -> SemanticSTF | `scripts/train.py + scripts/test.py` | Main comparison table |
| Dense fog | SemanticSTF dense fog subset | `scripts/test.py --subset dense_fog` | Weather-specific table |
| Light fog | SemanticSTF light fog subset | `scripts/test.py --subset light_fog` | Weather-specific table |
| Rain | SemanticSTF rain subset | `scripts/test.py --subset rain` | Weather-specific table |
| Snow | SemanticSTF snow subset | `scripts/test.py --subset snow` | Weather-specific table |
| Ablation | Module ablation settings | Config modifications | Ablation table |

### Ablation settings

To reproduce the ablation study, enable or disable MANF, RADM, and WGRG in `configs/model.yaml`.

- Baseline: backbone only
- + MANF
- + RADM
- + WGRG
- Full model

## Configuration

Main configuration files:

- `configs/train.yaml`: training schedule, optimizer, AMP, gradient clipping, epochs, etc.
- `configs/data.yaml`: dataset path, number of points, batch size, workers, augmentation, dataset split
- `configs/model.yaml`: hidden dimension, stage depths, dropout, neighborhood size, module switches

## Outputs

Training and evaluation outputs are saved to:

```text
outputs/weathermamba/<run_name>/
├── checkpoints/
├── logs/
├── predictions/
├── model_resolved.yaml
├── data_resolved.yaml
└── train_resolved.yaml
```

## Checkpoints

If pretrained checkpoints are provided, please place them under:

```text
checkpoints/
```

Or specify the path directly in the testing command:

```bash
python scripts/test.py --checkpoint /path/to/checkpoint.pth
```


## Notes on Reproducibility

To improve reproducibility, we recommend:

- Using the same environment versions listed above
- Verifying dataset paths and label mappings before training
- Keeping the training and evaluation configuration files unchanged when reproducing the reported numbers
- Testing the exact checkpoint corresponding to each reported experiment

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{weathermamba2026,
  title={WeatherMamba: Robust LiDAR Point Cloud Segmentation for Autonomous Driving under Adverse Weather Conditions},
  author={He Huang and Xintai Zhang and Yidan Zhang and Junxing Yang and Yu Liang},
  year={2026},
  note={Manuscript and code release},
  doi={10.5281/zenodo.19231095},
  url={https://doi.org/10.5281/zenodo.19231095}
}
```

## Data Acknowledgements

We would like to thank the authors of the following datasets and related resources:

```bibtex
@inproceedings{behley2019semantickitti,
  author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

Please also acknowledge the official sources of SemanticSTF and SynLiDAR in your final repository version if they are used in your experiments.
