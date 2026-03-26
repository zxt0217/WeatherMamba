# WeatherMamba
WeatherMamba: Robust LiDAR Point Cloud Segmentation for Autonomous Driving under Adverse Weather Conditions
```text
.
├── configs/                      # Training/data/model configs (default)
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── scripts/                      # Entry scripts
│   └── train.py
├── weathermamba/                 # Core library
│   ├── __init__.py
│   ├── cli/                      # CLI training logic
│   │   ├── train.py
│   │   └── __init__.py
│   ├── configs/                  # In-package config copies
│   │   ├── data.yaml
│   │   ├── model.yaml
│   │   ├── train.yaml
│   │   └── __init__.py
│   ├── data/                     # Dataset loading and augmentation
│   │   ├── augmentation.py
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── engine/                   # Trainer and training loop
│   │   ├── trainer.py
│   │   └── __init__.py
│   ├── models/                   # Model definitions
│   │   ├── mamba_mock.py
│   │   ├── weather_mamba.py
│   │   └── __init__.py
│   └── utils/                    # Utility helpers
│       ├── config.py
│       ├── runtime.py
│       └── __init__.py
└── requirements.txt              # Dependencies
```
1.Clone the repository
git clone https://github.com/<your-org>/WeatherMamba_Pro.git
cd WeatherMamba_Pro

2.Create and activate environment
conda create -n weathermamba python=3.8 -y
conda activate weathermamba

3.Install dependencies
pip install -r requirements.txt

Dataset Preparation
dataset_path: ""
train_split: train
val_split: val
Expected structure:
<dataset_root>/
├── train/
│   └── ... (recursive point cloud files)
└── val/
    └── ... (recursive point cloud files)
Supported input files:

.bin point clouds (float32, reshaped to N x 4)
.txt point clouds (N x 4 or N x 5; the 5th column is label if present)
For .bin samples, labels are loaded from .label files when available.

Training
Basic training:
python scripts/train.py --dataset-path ./your_dataset_root

Dry run (sanity check only):
python scripts/train.py --dataset-path ./your_dataset_root --dry-run

Example with common overrides:
python scripts/train.py \
  --dataset-path ./your_dataset_root \
  --epochs 100 \
  --batch-size 4 \
  --lr 5e-4 \
  --hidden-dim 512 \
  --stage-depths 3,3,4

Hyperparameter Tuning
Main files to tune:

configs/train.yaml
epochs, lr, weight_decay, amp, grad_clip, save_interval

configs/data.yaml
num_points, loading.batch_size, loading.num_workers, augmentation.*

configs/model.yaml
hidden_dim, stage_depths, dropout, k_small, k_medium, k_large

Outputs
Training outputs are saved to:
outputs/weathermamba_pro/<run_name>/
├── checkpoints/
├── model_resolved.yaml
├── data_resolved.yaml
└── train_resolved.yaml

Note
Weather type is inferred from file path keywords：
rain -> 0
snow -> 1
fog -> 2
otherwise unknown_weather_index from config

Default entrypoint
scripts/train.py -> weathermamba/cli/train.py
