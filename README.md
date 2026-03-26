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
