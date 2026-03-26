import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _CudaMamba
except ImportError:
    _CudaMamba = None

try:
    from .mamba_mock import Mamba as _MockMamba
except ImportError:
    _mock_path = Path(__file__).resolve().parent / "mamba_mock.py"
    _spec = importlib.util.spec_from_file_location("weather_mamba_mock", _mock_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _MockMamba = _mod.Mamba


class _AutoMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: float):
        super().__init__()
        self.cuda_mamba = None
        if _CudaMamba is not None:
            self.cuda_mamba = _CudaMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mock_mamba = _MockMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cuda_mamba is not None and x.is_cuda:
            return self.cuda_mamba(x)
        return self.mock_mamba(x)


def _build_mamba(d_model: int, d_state: int, d_conv: int, expand: float):
    return _AutoMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


def _stack_sequence_neighbors(values: torch.Tensor, k: int) -> torch.Tensor:
    _, npts, _ = values.shape
    if npts <= 1 or k <= 1:
        return values.unsqueeze(2)

    pad_size = max(k // 2, 1)
    padded = F.pad(values.transpose(1, 2), (pad_size, pad_size), mode="replicate").transpose(1, 2)

    neighbors = []
    for offset in range(-pad_size, pad_size + 1):
        if offset == 0:
            continue
        shifted = padded[:, pad_size + offset : pad_size + offset + npts, :]
        neighbors.append(shifted)

    if not neighbors:
        return values.unsqueeze(2)

    return torch.stack(neighbors, dim=2)


class MambaBlock(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _build_mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class BidirectionalMambaBlock(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)

        self.forward_mamba = _build_mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = _build_mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.pre_norm(x)

        x_fwd = self.forward_mamba(x_norm)

        x_rev = torch.flip(x_norm, dims=[1])
        x_rev = self.backward_mamba(x_rev)
        x_bwd = torch.flip(x_rev, dims=[1])

        fused = self.fuse(torch.cat([x_fwd, x_bwd], dim=-1))
        return residual + self.dropout(fused)


class HierarchicalMambaStage(nn.Module):

    def __init__(
        self,
        d_model: int,
        depth: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        dropout: float = 0.1,
        use_local_mixing: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.use_local_mixing = use_local_mixing
        if use_local_mixing:
            self.local_mixing = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                groups=d_model,
                bias=True,
            )
        else:
            self.local_mixing = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        if self.local_mixing is not None:
            mixed = self.local_mixing(x.transpose(1, 2)).transpose(1, 2)
            x = x + mixed

        return self.norm(x)


class HierarchicalWeatherMambaBackbone(nn.Module):

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        stage_depths: Sequence[int] = (2, 2, 2),
        dropout: float = 0.1,
        use_local_mixing: bool = True,
    ):
        super().__init__()

        if len(stage_depths) < 2:
            raise ValueError("HierarchicalWeatherMambaBackbone requires at least 2 stages.")

        self.stage_depths = tuple(int(max(1, d)) for d in stage_depths)

        self.stages = nn.ModuleList(
            [
                HierarchicalMambaStage(
                    d_model=d_model,
                    depth=depth,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_local_mixing=use_local_mixing,
                )
                for depth in self.stage_depths
            ]
        )

        self.transitions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                )
                for _ in range(len(self.stage_depths) - 1)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_multi_stage: bool = False):
        stage_features = []

        for idx, stage in enumerate(self.stages):
            x = stage(x)
            stage_features.append(x)
            if idx < len(self.transitions):
                x = self.transitions[idx](x)

        x = self.final_norm(x)

        if return_multi_stage:
            return x, stage_features
        return x


class MANF(nn.Module):

    def __init__(
        self,
        channels: int,
        k_small: int = 8,
        k_medium: int = 16,
        k_large: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k_small = k_small
        self.k_medium = k_medium
        self.k_large = k_large

        self.neighbor_mlp_small = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.neighbor_mlp_medium = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.neighbor_mlp_large = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        self.geo_encoder = nn.Sequential(
            nn.Linear(10, max(channels // 4, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 16), channels),
        )

        self.fusion = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

        self.norm = nn.LayerNorm(channels)

    def _aggregate_neighbors(self, x: torch.Tensor, k: int, mlp: nn.Module) -> torch.Tensor:
        neighbors = _stack_sequence_neighbors(x, k)
        edge_feat = (neighbors - x.unsqueeze(2)).mean(dim=2)
        combined = torch.cat([x, edge_feat], dim=-1)
        return mlp(combined)

    def _compute_geometric_features(self, coords: torch.Tensor, k: int) -> torch.Tensor:
        neighbors = _stack_sequence_neighbors(coords, k)

        local_mean = neighbors.mean(dim=2)
        local_std = neighbors.std(dim=2, unbiased=False)
        density = torch.norm(neighbors - coords.unsqueeze(2), dim=-1).mean(dim=-1, keepdim=True)

        geo_feat = torch.cat([coords, local_mean, local_std, density], dim=-1)
        return self.geo_encoder(geo_feat)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, npts, _ = x.shape

        k_small = min(max(2, self.k_small), max(2, npts))
        k_medium = min(max(2, self.k_medium), max(2, npts))
        k_large = min(max(2, self.k_large), max(2, npts))

        feat_small = self._aggregate_neighbors(x, k_small, self.neighbor_mlp_small)
        feat_medium = self._aggregate_neighbors(x, k_medium, self.neighbor_mlp_medium)
        feat_large = self._aggregate_neighbors(x, k_large, self.neighbor_mlp_large)

        if coords is None:
            geo_feat = torch.zeros_like(x)
        else:
            geo_feat = self._compute_geometric_features(coords, k_medium)

        feat_sum = feat_small + feat_medium + feat_large
        cross_scale_diff = feat_large - feat_small

        merged = torch.cat([x, feat_sum, geo_feat, cross_scale_diff], dim=-1)
        output = self.fusion(merged)

        return self.norm(x + output)


class RADM(nn.Module):

    def __init__(self, channels: int, k: int = 16, dropout: float = 0.1):
        super().__init__()
        self.k = k
        stat_dim = channels + 4

        self.noise_detector = nn.Sequential(
            nn.Linear(stat_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

        self.refinement_branch = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )

        conf_hidden = max(channels // 2, 32)
        self.confidence_net = nn.Sequential(
            nn.Linear(channels + 1, conf_hidden),
            nn.LayerNorm(conf_hidden),
            nn.GELU(),
            nn.Linear(conf_hidden, 1),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(channels)

    def _compute_local_statistics(self, x: torch.Tensor, k: int):
        neighbors = _stack_sequence_neighbors(x, k)

        local_mean_feat = neighbors.mean(dim=2)
        local_std_feat = neighbors.std(dim=2, unbiased=False)

        local_mean = local_mean_feat.mean(dim=-1, keepdim=True)
        local_std = local_std_feat.mean(dim=-1, keepdim=True)
        local_density = torch.norm(neighbors - x.unsqueeze(2), dim=-1).mean(dim=-1, keepdim=True)
        sim_var = ((neighbors - x.unsqueeze(2)) ** 2).mean(dim=(2, 3), keepdim=False).unsqueeze(-1)

        return local_mean, local_std, local_density, sim_var, local_mean_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, npts, _ = x.shape
        k_eff = min(max(2, self.k), max(2, npts))

        local_mean, local_std, local_density, sim_var, local_mean_feat = self._compute_local_statistics(x, k_eff)

        stat_feat = torch.cat([x, local_mean, local_std, local_density, sim_var], dim=-1)
        noise_prob = self.noise_detector(stat_feat)

        refinement_input = torch.cat([x, local_mean_feat], dim=-1)
        refined_feat = self.refinement_branch(refinement_input)

        denoised_feat = (1.0 - noise_prob) * x + noise_prob * refined_feat

        confidence = self.confidence_net(torch.cat([x, noise_prob], dim=-1))
        out = (1.0 - confidence) * x + confidence * denoised_feat

        return self.norm(out)


class WGRG(nn.Module):

    def __init__(self, channels: int, num_weather_types: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_weather_types = num_weather_types
        gate_hidden = max(channels // 4, 32)

        self.weather_embedding = nn.Embedding(num_weather_types, channels)
        self.weather_scale = nn.Embedding(num_weather_types, channels)
        self.weather_shift = nn.Embedding(num_weather_types, channels)

        self.geo_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.ref_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )

        self.geo_gate = nn.Sequential(
            nn.Linear(channels * 2, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid(),
        )
        self.ref_gate = nn.Sequential(
            nn.Linear(channels * 2, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )
        self.norm = nn.LayerNorm(channels)

        nn.init.ones_(self.weather_scale.weight)
        nn.init.zeros_(self.weather_shift.weight)

    def forward(self, x: torch.Tensor, weather_type: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, _, _ = x.shape

        if weather_type is None:
            weather_type = torch.zeros(bsz, dtype=torch.long, device=x.device)

        if weather_type.dim() == 0:
            weather_type = weather_type.view(1).expand(bsz)
        elif weather_type.dim() > 1:
            weather_type = weather_type.view(bsz, -1)[:, 0]

        weather_type = weather_type.long().clamp(0, self.num_weather_types - 1)

        weather_embed = self.weather_embedding(weather_type)
        weather_scale = self.weather_scale(weather_type)
        weather_shift = self.weather_shift(weather_type)

        x_cal = x * weather_scale.unsqueeze(1) + weather_shift.unsqueeze(1)

        geo_feat = self.geo_proj(x_cal)
        ref_feat = self.ref_proj(x_cal)

        global_ctx = x_cal.mean(dim=1)
        gate_input = torch.cat([global_ctx, weather_embed], dim=-1)

        alpha_g = self.geo_gate(gate_input).unsqueeze(1)
        alpha_r = self.ref_gate(gate_input).unsqueeze(1)

        fused = alpha_g * geo_feat + alpha_r * ref_feat
        fused = self.out_proj(fused)

        return self.norm(x + fused)


class MultiScalePyramid(nn.Module):

    def __init__(self, channels: int, num_scales: int = 3):
        super().__init__()
        _ = num_scales
        self.manf = MANF(channels=channels)

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.manf(x, coords)


class WeatherMambaSegmentationFinal(nn.Module):

    def __init__(
        self,
        num_classes: int = 20,
        input_dim: int = 4,
        hidden_dim: int = 384,
        d_state: int = 16,
        d_conv: int = 4,
        expand: float = 2.0,
        num_weather_types: int = 4,
        k_small: int = 8,
        k_medium: int = 16,
        k_large: int = 32,
        stage_depths: Sequence[int] = (2, 2, 2),
        dropout: float = 0.1,
        use_deep_supervision: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_deep_supervision = use_deep_supervision

        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.manf = MANF(
            channels=hidden_dim,
            k_small=k_small,
            k_medium=k_medium,
            k_large=k_large,
            dropout=dropout,
        )

        self.radm = RADM(channels=hidden_dim, k=k_medium, dropout=dropout)

        self.pre_backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.backbone = HierarchicalWeatherMambaBackbone(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            stage_depths=stage_depths,
            dropout=dropout,
            use_local_mixing=True,
        )

        self.wgrg = WGRG(
            channels=hidden_dim,
            num_weather_types=num_weather_types,
            dropout=dropout,
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.seg_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        points: torch.Tensor,
        weather_type: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if points.dim() != 3 or points.size(-1) < 4:
            raise ValueError(f"Expected points shape (B, N, 4+), got {tuple(points.shape)}")

        coords = points[..., :3]

        x = self.point_encoder(points)
        x = self.manf(x, coords)
        x = self.radm(x)
        x = self.pre_backbone(x)

        x, stage_features = self.backbone(x, return_multi_stage=True)
        x = self.wgrg(x, weather_type)

        global_feat = self.global_encoder(x.mean(dim=1))
        global_feat = global_feat.unsqueeze(1).expand(-1, x.size(1), -1)

        fused = torch.cat([x, global_feat], dim=-1)
        logits = self.seg_head(fused)

        if return_aux:
            aux_outputs: Dict[str, torch.Tensor] = {}
            for idx, feat in enumerate(stage_features, start=1):
                aux_outputs[f"stage{idx}"] = feat.mean(dim=-1)
            return logits, aux_outputs

        return logits

    def get_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weather_type: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ) -> torch.Tensor:
        _ = weather_type
        bsz, npts, num_classes = logits.shape
        logits_flat = logits.reshape(bsz * npts, num_classes)
        labels_flat = labels.reshape(bsz * npts)
        return F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction="mean")


class WeatherMambaSegmentation(WeatherMambaSegmentationFinal):
    pass


WeatherMamba = WeatherMambaSegmentationFinal

MambaBackbone = HierarchicalWeatherMambaBackbone
DynamicNeighborhoodFusion = MANF
PointCloudDenoising = RADM
WeatherAwareAttention = WGRG


def _distribute_depths(total_layers: int, num_stages: int = 3) -> Tuple[int, ...]:
    total_layers = max(int(total_layers), num_stages)
    base = total_layers // num_stages
    rem = total_layers % num_stages
    depths = [base] * num_stages
    for idx in range(rem):
        depths[idx] += 1
    return tuple(depths)


def create_weather_mamba_model(
    num_classes: int = 20,
    hidden_dim: int = 384,
    mamba_layers: Optional[int] = None,
    stage_depths: Optional[Sequence[int]] = None,
    **kwargs,
) -> WeatherMambaSegmentationFinal:
    if stage_depths is None:
        if mamba_layers is None:
            stage_depths = (2, 2, 2)
        else:
            stage_depths = _distribute_depths(mamba_layers, num_stages=3)

    return WeatherMambaSegmentationFinal(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        stage_depths=stage_depths,
        **kwargs,
    )


__all__ = [
    "MambaBlock",
    "BidirectionalMambaBlock",
    "HierarchicalMambaStage",
    "HierarchicalWeatherMambaBackbone",
    "MANF",
    "RADM",
    "WGRG",
    "WeatherMamba",
    "WeatherMambaSegmentation",
    "WeatherMambaSegmentationFinal",
    "MambaBackbone",
    "DynamicNeighborhoodFusion",
    "PointCloudDenoising",
    "WeatherAwareAttention",
    "MultiScalePyramid",
    "create_weather_mamba_model",
]
