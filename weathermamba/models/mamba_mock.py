import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True
        )

        self.ssm_proj = nn.Linear(d_inner, d_state, bias=False)
        self.ssm_out = nn.Linear(d_state, d_inner, bias=False)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.act = nn.SiLU()
        
    def forward(self, x):
        B, L, D = x.shape

        x_proj = self.in_proj(x)
        x_gate, x_ssm = x_proj.chunk(2, dim=-1)

        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)

        x_conv = self.act(x_conv)

        h = self.ssm_proj(x_conv)
        h = torch.tanh(h)
        y = self.ssm_out(h)

        y = y * self.act(x_gate)

        output = self.out_proj(y)
        
        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        
    def forward(self, x):
        return x + self.mamba(self.norm(x))
