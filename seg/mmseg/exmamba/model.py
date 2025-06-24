import math
import torch
import torch.nn as nn
from timm.layers import SqueezeExcite, trunc_normal_
from fvcore.nn import flop_count
from ..core import register
import torch.nn.functional as F


class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None

        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


class Stem(nn.Module):
    def __init__(self, in_dim=3, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2D(dim // 4, dim, kernel_size=3, stride=2, padding=1, act_layer=None))

    def forward(self, x):
        x = self.conv(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2D(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, groups=hidden_dim),
            SqueezeExcite(hidden_dim, .25),
            ConvLayer2D(hidden_dim, out_dim, kernel_size=1, act_layer=None)
        )

        self.dwconv1 = ConvLayer2D(in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class Slim_SSD(nn.Module):
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        self.h_proj = ConvLayer1D(d_model, self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        batch, _, L = x.shape
        H = int(math.sqrt(L))
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, H)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        AB = (A * B)
        h = x @ AB.transpose(-2, -1)
        h = self.h_proj(h)
        h = self.act(h)
        y = h @ C
        y = y.view(batch, -1, H, H).contiguous()
        return y, h


class Expert(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.w1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.w2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.w3 = nn.Linear(dim, int(mlp_ratio * dim))

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # return self.w2(F.silu(self.w1(x)))


class TopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k, filter_radio=0.62):
        super(TopkRouter, self).__init__()
        self.num_exports = num_experts
        self.filter_radio = filter_radio
        self.top_k = top_k
        self.load_lr = 0.001
        self.topkroute_linear = nn.Linear(dim, num_experts)
        self.register_buffer('bi', torch.zeros(num_experts))

    def forward(self, x):
        token_num = x.shape[0]
        logits = self.topkroute_linear(x)
        with torch.no_grad():
            bias = self.bi
            biased_logits = logits + bias
            _, indices = biased_logits.topk(self.top_k, dim=1)

        top_k_logits = torch.gather(logits, 1, indices)
        mask = (torch.rand_like(top_k_logits) > self.filter_radio).float()
        router_output = F.softmax(top_k_logits, dim=1) * mask

        c_avg = token_num / self.num_exports
        if not self.training:
            return router_output, indices
        with torch.no_grad():
            # Count tokens per expert
            c_i = torch.zeros(self.num_exports, device=x.device)  # shape: [num_experts]
            for i in range(self.top_k):
                expert_ids = indices[:, i]  # shape: [batch_size]
                c_i += torch.bincount(expert_ids, minlength=self.num_exports)

            # Compute load violation: e_i = c_avg - c_i
            e_i = c_avg - c_i

            # Update bias: b_i = b_i + u * sign(e_i)
            self.bi += self.load_lr * torch.sign(e_i)
        return router_output, indices


class NoisyTopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(dim, num_experts)
        self.noise_linear = nn.Linear(dim, num_experts)

    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=1)
        return router_output, indices


class Top1Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x):
        logits = self.gate(x)  # [N, num_experts]
        scores = torch.softmax(logits, dim=-1)
        top1_weights, top1_indices = torch.max(scores, dim=-1)  # [N]
        return top1_weights, top1_indices


class SparseMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k, mlp_ratio, router, shared=False, filter_radio=0):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.shared = shared
        if router == 'noisy':
            self.router = NoisyTopkRouter(dim, num_experts, top_k)
        elif router == 'topk':
            self.router = TopkRouter(dim, num_experts, top_k, filter_radio)
        elif router == 'top1':
            self.router = Top1Router(dim, num_experts)
        else:
            raise ValueError('Illegal router type')
        self.experts = nn.ModuleList([Expert(dim, mlp_ratio / top_k) for _ in range(num_experts)])
        if shared:
            self.shared_expert = Expert(dim, mlp_ratio / top_k)
        self.top_k = top_k

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        weights, indices = self.router(x)  # weights: [B, top_k], indices: [B, top_k]
        y = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i)  # shape: [B, top_k]
            if not expert_mask.any():
                continue
            pos_b, pos_k = torch.where(expert_mask)  # 行索引 / 路由位置索引
            x_selected = x[pos_b]  # [num_selected, dim]
            y[pos_b] += expert(x_selected) * weights[pos_b, pos_k].unsqueeze(1)
        if self.shared:
            y += self.shared_expert(x)

        y = y.reshape(shape).permute(0, 3, 1, 2)
        return y


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.w1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.w2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.w3 = nn.Linear(dim, int(mlp_ratio * dim))
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))   
        y = y.reshape(shape).permute(0, 3, 1, 2)
        return y 
    
class Block(nn.Module):
    def __init__(self, dim, num_experts, top_k, mlp_ratio=4., ssd_expand=1,
                 state_dim=64, router=None, dropout=0.1, shared=False, filter_radio=0):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.mixer = Slim_SSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)

        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)

        self.moe = SparseMoE(dim, num_experts, top_k, mlp_ratio, router, shared, filter_radio)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # DWconv1
        x = x + self.dwconv1(x)

        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x.flatten(2)))
        x = x_prev + x
        x = x + self.dwconv2(x)
        # MoE
        x = x + self.moe(x)
        x = self.dropout(x)
        return x, h


class Stage(nn.Module):
    def __init__(self, in_dim, out_dim, depth, num_experts, top_k, mlp_ratio=4., downsample=None,
                 ssd_expand=1, state_dim=64, router=None, dropout=0.1, shared=False, filter_radio=0):
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            Block(in_dim, num_experts, top_k, mlp_ratio, ssd_expand, state_dim, router, dropout, shared, filter_radio)
            for _ in range(depth)])
        self.downsample = downsample(in_dim=in_dim, out_dim=out_dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x, h = blk(x)

        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x_out, h


@register()
class ExMamba(nn.Module):
    def __init__(self, in_dim=3, embed_dim=[128, 256, 512], depths=[2, 2, 2], num_experts=10, top_k=2, mlp_ratio=4.,
                 ssd_expand=1, state_dim=[49, 25, 9], distillation=False, router=None, dropout=0.1, shared=False,
                 checkpoint=None, filter_radio=0, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.distillation = distillation
        self.patch_embed = Stem(in_dim=in_dim, dim=embed_dim[0])
        PatchMergingBlock = PatchMerging

        # build stages
        self.stages = nn.ModuleList()
        for i_layer in range(self.num_layers):
            stage = Stage(in_dim=int(embed_dim[i_layer]),
                          out_dim=int(embed_dim[i_layer + 1]) if (i_layer < self.num_layers - 1) else None,
                          depth=depths[i_layer],
                          num_experts=num_experts,
                          top_k=top_k,
                          mlp_ratio=mlp_ratio,
                          downsample=PatchMergingBlock if (i_layer == 0 or i_layer == 1) else None,
                          ssd_expand=ssd_expand,
                          state_dim=state_dim[i_layer],
                          router=router,
                          dropout=dropout,
                          shared=shared,
                          filter_radio=filter_radio)
            self.stages.append(stage)

    def forward(self, x):
        x = self.patch_embed(x)
        out = []
        out.append(x)
        for i, stage in enumerate(self.stages):
            x, x_out, h = stage(x)
            out.append(x)
        return out[-3:]
