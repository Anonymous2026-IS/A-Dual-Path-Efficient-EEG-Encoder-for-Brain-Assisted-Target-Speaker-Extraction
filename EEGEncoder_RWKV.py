import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from torch.utils.cpp_extension import load
from cuda_new.drop import DropPath

# Bidirectional WKV CUDA operator
bi_wkv_cuda = load(
    name="bi_wkv",
    sources=[
        "/br/DualPathEEG/cuda_new/bi_wkv.cpp",
        "/br/DualPathEEG/cuda_new/bi_wkv_kernel.cu"
    ],
    verbose=True,
    extra_cuda_cflags=[
        '-res-usage',
        '--maxrregcount 60',
        '--use_fast_math',
        '-O3',
        '-Xptxas -O3',
        '-gencode arch=compute_86,code=sm_86'
    ]
)


class BiWKVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)

        ctx.save_for_backward(w, u, k, v)

        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()

        y = bi_wkv_cuda.bi_wkv_forward(w, u, k, v)

        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()

        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)

        gw, gu, gk, gv = bi_wkv_cuda.bi_wkv_backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous()
        )

        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gkv.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def run_bi_wkv(w, u, k, v):
    return BiWKVFunction.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def eeg_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4

    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(
        B, C, patch_resolution[0], patch_resolution[1]
    )

    B, C, H, W = input.shape
    output = torch.zeros_like(input)

    output[:, 0:int(C * gamma), :, shift_pixel:W] = \
        input[:, 0:int(C * gamma), :, 0:W - shift_pixel]

    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = \
        input[:, int(C * gamma):int(C * gamma * 2), :, shift_pixel:W]

    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = \
        input[:, int(C * gamma * 2):int(C * gamma * 3), 0:H - shift_pixel, :]

    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = \
        input[:, int(C * gamma * 3):int(C * gamma * 4), shift_pixel:H, :]

    output[:, int(C * gamma * 4):, ...] = \
        input[:, int(C * gamma * 4):, ...]

    return output.flatten(2).transpose(1, 2)


class CorticalInteractionModule(nn.Module):
    """
    Cortical Interaction Module.

    This module corresponds to the spatial path in the paper.
    It models cross-electrode EEG dependencies using the bidirectional
    weighted key-value interaction.
    """

    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode='eeg_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        init_mode='fancy',
        key_norm=False,
        with_cp=False
    ):
        super().__init__()

        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None

        attn_sz = n_embd

        self._init_weights(init_mode)

        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode

        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.cortical_mix_k = None
            self.cortical_mix_v = None
            self.cortical_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None

        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd - 1)) ** (
                        0.7 + 1.3 * ratio_0_to_1
                    )

                self.cortical_decay = nn.Parameter(decay_speed)

                zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.n_embd)]) * 0.5
                self.cortical_first = nn.Parameter(
                    torch.ones(self.n_embd) * math.log(0.3) + zigzag
                )

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd

                self.cortical_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.cortical_mix_v = nn.Parameter(
                    torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                )
                self.cortical_mix_r = nn.Parameter(
                    torch.pow(x, 0.5 * ratio_1_to_almost0)
                )

        elif init_mode == 'local':
            self.cortical_decay = nn.Parameter(torch.ones(self.n_embd))
            self.cortical_first = nn.Parameter(torch.ones(self.n_embd))
            self.cortical_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.cortical_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.cortical_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))

        elif init_mode == 'global':
            self.cortical_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.cortical_first = nn.Parameter(torch.zeros(self.n_embd))
            self.cortical_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.cortical_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.cortical_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        B, T, C = x.size()

        if self.shift_pixel > 0:
            xx = self.shift_func(
                x,
                self.shift_pixel,
                self.channel_gamma,
                patch_resolution
            )

            xk = x * self.cortical_mix_k + xx * (1 - self.cortical_mix_k)
            xv = x * self.cortical_mix_v + xx * (1 - self.cortical_mix_v)
            xr = x * self.cortical_mix_r + xx * (1 - self.cortical_mix_r)

        else:
            xk = x
            xv = x
            xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, patch_resolution)

            x = run_bi_wkv(
                self.cortical_decay / T,
                self.cortical_first / T,
                k,
                v
            )

            if self.key_norm is not None:
                x = self.key_norm(x)

            x = sr * x
            x = self.output(x)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class IntraElectrodeTemporalModule(nn.Module):
    """
    Intra-Electrode Temporal Module.

    This module corresponds to the temporal/channel path in the paper.
    It captures local temporal dynamics and channel-wise nonlinear interaction
    within each EEG representation.
    """

    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode='eeg_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        key_norm=False,
        with_cp=False
    ):
        super().__init__()

        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp

        self._init_weights(init_mode)

        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode

        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.temporal_mix_k = None
            self.temporal_mix_r = None

        hidden_sz = hidden_rate * n_embd

        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad():
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)

                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd

                self.temporal_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.temporal_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        elif init_mode == 'local':
            self.temporal_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.temporal_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))

        elif init_mode == 'global':
            self.temporal_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.temporal_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

        else:
            raise NotImplementedError

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(
                    x,
                    self.shift_pixel,
                    self.channel_gamma,
                    patch_resolution
                )

                xk = x * self.temporal_mix_k + xx * (1 - self.temporal_mix_k)
                xr = x * self.temporal_mix_r + xx * (1 - self.temporal_mix_r)

            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = torch.square(torch.relu(k))

            if self.key_norm is not None:
                k = self.key_norm(k)

            kv = self.value(k)

            x = torch.sigmoid(self.receptance(xr)) * kv

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class Block(nn.Module):
    """
    Dual-Path EEG Block.

    This block contains:
    1. Cortical Interaction Module for cross-electrode spatial modeling.
    2. Intra-Electrode Temporal Module for local temporal/channel modeling.
    """

    def __init__(
        self,
        n_embd,
        n_layer,
        layer_id,
        shift_mode='eeg_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        drop_path=0.,
        hidden_rate=4,
        init_mode='fancy',
        init_values=None,
        post_norm=False,
        key_norm=False,
        with_cp=False
    ):
        super().__init__()

        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.cim = CorticalInteractionModule(
            n_embd,
            n_layer,
            layer_id,
            shift_mode,
            channel_gamma,
            shift_pixel,
            init_mode,
            key_norm=key_norm
        )

        self.ietm = IntraElectrodeTemporalModule(
            n_embd,
            n_layer,
            layer_id,
            shift_mode,
            channel_gamma,
            shift_pixel,
            hidden_rate,
            init_mode,
            key_norm=key_norm
        )

        self.layer_scale = init_values is not None
        self.post_norm = post_norm

        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                init_values * torch.ones((n_embd)),
                requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                init_values * torch.ones((n_embd)),
                requires_grad=True
            )

        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)

            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(
                        self.gamma1 * self.ln1(self.cim(x, patch_resolution))
                    )
                    x = x + self.drop_path(
                        self.gamma2 * self.ln2(self.ietm(x, patch_resolution))
                    )
                else:
                    x = x + self.drop_path(
                        self.ln1(self.cim(x, patch_resolution))
                    )
                    x = x + self.drop_path(
                        self.ln2(self.ietm(x, patch_resolution))
                    )
            else:
                if self.layer_scale:
                    x = x + self.drop_path(
                        self.gamma1 * self.cim(self.ln1(x), patch_resolution)
                    )
                    x = x + self.drop_path(
                        self.gamma2 * self.ietm(self.ln2(x), patch_resolution)
                    )
                else:
                    x = x + self.drop_path(
                        self.cim(self.ln1(x), patch_resolution)
                    )
                    x = x + self.drop_path(
                        self.ietm(self.ln2(x), patch_resolution)
                    )

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


