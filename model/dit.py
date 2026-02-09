from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from model.modules import (
    AdaLayerNorm_Final,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding
)
from model.harmonic import HarmonicAttention


class HarmonicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ha = HarmonicAttention()

    def forward(self, inp, seq_len, Q, drop_harmonic=False):
        if drop_harmonic:
            inp = torch.zeros_like(inp)

        inp = self.ha(inp, Q) * inp

        return inp


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 4, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, cond_noisy, harmonic_embed, drop_audio_cond=False, drop_noisy_audio_cond=False):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)
        if drop_noisy_audio_cond:
            cond_noisy = torch.zeros_like(cond_noisy)
        x = self.proj(torch.cat((x, cond, cond_noisy, harmonic_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        qk_norm=None,
        pe_attn_head=None,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.harmonic_embed = HarmonicEncoder()
        self.harmonic_cond, self.harmonic_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def clear_cache(self):
        self.harmonic_cond, self.harmonic_uncond = None, None

    def forward(
        self,
        x,
        cond,
        cond_noisy,
        inp,
        time,
        drop_audio_cond,
        drop_noisy_audio_cond,
        drop_harmonic,
        Q,
        mask=None,
        cache=False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        if cache:
            if drop_harmonic:
                if self.harmonic_uncond is None:
                    self.harmonic_uncond = self.harmonic_embed(inp, seq_len, Q, drop_harmonic=True)
                harmonic_embed = self.harmonic_uncond
            else:
                if self.harmonic_cond is None:
                    self.harmonic_cond = self.harmonic_embed(inp, seq_len, Q, drop_harmonic=False)
                harmonic_embed = self.harmonic_cond
        else:
            harmonic_embed = self.harmonic_embed(inp, seq_len, Q, drop_harmonic=drop_harmonic)
        x = self.input_embed(x, cond, cond_noisy, harmonic_embed, drop_audio_cond=drop_audio_cond, drop_noisy_audio_cond=drop_noisy_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
