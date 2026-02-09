from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from model.modules import MelSpec


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


class CFM(nn.Module):
    def __init__(
            self,
            transformer: nn.Module,
            sigma=0.0,
            odeint_kwargs: dict = dict(
                method="euler"
            ),
            audio_drop_prob=0.5,
            cond_drop_prob=0.2,
            num_channels=None,
            mel_spec_module: nn.Module | None = None,
            mel_spec_kwargs: dict = dict(),
            frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
            frac_lengths_mask_noisy: tuple[float, float] = (0.5, 1.0),
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        self.frac_lengths_mask_noisy = frac_lengths_mask_noisy

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
            self,
            cond: float["b n d"] | float["b nw"],  # noqa: F722
            cond_noisy: float["b n d"] | float["b nw"],  # noqa: F722
            inp: int["b nt"] | list[str],  # noqa: F722
            *,
            Q,
            steps=32,
            cfg_strength=1.0,
            sway_sampling_coef=None,
            seed: int | None = None,
            max_duration=6000,
            t_inter=0.1,
            edit_mask=None,
    ):
        self.eval()
        # raw wave
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        if cond_noisy.ndim == 2:
            cond_noisy = self.mel_spec(cond_noisy)
            cond_noisy = cond_noisy.permute(0, 2, 1)
            assert cond_noisy.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        cond_noisy = cond_noisy.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        _, noisy_cond_seq_len = cond_noisy.shape[:2]

        # duration
        duration = noisy_cond_seq_len

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        step_cond = torch.zeros_like(cond)

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # neural ode

        def fn(t, x):
            pred = self.transformer(
                x=x, cond=step_cond, cond_noisy=cond_noisy, inp=inp, time=t, mask=mask, drop_audio_cond=False,
                drop_noisy_audio_cond=False, drop_harmonic=False, cache=True,
                Q=Q
            )

            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, cond_noisy=cond_noisy, inp=inp, time=t, mask=mask, drop_audio_cond=True,
                drop_noisy_audio_cond=True, drop_harmonic=True, cache=True,
                Q=Q
            )

            return pred + (pred - null_pred) * cfg_strength

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        return out, trajectory

    def forward(
            self,
            inp,
            noisy_inp,
            harmonic_inp,
            *,
            Q,
            lens,
            noise_scheduler,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        if noisy_inp.ndim == 2:
            noisy_inp = self.mel_spec(noisy_inp)
            noisy_inp = noisy_inp.permute(0, 2, 1)
            assert noisy_inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        drop_audio_cond = random() < self.audio_drop_prob
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_noisy_audio_cond = True
            drop_harmonic = True
        else:
            drop_noisy_audio_cond = False
            drop_harmonic = False

        pred = self.transformer(
            x=φ, cond=cond, cond_noisy=noisy_inp, inp=harmonic_inp, time=time, drop_audio_cond=drop_audio_cond,
            drop_noisy_audio_cond=drop_noisy_audio_cond, drop_harmonic=drop_harmonic,
            Q=Q
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
