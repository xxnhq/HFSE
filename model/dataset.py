import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader, SequentialSampler
from tqdm import tqdm
import random
from model.modules import MelSpec
from model.utils import default
import numpy as np
import math
import librosa
from typing import List, Dict, Any, Optional
from pathlib import Path


def parse_scp(scp_path: str, split_token: str = " ") -> List[Dict[str, Any]]:
    path_list = []
    scp_path = Path(scp_path)

    if not scp_path.exists():
        raise FileNotFoundError(f"SCP file not found: {scp_path}")

    with open(scp_path, 'r', encoding='utf-8') as fid:
        for line_num, line in enumerate(fid, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(split_token)
            if len(parts) >= 3:
                try:
                    path_list.append({
                        "name": parts[0],
                        "clean_path": parts[1],
                        "noisy_path": parts[2],
                        "duration": float(parts[3])
                    })
                except (ValueError, IndexError) as e:
                    warnings.warn(f"Line {line_num} in {scp_path} is invalid: {line}. Error: {e}")
                    continue
            elif len(parts) == 1:
                path_list.append({"inputs": parts[0]})
            else:
                warnings.warn(f"Line {line_num} in {scp_path} has unexpected format: {line}")

    return path_list


def get_audio(path, fs=16000):
    wave_data, sr = torchaudio.load(path)
    if sr != fs:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=fs)
        wave_data = resampler(wave_data)
    wave_data = wave_data[0, :]
    return wave_data


class CustomDataset(Dataset):
    def __init__(
            self,
            target_sample_rate=24_000,
            hop_length=256,
            n_mel_channels=100,
            n_fft=1024,
            win_length=1024,
            mel_spec_type="bigvgan",
            mel_spec_module: Optional[nn.Module] = None,
            min_duration: float = 0.5,
            max_duration: float = 30.0
    ):
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.min_duration = min_duration
        self.max_duration = max_duration

        self._data_list = None
        self._Q_matrix = None

        self.mel_spectrogram = default(
            mel_spec_module,
            MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
        )

        self._resampler = None

    @property
    def data_list(self) -> List[Dict[str, Any]]:
        if self._data_list is None:
            self._data_list = parse_scp('datalist/mixed_audio_large.scp')
        return self._data_list

    @property
    def resampler(self) -> torchaudio.transforms.Resample:
        if self._resampler is None:
            self._resampler = torchaudio.transforms.Resample(16000, self.target_sample_rate)
        return self._resampler

    def get_frame_len(self, index):
        return self.data_list[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data_list)

    def get_comb_pitch_matrix_mel(self, sr=24000, F_bins=513, R=1, f_min=60, f_max=420, n_mels=100):
        if self._Q_matrix is not None:
            return self._Q_matrix
        
        pitch_candidates = np.arange(f_min, f_max + 1, R)
        Nc = len(pitch_candidates)
        Q_linear = np.zeros((Nc, F_bins), dtype=np.float32)

        for j, fc in enumerate(pitch_candidates):
            loc_last, peak_last = -1, 0.0
            max_p = int(sr // (2 * fc))
            for p in range(1, max_p + 1):
                loc = int(round(fc * p * F_bins / (sr / 2)))
                if loc >= F_bins:
                    continue
                peak = 1.0 / math.sqrt(p)
                Q_linear[j, loc] = peak
                if loc_last >= 0 and loc > loc_last + 1:
                    gap = loc - loc_last
                    for k in range(1, gap):
                        interp_loc = loc_last + k
                        alpha = k / gap
                        interp_val = peak_last * (1 - alpha) + peak * alpha
                        smooth_factor = 0.5 * (1 + np.cos(np.pi * alpha))
                        Q_linear[j, interp_loc] = interp_val * smooth_factor
                loc_last, peak_last = loc, peak

        mel_filters = librosa.filters.mel(sr=sr, n_fft=(F_bins - 1) * 2, n_mels=n_mels)
        Q_mel = np.dot(Q_linear, mel_filters.T)

        self._Q_matrix = torch.from_numpy(Q_mel)
        return self._Q_matrix

    def __getitem__(self, index):
        data = self.data_list[index]
        clean_path = data["clean_path"]
        noisy_path = data["noisy_path"]
        duration = data["duration"]

        degraded_audio = get_audio(noisy_path)
        audio = get_audio(clean_path)

        assert degraded_audio.shape == audio.shape

        if self.target_sample_rate != 16000:
            audio = self.resampler(audio)
            degraded_audio = self.resampler(degraded_audio)

        # to mel spectrogram
        mel_spec = self.mel_spectrogram(audio)
        degraded_mel_spec = self.mel_spectrogram(degraded_audio)
        
        Q_matrix = self.get_comb_pitch_matrix_mel()

        return {
            "audio_path": clean_path,
            "clean_audio": audio,
            "noisy_audio": degraded_audio,
            "mel_spec": mel_spec,
            "noisy_mel_spec": degraded_mel_spec,
            "Q": Q_matrix
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):

    def __init__(
            self,
            sampler: Sampler[int],
            frames_threshold: int,
            max_samples: int = 0,
            random_seed: Optional[int] = None,
            drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        self.epoch = 0

        self.batches = self._precompute_batches()
        self.drop_last = True

    def _precompute_batches(self) -> List[List[int]]:
        indices = []
        data_source = self.sampler.data_source

        for idx in tqdm(self.sampler, desc="Sorting samples by duration"):
            frame_len = data_source.get_frame_len(idx)
            indices.append((idx, frame_len))

        indices.sort(key=lambda x: x[1])

        batches = []
        current_batch = []
        current_frames = 0

        for idx, frame_len in tqdm(indices, desc=f"Batching with threshold {self.frames_threshold}"):
            can_add = (
                    (current_frames + frame_len <= self.frames_threshold) and
                    (self.max_samples == 0 or len(current_batch) < self.max_samples)
            )

            if can_add:
                current_batch.append(idx)
                current_frames += frame_len
            else:
                if current_batch:
                    batches.append(current_batch)

                if frame_len <= self.frames_threshold:
                    current_batch = [idx]
                    current_frames = frame_len
                else:
                    current_batch = []
                    current_frames = 0
                    warnings.warn(f"Sample {idx} exceeds frame threshold: {frame_len} > {self.frames_threshold}")

        if not self.drop_residual and current_batch:
            batches.append(current_batch)

        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.random_seed + self.epoch)
            indices = torch.randperm(len(self.batches), generator=generator).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches.copy()
            random.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


# collation
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Batch is empty")

    audio_paths = [item["audio_path"] for item in batch]
    clean_audios = [item["clean_audio"].squeeze(0) for item in batch]
    noisy_audios = [item["noisy_audio"].squeeze(0) for item in batch]
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    noisy_mel_specs = [item["noisy_mel_spec"].squeeze(0) for item in batch]
    Q_matrices = [item["Q"] for item in batch]

    audio_lengths = torch.tensor([audio.shape[-1] for audio in clean_audios], dtype=torch.long)
    max_audio_len = audio_lengths.max().item()

    padded_clean = torch.zeros(len(clean_audios), max_audio_len, dtype=clean_audios[0].dtype)
    padded_noisy = torch.zeros(len(noisy_audios), max_audio_len, dtype=noisy_audios[0].dtype)

    for i, (clean, noisy) in enumerate(zip(clean_audios, noisy_audios)):
        if clean.shape[-1] != noisy.shape[-1]:
            warnings.warn(
                f"Audio length mismatch in {audio_paths[i]}: clean={clean.shape[-1]}, noisy={noisy.shape[-1]}")
            min_len = min(clean.shape[-1], noisy.shape[-1])
            clean, noisy = clean[..., :min_len], noisy[..., :min_len]
            audio_lengths[i] = min_len

        padded_clean[i, :clean.shape[-1]] = clean
        padded_noisy[i, :noisy.shape[-1]] = noisy

    mel_lengths = torch.tensor([spec.shape[-1] for spec in mel_specs], dtype=torch.long)
    max_mel_len = mel_lengths.max().item()

    mel_dim = mel_specs[0].shape[0]
    padded_mel = torch.zeros(len(mel_specs), mel_dim, max_mel_len, dtype=mel_specs[0].dtype)
    padded_noisy_mel = torch.zeros(len(noisy_mel_specs), mel_dim, max_mel_len, dtype=noisy_mel_specs[0].dtype)

    for i, (mel, noisy_mel) in enumerate(zip(mel_specs, noisy_mel_specs)):
        if mel.shape[-1] != noisy_mel.shape[-1]:
            warnings.warn(
                f"Mel length mismatch in {audio_paths[i]}: clean={mel.shape[-1]}, noisy={noisy_mel.shape[-1]}")
            min_len = min(mel.shape[-1], noisy_mel.shape[-1])
            mel, noisy_mel = mel[..., :min_len], noisy_mel[..., :min_len]
            mel_lengths[i] = min_len

        padded_mel[i, :, :mel.shape[-1]] = mel
        padded_noisy_mel[i, :, :noisy_mel.shape[-1]] = noisy_mel

    Q_tensor = torch.stack(Q_matrices)

    return {
        "clean_audio": padded_clean,
        "noisy_audio": padded_noisy,
        "audio_lengths": audio_lengths,
        "mel": padded_mel,
        "noisy_mel": padded_noisy_mel,
        "mel_lengths": mel_lengths,
        "Q": Q_tensor,
        "audio_paths": audio_paths,
    }


# Load dataset
def load_dataset(
        mel_spec_module: Optional[nn.Module] = None,
        mel_spec_kwargs: Optional[Dict[str, Any]] = None
) -> CustomDataset:
    if mel_spec_kwargs is None:
        mel_spec_kwargs = {}

    return CustomDataset(
        mel_spec_module=mel_spec_module,
        **mel_spec_kwargs
    )
