from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from model.cfm import CFM
from model.dataset import DynamicBatchSampler, collate_fn
from model.cfm import default, exists
from pathlib import Path
from model.modules import MelSpec
from model.eval import eval_dnsmos


def parse_scp(scp_path: str, split_token: str = " "):
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

            path_list.append({
                "name": parts[0],
                "sr": parts[1],
                "clean_path": parts[2],
                "noisy_path": parts[3],
                "duration": float(parts[4])
            })

    return path_list


def load_vocoder(vocoder_name="bigvgan", is_local=False, local_path="", device=None, hf_cache_dir=None):
    if vocoder_name == "bigvgan":
        from model.vocoder.BigVGAN.bigvgan import BigVGAN
        if is_local:
            vocoder = BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    else:
        raise ValueError(f"Vocoder {vocoder_name} not supported")
    return vocoder


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        log_samples: bool = False,
        log_sample_rate: int = 24000,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        mel_spec_type: str = "bigvgan",
        is_local_vocoder: bool = True,
        local_vocoder_path: str = "",
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.log_samples = log_samples
        self.log_sample_rate = log_sample_rate

        self.accelerator = Accelerator(
            log_with=None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=f"tensorboard")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = checkpoint_path

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.data_list = parse_scp('datalist/no_rev.scp')
        self.rev_data_list = parse_scp('datalist/with_rev.scp')
        self.resampler = torchaudio.transforms.Resample(16000, 24000)
        self._resampler = torchaudio.transforms.Resample(24000, 16000)
        self.mel_spectrogram = MelSpec(
                            n_fft=1024,
                            hop_length=256,
                            win_length=1024,
                            n_mel_channels=100,
                            target_sample_rate=24000,
                            mel_spec_type="bigvgan",
                        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process


    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")


    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "SenSE WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update


    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        self.accelerator.even_batches = False
        sampler = SequentialSampler(train_dataset)
        batch_sampler = DynamicBatchSampler(
            sampler,
            self.batch_size_per_gpu,
            max_samples=self.max_samples,
            random_seed=resumable_with_seed,
            drop_residual=False,
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
            batch_sampler=batch_sampler,
        )

        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )

        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    noisy_mel_spec = batch["noisy_mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    Q = batch["Q"]

                    loss, cond, pred = self.model(
                        mel_spec, noisy_mel_spec, inp=noisy_mel_spec, lens=mel_lengths, Q=Q
                    ) 
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(
                        update=str(global_update),
                        loss=f"{loss.item():.4f}",
                        lr=f"{current_lr:.2e}"
                    )

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                ########################infer####################################
                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        target_dir = f"samples/{global_update}/no_rev"
                        rev_target_dir = f"samples/{global_update}/with_rev"
                        os.makedirs(target_dir, exist_ok=True)
                        os.makedirs(rev_target_dir, exist_ok=True)

                        ######################## no_rev ####################################
                        for index in tqdm(range(len(self.data_list))):
                            data = self.data_list[index]
                            noisy_path = data["noisy_path"]
                            name = data["name"]

                            degraded_audio, sr = torchaudio.load(noisy_path)
                            if sr != 24000:
                                degraded_audio = self.resampler(degraded_audio)

                            noisy_mel_spec = self.mel_spectrogram(degraded_audio).to(Q[0].device)

                            with torch.inference_mode():
                                generated_noref, _ = self.accelerator.unwrap_model(self.model).sample(
                                    cond=noisy_mel_spec.transpose(-1, -2),
                                    cond_noisy=noisy_mel_spec.transpose(-1, -2),
                                    inp=noisy_mel_spec.transpose(-1, -2),
                                    steps=8,
                                    cfg_strength=0.5,
                                    sway_sampling_coef=-1,
                                    Q=Q[0].unsqueeze(0),
                                )
                                generated_noref = generated_noref.to(torch.float32).permute(0, 2, 1)
                                gen_audio_noref = vocoder(generated_noref).squeeze(0).cpu()

                            torchaudio.save(
                                f"{target_dir}/{name}.wav", self._resampler(gen_audio_noref), 16000
                            )
                        sig, bak, ovrl = eval_dnsmos(target_dir)
                        print(global_update, sig, bak, ovrl)
                        adjust_gain(target_dir, target_dir, -25)
                        sig, bak, ovrl = eval_dnsmos(target_dir)
                        print(global_update, sig, bak, ovrl)
                        with open(f"{target_dir}/mos.txt", 'w') as f:
                            f.write(f"sig: {sig}, bak: {bak}, ovrl: {ovrl}")

                        ######################## with_rev ####################################
                        for index in tqdm(range(len(self.data_list))):
                            data = self.rev_data_list[index]
                            noisy_path = data["noisy_path"]
                            name = data["name"]

                            degraded_audio, sr = torchaudio.load(noisy_path)
                            if sr != 24000:
                                degraded_audio = self.resampler(degraded_audio)

                            noisy_mel_spec = self.mel_spectrogram(degraded_audio).to(Q[0].device)

                            with torch.inference_mode():
                                generated_noref, _ = self.accelerator.unwrap_model(self.model).sample(
                                    cond=noisy_mel_spec.transpose(-1, -2),
                                    cond_noisy=noisy_mel_spec.transpose(-1, -2),
                                    inp=noisy_mel_spec.transpose(-1, -2),
                                    steps=8,
                                    cfg_strength=0.5,
                                    sway_sampling_coef=-1,
                                    Q=Q[0].unsqueeze(0),
                                )
                                generated_noref = generated_noref.to(torch.float32).permute(0, 2, 1)
                                gen_audio_noref = vocoder(generated_noref).squeeze(0).cpu()

                            torchaudio.save(
                                f"{rev_target_dir}/{name}.wav", self._resampler(gen_audio_noref), 16000
                            )
                        sig, bak, ovrl = eval_dnsmos(rev_target_dir)
                        print(global_update, sig, bak, ovrl)
                        adjust_gain(rev_target_dir, rev_target_dir, -25)
                        sig, bak, ovrl = eval_dnsmos(rev_target_dir)
                        print(global_update, sig, bak, ovrl)
                        with open(f"{rev_target_dir}/mos.txt", 'w') as f:
                            f.write(f"sig: {sig}, bak: {bak}, ovrl: {ovrl}")

                        self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
