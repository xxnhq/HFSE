import hydra
from model.cfm import CFM
from model.dit import DiT
from model.trainer import Trainer
from model.dataset import load_dataset


@hydra.main(version_base="1.3", config_path="configs", config_name="HFMSE.yaml")
def main(model_cfg):
    model_arc = model_cfg.model.arch
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    # set model
    model = CFM(
        transformer=DiT(**model_arc, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
    )

    # init trainer
    trainer = Trainer_CFM(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=f'{model_cfg.model.name}',
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        log_sample_rate=model_cfg.ckpts.log_sample_rate,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
    )

    train_dataset = load_dataset(
        mel_spec_kwargs=model_cfg.model.mel_spec
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()