from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MuToRArguments:
    wandb: Optional[int] = field(
        default=0,
    )
    wandb_project: Optional[str] = field(
        default="MuToR",
    )
    wandb_name: Optional[str] = field(
        default="default",
    )
    save_dir: Optional[str] = field(
        default=None,
    )
    seed: Optional[int] = field(
        default=42,
    )
    model_hf: Optional[str] = field(
        default="google/gemma-2b",
    )
    dataset: Optional[str] = field(
        default="gsm8k",
    )
    batch_size: Optional[int] = field(
        default=1,
    )
    num_epochs: Optional[int] = field(
        default=5,
    )
    seq_length: Optional[int] = field(
        default=512,
    )
    beta1: Optional[float] = field(
        default=0.9,
    )
    beta2: Optional[float] = field(
        default=0.999,
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
    )
    num_warmup_steps: Optional[int] = field(
        default=1000,
    )
    max_train_steps: Optional[int] = field(
        default=10000,
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
    )
    weight_decay: Optional[float] = field(
        default=0.0,
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=10,
    )
    method_name: Optional[str] = field(
        default="MuToR",
    )
    max_offset: Optional[int] = field(
        default=3
    )
    min_offset: Optional[int] = field(
        default=0
    )
    aux_loss_coeff: Optional[float] = field(
        default=0.3
    )
    num_ckpts_saved: Optional[int] = field(
        default=5
    )
    math_split: Optional[int] = field(
        default=0
    )
    fsdp_distributed: Optional[int] = field(
        default=0
    )
    compile: Optional[int] = field(
        default=0
    )


def update_config(config, args):
    # put all args into config
    config.update(vars(args))
    return config
