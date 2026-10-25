import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, get_args, get_origin

import wandb
from pydantic import BaseModel
from torch import nn


def parse_args[T: BaseModel](cfg_t: type[T]) -> T:
    """
    Parses arguments using a Pydantic model, similar to `clap-rs`.
    This could be replaced with `pydantic-settings`, but that package requires more configuration.
    """
    parser = ArgumentParser()
    for k, v in cfg_t.model_fields.items():
        flag_name = f"--{k.replace('_', '-')}"
        help_text = v.description
        if v.annotation is bool:
            parser.add_argument(
                flag_name, default=v.default, action="store_true", help=help_text
            )
        elif get_origin(v.annotation) == Literal:
            choices = list(get_args(v.annotation))
            parser.add_argument(
                flag_name,
                default=v.default,
                choices=choices,
                type=type(choices[0]),
                help=help_text,
            )
        else:
            assert v.annotation is not None
            parser.add_argument(
                flag_name,
                default=v.default,
                type=v.annotation,  # pyrefly: ignore[bad-argument-type]
                help=help_text,
            )
    args = parser.parse_args()
    cfg = cfg_t(**args.__dict__)
    return cfg


def create_directory[T: BaseModel](out_dir: str | os.PathLike, meta: T) -> Path:
    """Creates and returns a new directory for a run, and returns the checkpoints path."""
    # Run is not assigned a name immediately, so we poll
    for _ in range(100):
        if wandb.run and wandb.run.name not in ["" or None]:
            break

    # If a run doesn't get a name (e.g. if WAND_MODE=offline), use the testing directory instead
    if wandb.run and wandb.run.name not in ["" or None]:
        out_id = wandb.run.name
    else:
        out_id = "testing"

    out_dir = Path(out_dir)
    exp_dir = out_dir / out_id
    try:
        os.mkdir(exp_dir)
    except OSError as e:
        print(e)
    with open(exp_dir / "meta.json", "w") as f:
        f.write(meta.model_dump_json(indent=2))

    chkpt_path = exp_dir / "checkpoints"
    try:
        os.mkdir(chkpt_path)
    except OSError as e:
        print(e)
    return chkpt_path


def polyak_avg(src: nn.Module, dest: nn.Module, p: float):
    """
    Smoothly copies params from one model to another.
    At `p` = 1, `src` overwrites `dest`, at `p` = 0, nothing happens.
    """
    for dest_, src_ in zip(dest.parameters(), src.parameters()):
        dest_.data.copy_(p * src_.data + (1.0 - p) * dest_.data)
