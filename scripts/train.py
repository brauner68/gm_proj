#!/usr/bin/env python3
"""
Non-interactive training entrypoint.

- All defaults (including paths) live in DEFAULTS.
- CLI arguments override DEFAULTS if provided.
- plot_loss_curve() saves to output_dir (safe for headless runs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.trainer import DiffusionTrainer
from src.config import get_config


# ----------------------------
# Defaults (EDIT HERE ONLY)
# ----------------------------
DEFAULTS = {
    "data_path": "~/gm_proj/data/nsynth_data/nsynth-valid",
    "output_dir": "~/gm_proj/results/runai_results/02",

    "lr": 1e-4,
    "epochs": 50,
    "batch_size": 8,
    "cfg_prob": 0.0,
    "save_interval": 10,
    "max_samples": None,
    "selected_families": ["mallet", "guitar"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train diffusion model (non-interactive).")

    # Paths (strings → expanded later)
    p.add_argument("--data-path", type=str, default=DEFAULTS["data_path"])
    p.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"])

    # Training params
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--cfg-prob", type=float, default=DEFAULTS["cfg_prob"])
    p.add_argument("--save-interval", type=int, default=DEFAULTS["save_interval"])
    p.add_argument("--max-samples", type=int, default=DEFAULTS["max_samples"])

    p.add_argument(
        "--families",
        nargs="*",
        default=DEFAULTS["selected_families"],
        help="NSynth families, e.g. --families string mallet guitar",
    )

    return p.parse_args()


def validate_nsynth_layout(data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")
    if not (data_path / "audio").exists():
        raise FileNotFoundError(f"Expected 'audio/' under: {data_path}")
    if not (data_path / "examples.json").exists():
        raise FileNotFoundError(f"Expected 'examples.json' under: {data_path}")


def main() -> None:
    args = parse_args()

    # ---- Resolve paths (expand ~ once, centrally) ----
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # ---- Build config ----
    config = get_config()
    config["data_path"] = str(data_path)
    config["output_dir"] = str(output_dir)

    config["lr"] = args.lr
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["cfg_prob"] = args.cfg_prob
    config["save_interval"] = args.save_interval
    config["max_samples"] = args.max_samples
    config["selected_families"] = args.families

    # ---- Sanity checks ----
    validate_nsynth_layout(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting Training...")
    print(f"   data_path:     {data_path}")
    print(f"   output_dir:    {output_dir}")
    print(f"   families:      {config['selected_families']}")
    print(f"   batch_size:    {config['batch_size']}")
    print(f"   epochs:        {config['epochs']}")
    print(f"   lr:            {config['lr']}")
    print(f"   cfg_prob:      {config['cfg_prob']}")
    print(f"   max_samples:   {config['max_samples']}")
    print(f"   save_interval: {config['save_interval']}")

    # ---- Run ----
    trainer = DiffusionTrainer(config)
    trainer.train()
    trainer.plot_loss_curve()   # saves to output_dir


if __name__ == "__main__":
    main()
