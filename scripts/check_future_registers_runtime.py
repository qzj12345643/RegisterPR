"""Check the selected platform environment before allocating the frozen DiT."""
import argparse
import importlib
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--expected-env', required=True)
args = parser.parse_args()
if Path(sys.prefix).resolve() != Path(args.expected_env).resolve():
    raise SystemExit(f'Wrong environment: {sys.prefix}; expected {args.expected_env}')
print(f'Python: {sys.executable}', flush=True)
for name in ('torch', 'nuplan', 'diffusers', 'hydra', 'pytorch_lightning',
             'navsim.agents.worlddrive.future_register_agent'):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f'Cannot import {name}: {exc}. Install NAVSIM and CogVideoX dependencies in this environment.') from exc
    print(f'{name}: {getattr(module, "__file__", "namespace")}', flush=True)
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the selected environment.')
visible = torch.cuda.device_count()
print(f'Visible GPUs: {visible}; CUDA_VISIBLE_DEVICES={os.getenv("CUDA_VISIBLE_DEVICES")}', flush=True)
if visible < 4:
    raise SystemExit('This launcher requires four visible GPUs.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Default bf16-mixed configuration requires BF16-capable GPUs.')
x = torch.ones(8, device='cuda')
assert (x + x).sum().item() == 16
print('Runtime check passed.', flush=True)
