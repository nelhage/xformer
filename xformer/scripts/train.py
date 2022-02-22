import xformer
import xformer.data
import xformer.loss

import os
import torch
import time
import itertools
import argparse
import glob
import wandb
from contextlib import nullcontext

from torch.profiler import profile, ProfilerAction

def main():
  parser = argparse.ArgumentParser(description="Train a transformer")
  parser.add_argument('--layers', type=int, default=2, help="Number of layers")
  parser.add_argument('--d_model', type=int, default=None, help="embedding dimension")
  parser.add_argument('--d_head', type=int, default=32, help="head dimension")
  parser.add_argument('--n_ctx', type=int, default=1024, help="maximum context length")
  parser.add_argument('--data', type=str, default='data/pile/chunked/train-*.pt', help="datasource")
  parser.add_argument('--batch', type=int, default=64, help="batch size")
  parser.add_argument('--minibatch', type=int, default=4, help="minibatch")
  parser.add_argument('--device', type=str, choices=('cpu', 'cuda'), default='cuda', help="device")
  parser.add_argument('--wandb', action='store_true', default=False)
  parser.add_argument('--no-wandb', action='store_false', dest='wandb')
  parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
  parser.add_argument('--pe', type=str, default=None, help="positional encoding (sin, learned)")
  parser.add_argument('--steps', type=int, default=None)
  parser.add_argument('--profile-steps', type=str, default=None)
  parser.add_argument('--tokens', type=int, default=None)

  args = parser.parse_args()

  cfg = xformer.Config(
    n_layer = args.layers,
    d_model = args.d_model or 128*args.layers,
    d_head = args.d_head,
    n_ctx = args.n_ctx,
    n_vocab = 256,
  )
  if args.pe is not None:
    cfg.positional_encoding = args.pe

  files = sorted(glob.glob(args.data))
  ds = xformer.data.PTDataset(files)
  loader = torch.utils.data.DataLoader(ds, batch_size=args.minibatch, pin_memory=True, num_workers=1)

  model = xformer.Transformer(cfg, dtype=torch.float32, device=args.device)

  opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

  assert args.batch % args.minibatch == 0, "minibatch must divide batch"
  steps_per_batch = args.batch // args.minibatch

  data = iter(xformer.data.ARProcessor(loader, device='cuda'))

  if args.wandb:
    run = wandb.init()
    wandb.watch(model, log_freq=100, log='gradients')
    wandb.config.update(args)
    wandb.config.update({"n_parameters": cfg.n_parameters})

  model.init_weights()
  param_bytes = 0
  for p in model.parameters():
    param_bytes += p.numel() * p.element_size()

  print(f"Training a {cfg.n_layer}L model with {cfg.n_parameters:,} non-embedding parameters...")
  print(f" Model params use {param_bytes/1024**3:.2f}GiB on device")

  start = time.time()
  tokens = 0

  steps = range(args.steps) if args.steps is not None else itertools.count()

  profile_steps = set()
  if args.profile_steps is not None:
    profile_steps = set(int(s) for s in args.profile_steps.split(','))
  def schedule(step):
    if step in profile_steps:
      print(f"Profiling step {step}...")
      return ProfilerAction.RECORD_AND_SAVE
    if step+1 in profile_steps:
      return ProfilerAction.WARMUP
    return ProfilerAction.NONE

  def save_profile(prof):
    os.makedirs('profile', 0o755, True)
    prof.export_chrome_trace(f"profile/step_{step_i}.pt.trace.json")

  profiler = profile(schedule=schedule,
                     with_stack=True,
                     on_trace_ready=save_profile)

  loss_fn = xformer.loss.ARLoss()

  with profiler:
    for step_i in steps:
      step_start = time.time()

      avg_loss = torch.zeros((), device=args.device)
      opt.zero_grad(set_to_none=True)
      for _ in range(steps_per_batch):
        record = next(data)
        batch = record['input']

        logits = model(batch)

        loss = loss_fn(logits, record)
        avg_loss += loss
        tokens += batch.numel()
        (loss / steps_per_batch).backward()
      opt.step()
      profiler.step()

      now = time.time()
      avg_loss = (avg_loss/steps_per_batch).item()
      print(f"[step={step_i:06d} t={now-start:.1f}s tokens={tokens:08d}] loss={avg_loss:2.2f} ms_per_step={1000*(now-step_start):.0f}")
      if args.wandb:
        wandb.log({
          'tokens': tokens,
          'elapsed_time': now-start,
          'train_loss': avg_loss,
        }, step=step_i)
      if args.tokens is not None and tokens >= args.tokens:
        break

if __name__ == '__main__':
  main()
