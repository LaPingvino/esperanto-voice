#!/usr/bin/env python3
"""`python -m piper.train`, with two fixes and a readable progress log.

- Drops piper.train's "val_mos" ModelCheckpoint: with --model.mos_metric none
  that key is never logged and Lightning raises instead of skipping.
- Prints one plain line per optimizer step and per epoch to stdout:

      step 12  epoch 1  loss_g 31.20  loss_d 2.71  2.8 s/step  elapsed 0:01:05

  Lightning's progress bar writes nothing useful when stdout is a file, which
  left long runs with no sign of life. Now `tail -f train.log` shows speed and
  progress within a minute.

Usage is identical to `python -m piper.train`:  piper_train.py fit --data... --model...
"""

import sys
import time

from lightning.pytorch.callbacks import Callback

import piper.train.__main__ as piper_main


class ProgressPrinter(Callback):
    def __init__(self):
        self.t0 = self.last = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        m = trainer.callback_metrics
        parts = [f"step {trainer.global_step}", f"epoch {trainer.current_epoch}"]
        for k in ("loss_g", "loss_d", "train_mel"):
            if k in m:
                parts.append(f"{k} {float(m[k]):.2f}")
        parts.append(f"{now - self.last:.1f} s/step")
        parts.append(f"elapsed {time.strftime('%H:%M:%S', time.gmtime(now - self.t0))}")
        print("  ".join(parts), flush=True)
        self.last = now

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "val_mel" in m:
            print(f"epoch {trainer.current_epoch} done  val_mel {float(m['val_mel']):.4f}", flush=True)


def main():
    piper_main._DEFAULT_CALLBACKS[:] = [
        c for c in piper_main._DEFAULT_CALLBACKS if getattr(c, "monitor", None) != "val_mos"
    ] + [ProgressPrinter()]
    sys.argv[0] = "piper.train"
    piper_main.main()


if __name__ == "__main__":
    main()
