#!/usr/bin/env python3
"""`python -m piper.train`, with fixes, a readable progress log, and a fast
text-only training phase.

Fixes and additions over `python -m piper.train`:

- Drops piper.train's "val_mos" ModelCheckpoint: with --model.mos_metric none
  that key is never logged and Lightning raises instead of skipping.
- Prints one plain line per optimizer step and per validation to stdout
  (Lightning's progress bar writes nothing useful when stdout is a file).
- PIPER_CKPT_TOP_K (default 1): how many best-by-val_mel checkpoints to keep.
  piper keeps 5; at ~850 MB each that is real disk and I/O time per epoch.
- PIPER_PHASE=text: the fast phase. VITS splits into a language-independent
  acoustic side (posterior encoder: audio → latent; decoder/vocoder: latent →
  audio, with discriminators to sharpen it) and a language side (text
  encoder, flow, duration predictor: phonemes → latent). Learning Esperanto
  pronunciation only needs the language side, whose losses (KL + duration)
  never touch the decoder. This phase freezes the posterior encoder and
  decoder and skips the decoder and both discriminators entirely. Validation
  still runs the full model, so val_mel stays comparable across phases.
  Polish the voice afterwards with a normal (full) run from that checkpoint.

Usage is identical to `python -m piper.train`:  piper_train.py fit --data... --model...
"""

import math
import os
import sys
import time

import torch
from lightning.pytorch.callbacks import Callback

import piper.train.__main__ as piper_main
from piper.train.vits import commons, monotonic_align
from piper.train.vits.lightning import VitsModel
from piper.train.vits.losses import kl_loss


class ProgressPrinter(Callback):
    def __init__(self):
        self.t0 = self.last = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        m = trainer.callback_metrics
        parts = [f"step {trainer.global_step}", f"epoch {trainer.current_epoch}"]
        for k in ("loss_g", "loss_d", "train_mel", "train_kl", "train_dur"):
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


class SaveOnEnd(Callback):
    """Write last.ckpt when training stops, not only at epoch ends.

    With --trainer.max_time the run stops mid-epoch, and Lightning's own
    last.ckpt is then up to a full epoch old — for small datasets a big
    slice of the training time.
    """

    def on_train_end(self, trainer, pl_module):
        cb = trainer.checkpoint_callback
        if cb is not None and cb.dirpath:
            path = os.path.join(cb.dirpath, "last.ckpt")
            trainer.save_checkpoint(path)
            print(f"saved final checkpoint at step {trainer.global_step} → {path}", flush=True)


class SaveLatest(Callback):
    """Write latest.ckpt every PIPER_LATEST_EVERY epochs (default 5).

    Lightning's last.ckpt is only rewritten when the monitored val_mel
    improves; on a tiny validation set that can stall for many epochs, so
    mid-run peeks silently got stale checkpoints.
    """

    def __init__(self):
        self.every = int(os.environ.get("PIPER_LATEST_EVERY", "5"))

    def on_train_epoch_end(self, trainer, pl_module):
        cb = trainer.checkpoint_callback
        if self.every and cb is not None and cb.dirpath and (trainer.current_epoch + 1) % self.every == 0:
            tmp = os.path.join(cb.dirpath, "latest.ckpt.tmp")
            trainer.save_checkpoint(tmp)
            os.replace(tmp, os.path.join(cb.dirpath, "latest.ckpt"))   # atomic: peeks never see half a file


class FreezeAcoustic(Callback):
    """Text phase: freeze the language-independent acoustic modules."""

    def on_train_start(self, trainer, pl_module):
        g = pl_module.model_g
        for mod in (g.enc_q, g.dec):
            mod.requires_grad_(False)
            mod.eval()
        n = sum(p.numel() for p in g.parameters() if p.requires_grad)
        print(f"text phase: training {n/1e6:.1f}M generator parameters "
              f"(posterior encoder and decoder frozen, discriminators skipped)", flush=True)


def text_phase_training_step(self, batch, batch_idx):
    """KL + duration losses only; mirrors SynthesizerTrn.forward minus the decoder."""
    opt_g, _opt_d = self.optimizers()
    g_model = self.model_g
    x, x_lengths = batch.phoneme_ids, batch.phoneme_lengths
    spec, spec_lengths = batch.spectrograms, batch.spectrogram_lengths
    g = None
    if g_model.n_speakers > 1 and batch.speaker_ids is not None:
        g = g_model.emb_g(batch.speaker_ids).unsqueeze(-1)

    x, m_p, logs_p, x_mask = g_model.enc_p(x, x_lengths)
    with torch.no_grad():
        z, _m_q, logs_q, y_mask = g_model.enc_q(spec, spec_lengths, g=g)
    z_p = g_model.flow(z, y_mask, g=g)

    with torch.no_grad():
        s_p_sq_r = torch.exp(-2 * logs_p)
        neg_cent = (
            torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            + torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            + torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            + torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    if g_model.use_sdp:
        l_length = g_model.dp(x, x_mask, w, g=g) / torch.sum(x_mask)
    else:
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = g_model.dp(x, x_mask, g=g)
        l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
    loss_dur = torch.sum(l_length.float())
    loss = kl * self.hparams.c_kl + loss_dur

    self.log("loss_g", loss, batch_size=self.batch_size)
    self.log_dict({"train_kl": kl.detach(), "train_dur": loss_dur.detach()},
                  batch_size=self.batch_size)
    opt_g.zero_grad()
    self.manual_backward(loss)
    opt_g.step()


def main():
    callbacks = [c for c in piper_main._DEFAULT_CALLBACKS
                 if getattr(c, "monitor", None) != "val_mos"]
    top_k = int(os.environ.get("PIPER_CKPT_TOP_K", "1"))
    for c in callbacks:
        if getattr(c, "monitor", None) == "val_mel":
            c.save_top_k = top_k
    callbacks.append(ProgressPrinter())
    callbacks.append(SaveOnEnd())
    callbacks.append(SaveLatest())

    if os.environ.get("PIPER_PHASE") == "text":
        VitsModel.training_step = text_phase_training_step
        callbacks.append(FreezeAcoustic())

    piper_main._DEFAULT_CALLBACKS[:] = callbacks
    sys.argv[0] = "piper.train"
    piper_main.main()


if __name__ == "__main__":
    main()
