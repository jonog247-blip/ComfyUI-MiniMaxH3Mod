"""A stand-in for ``comfy.sd.VAE``'s MiniMax-H3 video VAE.

It reproduces the parts of the real contract the extractor depends on —
``[T, H, W, 3]`` pixels in, ``[1, 24, t, h, w]`` latents out, the pixel crop to
a multiple of the spatial ratio, the length-1 frame shortcut, ``output_device``,
and the per-sample encode-memory estimate — without the ~1 GB checkpoint, so
the CLI pipeline can be exercised end to end on a machine with no model and no
GPU.

``batch_unsafe=True`` leaks information across the batch dimension, which is
what a VAE that does not support a batch of independent stills would do. The
batched-encode guard is supposed to catch that and fall back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SPATIAL_RATIO = 16
LATENT_CHANNELS = 24


class _Patcher:
    def __init__(self, free_bytes):
        self.free_bytes = free_bytes

    def get_free_memory(self, device):
        return float(self.free_bytes)


class StubH3VAE:
    """Same call surface as ``comfy.sd.VAE`` for the paths the extractor uses."""

    def __init__(self, batch_unsafe: bool = False, free_bytes: int = 8 << 30,
                 device=None):
        self.crop_input = True
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.batch_unsafe = batch_unsafe
        self.patcher = _Patcher(free_bytes)
        self.calls = []            # shapes handed to the encoder, in order

    # ── comfy.sd.VAE surface ──────────────────────────────────────────
    def throw_exception_if_invalid(self):
        return None

    def spacial_compression_encode(self):
        return SPATIAL_RATIO

    @property
    def vae_dtype(self):
        return torch.float32

    def memory_used_encode(self, shape, dtype):
        # the real estimate for a single frame is ~110 MB of fixed cost plus
        # ~7 values per pixel; only the order of magnitude matters here
        return float((7 * max(1, shape[2]) * shape[3] * shape[4] + 110_000_000) * 4)

    def encode(self, pixel_samples):
        pixels = pixel_samples
        if self.crop_input:
            dims = pixels.shape[1:-1]
            for d in range(len(dims)):
                size = dims[d]
                cropped = (size // SPATIAL_RATIO) * SPATIAL_RATIO
                if cropped != size:
                    pixels = pixels.narrow(d + 1, (size % SPATIAL_RATIO) // 2, cropped)
        pixels = pixels[..., :3].movedim(-1, 1)
        if pixels.ndim == 4:  # [T, C, H, W] is one video, not a batch
            pixels = pixels.movedim(1, 0).unsqueeze(0)
        n, c, t, h, w = pixels.shape
        if t <= 0:
            raise RuntimeError("stub VAE got an empty frame axis (crop trap)")
        lat_t = 1 if t == 1 else max(1, (t - 5) // 17 * 5 + 2)
        self.calls.append(tuple(pixels.shape))
        # deterministic content summary: one channel, then broadcast to 24
        moment = pixels.float().mean(1, keepdim=True)
        latent = F.adaptive_avg_pool3d(moment, (lat_t, max(1, h // SPATIAL_RATIO),
                                                max(1, w // SPATIAL_RATIO)))
        latent = latent.repeat(1, LATENT_CHANNELS, 1, 1, 1)
        if self.batch_unsafe and latent.shape[0] > 1:
            latent = latent + latent.mean(0, keepdim=True)
        return latent.to(self.output_device)
