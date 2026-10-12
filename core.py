"""
core.py — "RefMod" adapter for MiniMax H3 (no training required)

The reference-video path of MiniMax H3 works by injecting *reference tokens*
into the packed sequence: the ref is VAE-encoded, patchified, projected and
placed on the 3D RoPE grid, then every DiT block attends to it.  A video ref
is expensive because it contributes thousands of tokens.

Mode naming: ``encode`` (was ``full``) = straight full-res VAE encode;
``training`` (was ``pooled``) = compressed grid refined by gradient steps
(the only mode that "trains" — the refinement loop).  The old names are
accepted everywhere as legacy aliases so saved workflows and mods keep
working.

A RefMod is the same reference, compressed to a handful of tokens:

  * the ref is VAE-encoded to its full latent [1, 24, T, H, W],
  * the latent is average-pooled to a tiny grid (default 4x4) and a few
    latent frames, so the patchified token count drops to ~4-16,
  * optionally the small latent is refined with a few gradient steps that
    reconstruct the full latent (still no model weights involved).

At generation time the pooled latent is handed back to the model through the
native ``refs`` payload, so it flows through the exact same per-block
attention machinery as a full reference — the only thing that changes is the
token budget.

This is the MiniMax H3 analog of the LTX "Mod" concept tokens + per-block
injection: instead of a trained hypernetwork predicting AdaLN deltas, the
model's own cross-attention over the compressed ref tokens does the work.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Metadata key embedded in the safetensors header (keeps a mod in one file,
# so it can be shared/uploaded as a single artifact).
META_KEY = "refmod_meta"

# What a mod is standing in for. This is documentation + a routing hint, not
# a hard constraint on the tensors — but it drives two things: (1) the
# "identity in training mode" warning in nodes.py, and (2) the loader's
# prompt_hint output, which merges each loaded mod's concept_type +
# description into a string you can concat onto your CLIP Text Encode prompt.
# There's no CLIP-Vision / image-embedding injection point on H3's ref2va
# path to hang a "visual clue" off of — text is the only channel the model's
# text encoder reads, so that's the channel this uses.
CONCEPT_TYPES = (
    "generic",       # unspecified / mixed
    "identity",      # a specific person/character — use mode="encode" for this
    "pose_motion",   # a pose, dance, gesture, camera move
    "clothing",      # an outfit / garment, decoupled from who's wearing it
    "background",    # environment / set / location plate
    "voice", "singing", "music_style", "sound_fx", "ambience",
    "style",         # look/grade/animation style, not a concrete subject
)

# Spelling variants accepted everywhere a concept type is entered (CLI flags,
# node widgets, saved metadata). The RefMod creation guide documents
# ``identity`` / ``style`` / ``general`` — ``general`` is this pack's
# ``generic`` — and users reach for the plain-English names below. Everything
# is stored canonically, so a mod written as "general" reads back as "generic"
# and old library filters keep matching.
CONCEPT_TYPE_ALIASES = {
    "general": "generic",
    "mixed": "generic",
    "unknown": "generic",
    "character": "identity",
    "person": "identity",
    "face": "identity",
    "portrait": "identity",
    "pose": "pose_motion",
    "motion": "pose_motion",
    "dance": "pose_motion",
    "camera_move": "pose_motion",
    "outfit": "clothing",
    "garment": "clothing",
    "environment": "background",
    "set": "background",
    "location": "background",
    "art_style": "style",
    "look": "style",
    "grade": "style",
    "speech": "voice",
    "vocal": "singing",
    "song": "music_style",
    "music": "music_style",
    "sfx": "sound_fx",
    "ambient": "ambience",
}

# What the CLIs offer: canonical names plus the accepted aliases.
CONCEPT_TYPE_CHOICES = tuple(CONCEPT_TYPES) + tuple(sorted(CONCEPT_TYPE_ALIASES))


def _slug(value) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_concept_type(value) -> Optional[str]:
    """Canonical concept type for ``value``, or None if it isn't one of ours.

    Unknown strings return None rather than being coerced, so callers can
    decide: loaders keep a foreign/custom label as-is, while the CLIs reject
    it with the list of valid names.
    """
    if not isinstance(value, str):
        return None
    key = _slug(value)
    if not key:
        return None
    if key in CONCEPT_TYPES:
        return key
    return CONCEPT_TYPE_ALIASES.get(key)


def _blur_latent(z: torch.Tensor, factor: int = 8) -> torch.Tensor:
    """Heavy spatial low-pass (downsample then upsample) used as the
    weakening target for ``ref_block``'s ``strength`` below ``1.0``.

    Originally this mixed toward ``torch.randn()`` noise, mirroring what the
    docstring below describes as the model's own ``visual_cond_noise_aug``
    mechanism. In practice, at the retention range people actually use this
    dial for (0.15-0.7, i.e. large noise fractions), it produces a visible
    woven/static texture instead of a gracefully weaker reference: a VAE
    latent's channels are correlated, so raw iid noise isn't read as "less
    reference," it's read as real-but-garbled content and gets rendered as
    an actual (wrong) texture. ``visual_cond_noise_aug`` in the model's own
    training is likely a small-magnitude robustness augmentation, not
    something swept this far — so copying its math doesn't reproduce its
    behavior at these strengths. A blurred copy stays on the latent
    manifold (smooth, plausible) while still discarding detail as strength
    drops, which is what "weaker reference" should actually look like.
    """
    if z.device.type == "mps":
        # MPS lacks adaptive_avg_pool3d (and some non-divisible pooling
        # shapes). Keep this small preprocessing chain on CPU explicitly,
        # then restore the caller's device; global MPS fallback is not needed.
        return _blur_latent(z.cpu(), factor).to(z.device)
    if z.dim() == 4:
        b, c, stereo, t = z.shape
        if t <= 1:
            return z
        flat = z.reshape(b * c * stereo, 1, t).float()
        down = F.adaptive_avg_pool1d(flat, max(1, t // factor))
        return F.interpolate(down, size=t, mode="linear", align_corners=False).reshape_as(z).to(z.dtype)
    if z.dim() != 5:
        raise ValueError(f"Unsupported RefMod latent shape: {tuple(z.shape)}")
    t, h, w = z.shape[2], z.shape[3], z.shape[4]
    sh, sw = max(1, h // factor), max(1, w // factor)
    down = F.adaptive_avg_pool3d(z.float(), (t, sh, sw))
    up = F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False)
    return up.to(z.dtype)


# Memoized blur for the generation hot path. The Step Curve wrapper re-mixes
# every ref latent once per denoising step (and once per CFG branch), but the
# blurred copy of a given latent never changes — it only depends on the latent
# itself. Caching it turns ``steps x refs x cfg`` blurs into one per ref, which
# matters most on MPS where _blur_latent has to round-trip through CPU.
#
# Entries are keyed by object identity *and* keep a reference to the source
# tensor, so a recycled id()/data_ptr() can never alias a stale blur. Bounded
# by count and by total bytes; Step Curve clears it when a new wrapper is
# built so a long-running server doesn't accumulate ref latents.
_BLUR_CACHE: Dict = {}
_BLUR_CACHE_MAX = 12
_BLUR_CACHE_BYTES = 512 * 1024 * 1024
_BLUR_CACHE_FACTOR = 8


def clear_blur_cache() -> None:
    """Drop every cached blurred latent (call between sampling runs)."""
    _BLUR_CACHE.clear()


def blur_latent_cached(z: torch.Tensor, factor: int = _BLUR_CACHE_FACTOR) -> torch.Tensor:
    """``_blur_latent(z)`` memoized per source tensor object.

    Same result as calling ``_blur_latent`` directly; only safe for latents
    that are never mutated in place (RefMod payload latents aren't — the
    strength mix always allocates a new tensor).
    """
    if factor != _BLUR_CACHE_FACTOR:
        return _blur_latent(z, factor)
    device = z.device
    key = (id(z), z.data_ptr(), tuple(z.shape), z.dtype,
           device.type, device.index if device.type != "cpu" else -1)
    hit = _BLUR_CACHE.get(key)
    if hit is not None and hit[0] is z:
        return hit[1]
    blurred = _blur_latent(z, factor)
    _BLUR_CACHE[key] = (z, blurred)
    while _BLUR_CACHE and (
            len(_BLUR_CACHE) > _BLUR_CACHE_MAX
            or sum(v[1].numel() * v[1].element_size()
                   for v in _BLUR_CACHE.values()) > _BLUR_CACHE_BYTES):
        _BLUR_CACHE.pop(next(iter(_BLUR_CACHE)))
    return blurred


def read_refmod_meta(path_no_ext: str) -> Optional[Dict]:
    """
    Read the metadata block from a mod/preset file without loading tensors.

    Prefers the metadata embedded in the safetensors header; falls back to
    the legacy sidecar ``{path}.json`` written by older versions.
    """
    try:
        with safe_open(path_no_ext + ".safetensors", framework="pt") as f:
            meta = f.metadata()
        if meta:
            for key in (META_KEY, "audio_refmod_meta"):
                if key in meta:
                    return json.loads(meta[key])
    except Exception:
        pass
    jpath = path_no_ext + ".json"
    if os.path.isfile(jpath):
        try:
            with open(jpath) as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ═══════════════════════════════════════════════════════════════════════════
# Latent compression
# ═══════════════════════════════════════════════════════════════════════════

MODE_ALIASES = {"full": "encode", "pooled": "training",
                "Full Reference": "encode", "Compressed Reference": "training"}


def normalize_mode(mode: str) -> str:
    """Map legacy mode names (``full``/``pooled``) to the current ones."""
    return MODE_ALIASES.get(mode, mode)


def aspect_grid(pool_h: int, pool_w: int, aspect: float):
    """Even pool grid dims that match ``aspect`` (h/w) within the dial caps.

    The DiT patches each 2x2 latent cell into one token and the rope grid is
    area-normalized per axis, so a pooled grid that ignores the source aspect
    squishes the subject: a portrait latent (tall person) forced into a square
    16x16 grid is compressed 5x in height but only 3x in width, and comes out
    "fat".  The official ref2video node never pools — it feeds the model
    aspect-correct latent dims, which is what the DiT was trained on.

    The long edge follows the user's dial (``max(pool_h, pool_w)``) and the
    short edge is derived from the source aspect, both rounded to even for the
    2x2 patch.  Square sources keep the exact dial value (16x16 stays 16x16).
    """
    long_edge = max(pool_h, pool_w)
    if aspect >= 1.0:
        h, w = long_edge, long_edge / aspect
    else:
        w, h = long_edge, long_edge * aspect
    h = max(2, round(h / 2) * 2)
    w = max(2, round(w / 2) * 2)
    return h, w


def pool_latent(
    z: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """
    Average-pool a VAE latent ``[1, 24, T, H, W]`` down to a tiny grid.

    ``latent_h/latent_w`` are the *latent* grid dims (the DiT patches each 2x2
    latent cell into one token), so a 4x4 latent = 2x2 = 4 tokens per frame.
    Dims must be even (the DiT's 2x2 patch).  Pooling runs in fp32; the result
    keeps the input dtype.
    """
    if z.shape[2] == latent_t and z.shape[3] == latent_h and z.shape[4] == latent_w:
        return z
    if latent_h % 2 != 0 or latent_w % 2 != 0:
        raise ValueError(f"pool_h/pool_w must be even (got {latent_h}x{latent_w})")
    # Normal ComfyUI intermediates are already on CPU. GPU-only workflows can
    # supply MPS latents, where adaptive_avg_pool3d is not implemented.
    work = z.cpu() if z.device.type == "mps" else z
    pooled = F.adaptive_avg_pool3d(work.float(), (latent_t, latent_h, latent_w))
    return pooled.to(device=z.device, dtype=z.dtype)


def _make_optimizer(params, lr: float, device: Optional[torch.device]):
    """Adam, using CUDA's fused kernel when it applies.

    The refinement loop is hundreds of tiny steps on a ~1-2K parameter tensor,
    so it is pure kernel-launch overhead on a GPU. ``fused=True`` collapses
    Adam's per-parameter update into one kernel; ``foreach`` is the next best
    thing on other backends. Numerically the same update rule, just fewer
    launches (measured ~3x fewer CPU-side dispatches per step on CUDA).
    """
    if device is not None and device.type == "cuda":
        try:
            return torch.optim.Adam(params, lr=lr, fused=True)
        except (RuntimeError, TypeError, ValueError):
            pass
    try:
        return torch.optim.Adam(params, lr=lr, foreach=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.optim.Adam(params, lr=lr)


def optimize_latent(
    z_small: torch.Tensor,
    z_full: torch.Tensor,
    steps: int = 150,
    lr: float = 0.02,
    device: Optional[torch.device] = None,
    progress_every: int = 0,
    tol: float = 0.0,
) -> torch.Tensor:
    """
    Model-free refinement of the compressed latent.

    Optimizes the small latent so its trilinearly upsampled reconstruction
    matches the full reference latent.  This pulls the pooled representation
    closer to what the DiT would see from the full ref, with nothing but the
    tiny latent trainable (~1-2K params) and no diffusion model loaded.

    ``progress_every`` > 0 prints a ``[RefMod] identity <step>/<steps>`` line
    every N steps so long pooled-mode refinement isn't silent (the default
    ``identity=500`` takes a while per ref on CPU/VRAM-bound setups).

    ``tol`` > 0 stops early once the loss moves less than that *relative*
    amount between checks (every 10 steps): Adam on this objective flattens
    out long before the step budget, so the usual 500-step identity refine
    finishes in a fraction of the steps with the same converged latent.
    0 (the default) keeps the exact old "run every step" behavior.

    Returns the refined latent detached, same shape/dtype as ``z_small``.
    """
    if steps <= 0:
        return z_small
    device = device or z_full.device
    # Node execution runs under ComfyUI's global inference mode, which would
    # disable the autograd this refine loop needs.  Re-enable it for this
    # scope and clone the inputs to drop the inference flag.
    with torch.inference_mode(False), torch.set_grad_enabled(True):
        target = z_full.clone().float().to(device)
        param = nn.Parameter(z_small.clone().float().to(device))
        opt = _make_optimizer([param], lr, device)
        size = tuple(target.shape[2:])
        last = None
        for i in range(steps):
            opt.zero_grad()
            up = F.interpolate(param, size=size, mode="trilinear", align_corners=False)
            loss = F.mse_loss(up, target)
            loss.backward()
            opt.step()
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[RefMod] identity {i + 1}/{steps}")
            if tol > 0 and (i + 1) % 10 == 0:
                value = float(loss.detach())
                if last is not None and abs(last - value) <= tol * max(last, 1e-12):
                    print(f"[RefMod] identity converged at step {i + 1}/{steps} "
                          f"(loss {value:.3e}, tol {tol:g})")
                    break
                last = value
        # materialize inside the scope so the result is a normal tensor, not
        # an inference-mode tensor (it gets stored in the mod and reused)
        refined = param.detach().to(z_small.dtype)
    return refined


def optimize_latent_multi(z_init, targets, steps=150, lr=0.02, device=None,
                          progress_every=0, strategy="grouped", tol=0.0):
    """Mean reconstruction objective with bounded activation memory.

    grouped: average targets with identical shapes on CPU before refinement.
    stream: transfer one target per gradient contribution.
    resident: retain targets on the compute device, backward one at a time.
    All strategies optimize the same mean MSE; grouping drops a constant term.
    ``tol`` > 0 stops early on a flat loss (see ``optimize_latent``).
    """
    if not targets or steps <= 0:
        return z_init
    if strategy not in ("grouped", "stream", "resident"):
        raise ValueError("Unknown multi-reference optimization strategy.")
    device = device or z_init.device
    with torch.inference_mode(False), torch.set_grad_enabled(True):
        count = len(targets)
        if strategy == "grouped":
            groups = {}
            for target in targets:
                key = tuple(target.shape)
                value = target.detach().to(device="cpu", dtype=torch.float32).clone()
                if key in groups:
                    groups[key][0].add_(value)
                    groups[key][1] += 1
                else:
                    groups[key] = [value, 1]
            prepared = [(total.div_(n), n / count) for total, n in groups.values()]
        else:
            target_device = device if strategy == "resident" else "cpu"
            prepared = [(t.detach().to(device=target_device, dtype=torch.float32).clone(), 1 / count) for t in targets]
        param = nn.Parameter(z_init.clone().float().to(device))
        opt = _make_optimizer([param], lr, device)
        last = None
        for i in range(steps):
            opt.zero_grad()
            # only read the loss back (a device sync per target) on the steps
            # the early-exit check actually looks at
            watch = tol > 0 and (i + 1) % 10 == 0
            total = None
            for target, weight in prepared:
                current = target.to(device)
                up = F.interpolate(param, size=tuple(current.shape[2:]), mode="trilinear", align_corners=False)
                term = F.mse_loss(up, current) * weight
                term.backward()
                if watch:
                    total = float(term.detach()) if total is None else total + float(term.detach())
                del up, current
            opt.step()
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[RefMod] merge {i + 1}/{steps}")
            if watch and total is not None:
                if last is not None and abs(last - total) <= tol * max(last, 1e-12):
                    print(f"[RefMod] merge converged at step {i + 1}/{steps} "
                          f"(loss {total:.3e}, tol {tol:g})")
                    break
                last = total
        return param.detach().to(z_init.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Per-frame strength curve
# ═══════════════════════════════════════════════════════════════════════════

# Direction names describe where the *concept* shows up in the output, and
# the envelope is strong exactly where the concept lands: concept_at_start
# keeps the ref at full strength at the START of the video and releases it
# toward the end, concept_at_end fades it in toward the END. The old
# "decrease"/"increase" envelopes are kept as legacy aliases (decrease =
# concept_at_start, increase = concept_at_end) so pre-rename workflows keep
# their behavior. shape is how the envelope travels between its endpoints;
# value is the non-zero endpoint ("user input").
CURVE_DIRECTIONS = ("constant", "concept_at_start", "concept_at_middle",
                    "concept_at_end", "concept_at_ends")
CURVE_SHAPES = ("linear", "ease", "sigmoid", "tanh", "quadratic", "cubic",
                "exponential", "stair", "elastic", "bump", "dip")

# old single-combo preset names -> (direction, shape, value) so workflows
# saved before the curve was split still resolve
_LEGACY_CURVES = {
    "flat": ("constant", "linear", 1.0),
    # keep each preset's historical envelope: fade_in = crescent (0 -> value),
    # fade_out = decrescent (value -> 0), bump = mid peak
    "fade_in": ("concept_at_end", "linear", 1.0),
    "fade_out": ("concept_at_start", "linear", 1.0),
    "bump": ("concept_at_end", "bump", 1.0),
    "dip": ("constant", "dip", 1.0),
}


def _ease(shape: str, x: float) -> float:
    """Shape a progress ``x`` in [0, 1] per the easing name (may overshoot)."""
    if shape == "linear":
        return x
    if shape == "ease":  # smoothstep
        return x * x * (3.0 - 2.0 * x)
    if shape == "sigmoid":  # logistic S-curve
        return 1.0 / (1.0 + math.exp(-12.0 * (x - 0.5)))
    if shape == "tanh":  # sharper S-curve (steeper knee than sigmoid)
        return 0.5 * (math.tanh(8.0 * (x - 0.5)) + 1.0)
    if shape == "quadratic":
        return x * x
    if shape == "cubic":
        return x * x * x
    if shape == "exponential":
        return 2.0 ** x - 1.0
    if shape == "stair":
        return min(1.0, math.floor(x * 4) / 3.0)
    if shape == "elastic":
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        return 2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
    if shape == "bump":
        return 1.0 - abs(2.0 * x - 1.0)
    if shape == "dip":
        return abs(2.0 * x - 1.0)
    return x


def curve_strengths(spec, t: int) -> Optional[List[float]]:
    """Resolve a curve spec to ``t`` per-frame strength multipliers in [0, 1].

    Kept as plain Python so the Apply node exposes the curve as normal
    combo/float widgets instead of depending on ComfyUI's new Curve widget.
    Accepts:

      * a ``(direction, shape, value)`` tuple — direction is ``constant``
        (one strength everywhere), ``concept_at_start`` (value -> 0, a
        decrescent — the concept shows in the first half; legacy
        ``decrease``), ``concept_at_end`` (0 -> value, a crescent — the
        concept shows in the second half; legacy ``increase``),
        ``concept_at_middle`` ([0..1..0] — concept peaks mid-timeline) or
        ``concept_at_ends`` ([1..0..1] — concept at both ends); shape is how
        the envelope travels between its endpoints (see ``_ease``); value is
        the non-zero endpoint (1.0 = full strength there);
      * a legacy preset name (``flat``/``fade_in``/``fade_out``/``bump``/
        ``dip``) from before the split;
      * a list of per-frame floats (len == t), used as-is;
      * a list of ``(x, y)`` control points (x in [0, 1]), linearly
        interpolated across the frames;
      * any object with ``interp(x) -> float`` (legacy CurveInput saved in an
        old workflow).

    Returns None for flat/no curve so the caller keeps its single-strength
    path unchanged (a ``constant`` curve at value 1.0 included).
    """
    if t <= 1 or spec is None or spec == "":
        return None
    if isinstance(spec, str):
        legacy = _LEGACY_CURVES.get(spec)
        return curve_strengths(legacy, t) if legacy is not None else None
    if (isinstance(spec, tuple) and len(spec) == 3
            and isinstance(spec[0], str) and isinstance(spec[1], str)):
        direction, shape, value = spec
        value = float(value)
        if direction == "constant":
            # linear = flat at ``value`` (1.0 == no curve); the other shapes
            # modulate around the constant, so constant + dip is a trough,
            # constant + bump is a peak (both endpoints equal)
            if shape == "linear":
                return None if value >= 1.0 else [value] * t
            p = [_ease(shape, i / (t - 1)) for i in range(t)]
            return [max(0.0, min(1.0, value * y)) for y in p]
        p = [_ease(shape, i / (t - 1)) for i in range(t)]
        # legacy "decrease"/"increase" (pre-rename workflows) keep their
        # original envelopes: decrease = strong at the start, increase =
        # strong at the end.  The concept_* names mean exactly what they say.
        if direction in ("concept_at_end", "increase"):
            return [max(0.0, min(1.0, value * y)) for y in p]
        if direction in ("concept_at_start", "decrease"):
            return [max(0.0, min(1.0, value * (1.0 - y))) for y in p]
        if direction == "concept_at_middle":  # [0..1..0]: concept peaks mid-timeline
            return [max(0.0, min(1.0, value * (1.0 - abs(2.0 * y - 1.0)))) for y in p]
        if direction == "concept_at_ends":  # [1..0..1]: concept at both ends
            return [max(0.0, min(1.0, value * abs(2.0 * y - 1.0))) for y in p]
        return None  # unknown direction: treat as flat
    if isinstance(spec, (list, tuple)):
        if len(spec) == t and all(isinstance(v, (int, float)) for v in spec):
            return [float(v) for v in spec]
        pts = [(float(x), float(y)) for x, y in spec
               if isinstance(x, (int, float)) and isinstance(y, (int, float))]
        if pts:
            pts = sorted(pts, key=lambda p: p[0])
            out = []
            for i in range(t):
                x = i * (1.0 / (t - 1))
                if x <= pts[0][0]:
                    out.append(pts[0][1])
                elif x >= pts[-1][0]:
                    out.append(pts[-1][1])
                else:
                    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                        if x0 <= x <= x1:
                            out.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
                            break
            return out
    interp = getattr(spec, "interp", None)
    if callable(interp):
        return [float(interp(i / (t - 1))) for i in range(t)]
    return None


def curve_value_at(spec, x: float) -> float:
    """Per-progress strength of a curve spec at one point ``x`` in [0, 1].

    The continuous analog of ``curve_strengths`` — same direction/shape/value
    math, but evaluated at a single progress value instead of discretized over
    frames.  The step-curve node uses it to re-mix the ref latents once per
    denoising step, where ``x`` is the denoise progress (0 = first step,
    1 = last step).  Returns 1.0 (no weakening) for anything unrecognized.
    """
    if spec is None or spec == "":
        return 1.0
    if isinstance(spec, str):
        legacy = _LEGACY_CURVES.get(spec)
        return curve_value_at(legacy, x) if legacy is not None else 1.0
    if (isinstance(spec, tuple) and len(spec) == 3
            and isinstance(spec[0], str) and isinstance(spec[1], str)):
        direction, shape, value = spec
        value = float(value)
        if direction == "constant":
            if shape == "linear":
                return value
            return max(0.0, min(1.0, value * _ease(shape, x)))
        y = _ease(shape, x)
        if direction in ("concept_at_end", "increase"):
            return max(0.0, min(1.0, value * y))
        if direction in ("concept_at_start", "decrease"):
            return max(0.0, min(1.0, value * (1.0 - y)))
        if direction == "concept_at_middle":  # [0..1..0]: concept peaks mid-timeline
            return max(0.0, min(1.0, value * (1.0 - abs(2.0 * y - 1.0))))
        if direction == "concept_at_ends":  # [1..0..1]: concept at both ends
            return max(0.0, min(1.0, value * abs(2.0 * y - 1.0)))
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Token budget cap (shared by the Extract node and extract_mod.py)
# ═══════════════════════════════════════════════════════════════════════════


def dedup_frame_indices(z: torch.Tensor, threshold: float = 0.02) -> List[int]:
    """Indices of latent frames kept by greedy temporal dedup.

    Each frame is compared to the last *kept* frame: if the mean-abs
    difference between them, normalized by the frames' own magnitude, is
    below ``threshold`` the frame is dropped as a (near-)duplicate.  Video
    refs (a dance loop, a static shot, a talking head) contain long runs of
    frames that differ only by codec noise — each one still costs a token
    per spatial patch in every DiT block, so cutting them is the cheapest
    way to honor a token cap.  Stacked image refs survive: different
    angles/expressions land well above the threshold.
    """
    t = z.shape[2]
    if t <= 1:
        return list(range(t))
    flat = z[0].float()  # [24, T, H, W]
    kept = [0]
    prev = flat[:, 0]
    for i in range(1, t):
        cur = flat[:, i]
        denom = (cur.abs().mean() + prev.abs().mean()) / 2 + 1e-6
        diff = (cur - prev).abs().mean() / denom
        if diff >= threshold:
            kept.append(i)
            prev = cur
    return kept


def select_frames_diverse(z: torch.Tensor, keep: int) -> List[int]:
    """``keep`` frame indices chosen to stay as different from each other as possible.

    Greedy k-center: start from the frame furthest from the stack's mean, then
    repeatedly add the frame whose distance to the already-chosen set is
    largest.  When a token cap forces a cut on a stack of *still* references
    (the identity case: 8-20 angles/expressions of one person), uniform
    subsampling keeps whatever happens to sit at even positions — three
    near-identical front views and no profile.  Spreading the picks over the
    set keeps more of the concept per token.  For a video reference the
    uniform order is usually what you want, so callers choose.
    """
    t = z.shape[2]
    if keep >= t:
        return list(range(t))
    if keep <= 1:
        return [0]
    # z[0] is [C, T, H, W]: frames have to be the leading axis to compare them
    flat = z[0].float().permute(1, 0, 2, 3).flatten(1)   # [T, C*H*W]
    flat = flat - flat.mean(1, keepdim=True)
    norms = flat.norm(dim=1)
    order = torch.argsort(norms, descending=True)
    chosen = [int(order[0])]
    dist = (flat - flat[chosen[0]]).norm(dim=1)
    for _ in range(keep - 1):
        dist[chosen] = -1.0
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, (flat - flat[nxt]).norm(dim=1))
    return sorted(chosen)


def fit_token_budget(latent: torch.Tensor, budget: int, label: str,
                     selection: str = "uniform") -> torch.Tensor:
    """Bring a stacked latent's injected token count under ``budget``.

    Tokens = (H/2)*(W/2) per frame (the DiT's 2x2 patch).  When the stack
    exceeds the budget it is cut in two cheap, loss-ordered passes:

      1. temporal dedup — drop latent frames that are (near-)identical to
         the last kept frame (free: they carry no new reference info);
      2. budget-fit resample — subsample the remaining frames down to the
         largest count that fits the budget.  ``selection="uniform"`` (the
         default, and what saved workflows have always done) takes evenly
         spaced frames, which preserves a clip's timeline;
         ``selection="diverse"`` spreads the picks over the stack so a
         multi-angle still set keeps more distinct views per token (see
         ``select_frames_diverse``).

    The spatial dims are never touched: resizing a latent's H/W changes the
    rope grid and degrades the reference, so time is the only honest lever.
    ``label`` is the mod name for the console notes.
    """
    if budget <= 0:
        return latent
    if selection not in ("uniform", "diverse"):
        raise ValueError(f"Unknown frame selection '{selection}'.")
    h, w = latent.shape[3], latent.shape[4]
    per_frame = (h // 2) * (w // 2)
    t = latent.shape[2]
    if per_frame > budget:
        raise ValueError(f"{label}: one frame costs {per_frame} tokens, above budget {budget}. Lower ref_resolution or pool size.")
    if per_frame * t <= budget:
        return latent
    kept = dedup_frame_indices(latent)
    if len(kept) < t:
        print(f"[RefMod] {label}: over {budget}-token cap, "
              f"dropped {t - len(kept)} near-duplicate frame(s) "
              f"({t} -> {len(kept)})")
        latent = latent[:, :, kept]
        t = latent.shape[2]
    if per_frame * t > budget:
        fit_t = max(1, budget // per_frame)
        if fit_t < t:
            if selection == "diverse" and fit_t > 1:
                idx = torch.tensor(select_frames_diverse(latent, fit_t), dtype=torch.long)
                print(f"[RefMod] {label}: still over {budget}-token cap, "
                      f"kept the {fit_t} most distinct of {t} frame(s) to fit")
            else:
                idx = torch.linspace(0, t - 1, fit_t).round().long()
                print(f"[RefMod] {label}: still over {budget}-token cap, "
                      f"resampled {t} -> {fit_t} frame(s) to fit")
            latent = latent[:, :, idx]
        if per_frame * latent.shape[2] > budget:
            print(f"[RefMod] {label}: one frame alone is {per_frame} tokens > "
                  f"{budget} — lower ref_resolution or pool_h/pool_w to meet "
                  f"the cap.")
    return latent


# ═══════════════════════════════════════════════════════════════════════════
# H3RefMod — the saved artifact
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class H3RefMod:
    """
    A compressed reference for MiniMax H3.

    ``latent`` is the VAE latent ``[1, 24, latent_t, latent_h, latent_w]`` — a
    full-resolution encode (``mode="encode"``) or a pooled thumbnail
    (``mode="training"``).  ``kind`` is ``"image"`` (single latent frame) or
    ``"video"`` (a few frames), matching the native ref block kinds the
    model's ``PackedLayout`` understands.

    ``mode="encode"`` stores the encode at the resolution the official ref2video
    path uses, so the injected ref carries real identity detail;
    ``mode="training"`` stores a tiny average-pooled grid refined by gradient
    steps (concept/motion, or identity at high pool sizes).
    """

    name: str
    kind: str
    latent: torch.Tensor
    latent_h: int = 4
    latent_w: int = 4
    latent_t: int = 1
    mode: str = "training"
    source: str = ""          # "image" | "video" | "manual"
    source_shape: str = ""    # original latent dims as "TxHxW"
    pool: str = "4x4x1"       # pool_t x pool_h x pool_w
    optimize_steps: int = 0
    tags: List[str] = field(default_factory=list)
    description: str = ""     # optional text describing the concept (emitted by the loaders)
    concept_type: str = "generic"  # what this mod represents; see CONCEPT_TYPES above
    config: Dict = field(default_factory=dict)
    # Recommended Apply/Step-Curve settings baked in by the "Fix H3 RefMod
    # Config" node: {"retention": float, "curve": [direction, shape, value],
    # "step_curve": [direction, shape, value]}.  The Apply / Step Curve
    # nodes' ``override`` toggle reads it back so a concept ships with the
    # settings that make it work — newbies never have to tune.
    sample_rate: int = 32000
    path: str = ""            # path_no_ext the mod was loaded from / saved to (for re-saving)
    bundle_index: int = -1  # index in the container at path; -1 for standalone refs

    def __post_init__(self):
        if self.kind == "audio":
            if self.latent.ndim != 4 or tuple(self.latent.shape[:3]) != (1, 32, 2) or self.latent.shape[-1] < 1:
                raise ValueError("H3 audio RefMod must contain [1,32,2,T] with T >= 1.")
            self.latent_t = self.latent.shape[-1]
            self.latent_h = self.latent_w = 0
            return
        if self.kind not in ("image", "video"):
            raise ValueError(f"kind must be 'image' or 'video' (got {self.kind!r})")
        if self.latent.ndim != 5 or self.latent.shape[0] != 1:
            raise ValueError(
                f"RefMod '{self.name}': expected a visual latent [1, 24, T, H, W], "
                f"got {tuple(self.latent.shape)}.")
        if not torch.isfinite(self.latent).all():
            raise ValueError(
                f"RefMod '{self.name}': the stored latent contains NaN/Inf values — "
                f"the encode that produced it failed. Re-extract it (see the "
                f"RefMod creation guide's troubleshooting section).")
        # The DiT patchifies each 2x2 latent cell into one token with integer
        # division, so an odd latent edge silently drops a row/column of the
        # reference instead of failing. Every writer in this pack rounds to an
        # even grid; this catches hand-edited or foreign files at load time.
        if self.latent.shape[3] % 2 or self.latent.shape[4] % 2:
            raise ValueError(
                f"RefMod '{self.name}': latent grid {self.latent.shape[3]}x"
                f"{self.latent.shape[4]} is not even. H3 patches latents in 2x2 "
                f"cells, so both edges must be even — re-extract this mod.")
        if self.kind == "image":
            if self.latent.shape[2] != 1:
                raise ValueError(
                    f"RefMod '{self.name}': kind='image' but the latent holds "
                    f"{self.latent.shape[2]} frames. Use kind='video' (the "
                    f"extractors pick this from the frame count automatically).")
            self.latent_t = 1

    # ── token budget ──────────────────────────────────────────────────

    @property
    def token_count(self) -> int:
        """Number of patchified tokens the mod injects into the packed sequence."""
        if self.kind == "audio":
            return 2 * self.latent_t
        per_frame = (self.latent_h // 2) * (self.latent_w // 2)
        return self.latent_t * per_frame

    # ── native ref block ──────────────────────────────────────────────

    def ref_block(self, strength: float = 1.0,
                  curve=None) -> Optional[Dict]:
        """
        Build the ref block dict the model's ``PackedLayout`` / payload consumes.

        The shape mirrors what the native ref2va nodes emit, so the pooled
        latent rides the exact same path: ``cond_video_latents`` -> patchify ->
        ``video_patch_proj`` -> packed sequence with a 3D RoPE grid.

        ``strength`` weakens the ref by mixing the latent toward a heavily
        blurred copy of itself (see ``_blur_latent``) rather than toward
        zero or random noise: scaling toward zero pushes values out of the
        normalized latent distribution (reads as grey output), and mixing
        toward random noise reads as real-but-garbled content and decodes
        as a wrong texture rather than "weaker" — a blurred copy stays
        on-manifold while still discarding the specific detail that makes a
        reference strong. ``strength <= 0`` drops the block entirely (no
        tokens injected).

        ``curve`` (optional) is a per-frame strength spec for the ref's
        latent timeline — a ``(direction, shape, value)`` tuple (e.g.
        ``("concept_at_end", "ease", 1.0)``), a legacy preset name, a list of
        per-frame floats, ``(x, y)`` control points, or any object with
        ``interp(x)`` (see ``curve_strengths``).  Each frame is mixed with
        its own strength ``retention * curve(x)`` instead of one flat value,
        so the ref can fade in/out across the video.  A flat curve is
        identical to passing ``strength`` alone.
        """
        if strength <= 0.0:
            return None
        latent = self.latent
        strengths = curve_strengths(curve, self.latent_t) if curve is not None else None
        if strengths is not None:
            weights = [max(0.0, min(1.0, strength * value)) for value in strengths]
            if any(value < 1.0 for value in weights):
                shape = (1, 1, 1, self.latent_t) if self.kind == "audio" else (1, 1, self.latent_t, 1, 1)
                st = latent.new_tensor(weights).view(shape)
                latent = st * latent + (1.0 - st) * _blur_latent(latent)
        elif strength < 1.0:
            latent = strength * latent + (1.0 - strength) * _blur_latent(latent)
        if self.kind == "audio":
            return {"kind": "audio", "ref_audio_t": self.latent_t, "audio_latent": latent}
        block: Dict = {
            "kind": self.kind,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "latent": latent,
        }
        if self.kind == "video":
            block["latent_t"] = self.latent_t
            block["ref_audio_t"] = 0
            block["audio_latent"] = None
        return block

    # ── serialization ─────────────────────────────────────────────────

    def metadata(self) -> dict:
        meta = {
            "name": self.name,
            "kind": self.kind,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "latent_t": self.latent_t,
            "mode": self.mode,
            "source": self.source,
            "source_shape": self.source_shape,
            "pool": self.pool,
            "optimize_steps": self.optimize_steps,
            "tags": self.tags,
            "description": self.description,
            "concept_type": self.concept_type,
            "_format_version": 4,
            "sample_rate": self.sample_rate,
        }
        if self.config:
            meta["refmod_config"] = json.dumps(self.config)
        return meta

    def save(self, path_no_ext: str) -> str:
        """Save standalone, or update this member without dropping its siblings."""
        if self.bundle_index >= 0 and os.path.normcase(os.path.abspath(path_no_ext)) == os.path.normcase(os.path.abspath(self.path)):
            from .bundle import update_member
            return update_member(self)
        os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
        meta = self.metadata()
        destination = path_no_ext + ".safetensors"
        fd, temporary = tempfile.mkstemp(prefix=".refmod-", suffix=".tmp", dir=os.path.dirname(destination) or ".")
        os.close(fd)
        try:
            save_file({"latent": self.latent.contiguous()}, temporary,
                      metadata={META_KEY: json.dumps(meta)})
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    @classmethod
    def load(cls, path_no_ext: str, device: str = "cpu") -> "H3RefMod":
        """Load from ``{path}.safetensors`` (metadata in header, or legacy .json)."""
        meta = read_refmod_meta(path_no_ext)
        if not isinstance(meta, dict):
            raise ValueError(
                f"{path_no_ext}.safetensors has no RefMod metadata "
                f"(header key '{META_KEY}' or sidecar .json missing).")
        if meta.get("kind") == "bundle":
            raise ValueError("This is a RefMod bundle. Use the current RefMod Loader to select its components.")
        # Clone releases the file mapping before a possible Windows overwrite.
        latent = load_file(path_no_ext + ".safetensors", device=device)["latent"].clone()
        return cls.from_metadata(meta, latent, path_no_ext)

    @classmethod
    def from_metadata(cls, meta, latent, path_no_ext="", bundle_index=-1):
        raw_config = meta.get("refmod_config")
        try:
            config = json.loads(raw_config) if isinstance(raw_config, str) else {}
        except ValueError:
            config = {}
        return cls(
            name=meta.get("name", os.path.basename(path_no_ext)),
            kind=meta.get("kind", "image"),
            latent=latent,
            latent_h=int(meta.get("latent_h", latent.shape[3] if latent.ndim == 5 else 0)),
            latent_w=int(meta.get("latent_w", latent.shape[4] if latent.ndim == 5 else 0)),
            latent_t=int(meta.get("latent_t", latent.shape[2] if latent.ndim == 5 else latent.shape[-1])),
            mode=normalize_mode(meta.get("mode", "training")),
            source=meta.get("source", ""),
            source_shape=meta.get("source_shape", ""),
            pool=meta.get("pool", ""),
            optimize_steps=int(meta.get("optimize_steps", 0)),
            tags=list(meta.get("tags", [])),
            description=str(meta.get("description", "") or ""),
            # canonicalize documented aliases ("general" -> "generic") but keep
            # an unrecognized label verbatim, so custom vocabularies survive a
            # round-trip through this pack
            concept_type=(normalize_concept_type(meta.get("concept_type"))
                          or str(meta.get("concept_type", "generic") or "generic")),
            config=config if isinstance(config, dict) else {},
            path=path_no_ext,
            bundle_index=bundle_index,
            sample_rate=int(meta.get("sample_rate", 32000)),
        )

