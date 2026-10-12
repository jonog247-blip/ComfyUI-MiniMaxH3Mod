"""
extract_mod.py — standalone MiniMax H3 RefMod extractor (no training)

Turns reference images/videos into a ``.safetensors`` mod that the ComfyUI
Load H3 RefMods / Apply nodes can inject into MiniMax H3 generation.  Only
the H3 video VAE is needed; the 29B DiT is never loaded.

To extract a whole dataset folder in one command (Method 1 of the RefMod
creation guide) use ``scripts/generate_refmod.py``, which wraps the
``extract``/``load_h3_vae`` helpers below and adds folder scanning,
``--dry-run`` and bulk mode.

Two modes:

  * ``encode`` (default) — refs are resized to ``--resolution`` short edge (down
    only) and encoded at that resolution, exactly like the official ref2video
    node.  This is what carries identity (a face, an outfit); files are
    ~0.2-1 MB per frame.  (Old name: ``full``.)
  * ``training`` — refs are resized to ``--resolution`` short edge too (the
    latent is pooled to a tiny grid anyway, so encoding at native resolution
    is wasted compute), then average-pooled to a small grid (4x4 by default
    = 4 tokens per latent frame) and optionally refined with a few gradient
    steps (still model-free; that refinement loop is the only "training" in
    the whole pack).  Nearly free to inject but only carries concept / motion,
    not fine identity.  (Old name: ``pooled``.)

Usage
-----
  # full-res identity mod (recommended for characters)
  python custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py \
      --image char.png --vae path/to/h3_video_vae.safetensors \
      --name my_character --mode encode --resolution 1024

  # tiny concept/motion mod
  python custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py \
      --video dance.mp4 --vae path/to/h3_video_vae.safetensors \
      --name dance --mode training --pool 4 --latent-frames 2

Multi-reference concept (each ref becomes its own latent frame):

  python custom_nodes/ComfyUI-MiniMaxH3Mod/extract_mod.py \
      --image face_a.png --image face_b.png --image full.png \
      --video dance.mp4 --vae path/to/h3_video_vae.safetensors \
      --name disney_char --mode encode --resolution 1024

Output goes to ``ComfyUI/models/refmods/`` by default (created on first run,
next to loras/ and unet/).

GPU
---
On CUDA the extractor turns on TF32 + cuDNN autotuning for the VAE pass,
keeps the encode output on the GPU (so a latent doesn't cross PCIe twice),
resizes stills on the GPU and encodes same-shape stills in one batched call.
Each of those falls back to the old path when it doesn't apply, and each has
an escape hatch: ``REFMOD_TF32=0``, ``REFMOD_CUDNN_BENCHMARK=0``,
``REFMOD_VAE_GPU_IO=0``, ``--encode-batch 1``, ``--resize-on-gpu never``.

Run it with the same Python/venv that runs ComfyUI (it imports ``comfy``
from the install the script lives in).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from common import (cuda_fast_math, encode_ref_stills, ensure_min_size,
                    load_image_file, load_video_file, mod_output_path,
                    resolve_output_dir, resize_ref as _resize_ref,
                    sample_video_for_vae, vae_output_on,
                    warn_if_implausible_size)
from core import (CONCEPT_TYPE_CHOICES, H3RefMod, aspect_grid, fit_token_budget,
                  normalize_concept_type, normalize_mode, optimize_latent,
                  pool_latent)

MAX_VIDEO_FRAMES = 60  # uniform sample cap; temporal pooling averages anyway

# Token budget the RefMod creation guide documents: 8 frames x 1024 tokens
# (a 1024px square ref is a 64x64 latent = 32x32 patches) = 8192.
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MODE = "encode"          # what carries identity; the guide's default
DEFAULT_RESOLUTION = 1024        # guide: short edge, multiples of 32
IDENTITY_IMAGE_RANGE = (8, 20)   # guide's recommended identity dataset size


# ═══════════════════════════════════════════════════════════════════════════
# Media loading (delegates to common.py: image/video -> [B, H, W, 3] float32)
# ═══════════════════════════════════════════════════════════════════════════

def _load_image(path: str, max_edge: int, device=None) -> torch.Tensor:
    return load_image_file(path, max_edge=max_edge, device=device)


def _load_video(path: str, max_edge: int,
                max_frames: int = MAX_VIDEO_FRAMES) -> torch.Tensor:
    # decoded on the CPU on purpose: a 60-frame clip at 1536px is ~1 GB of
    # fp32, so videos resize where they are and only the latent moves to the
    # encode device
    return load_video_file(path, max_frames=max_frames, max_edge=max_edge)


def find_near_duplicates(paths, threshold: float = 0.02) -> list:
    """(index, index, distance) pairs of source images that are near-identical.

    Two copies of the same shot cost two frames of the token budget and add no
    reference information, so the guide's 8-20 image datasets are worth a
    duplicate pass before encoding. Comparison is a downscaled grayscale mean
    difference — cheap, and only used to *warn*, never to drop anything.
    """
    import numpy as np
    from PIL import Image
    thumbs = []
    for path in paths:
        try:
            with Image.open(path) as img:
                img = img.convert("L").resize((32, 32), Image.BILINEAR)
                thumbs.append(torch.from_numpy(np.asarray(img).copy()).float() / 255.0)
        except Exception:
            thumbs.append(None)
    pairs = []
    for i in range(len(thumbs)):
        for j in range(i + 1, len(thumbs)):
            a, b = thumbs[i], thumbs[j]
            if a is None or b is None:
                continue
            denom = (a.abs().mean() + b.abs().mean()) / 2 + 1e-6
            distance = float((a - b).abs().mean() / denom)
            if distance < threshold:
                pairs.append((i, j, distance))
    return pairs


def resolve_device(name: str = "auto"):
    """Torch device for the VAE: ComfyUI's choice for 'auto', else as named."""
    if name == "auto":
        import comfy.model_management
        return comfy.model_management.get_torch_device()
    return torch.device(name)


def load_h3_vae(path: str, device):
    """Load the MiniMax H3 video VAE onto ``device`` (never the DiT)."""
    import comfy.sd
    import comfy.utils
    print(f"[extract] loading VAE {path} (device={device})")
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=device)
    vae.throw_exception_if_invalid()
    return vae


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract a MiniMax H3 RefMod from one or more images/videos "
                    "of the same concept (they are stacked into a video-kind mod)")
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="reference image path (repeatable for multi-view concepts)")
    ap.add_argument("--video", action="append", default=[], metavar="PATH",
                    help="reference video path (repeatable)")
    ap.add_argument("--vae", default="",
                    help="MiniMax H3 video VAE .safetensors (required; optional here only "
                         "so callers such as scripts/generate_refmod.py can build the "
                         "namespace programmatically)")
    ap.add_argument("--name", default=None, help="mod name (default: first source file stem)")
    ap.add_argument("--subfolder", default="", help="optional subfolder inside models/refmods")
    ap.add_argument("--output", default=None,
                    help="output dir (default: ComfyUI/models/refmods). A relative path that "
                         "exists under --comfy-dir is resolved against it, so the guide's "
                         "'--output models/refmods' works from any working directory.")
    ap.add_argument("--comfy-dir", default=None,
                    help="ComfyUI root, used to resolve a relative --output (default: the "
                         "install this script lives in)")
    ap.add_argument("--mode", choices=["training", "encode", "full", "pooled"],
                    default=DEFAULT_MODE,
                    help="encode = straight full-res VAE encode, ~1K tokens/img (default; what "
                         "carries a face); training = compressed grid refined by --identity; "
                         "full/pooled = old names, still accepted")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help="both modes: target short edge in px, downscale only (default 1024; 2048 = max fidelity). training mode encodes at this size too — the main speed dial for it")
    ap.add_argument("--pool", type=int, default=16, help="training mode: spatial latent grid (even, default 16)")
    ap.add_argument("--pool-w", type=int, default=None, help="pool width (default: == --pool)")
    ap.add_argument("--latent-frames", type=int, default=16,
                    help="frames kept per video ref (default 16; images use 1): training mode pools latent frames; encode mode samples source frames on H3's 17k+5 grid (16 selects 5, below 5 selects the first image)")
    ap.add_argument("--identity", type=int, default=500,
                    help="training mode: how tightly the mod clings to the reference (gradient refinement steps; default 500, 0 = pure pooling)")
    ap.add_argument("--refine-tol", type=float, default=1e-4,
                    help="training mode: stop refining early once the loss stops improving by "
                         "this relative amount (default 1e-4; 0 = always run every --identity "
                         "step). Adam on this objective converges well before the step budget.")
    ap.add_argument("--multiplier", type=int, default=1,
                    help="data multiplier: repeat the extracted ref N times along time so a short video/GIF isn't drowned out by the main video's tokens (default 1 = no repeat)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"hard cap on the total injected tokens (0 = off; default {DEFAULT_MAX_TOKENS}, "
                         f"the budget the creation guide documents): drops near-duplicate latent "
                         f"frames first, then trims the rest to fit")
    ap.add_argument("--frame-selection", choices=["diverse", "uniform"], default="diverse",
                    help="which frames survive --max-tokens: 'diverse' (default) spreads the "
                         "picks over the stack so a multi-angle still set keeps the most "
                         "distinct views; 'uniform' keeps evenly spaced frames (the old "
                         "behavior, better for preserving a clip's timeline)")
    ap.add_argument("--max-edge", type=int, default=1536,
                    help="resize source so the longest edge is <= this before encoding (default 1536)")
    ap.add_argument("--max-frames", type=int, default=60,
                    help="max video frames sampled (uniform) before encoding (default 60; lower for CPU)")
    ap.add_argument("--description", default="",
                    help="optional text describing the concept (e.g. 'a ginger woman with messy hair', "
                         "'an animation style'). Stored in the mod and emitted by the loaders so it "
                         "can be merged into the prompt without retyping it.")
    ap.add_argument("--concept-type", choices=list(CONCEPT_TYPE_CHOICES), default="generic",
                    help="what this mod represents (default: generic). Documented aliases are "
                         "accepted and stored canonically ('general' -> generic, 'character' -> "
                         "identity, ...). 'identity' in --mode training prints a warning: pooling "
                         "is lossy in exactly the way that destroys faces — use --mode encode for people.")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu (VAE device)")
    ap.add_argument("--encode-batch", type=int, default=4,
                    help="still refs encoded per VAE call on a GPU (default 4, 1 = one at a time). "
                         "Batching is verified against a single encode the first time and falls "
                         "back automatically; it is also capped by the VAE's own free-memory estimate.")
    ap.add_argument("--resize-on-gpu", choices=["auto", "always", "never"], default="auto",
                    help="resize stills on the encode device instead of the CPU (default auto = "
                         "on for CUDA). Faster for large sources; 'never' keeps every pixel "
                         "operation on the CPU.")
    return ap


def prepare(args) -> None:
    """Normalize ``args`` in place and print the guide's dataset guidance."""
    args.mode = normalize_mode(args.mode)  # accept legacy 'full'/'pooled'
    args.concept_type = normalize_concept_type(args.concept_type) or "generic"
    if not args.image and not args.video:
        raise ValueError("provide at least one --image or --video")
    if args.concept_type == "identity" and args.mode == "training" and args.pool < 16:
        print(
            f"[extract] warning: --concept-type identity with --mode training at "
            f"--pool {args.pool} — pooling averages away exactly the detail that "
            f"carries a face. Use --mode encode for people, or at least --pool 32."
        )
    if args.resolution % 32:
        print(f"[extract] note: --resolution {args.resolution} is not a multiple of 32 "
              f"(H3's patch grid); sources are snapped to the nearest multiple.")
    low, high = IDENTITY_IMAGE_RANGE
    if args.concept_type == "identity" and not (low <= len(args.image) <= high):
        print(f"[extract] note: the creation guide recommends {low}-{high} images for an "
              f"identity RefMod (40% front, 30% three-quarter, 20% profile, 10% "
              f"waist-up); you passed {len(args.image)}.")
    for i, j, distance in (find_near_duplicates(args.image) if len(args.image) > 1 else []):
        print(f"[extract] warning: {os.path.basename(args.image[i])} and "
              f"{os.path.basename(args.image[j])} look like the same shot "
              f"(distance {distance:.4f}) — they spend two frames of the token "
              f"budget on one view.")


def extract(args, vae, device) -> H3RefMod:
    """Encode the configured refs and save the mod. Returns the saved RefMod.

    ``vae``/``device`` come from ``load_h3_vae``/``resolve_device`` so a bulk
    run can load the VAE once and reuse it for every dataset.
    """
    started = time.perf_counter()
    resize_device = device if (args.resize_on_gpu == "always"
                               or (args.resize_on_gpu == "auto" and device.type == "cuda")) else None
    if resize_device is not None:
        resize_device = torch.device(resize_device)

    pool_w = args.pool_w or args.pool
    # both modes resize to --resolution before encode, so the load-time
    # longest-edge cap just needs to stay ahead of that target
    load_max_edge = max(args.resolution * 2, args.max_edge)
    # full-res refs must share one spatial canvas so the stacked latent has a
    # single H/W: anchor on the first source, cover-crop the rest to it
    canvas = None
    if args.mode == "encode" and len(args.image) + len(args.video) > 1:
        if args.image:
            first = _load_image(args.image[0], load_max_edge, resize_device)
        else:
            first = _load_video(args.video[0], load_max_edge, args.max_frames)
        h, w = first.shape[1], first.shape[2]
        scale = min(1.0, args.resolution / min(h, w))
        canvas = (max(32, round(w * scale / 32) * 32),
                  max(32, round(h * scale / 32) * 32))
        del first
    # training mode: anchor the pool grid to the first source's aspect so a
    # portrait person isn't squished into a square grid ("fat" distortion).
    pool_grid = None
    if args.mode == "training":
        if args.image:
            first = _load_image(args.image[0], load_max_edge, resize_device)
        else:
            first = _load_video(args.video[0], load_max_edge, args.max_frames)
        h0, w0 = first.shape[1], first.shape[2]
        pool_grid = aspect_grid(args.pool, pool_w, h0 / w0)
        del first
        if pool_grid != (args.pool, pool_w):
            print(f"[extract] pooled grid {args.pool}x{pool_w} -> "
                  f"{pool_grid[0]}x{pool_grid[1]} to match source aspect "
                  f"{w0}x{h0} (avoids squishing the subject wide)")
    gh, gw = pool_grid if pool_grid is not None else (args.pool, pool_w)
    # how many same-canvas stills share one VAE call. Batching pays off most
    # where the GPU idles between the VAE's small internal tiles, but it is
    # honored on any device (ComfyUI's VAE chunks by free memory), so the
    # option stays meaningful -- and testable -- without a GPU.
    batch = max(1, int(args.encode_batch))

    def refine(pooled, z):
        """Gradient-refined pooled latent (training mode), else the pool itself."""
        if args.identity <= 0 or args.mode != "training":
            return pooled
        return optimize_latent(pooled, z, steps=args.identity, device=device,
                               progress_every=100, tol=args.refine_tol)

    frames = []
    shapes = []
    n_img = n_vid = 0
    with cuda_fast_math(verbose=True):
        for start in range(0, len(args.image), batch):
            chunk = args.image[start:start + batch]
            sources = []
            for path in chunk:
                src = _load_image(path, load_max_edge, resize_device)
                src = ensure_min_size(_resize_ref(src, args.resolution, canvas, resize_device))
                print(f"[extract] image {path}: {tuple(src.shape)} (mode={args.mode})")
                sources.append(src)
            with vae_output_on(vae, device), torch.no_grad():
                zs = encode_ref_stills(vae, sources, device=device, max_batch=batch)
            del sources
            for z in zs:
                pooled = (z.to(torch.float16) if args.mode == "encode"
                          else refine(pool_latent(z.float(), 1, gh, gw).to(torch.float16), z))
                frames.append(pooled.cpu())
                shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")
                n_img += 1
                del z, pooled
        for path in args.video:
            src = _load_video(path, load_max_edge, args.max_frames)
            src = sample_video_for_vae(src, args.latent_frames if args.mode == "encode" else None)
            src = ensure_min_size(_resize_ref(src, args.resolution, canvas))
            print(f"[extract] video {path}: {tuple(src.shape)} (mode={args.mode})")
            with vae_output_on(vae, device), torch.no_grad():
                z = vae.encode(src)
            del src
            if args.mode == "encode":
                frames.append(z.to(torch.float16).cpu())
            else:
                pool_t = min(args.latent_frames, z.shape[2])
                pooled = pool_latent(z.float(), pool_t, gh, gw).to(torch.float16)
                frames.append(refine(pooled, z).cpu())
            shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")
            n_vid += 1
            del z

    if args.mode == "encode" and args.identity > 0:
        print(f"[extract] warning: --identity only applies to training mode — "
              f"encode mode stores the actual encode, so --identity {args.identity} "
              f"was ignored.")

    latent = torch.cat([f.cpu() for f in frames], dim=2)
    if args.multiplier > 1:
        latent = latent.repeat(1, 1, args.multiplier, 1, 1)
    name = args.name or (os.path.splitext(os.path.basename(args.image[0]))[0]
                         if args.image else os.path.splitext(os.path.basename(args.video[0]))[0])
    dropped = 0
    if args.max_tokens > 0:
        before = latent.shape[2]
        latent = fit_token_budget(latent, args.max_tokens, name,
                                  selection=args.frame_selection)
        dropped = before - latent.shape[2]
    total_t = latent.shape[2]
    kind = "video" if total_t > 1 else "image"

    out_dir = resolve_output_dir(getattr(args, "output", None),
                                 getattr(args, "comfy_dir", None) or ROOT)
    os.makedirs(out_dir, exist_ok=True)
    px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
    mod = H3RefMod(
        name=name,
        kind=kind,
        latent=latent,
        latent_h=latent.shape[3],
        latent_w=latent.shape[4],
        latent_t=total_t,
        mode=args.mode,
        source="stack" if len(frames) > 1 else ("video" if n_vid else "image"),
        source_shape=" +".join(shapes),
        pool=f"full-res {px_w}x{px_h}px (short-edge cap {args.resolution}px)" if args.mode == "encode" else f"{total_t}x{gh}x{gw}",
        optimize_steps=args.identity if args.mode == "training" else 0,
        tags=[f"{n_img} img, {n_vid} vid"] + ([f"x{args.multiplier} repeat"] if args.multiplier > 1 else []),
        description=args.description.strip(),
        concept_type=args.concept_type,
    )
    # so Fix H3 RefMod Config can re-save in place
    mod.path = (os.path.join(out_dir, name) if getattr(args, "output", None)
                else mod_output_path(name, args.subfolder))
    path = mod.save(mod.path)
    elapsed = time.perf_counter() - started
    warn_if_implausible_size(path, mod.token_count)
    print(f"[extract] saved {kind} mod '{name}' "
          f"({mod.token_count} tokens, "
          f"{latent.numel() * latent.element_size() / 1024 / 1024:.2f} MB latent, "
          f"{os.path.getsize(path) / 1024 / 1024:.2f} MB on disk) -> {path}")
    if dropped:
        print(f"[extract] note: {dropped} of {n_img + n_vid} reference frame(s) were dropped "
              f"to stay under --max-tokens {args.max_tokens} "
              f"(selection={args.frame_selection}).")
    print(f"[extract] {n_img + n_vid} ref(s) in {elapsed:.1f}s on {device}"
          + (f" (encode batch={batch})" if batch > 1 else ""))
    print(f"[extract] load it in ComfyUI with the Load H3 RefMods node "
          f"(dropdown '{name}', strength 1.0 = full ref), then chain "
          f"Apply H3 RefMod (Cond) into your sampling.")
    return mod


def main(argv=None) -> H3RefMod:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.vae:
        parser.error("--vae is required: path to the MiniMax H3 video VAE .safetensors")
    prepare(args)
    device = resolve_device(args.device)
    vae = load_h3_vae(args.vae, device)
    return extract(args, vae, device)


if __name__ == "__main__":
    main()
