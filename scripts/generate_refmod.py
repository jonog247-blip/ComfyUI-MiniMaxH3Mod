#!/usr/bin/env python
"""
generate_refmod.py — automated whole-folder RefMod extraction.

This is **Method 1** of the MiniMax H3 RefMod creation guide: point it at a
dataset folder and it scans the images, works out the canvas/aspect, allocates
the token budget and writes a ``.safetensors`` straight into
``ComfyUI/models/refmods/``.

    # ComfyUI Easy-Install / Portable (Windows), from the ComfyUI root
    python_embeded\\python.exe scripts\\generate_refmod.py "E:\\Datasets\\GabrielleAplin"

    # standard venv install
    python scripts/generate_refmod.py /data/datasets/GabrielleAplin

    # preview what would be encoded, without touching the VAE
    python scripts/generate_refmod.py /data/datasets/GabrielleAplin --dry-run

    # bulk (the guide's Method 3): one mod per subfolder, VAE loaded once
    python scripts/generate_refmod.py /data/datasets --each-subfolder

The script lives in the custom node pack. The guide shows it at
``ComfyUI/scripts/generate_refmod.py`` — copy or symlink it there and the
commands above work verbatim, or run it from this folder; either way it finds
its ComfyUI root by walking up from ``__file__`` (override with
``--comfy-dir``).

Output naming, resolution, token budget and concept types follow the guide:
``minimaxh3_<dataset>_v1_refmod``, 1024px short edge, an 8192-token cap, and
``identity`` / ``style`` / ``general`` (stored as ``generic``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACK_DIR = os.path.abspath(os.path.join(HERE, ".."))


# ═══════════════════════════════════════════════════════════════════════════
# Environment: locate ComfyUI, then import the pack's own extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def looks_like_comfy_root(path: str) -> bool:
    return (os.path.isfile(os.path.join(path, "folder_paths.py"))
            and os.path.isdir(os.path.join(path, "comfy")))


def find_comfy_root(start: str = PACK_DIR) -> str:
    """Nearest ancestor that is a ComfyUI install.

    Works from ``ComfyUI/custom_nodes/<pack>/scripts`` and from a copy at
    ``ComfyUI/scripts``; raises with an actionable message if neither parent
    chain reaches one (``--comfy-dir`` is the escape hatch).
    """
    current = os.path.abspath(start)
    for _ in range(8):
        if looks_like_comfy_root(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise SystemExit(
        "Could not find a ComfyUI installation above "
        f"'{os.path.abspath(start)}'.\n"
        "Pass --comfy-dir /path/to/ComfyUI (the folder that contains comfy/ and "
        "models/).")


def load_pipeline(comfy_root: str):
    """Import the pack's extractor with ComfyUI importable. Returns the module."""
    for path in (comfy_root, PACK_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
    import extract_mod  # noqa: PLC0415  (path is only complete now)
    return extract_mod


# ═══════════════════════════════════════════════════════════════════════════
# Dataset discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover(folder: str, recursive: bool = False, image_exts=None,
             video_exts=None, include_videos: bool = True):
    """(images, videos) under ``folder``, sorted by name.

    Skips the obvious non-dataset directories (thumbnails, masks, caches) so
    pointing it at a training-data root doesn't encode mask PNGs as reference
    frames.
    """
    image_exts = image_exts or {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    video_exts = video_exts or {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
    skip_dirs = {"__pycache__", ".git", "mask", "masks", "alpha", "caption",
                 "captions", "text", "thumbs", ".ipynb_checkpoints"}
    images, videos = [], []

    def scan(directory: str) -> None:
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            return
        for entry in entries:
            path = os.path.join(directory, entry)
            ext = os.path.splitext(entry)[1].lower()
            if os.path.isdir(path):
                if recursive and entry.lower() not in skip_dirs and not entry.startswith("."):
                    scan(path)
                continue
            if ext in image_exts:
                images.append(path)
            elif include_videos and ext in video_exts:
                videos.append(path)

    scan(folder)
    return images, videos


def derive_name(folder: str, suffix: str = "v1") -> str:
    """``minimaxh3_<dataset>_v1_refmod`` — the guide's auto-derived name."""
    base = os.path.basename(os.path.normpath(os.path.abspath(folder)))
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_").lower() or "dataset"
    return f"minimaxh3_{slug}_{suffix}_refmod" if suffix else f"minimaxh3_{slug}_refmod"


def default_vae(comfy_root: str, requested: str = "") -> str:
    """The H3 video VAE: the guide's filename, else the best ``models/vae`` match."""
    if requested:
        if not os.path.isfile(requested):
            raise SystemExit(f"VAE not found: {requested}")
        return requested
    vae_dir = os.path.join(comfy_root, "models", "vae")
    preferred = os.path.join(vae_dir, "minimax_h3_video_vae_fp16.safetensors")
    if os.path.isfile(preferred):
        return preferred
    try:
        candidates = sorted(f for f in os.listdir(vae_dir) if f.endswith(".safetensors"))
    except OSError:
        candidates = []
    matches = [f for f in candidates if "h3" in f.lower() and "audio" not in f.lower()]
    if matches:
        return os.path.join(vae_dir, matches[0])
    raise SystemExit(
        f"No MiniMax H3 video VAE in {vae_dir}.\n"
        "Expected models/vae/minimax_h3_video_vae_fp16.safetensors — or pass "
        "--vae /path/to/vae.safetensors.")


# ═══════════════════════════════════════════════════════════════════════════
# Dry-run plan
# ═══════════════════════════════════════════════════════════════════════════

def _canvas_for(images, resolution: int):
    """Shared (w, h) encode canvas, anchored on the first image's aspect."""
    from PIL import Image
    with Image.open(images[0]) as img:
        w, h = img.size
    scale = min(1.0, resolution / min(h, w))
    return (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32))


def print_plan(folder: str, name: str, images, videos, args, comfy_root: str,
               out_dir: str) -> None:
    """The ``--dry-run`` report: what would be encoded, and what it would cost."""
    print(f"\nRefMod plan: {name}")
    print(f"  folder        : {folder}")
    print(f"  source        : {len(images)} image(s)"
          + (f", {len(videos)} video(s)" if videos else ""))
    print(f"  output        : {os.path.join(out_dir, name)}.safetensors")
    print(f"  concept type  : {args.concept_type}  (mode={args.mode})")
    print(f"  short edge    : {args.resolution}px, token cap {args.max_tokens}")
    if not images:
        print("  no images to encode")
        return
    tw, th = _canvas_for(images, args.resolution)
    lat_w, lat_h = tw // 16, th // 16
    per_frame = (lat_w // 2) * (lat_h // 2)
    print(f"  canvas        : {tw}x{th}px -> {lat_w}x{lat_h} latent "
          f"-> {per_frame} tokens/frame")
    for path in images:
        print(f"    [image] {os.path.relpath(path, folder)}")
    for path in videos:
        print(f"    [video] {os.path.relpath(path, folder)}"
              f"  (tokens depend on the sampled frame count)")
    n_refs = len(images) + len(videos)
    total = per_frame * n_refs
    kept = n_refs
    if args.max_tokens and total > args.max_tokens:
        kept = max(1, args.max_tokens // per_frame)
        print(f"  budget        : {total} tokens > {args.max_tokens} cap -> "
              f"about {kept} of {n_refs} frame(s) survive "
              f"(selection={args.frame_selection})")
    else:
        print(f"  budget        : {total} tokens (fits the {args.max_tokens} cap)")
    # latent elements per frame = 24 channels x latent_h x latent_w = per_frame*4
    mb = per_frame * 4 * kept * 24 * 2 / 1024 / 1024
    print(f"  est. size     : ~{mb:.2f} MB latent on disk (fp16; the guide's "
          f"1.1-1.6 MB is 8 x 1024px square stills)")
    print(f"  comfy root    : {comfy_root}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser(extract_mod) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="generate_refmod.py",
        description="Scan a dataset folder and write a MiniMax H3 RefMod into "
                    "ComfyUI/models/refmods/ (Method 1 of the creation guide).")
    ap.add_argument("folders", nargs="+", metavar="FOLDER",
                    help="dataset folder(s) of reference images")
    ap.add_argument("--comfy-dir", default=None,
                    help="path to the ComfyUI root (default: auto-detected from this file)")
    ap.add_argument("--name", default=None,
                    help="output RefMod name (default: minimaxh3_<folder>_v1_refmod). "
                         "Ignored with --each-subfolder or multiple folders.")
    ap.add_argument("--resolution", type=int, default=extract_mod.DEFAULT_RESOLUTION,
                    help="short-edge resolution cap in px, multiples of 32 (default 1024)")
    ap.add_argument("--max-tokens", type=int, default=extract_mod.DEFAULT_MAX_TOKENS,
                    help=f"maximum token budget (default {extract_mod.DEFAULT_MAX_TOKENS})")
    ap.add_argument("--concept-type", choices=list(extract_mod.CONCEPT_TYPE_CHOICES),
                    default="identity",
                    help="identity (default), style or general — documented aliases are "
                         "accepted and stored canonically")
    ap.add_argument("--dry-run", action="store_true",
                    help="preview the discovered files and the token plan without encoding")
    # ── beyond the guide's table, same knobs as extract_mod.py ──
    ap.add_argument("--vae", default="",
                    help="H3 video VAE path (default: models/vae/minimax_h3_video_vae_fp16.safetensors)")
    ap.add_argument("--mode", choices=["encode", "training", "full", "pooled"],
                    default=extract_mod.DEFAULT_MODE,
                    help="encode = full-res VAE encode (default, carries identity); "
                         "training = pooled grid refined by --identity")
    ap.add_argument("--pool", type=int, default=16, help="training mode: spatial latent grid")
    ap.add_argument("--identity", type=int, default=500,
                    help="training mode: gradient refinement steps")
    ap.add_argument("--refine-tol", type=float, default=1e-4,
                    help="training mode: stop refining once the loss is flat (0 = off)")
    ap.add_argument("--latent-frames", type=int, default=16, help="frames kept per video ref")
    ap.add_argument("--max-frames", type=int, default=60, help="max video frames sampled")
    ap.add_argument("--frame-selection", choices=["diverse", "uniform"], default="diverse",
                    help="how frames are chosen when the token cap forces a cut")
    ap.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    ap.add_argument("--encode-batch", type=int, default=4,
                    help="still refs per VAE call on a GPU (1 = one at a time)")
    ap.add_argument("--resize-on-gpu", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--description", default="", help="concept description stored in the mod")
    ap.add_argument("--output", default=None,
                    help="output dir (default: <comfy>/models/refmods)")
    ap.add_argument("--recursive", action="store_true",
                    help="also scan subfolders of each dataset folder")
    ap.add_argument("--no-videos", action="store_true",
                    help="ignore video files (images only)")
    ap.add_argument("--each-subfolder", action="store_true",
                    help="bulk mode (the guide's Method 3): treat every immediate "
                         "subfolder as its own dataset, loading the VAE once")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract even when the output file already exists "
                         "(default: skip, as in the guide's bulk example)")
    return ap


def extract_args_for(parser_args, extract_mod, images, videos, name: str,
                     out_dir: str) -> argparse.Namespace:
    """A ready-to-run ``extract_mod`` namespace for one dataset."""
    args = extract_mod.build_parser().parse_args([])
    args.image = list(images)
    args.video = list(videos)
    args.vae = parser_args.vae_path
    args.name = name
    args.output = out_dir
    args.comfy_dir = parser_args.comfy_dir
    args.mode = parser_args.mode
    args.resolution = parser_args.resolution
    args.max_tokens = parser_args.max_tokens
    args.concept_type = parser_args.concept_type
    args.pool = parser_args.pool
    args.pool_w = parser_args.pool
    args.identity = parser_args.identity
    args.refine_tol = parser_args.refine_tol
    args.latent_frames = parser_args.latent_frames
    args.max_frames = parser_args.max_frames
    args.frame_selection = parser_args.frame_selection
    args.device = parser_args.device
    args.encode_batch = parser_args.encode_batch
    args.resize_on_gpu = parser_args.resize_on_gpu
    args.description = parser_args.description
    extract_mod.prepare(args)
    return args


def collect_jobs(parser_args, extract_mod):
    """[(folder, images, videos, name)] for every dataset to extract."""
    jobs = []
    for folder in parser_args.folders:
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            raise SystemExit(f"Not a folder: {folder}")
        if parser_args.each_subfolder:
            subfolders = [os.path.join(folder, entry)
                          for entry in sorted(os.listdir(folder))
                          if os.path.isdir(os.path.join(folder, entry))
                          and not entry.startswith(".")]
            if not subfolders:
                raise SystemExit(f"{folder} has no subfolders to scan "
                                 f"(drop --each-subfolder to scan the folder itself)")
            for sub in subfolders:
                images, videos = discover(sub, parser_args.recursive,
                                          include_videos=not parser_args.no_videos)
                if not images and not videos:
                    print(f"[generate] skipping {os.path.basename(sub)}: no images or videos")
                    continue
                jobs.append((sub, images, videos, derive_name(sub)))
            continue
        images, videos = discover(folder, parser_args.recursive,
                                  include_videos=not parser_args.no_videos)
        if not images and not videos:
            raise SystemExit(
                f"No images (or videos) found in {folder}.\n"
                "Supported: .png .jpg .jpeg .webp .bmp"
                + ("" if parser_args.no_videos else " .mp4 .webm .mov .mkv .avi .m4v")
                + ("\nTry --recursive if the images are in subfolders."
                   if not parser_args.recursive else ""))
        name = parser_args.name if (parser_args.name and len(parser_args.folders) == 1) \
            else derive_name(folder)
        jobs.append((folder, images, videos, name))
    return jobs


def main(argv=None) -> int:
    # resolve the ComfyUI root before importing anything that needs it
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--comfy-dir", default=None)
    known, _ = pre.parse_known_args(argv)
    comfy_root = os.path.abspath(known.comfy_dir) if known.comfy_dir else find_comfy_root()
    if not looks_like_comfy_root(comfy_root):
        raise SystemExit(f"{comfy_root} does not look like a ComfyUI installation "
                         f"(no comfy/ + folder_paths.py).")

    extract_mod = load_pipeline(comfy_root)
    args = build_parser(extract_mod).parse_args(argv)
    args.comfy_dir = comfy_root

    from common import resolve_output_dir  # noqa: PLC0415  (needs the ComfyUI root)

    out_dir = resolve_output_dir(args.output, comfy_root)
    jobs = collect_jobs(args, extract_mod)

    if args.dry_run:
        for folder, images, videos, name in jobs:
            print_plan(folder, name, images, videos, args, comfy_root, out_dir)
        print(f"\n[generate] dry run: {len(jobs)} dataset(s), nothing written.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    args.vae_path = default_vae(comfy_root, args.vae)
    pending = []
    for folder, images, videos, name in jobs:
        target = os.path.join(out_dir, name + ".safetensors")
        if os.path.exists(target) and not args.overwrite:
            print(f"[generate] skipping {name}: {target} exists (use --overwrite)")
            continue
        pending.append((folder, images, videos, name))
    if not pending:
        print("[generate] nothing to do.")
        return 0

    started = time.perf_counter()
    device = extract_mod.resolve_device(args.device)
    vae = extract_mod.load_h3_vae(args.vae_path, device)  # loaded once for every dataset
    results = []
    for folder, images, videos, name in pending:
        print(f"\n[generate] === {name} <- {folder} "
              f"({len(images)} image(s)" + (f", {len(videos)} video(s)" if videos else "") + ")")
        try:
            mod = extract_mod.extract(
                extract_args_for(args, extract_mod, images, videos, name, out_dir),
                vae, device)
        except Exception as exc:  # keep bulk runs going, report at the end
            print(f"[generate] FAILED {name}: {exc}")
            results.append((name, None))
            continue
        results.append((name, mod))
        print(f"[generate] Generated {name} "
              f"({len(images) + len(videos)} sources, {mod.token_count} tokens)")

    ok = [name for name, mod in results if mod is not None]
    failed = [name for name, mod in results if mod is None]
    print(f"\n[generate] {len(ok)}/{len(results)} RefMod(s) in "
          f"{time.perf_counter() - started:.1f}s -> {out_dir}")
    for name in failed:
        print(f"[generate] failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
