"""Shared helpers for the RefMod pack: media loading and the refmods folder."""

from __future__ import annotations

import os
import ntpath
import math
import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import comfy.utils

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}

# Every knob below is an escape hatch, not a required setting: the defaults are
# what the CLIs/nodes use, and each one falls back to the old behavior when the
# machine or the VAE doesn't cooperate.
def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


@contextmanager
def cuda_fast_math(verbose: bool = False):
    """Enable TF32 + cuDNN autotuning for the duration of a VAE pass.

    Both are process-global torch settings, so this is a context manager that
    restores whatever was there before instead of a set-and-forget call: a
    custom node that flipped them permanently would change the numerics and
    the autotune behavior of every other model in the same ComfyUI process.

      * TF32 (Ampere+) runs the VAE's fp32 convs/matmuls on the tensor cores
        at ~1/8 the mantissa. A VAE encode is not a precision-critical
        accumulation — the result is stored as fp16 anyway — and the speedup
        on a 3D conv stack is large.
      * ``cudnn.benchmark`` autotunes the conv algorithms for each new input
        shape. Extraction hits a handful of repeated shapes (every ref of a
        dataset is the same canvas), so the one-time search pays for itself
        immediately; it would be a pessimization for constantly changing
        shapes, which is why it is scoped here.

    No-op on CPU/MPS. ``REFMOD_TF32=0`` / ``REFMOD_CUDNN_BENCHMARK=0`` keep
    the exact previous numerics/algorithms for A/B comparisons.
    """
    if not (torch.cuda.is_available() and torch.backends.cuda.is_built()):
        yield False
        return
    tf32 = _env_flag("REFMOD_TF32")
    bench = _env_flag("REFMOD_CUDNN_BENCHMARK")
    saved = (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark)
    try:
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if bench:
            torch.backends.cudnn.benchmark = True
        if verbose:
            print(f"[RefMod] cuda fast-math: tf32={tf32} cudnn_benchmark={bench} "
                  f"({torch.cuda.get_device_name()})")
        yield True
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = saved


@contextmanager
def vae_output_on(vae, device):
    """Keep a VAE's encode output on ``device`` instead of the CPU.

    ComfyUI parks intermediates on the CPU (``intermediate_device``), so an
    extraction that refines a latent pays a full-res latent D2H and then H2D
    again, and one that stores fp16 pays twice the PCIe bytes it needs. This
    points ``output_device`` at the compute device for the duration of the
    block and restores it afterwards.

    No-op when the VAE has no such attribute or the device is the CPU.
    ``REFMOD_VAE_GPU_IO=0`` disables it.
    """
    if (device is None or not hasattr(vae, "output_device")
            or not _env_flag("REFMOD_VAE_GPU_IO")):
        yield False
        return
    target = torch.device(device)
    if target.type == "cpu":
        yield False
        return
    if target.type == "cuda":
        # The point of ComfyUI's tiled-encode fallback is that VRAM ran out;
        # parking the result on the GPU is only safe while there is room for
        # both the activations and the (small) latent. Under pressure, keep
        # the old CPU destination instead of making an OOM more likely.
        try:
            free, _total = torch.cuda.mem_get_info(target)
        except Exception:
            free = None
        if free is not None and free < 1.5 * 1024 ** 3:
            yield False
            return
    saved = vae.output_device
    try:
        vae.output_device = target
        yield True
    finally:
        vae.output_device = saved


def _vae_spatial_ratio(vae) -> int:
    try:
        return int(vae.spacial_compression_encode())
    except Exception:
        ratio = getattr(vae, "downscale_ratio", 16)
        return int(ratio[-1]) if isinstance(ratio, (tuple, list)) else int(ratio or 16)


# Batched-encode verdict per VAE object: (vae, supported). The reference is
# held so the id can't be recycled into a stale verdict.
_BATCH_VERDICTS: Dict[int, Tuple[object, bool]] = {}


def _batch_verdict(vae) -> Optional[bool]:
    entry = _BATCH_VERDICTS.get(id(vae))
    return None if entry is None or entry[0] is not vae else entry[1]


def _set_batch_verdict(vae, supported: bool) -> None:
    _BATCH_VERDICTS[id(vae)] = (vae, supported)


def _encode_batch_limit(vae, device, per_sample: Tuple[int, int, int]) -> int:
    """How many single-frame refs fit in one encode, by the VAE's own estimate."""
    try:
        needed = float(vae.memory_used_encode((1, 3, 1) + tuple(per_sample[1:]), vae.vae_dtype))
        free = float(vae.patcher.get_free_memory(device))
    except Exception:
        return 1 << 30
    if needed <= 0 or not math.isfinite(free):
        return 1 << 30
    # leave headroom: the estimate is per sample and ComfyUI's own batching
    # inside VAE.encode does not know we grew the batch dimension
    return max(1, int(free * 0.6 / needed))


def encode_ref_stills(vae, sources: Sequence[torch.Tensor], device=None,
                      max_batch: int = 4, verify: bool = True,
                      log=print) -> List[torch.Tensor]:
    """VAE-encode single-frame references, batching equal-shape ones together.

    ``sources`` are ``[1, H, W, 3]`` tensors (one still each); the result is a
    list of ``[1, C, 1, h, w]`` latents in the same order. Encoding N stills
    as one ``[N, 1, H, W, 3]`` call turns N passes over the VAE's 256px tile
    stack into one, which is a large win on a GPU: each tile is far too small
    to fill the SMs on its own.

    This is guarded, because batching rides a less-travelled path:

      * only refs whose H/W are already multiples of the VAE's spatial ratio
        are candidates (``resize_ref`` snaps to 32, so this holds in
        practice) — that lets the encode skip ComfyUI's pixel crop, which
        would otherwise truncate the length-1 frame axis of a 5D input;
      * the first batch is checked against a standalone encode of the same
        still and batching is disabled for that VAE if they disagree;
      * the batch is sized by the VAE's own memory estimate, so an 8 GB card
        doesn't OOM on a 20-image dataset;
      * any failure falls back to per-image encodes, i.e. exactly the old
        behavior.

    ``max_batch < 2`` disables batching. ``REFMOD_ENCODE_BATCH`` sets the
    default cap when the caller passes 0.
    """
    results: List[Optional[torch.Tensor]] = [None] * len(sources)
    if max_batch == 0:
        max_batch = int(os.environ.get("REFMOD_ENCODE_BATCH", "4") or 4)
    pending = []          # indices that still need an individual encode
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, s in enumerate(sources):
        if torch.is_tensor(s) and s.dim() == 4 and s.shape[0] == 1 and max_batch >= 2:
            groups.setdefault((int(s.shape[1]), int(s.shape[2])), []).append(i)
        else:
            pending.append(i)

    ratio = _vae_spatial_ratio(vae)
    for (h, w), idxs in groups.items():
        if (len(idxs) < 2 or h % ratio or w % ratio
                or _batch_verdict(vae) is False):
            pending.extend(idxs)
            continue
        limit = min(max_batch, _encode_batch_limit(vae, device, (1, h, w)), len(idxs))
        if limit < 2:
            pending.extend(idxs)
            continue
        for start in range(0, len(idxs), limit):
            chunk = idxs[start:start + limit]
            if len(chunk) < 2 or _batch_verdict(vae) is False:
                pending.extend(chunk)
                continue
            stacked = torch.stack([sources[i][0][..., :3] for i in chunk], 0).unsqueeze(1)
            saved_crop = getattr(vae, "crop_input", None)
            try:
                # dims are pre-aligned, so the crop is a no-op for the spatial
                # axes — and skipping it is what keeps the length-1 frame axis
                # of a 5D input intact
                if saved_crop is not None:
                    vae.crop_input = False
                encoded = vae.encode(stacked)
            except Exception as exc:
                _set_batch_verdict(vae, False)
                log(f"[RefMod] batched VAE encode unavailable ({exc}); "
                    f"encoding the remaining ref(s) individually")
                pending.extend(chunk)
                continue
            finally:
                if saved_crop is not None:
                    vae.crop_input = saved_crop
            expected = (len(chunk), encoded.shape[1], 1,
                        max(1, h // ratio), max(1, w // ratio))
            if tuple(encoded.shape) != expected:
                _set_batch_verdict(vae, False)
                log(f"[RefMod] batched VAE encode returned {tuple(encoded.shape)}, "
                    f"expected {expected}; encoding individually")
                pending.extend(chunk)
                continue
            if verify and _batch_verdict(vae) is None:
                reference = vae.encode(sources[chunk[0]][..., :3])
                same = (tuple(reference.shape) == tuple(encoded[0:1].shape) and
                        torch.allclose(reference.float(), encoded[0:1].float(),
                                       rtol=1e-2, atol=1e-2))
                _set_batch_verdict(vae, same)
                if not same:
                    log("[RefMod] batched VAE encode disagrees with a single "
                        "encode; using per-image encodes for this VAE")
                    pending.extend(chunk)
                    continue
                log(f"[RefMod] verified batched VAE encode against a single "
                    f"encode (up to {limit} refs per call)")
            for n, i in enumerate(chunk):
                results[i] = encoded[n:n + 1]
    for i in sorted(set(pending)):
        if results[i] is None:
            results[i] = vae.encode(sources[i])
    missing = [i for i, r in enumerate(results) if r is None]
    if missing:  # unreachable; keep the contract explicit rather than returning None
        raise RuntimeError(f"encode_ref_stills left {len(missing)} reference(s) unencoded")
    return results


def refmods_dir() -> str:
    """First registered RefMod root, or the default; selection does not write."""
    import folder_paths
    try:
        roots = folder_paths.get_folder_paths("refmods")
    except KeyError:
        roots = []
    return roots[0] if roots else os.path.join(folder_paths.models_dir, "refmods")


def list_media_files(folder: str) -> Tuple[List[str], List[str]]:
    """(images, videos) directly under ``folder`` (top level only), sorted by name."""
    images, videos = [], []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            ext = os.path.splitext(fn)[1].lower()
            p = os.path.join(folder, fn)
            if os.path.isfile(p):
                if ext in IMAGE_EXTS:
                    images.append(p)
                elif ext in VIDEO_EXTS:
                    videos.append(p)
    return images, videos


def load_image_file(path: str, max_edge: Optional[int] = None,
                    device=None) -> torch.Tensor:
    """Load one image file -> [1, H, W, 3] float32 in [0, 1].

    ``max_edge`` downscales on the CPU while decoding (the cheap path).
    Pass ``device`` to hand the pixels to a GPU instead: the expensive
    lanczos resize then happens there (see ``resize_ref``), so this only
    bounds the upload — a source far larger than ``max_edge`` gets a fast
    bilinear pass first, and the precise resize still runs at >= 2x the
    target size.
    """
    import numpy as np
    from PIL import Image
    gpu = device is not None and torch.device(device).type != "cpu"
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max_edge is not None:
            # CPU path: land exactly on the cap. GPU path: only bound the
            # transfer, keep at least 2x the target for the lanczos pass.
            limit = max_edge * 2 if gpu else max_edge
            scale = min(1.0, limit / max(w, h))
            if scale < 1.0:
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                 Image.BILINEAR if gpu else Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
    if gpu:
        arr = arr.to(device)
    return arr.unsqueeze(0)  # [1, H, W, 3]


def _sample_video_frames(frames, total, max_frames, max_edge):
    """Uniform known-length sampling; bounded deterministic reservoir otherwise."""
    import numpy as np
    from PIL import Image
    known = math.isfinite(total) and total > 0
    total = int(total) if known else 0
    targets = None
    if total > max_frames:
        targets = {round(i * (total - 1) / max(1, max_frames - 1)) for i in range(max_frames)}
    kept = []
    rng = random.Random(0)
    for index, frame in enumerate(frames):
        if targets is not None and index not in targets:
            continue
        # Treat underestimated counts as unknown after filling the reservoir.
        slot = len(kept) if len(kept) < max_frames else rng.randrange(index + 1)
        if slot >= max_frames:
            continue
        frame = np.asarray(frame)
        if max_edge is not None:
            h, w = frame.shape[:2]
            scale = min(1.0, max_edge / max(h, w))
            if scale < 1.0:
                frame = np.asarray(Image.fromarray(frame).resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR))
        item = (index, frame.copy())
        if slot == len(kept):
            kept.append(item)
        else:
            kept[slot] = item
        if targets is not None and len(kept) == len(targets):
            break
    if not kept:
        raise ValueError("Video contains no decodable frames.")
    return torch.from_numpy(np.stack([frame for _, frame in sorted(kept)])).float() / 255.0


def load_video_file(path: str, max_frames: int = 240,
                    max_edge: Optional[int] = None) -> torch.Tensor:
    """Load bounded, ordered RGB frames; release either decoder on every exit."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    errors = []
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            def frames():
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return _sample_video_frames(frames(), total, max_frames, max_edge)
        finally:
            cap.release()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(path)
        try:
            total = reader.get_meta_data().get("nframes", 0) or 0
            return _sample_video_frames(iter(reader), total, max_frames, max_edge)
        finally:
            reader.close()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        errors.append(str(exc))
    raise RuntimeError(f"Cannot decode video '{path}': " + "; ".join(errors))


def mod_output_path(name, subfolder=""):
    relative = "/".join(part for part in (subfolder.replace("\\", "/"), name) if part)
    if (ntpath.splitdrive(relative)[0] or relative.startswith("/") or
            any(p in ("..", ".", "graph_presets", ".git", "__pycache__") for p in relative.split("/"))):
        raise ValueError("Use a relative RefMod subfolder without '..' or reserved directories.")
    root = os.path.realpath(refmods_dir())
    path = os.path.join(root, relative)
    if os.path.commonpath((root, os.path.realpath(path + ".safetensors"))) != root:
        raise ValueError("RefMod save path leaves the model folder.")
    return path


def resize_ref(image, short_edge: int, canvas=None, device=None):
    """Aspect-preserving downscale (never upscale) to ``short_edge`` px; dims to /32.

    When several refs are stacked into one mod they must share a single spatial
    canvas, so ``canvas`` (tw, th) cover-crops each ref to it (like the official
    node's follower keyframes).  Mirrors the official ref2video node: refs are
    resized before VAE encode, so the stored latent rides the same
    full-resolution path the model was trained with (the pooled path below is
    the cheap "thumbnail" alternative).

    ``device`` moves the pixels to a GPU before resizing, which is where the
    lanczos pass belongs when a GPU is going to encode the result anyway: on a
    large source the CPU resize costs more than the extra PCIe transfer it
    saves. The returned tensor then lives on that device.
    """
    if device is not None:
        target = torch.device(device)
        if target.type != "cpu" and image.device.type == "cpu":
            image = image.to(target)
    h, w = image.shape[1], image.shape[2]
    if h <= 0 or w <= 0:
        raise ValueError(
            f"_resize_ref: source has an empty frame ({h}x{w}) before any "
            f"resize — the reference itself is invalid.")
    scale = min(1.0, short_edge / min(h, w))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    crop = "disabled"
    if canvas is not None:
        tw, th = canvas
        crop = "center"
    if tw <= 0 or th <= 0:
        raise ValueError(
            f"_resize_ref: computed a zero-size resize target ({tw}x{th}) "
            f"for a {h}x{w} source (short_edge={short_edge}, canvas={canvas}). "
            f"This should be impossible — please report this shape.")
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", crop)
    if samples.shape[2] <= 0 or samples.shape[3] <= 0:
        raise ValueError(
            f"_resize_ref: common_upscale produced an empty result "
            f"{tuple(samples.shape)} from a {h}x{w} source targeting "
            f"{tw}x{th} (crop={crop}, canvas={canvas}). This points to a bug "
            f"in comfy.utils.common_upscale for this input, not in RefMod's "
            f"own math.")
    return samples.movedim(1, -1)


def snap_to_causal_grid(n_frames: int) -> int:
    """Round down to H3's reference-video grid: 5, 22, 39, ... (17k+5).

    The VAE encodes 17-frame chunks, pads the final chunk and drops three
    trailing latent tokens. Its effective video grid is not the encoder's
    standalone 4x temporal stride. RefMod keeps its single-image fallback
    when fewer than five frames are selected.
    """
    if n_frames < 1:
        raise ValueError("A reference must contain at least one frame.")
    if n_frames < 5:
        return 1
    return ((n_frames - 5) // 17) * 17 + 5


def sample_video_for_vae(frames: torch.Tensor, max_frames: Optional[int] = None) -> torch.Tensor:
    """Sample the full clip onto H3's grid without exceeding the frame cap.

    Keep both endpoints whenever the selected count is greater than one.
    Already-aligned clips are returned unchanged. A cap/input shorter than
    five frames selects only the first image, using H3's still-image path.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    count = snap_to_causal_grid(min(frames.shape[0], max_frames)
                                if max_frames is not None else frames.shape[0])
    if count == frames.shape[0]:
        return frames
    indices = torch.linspace(0, frames.shape[0] - 1, count, device=frames.device).round().long()
    return frames[indices]


def resolve_output_dir(output: Optional[str] = None, comfy_root: Optional[str] = None) -> str:
    """Absolute RefMod output directory for a CLI ``--output`` value.

    The creation guide shows a *relative* ``--output models\\refmods`` run from
    the ComfyUI root, and full paths when run from anywhere else. A relative
    value that doesn't exist next to the CWD but does exist under the ComfyUI
    root is resolved against the root, so the documented command writes to the
    same place either way; otherwise it is CWD-relative (and created on save).
    """
    if not output:
        return refmods_dir()
    if os.path.isabs(output):
        return os.path.normpath(output)
    cwd_path = os.path.abspath(output)
    if not os.path.isdir(cwd_path) and comfy_root:
        rooted = os.path.abspath(os.path.join(comfy_root, output))
        if os.path.isdir(rooted):
            return rooted
    return cwd_path


# The creation guide's troubleshooting section: a valid visual RefMod is
# roughly 1.1-1.6 MB, and anything under 100 KB means the encode produced
# nothing usable.
SMALL_MOD_BYTES = 100 * 1024


def warn_if_implausible_size(path: str, tokens: int, kind: str = "visual",
                             log=print) -> bool:
    """True when the written file looks like a failed extraction.

    Returns whether the file looks healthy; the caller decides what to do with
    the warning (both CLIs print it and keep going).
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if kind == "audio":
        return True
    if size < SMALL_MOD_BYTES:
        log(f"[RefMod] WARNING: {os.path.basename(path)} is only "
            f"{size / 1024:.1f} KB. The creation guide expects ~1.1-1.6 MB for a "
            f"visual RefMod; under 100 KB usually means no valid images were "
            f"encoded. Check the source folder and re-run.")
        return False
    if tokens <= 0:
        log(f"[RefMod] WARNING: {os.path.basename(path)} stores {tokens} tokens — "
            f"it will inject nothing. Check --resolution/--pool.")
        return False
    return True


def ensure_min_size(image, floor: int = 320):
    """Upscale (never downscale) so both spatial dims are >= ``floor`` px.

    The MiniMax H3 VAE encodes with internal tiled_encode (~256px tiles). A
    reference smaller than the tile size in one dimension can make the tiler
    compute a zero-size edge tile, which crashes deep inside conv_in with a
    cryptic 'Expected 4D or 5D... but got [1,3,1,0,W]' error. This applies
    regardless of extraction mode ('encode' already resizes down to
    ref_resolution but never guarantees a floor; 'training' now resizes to
    the same cap, also without a floor), so it's a separate, unconditional
    safety net right before encode.
    """
    import comfy.utils
    h, w = image.shape[1], image.shape[2]
    if h >= floor and w >= floor:
        return image
    scale = floor / min(h, w)
    tw = max(floor, round(w * scale / 32) * 32)
    th = max(floor, round(h * scale / 32) * 32)
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
    return samples.movedim(1, -1)
