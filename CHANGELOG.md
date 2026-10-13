# Changelog

All notable changes are tracked here. Each version is also published as a
[GitHub Release](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/releases),
so you can keep using an older version if a new one changes something you
rely on.

## v0.2.9 — 2026-10-13

### One graph, one character

- Add `examples/workflows/refmod_character_master.json`: Load H3 RefMod Folder (stills **and** a short motion clip from one folder) + LoadAudio → Master (identity, 1024 px, 8192 tokens, `save_layout = bundle`, `audio_concept_type = voice`) → Inspect → PreviewImage and PreviewAudio. One queue produces one `models/refmods/<name>.safetensors` holding the visual reference and the voice, and Inspect decodes both members back so you can see and hear what was captured before spending GPU time on generation. Both VAELoaders are wired; no DiT and no CLIP are needed.
- Verified end to end against a live ComfyUI 0.39.0 server: the shipped graph, posted as the frontend would send it, returned `status: success` in 46 s and wrote a version-5 bundle whose visual member is tagged `3 img, 1 vid` (so the motion clip really went through the folder loader) with an audio member of 20 latent frames, both tensors finite, plus a 384×384 image preview and an audio preview. The run used fabricated miniature VAE weights on a 3.8 GB-RAM box with reduced resolution/budget widget values, so it proves the plumbing, not reference quality. On the same run the load-time latent guard fired on a NaN encode from those fabricated weights and failed loudly instead of saving a broken mod.
- Rebuild all three example graphs with numbered colour groups and a longer instruction Note, and extend `tests/test_example_workflows.py`: the character graph's links and widget values are pinned (folder → `refs_bundle`, `audio`, `vae`, `audio_vae`, `save_layout = bundle`, both previews fed by Inspect), and every graph is checked for layout — exactly one Note, every node inside a group box, no two groups overlapping.

### Speed

- Decode references ahead of the encode instead of one file at a time. `MiniMaxH3RefModFolderLoader` prepares the next stills on daemon threads (`FOLDER_LOADER_PREFETCH = 3`) while the VAE works, and `extract_mod.py` / `scripts/generate_refmod.py` get the same via `--prefetch` (default 2). Measured on a 2-core CPU box: 12 2048² JPEGs at `max_edge=1024` went 1.34 s → 0.64 s (≈2.1×, saturating at the core count); 12 refs at 512 px against a stub VAE that sleeps 60 ms per encode went 0.87 s → 0.79 s. Unreadable files are still reported and skipped rather than aborting the run, and videos stay sequential because one decode already buffers a whole clip.
- Encode byte-identical sources once and reuse the latent (`--dedupe on`, the default), so a dataset that contains the same shot twice — or the same file listed twice — costs one encode instead of two. `--dedupe off` restores the old loop.
- Both are asserted in tests to produce exactly the tensors the old sequential path produced, so a saved workflow's output does not change.

### Optional edge-aware refinement

- `optimize_latent` / `optimize_latent_multi` take `objective="mse"|"detail"` and `detail_weight`, exposed as `--refine-objective` and `--refine-detail-weight` (default 1.0) in both CLIs. `detail` adds a per-map Laplacian high-pass term to the reconstruction loss. On a structured synthetic latent over 200 steps at lr 0.05 it cut high-pass error from 0.08725 to 0.07922 (−9 %) while plain MSE rose from 0.01472 to 0.01807 (+23 %); at weight 5 it over-trades. Because it is a trade and not a free win, `mse` remains the default and the node API is unchanged.

### Video references on a stock install

- `common.load_video_file` now falls back to PyAV after OpenCV and imageio. ComfyUI itself depends on PyAV, so a fresh install can read `.mp4`/`.webm`/`.mov` references without installing anything; before this, only OpenCV and imageio were attempted and neither ships with ComfyUI, so every clip failed with "Cannot decode video". When no backend is available the error now names all three attempts. Covered by `VideoDecodeTests`, which encodes a real clip with PyAV and runs it with the optional backends forced absent.

### Tests

- `tests/test_core_refmod.py` 27 → 35 (pipelined-load order, single-worker inline path, off-thread execution, exception propagation, `duplicate_map`, and the `detail` objective), `tests/test_refmod_pipeline.py` 19 → 28 (dedupe on/off, prefetch parity, the `detail` objective through the CLI, folder-loader prefetch parity and skip-not-fatal, video backends), `tests/test_example_workflows.py` 7 → 14.

## v0.2.8 — 2026-10-12

### Guide compliance

- Add `scripts/generate_refmod.py`, the whole-folder extractor the creation guide documents as Method 1: folder scanning (with `--recursive`, mask/caption folder exclusion), `--dry-run` plan (canvas, tokens per frame, budget, estimated size), `--each-subfolder` bulk mode that loads the VAE once (the guide's Method 3), `--overwrite`, guide-conventional `minimaxh3_<dataset>_v1_refmod` naming, and the guide's CLI surface (`--comfy-dir`, `--name`, `--resolution`, `--max-tokens`, `--concept-type`, `--dry-run`). It locates its ComfyUI root itself, so it runs from the pack or from a copy in `ComfyUI/scripts/`.
- Accept the documented concept-type spellings everywhere (`general` -> `generic`, `character`/`person`/`face` -> `identity`, `pose` -> `pose_motion`, `outfit` -> `clothing`, `sfx` -> `sound_fx`, ...) and store them canonically. Mods that carry an unrecognized label keep it verbatim, so custom vocabularies survive a round trip.
- `extract_mod.py` now defaults to `--mode encode` (its own docstring and the guide already described encode as the default; the flag said `training`) and to `--max-tokens 8192`, the budget the guide documents. 5120 silently trimmed the guide's own 8-image example.
- `--output` resolves a relative path against `--comfy-dir` when it exists there, so the guide's `--output models\refmods` writes to the same place from any working directory.
- Add the guide's sanity checks to both CLIs: 8-20 images recommended for an identity dataset, a warning when two source files are the same shot, and a warning when the written mod is under 100 KB (a healthy 8 x 1024px identity mod is ~1.5 MB / 8192 tokens).
- Fix `issue_regression_test.py`, which could not be run as documented: the script directory shadowed ComfyUI's `nodes` module and every relative import in the pack failed. All 10 of its regressions run again.

### CUDA extraction speed

- Batch same-shape still references into one VAE call (`--encode-batch`, default 4) instead of one call each. Guarded: pre-aligned refs only, the first batch is verified against a single encode, the batch is sized by the VAE's own free-memory estimate, and any disagreement falls back to per-image encodes for that VAE. The Extract node uses the same helper for plain `encode`-mode stills; masked, `motion_only`, video and `training` refs keep their per-reference path.
- Scope TF32 + cuDNN autotuning to the encode (`REFMOD_TF32=0` / `REFMOD_CUDNN_BENCHMARK=0` to keep the old numerics) and restore both afterwards, so the pack never changes the numerics of the rest of the process.
- Keep the encode output on the GPU (`REFMOD_VAE_GPU_IO=0` to opt out) so a full-res latent is not copied to the CPU and back for refinement, and so the fp16 store copies half the bytes.
- Resize stills on the encode device when it is a GPU (`--resize-on-gpu never` to opt out).
- Stop the gradient refinement early once the loss is flat (`--refine-tol`, default 1e-4 in the CLIs; 0 keeps the old behavior) and use CUDA's fused Adam in both refinement loops.
- Memoize each reference's blurred latent for a sampling run in the Step Curve wrapper instead of recomputing it per denoising step and per CFG branch. Bounded LRU, cleared when a new wrapper is built; MPS benefits most because that blur round-trips through the CPU.

### Robustness and quality

- Reject invalid stored latents at load time with a specific error: non-even latent grids (H3 patches 2x2 cells, so an odd edge silently drops a row), NaN/Inf latents, `kind="image"` with several frames, and wrong tensor rank.
- Add `--frame-selection diverse` (default in both CLIs) for when the token cap forces a cut on a stack of stills: greedy k-center keeps the most distinct views instead of whatever sits at even positions. `uniform` remains available and is still what the node does, so saved workflows are unchanged.
- Reconcile the release metadata: `__init__.py` and `pyproject.toml` both report 0.2.8.
- Track `tests/` in git and add `tests/run_all.py`, so a clean clone can reproduce the checks: 88 python tests across seven suites plus the frontend slot test.

### Example workflows

- Add `examples/workflows/refmod_smoke_test.json`: folder loader -> Create H3 RefMod (Full Reference, identity, 1024px, 8192 tokens) -> Inspect -> PreviewImage. It needs the H3 video VAE and nothing else, so a fresh install can confirm extraction, saving, loading and decoding without the 29B DiT or the CLIP.
- Add `examples/workflows/refmod_identity_video.json`: UNETLoader -> Sigma Shift -> Image to Video -> Apply H3 RefMod -> KSampler -> VAE Decode (+ audio) -> Create Video -> Save Video, with the shipped `vanellope_example` mod in the loader slot.
- Add `tests/test_example_workflows.py`, which loads both files and checks them against the live node schemas — node types exist, socket names are real inputs, widget values line up positionally with the schema's widget list, every link resolves at both ends with a compatible type, and each graph reaches an output node — then converts each graph to the API prompt format and runs it through ComfyUI's own `execution.validate_prompt`. Placeholder model names keep it weight-free and GPU-free. Verified against ComfyUI 0.39.0; the smoke graph was also executed for real end to end (folder of stills -> 2816-token mod on disk -> decoded preview).
- Correct the stale node-default list in the README's minimal-workflow steps (the visual node defaults to Compressed Reference and `generic`, and the CLIs default the other way round because that is what the creation guide documents).

## v0.2.7 (unreleased work folded into 0.2.8)

- Correct visual reference sampling to H3's 17k+5 video grid in the node and CLI. Preserve all frames of aligned clips and both endpoints when uniformly sampling; references/caps below five frames use the first image.
- Support MPS-resident RefMod latents without global CPU fallback: pool and blur explicitly on CPU, preserving return device, dtype and autograd. Synchronize MPS in the refinement timing harness.

- Add a reference_map output to Inspect H3 RefMod for LLM prompt generation before Text Encode, using the same reference numbering (#19).
- Fix Loader/Axis settings resetting when switching workflow tabs or reopening workflows: defer slot regrouping until widget restoration finishes (#18).
- Repair Apply's reactive MatchType socket on older frontends whose canvas socket classes use private fields, preventing missing controls while preserving conditioning type matching (#17).

## v0.2.6 — 2026-09-12

- Make Loader/Axis additive: start with one visible slot, add/remove slots up to eight, group modality controls with their reference, preserve old workflow values and protect connected slots.
- Add version-5 single-file RefMod bundles containing independent image/video/audio members, with atomic saves and preserved member order.
- Add Save H3 RefMod Bundle as an output node and opt-in Master save_layout=bundle. Keep standalone saving and existing workflow defaults.
- Add All/Visual/Audio selection and separate modality strengths to every Loader/Axis slot. Skip excluded tensors and count only selected references in token budgets.
- Discover combined files in the library; preserve unselected members during Config updates and support export to standalone files.
- Document the container schema for third-party pickers. Bundling does not add voice binding, AV synchronization, or original-waveform passthrough.

## v0.2.5 — 2026-09-08

- Rename Extract nodes to Create, with unchanged node IDs. Full Reference and Compressed Reference replace the visible encode/training names; old mode values remain accepted. Rename the identity control to Refinement Steps.
- Add visual budget_policy: truncate preserves existing behavior; error stops before saving when the visual token budget is exceeded. Master inherits the option.
- Allow larger token/frame limits and preserve the requested frame count in the motion preset. Loader budgets default to 0 (disabled).
- Add experimental H3 RefMod Text Encode with numbered reference presentation and a reference map. Support compatible ClipProj encoders and native batched video-VAE output.
- Add optional full-video inspection with correct IMAGE output dimensions; retain the lightweight first-frame default.
- Add playable music/video README examples and exclude local tests from distribution.

Validation: 51 local regression tests passed; creation-label hooks checked with Node.js. Full GPU generation, projected-encoder quality and voice transfer remain unverified. Voice cloning is not claimed to work.

Text Encode already attaches references: do not inject the same bundle again with Apply/Bridge. Integrated Continuum samplers still need a dedicated integration to use this conditioning. Full visual previews and text/vision presentation increase memory use.

## v0.2.0 — 2026-09-07

### Known limitation: voice transfer

Audio reference support includes an observed music-transfer example, but current
speaker-identity tests failed. This release does not provide working voice cloning.
The cause has not been isolated between H3/checkpoint behavior and the reference
presentation difference in the current RefMod integration.

### Follow-up implementation (2026-09-07)

- Clarify Master saving logs: deferred extraction, Created/Replaced destinations,
  explicit save=False and a final token/file summary; report saved_paths in details.

- Add Save H3 RefMods as an output node: save mixed bundles without connecting
  a Preview or sampler, with prefix/subfolder options and copy deduplication.

- Save new mods/presets in the first registered refmods root (extra_model_paths);
  keep the standard models/refmods fallback and existing-file Config locations.
- Remove sibling-pack imports and startup dependency warning. Legacy av_encoder
  inputs use native ComfyUI video VAE loading; standard VAE inputs are preferred.

- Add Extract H3 RefMod Master: optional image/video and audio inputs, sequential
  extraction through the shared extractors, combined token budget, one bundle
  output and separate compatible visual/audio files under a shared name.

- Step Curve now mixes the current payload without retaining tensors and handles
  keyframe offsets, audio refs, empty conditioning and chained curves.
- Bridge uses the MODEL outer-sampling hook with finally cleanup; removes global
  state and third-party monkey patches. The old Disarm node is deprecated.
- Align image/mask center crops; share resize and causal-frame helpers with CLI.
- Invalidate loaded files and queue cache on changes; cap cache at 256 MiB;
  enforce copies/numeric limits; bound video decode buffers with unknown lengths.
- Add native audio RefMods, legacy AudioMod import, audio Extract, library UI,
  inspection/optional previews, save subfolders, explicit scramble modes,
  extraction presets and total-token budgets.
- Atomic saves, deduplicated config writes and grouped mean-MSE refinement.
- Historical entries below describe earlier unreleased revisions; the behavior
  above and the current README supersede their bridge/cache descriptions.

### Fixed: linked loader inputs and subfolders

- Both loaders accept unresolved/empty mod names and connected STRING/combo
  outputs. Fixed missing names still fail validation; connected names resolve
  at execution with a missing-file error when necessary.
- Discover RefMods recursively as `folder/name`, preserving metadata filtering
  and skipping graph presets. Listing cache tracks nested files, nanosecond
  timestamps, and legacy JSON sidecars.
- Normalize Windows separators and check relative paths at execution. Cache
  mods by file path so identical metadata names in different folders do not
  collide when saving their configuration.
- Add regression tests using real safetensors and ComfyUI queue validation.

### New: Continuum integration

- **Continuum RefMod Bridge** node — injects your mods into
  [ComfyUI-H3-Continuum](https://github.com/xiaolibai-sys/ComfyUI-H3-Continuum)'s
  chunked long-video samplers (V2/V3/V3.4), which build their conditioning
  internally and have no CONDITIONING socket. While armed, every chunk
  carries the bundle's ref blocks as native `minimax_refs` — persistent
  across all chunks, no Continuum markers so its continuity layout never
  re-times them. Version-checked against one public seam; if Continuum
  changes internally the bridge reports and degrades to no-injection rather
  than breaking a run. One active bundle per graph; per-run scramble seeds
  work normally. Wire it on the sampler's model line (Load Model -> bridge
  -> sampler `model`): ComfyUI never executes fully disconnected nodes, so
  the earlier "anywhere upstream" placement silently did nothing.

### Fixed: why insertions felt weak, random, or "possessed"

Three defects were found and fixed; each is reproducible with the new
`gauntlet_harness.py` (run it with ComfyUI's Python — no server needed).

- **Apply's default curve silently HALVED every multi-frame mod.** The old
  default (`concept_at_end` + `ease`) runs its envelope across the mod's own
  ref frames — mean multiplier ≈ **0.50** at any frame count — so video/
  stacked mods shipped half-blurred unless you touched widgets. Worse, ref
  tokens are not bound to output-video time at all: no frame curve can place
  a concept "at the end of the video"; it only weights which reference
  content dominates. Defaults are now `constant` + `linear` + `1.0`
  (= full strength, official-ref parity); saved workflows keep their saved
  values. The curve widgets remain for stack weighting, with honest tooltips.
  For real output-timing control use **H3 RefMod Step Curve** (denoise
  timeline).
- **Image-kind mods ignored every curve dial silently** (`curve_strengths`
  short-circuits at 1 latent frame), so single-image identity mods always
  injected at full strength — the "concept bleeds into everything" case.
  Now `curve_value` acts as a plain strength cap on single-frame mods
  (e.g. value 0.4 = the ref blends 40% toward its blurred self), with a
  console note.
- **H3 RefMod Step Curve leaked state across generations.** The wrapper
  cached pristine/blurred latents plus the marked-ref index mapping on the
  first forward and never reset them — but ComfyUI reuses the patched MODEL
  and conditioning payload across queued runs, so run #2 mixed run #1's
  cached tensors whenever the scramble seed/mods/strengths changed
  (corrupted, "ghost" output). The wrapper now keys its state on the
  injected refs' identities (which it never mutates), re-keying cleanly on
  every new run while keeping pristine stable within one; chaining multiple
  Step Curve nodes composes deterministically, and progress is normalized to
  each run's actual schedule start (partial denoise included).

Also new: Apply now prints each row's *effective* strength (row × retention
× curve-mean) and flags anything below 0.30 — the usual suspect when an
insertion feels weak.

### New: fixed configs, shipped with the mod

- **Fix H3 RefMod Config** node — bakes tuned Apply/Step-Curve settings
  (`retention` + `curve` + `step_curve`) into a mod's safetensors metadata
  and re-saves the file in place, so a concept ships with the settings that
  make it work.
- **`override` toggle** on Apply H3 RefMod and H3 RefMod Step Curve — reads
  the first mod in the bundle that carries a fixed config and uses those
  settings instead of the widgets; falls back to the manual parameters
  (with a console note) when no mod has one.
- The debug curve graph reflects the overridden curve, so what you see is
  what actually ran.

### New: merge mode on Extract H3 RefMod

- **`merge` toggle** (training mode): one shared latent refined jointly
  against every reference's full encode (mean reconstruction error) instead
  of stacking each ref's own pooled latent. A whole collection becomes ONE
  tiny consensus mod — pulled toward what's common across all the views
  (structure/motion/identity) rather than any single shot's framing or
  background, at one ref block's worth of tokens. `identity` dials the
  joint refinement; masks and `background_retention` still apply per ref.
- **`motion_only` toggle** (training mode, experimental): video refs are
  converted to per-frame temporal differences before encoding, so the mod
  carries where/how things move instead of what they look like — static
  appearance (lineart look, background) never enters the latent. A soft
  motion guide, not a ControlNet (the ref channel is content-based).

### Fixed

- **Curve directions now mean what they say.** `concept_at_end` actually
  puts the concept at the **end** of the video (and at the end of the
  denoise timeline on H3 RefMod Step Curve); `concept_at_start` puts it at
  the start. The old envelopes were inverted relative to the names. Legacy
  `decrease`/`increase` workflow values and the preset names
  (`fade_in`/`fade_out`/`bump`/`dip`) still resolve to their original
  envelopes, so old workflows keep their behavior.
- **H3 RefMod Step Curve no longer touches your input video.** The wrapper
  used to re-mix *every* ref latent, including the original video's
  identity anchor — that blurred the source subject and let a different
  person "pop". It now identifies the refs injected by Apply H3 RefMod (via
  a marker on the blocks) and only re-mixes those; native ref2va refs pass
  through untouched, and a console note explains when there is nothing to
  re-mix.

## v0.1.0 — 2026-08-16

First tagged release. Extract references once as tiny `.safetensors`
"mods" and inject them through conditioning — no full video/image loading
every generation, no training.

### Nodes

- **Extract H3 RefMod** — image / video / GIF → one small mod file. Modes:
  `training` (pooled concept/identity thumbnails; the `pool` dial trades
  concept ↔ identity) and `encode` (full-resolution VAE encode). Identity
  refinement steps, token cap with dedup, data multiplier for short clips,
  optional `av_encoder` input, and folder bulk-loading.
- **Load H3 RefMods** — LoRA-loader-style rows with a typed strength and a
  `copies` multiplier (2-10x — the manual row-duplication trick as a knob).
- **Load H3 RefMod Axis** — signed A/B sliders: negative picks the A mod,
  positive the B mod, one dial controls both.
- **Load H3 RefMod Folder** — every image/video in a folder as an ordered
  ref list.
- **Apply H3 RefMod** — one node for the pack's `MINIMAX_H3_COND` and the
  built-in `CONDITIONING`. `retention` master strength; a curve split into
  `curve_direction` (constant / concept_at_start / concept_at_middle /
  concept_at_end / concept_at_ends) + `curve_shape` (linear / ease /
  sigmoid / tanh / quadratic / cubic / exponential / stair / elastic /
  bump / dip) + `curve_value`; `scramble_seed`; optional curve-graph
  `debug` IMAGE output; shareable PNG graph presets (graph embedded in the
  image metadata, legacy `.json` still loads).
- **H3 RefMod Step Curve** — the same curve widgets, but over the **denoise
  timeline**: re-mixes every ref latent once per step (early steps lock
  composition/identity, late steps stay clean or refine detail) via a
  ComfyUI `DIFFUSION_MODEL` wrapper, attached between the model loader and
  the sampler.

### Reference math

- Weakening a ref blends toward a blurred copy of itself instead of noise
  or zero — stays on the latent manifold, so no grey/static output.
- The per-frame curve mixes each ref latent frame with
  `retention * curve(x)` instead of one flat strength.
- Greedy temporal dedup + budget-fit resampling make the token cap cheap.

### Misc

- `extract_mod.py` standalone CLI.
- Mods live in `ComfyUI/models/refmods/` (created on first run, next to
  loras/ and unet/); older mods in the pack's `mods/` folder still load.
