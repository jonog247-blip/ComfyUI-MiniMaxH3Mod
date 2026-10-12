"""End-to-end regressions for the creation-guide workflow (no weights, no GPU).

Runs the real CLI entry points (``scripts/generate_refmod.py`` and
``extract_mod.py``) against a stub H3 VAE that follows ``comfy.sd.VAE``'s
contract, so the whole path is covered: folder discovery, name derivation,
aspect/canvas handling, batched encode, token budget, safetensors write and the
guide's file-size/token expectations.

Run with ComfyUI's Python:  python tests/test_refmod_pipeline.py
"""

from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parents[1]))   # the ComfyUI install this pack lives in

from common import encode_ref_stills
from core import H3RefMod, read_refmod_meta
from h3_vae_stub import StubH3VAE

generate = importlib.import_module("generate_refmod")
extract_mod = importlib.import_module("extract_mod")


def make_dataset(folder: Path, count: int = 8, size=(1024, 1024), prefix="img"):
    """``count`` distinct images at ``size`` — the guide's identity dataset."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        band = (np.arange(size[0], dtype=np.int32) + i * 29) % 256
        band = band.astype(np.uint8)
        arr = np.stack([np.tile(band, (size[1], 1)),
                        np.tile(band[::-1], (size[1], 1)),
                        np.full((size[1], size[0]), (i * 17) % 256, np.uint8)], -1)
        path = folder / f"{prefix}{i + 1:02d}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths


class RefModPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        base = Path(cls.temp.name)
        cls.dataset = base / "GabrielleAplin"
        cls.images = make_dataset(cls.dataset, 8)
        cls.other = base / "SecondPersona"
        cls.other_images = make_dataset(cls.other, 3, size=(768, 1024))
        # a directory that looks like a ComfyUI install: comfy/ + folder_paths.py
        cls.comfy_root = base / "ComfyUI"
        cls.vae_dir = cls.comfy_root / "models" / "vae"
        cls.vae_dir.mkdir(parents=True)
        (cls.comfy_root / "comfy").mkdir()
        (cls.comfy_root / "folder_paths.py").write_text("")
        cls.vae_path = cls.vae_dir / "minimax_h3_video_vae_fp16.safetensors"
        cls.vae_path.write_bytes(b"stub")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        self.vae = StubH3VAE()
        loader = patch.object(extract_mod, "load_h3_vae", return_value=self.vae)
        device = patch.object(extract_mod, "resolve_device",
                              return_value=torch.device("cpu"))
        loader.start()
        device.start()
        self.addCleanup(loader.stop)
        self.addCleanup(device.stop)
        reset_batch_verdicts()          # each test starts with an undecided VAE
        self.addCleanup(reset_batch_verdicts)
        self.log = io.StringIO()
        out = patch("sys.stdout", self.log)
        out.start()
        self.addCleanup(out.stop)

    def output_path(self, name):
        return os.path.join(self.out.name, name + ".safetensors")

    # ── discovery / naming / defaults ─────────────────────────────────

    def test_name_follows_the_guide_convention(self):
        self.assertEqual(generate.derive_name("/data/GabrielleAplin"),
                         "minimaxh3_gabrielleaplin_v1_refmod")
        self.assertEqual(generate.derive_name("/data/0Done/My Char!", "v2"),
                         "minimaxh3_my_char_v2_refmod")

    def test_discovery_skips_helper_dirs_and_can_recurse(self):
        (self.dataset / "masks").mkdir(exist_ok=True)
        Image.new("RGB", (64, 64)).save(self.dataset / "masks" / "a.png")
        (self.dataset / "nested").mkdir(exist_ok=True)
        Image.new("RGB", (64, 64)).save(self.dataset / "nested" / "b.png")
        flat, _ = generate.discover(str(self.dataset))
        self.assertEqual(len(flat), 8)
        deep, _ = generate.discover(str(self.dataset), recursive=True)
        self.assertEqual(len(deep), 9)  # masks/ stays excluded either way

    def test_vae_default_is_the_guide_filename(self):
        self.assertEqual(generate.default_vae(str(self.comfy_root)), str(self.vae_path))
        with self.assertRaises(SystemExit):
            generate.default_vae(str(Path(self.temp.name) / "nothing"))

    def test_dry_run_reports_the_plan_and_writes_nothing(self):
        rc = generate.main([str(self.dataset), "--comfy-dir",
                            str(self.comfy_root), "--dry-run",
                            "--output", self.out.name])
        self.assertEqual(rc, 0)
        report = self.log.getvalue()
        self.assertIn("1024x1024px -> 64x64 latent -> 1024 tokens/frame", report)
        self.assertIn("8192 tokens (fits the 8192 cap)", report)
        self.assertEqual(os.listdir(self.out.name), [])

    # ── the guide's documented output ─────────────────────────────────

    def test_identity_dataset_produces_the_documented_mod(self):
        rc = generate.main([str(self.dataset), "--comfy-dir",
                            str(self.comfy_root), "--output", self.out.name])
        self.assertEqual(rc, 0)
        name = "minimaxh3_gabrielleaplin_v1_refmod"
        path = self.output_path(name)
        self.assertTrue(os.path.isfile(path), self.log.getvalue())
        mod = H3RefMod.load(path[:-len(".safetensors")])
        # guide: 8 stills at a 1024px short edge = 8 x 1024 tokens, ~1.5 MB
        self.assertEqual(mod.kind, "video")
        self.assertEqual(mod.latent_t, 8)
        self.assertEqual((mod.latent_h, mod.latent_w), (64, 64))
        self.assertEqual(mod.token_count, 8192)
        self.assertEqual(mod.concept_type, "identity")   # the guide's default
        self.assertEqual(mod.mode, "encode")
        self.assertEqual(mod.latent.dtype, torch.float16)
        size = os.path.getsize(path)
        self.assertGreaterEqual(size, 1.1 * 1024 * 1024)
        self.assertLessEqual(size, 1.6 * 1024 * 1024)
        meta = read_refmod_meta(path[:-len(".safetensors")])
        self.assertEqual(meta["concept_type"], "identity")
        self.assertEqual(meta["tags"], ["8 img, 0 vid"])
        self.assertEqual(meta["source"], "stack")

    def test_documented_concept_alias_general_is_stored_canonically(self):
        rc = generate.main([str(self.dataset), "--comfy-dir",
                            str(self.comfy_root), "--concept-type", "general",
                            "--name", "alias_check", "--output", self.out.name])
        self.assertEqual(rc, 0)
        mod = H3RefMod.load(self.output_path("alias_check")[:-len(".safetensors")])
        self.assertEqual(mod.concept_type, "generic")

    def test_token_cap_trims_to_the_budget_and_reports_it(self):
        many = Path(self.temp.name) / "BigSet"
        make_dataset(many, 12)
        rc = generate.main([str(many), "--comfy-dir", str(self.comfy_root),
                            "--output", self.out.name])
        self.assertEqual(rc, 0)
        mod = H3RefMod.load(
            self.output_path("minimaxh3_bigset_v1_refmod")[:-len(".safetensors")])
        self.assertLessEqual(mod.token_count, 8192)
        self.assertEqual(mod.latent_t, 8)          # 8192 // 1024 tokens per frame
        self.assertIn("dropped to stay under --max-tokens 8192", self.log.getvalue())

    def test_extract_mod_cli_defaults_match_the_guide(self):
        parser = extract_mod.build_parser()
        defaults = {a.dest: a.default for a in parser._actions}
        self.assertEqual(defaults["mode"], "encode")
        self.assertEqual(defaults["max_tokens"], 8192)
        self.assertEqual(defaults["resolution"], 1024)

    def test_extract_mod_cli_writes_a_loadable_mod(self):
        argv = ["--vae", str(self.vae_path),
                "--image", self.images[0], "--image", self.images[1],
                "--name", "cli_two", "--mode", "encode", "--resolution", "512",
                "--concept-type", "character", "--output", self.out.name,
                "--encode-batch", "2"]
        mod = extract_mod.main(argv)
        self.assertEqual(mod.concept_type, "identity")   # alias, canonicalized
        self.assertEqual(mod.latent_t, 2)
        self.assertTrue(os.path.isfile(self.output_path("cli_two")))
        # both stills shared one batched call
        self.assertIn((2, 3, 1, 512, 512), self.vae.calls)

    def test_guide_method_2_command_runs_verbatim(self):
        """The exact extract_mod.py invocation from section 5 of the guide."""
        cwd = os.getcwd()
        os.chdir(str(self.comfy_root))       # the guide runs from the ComfyUI root
        try:
            argv = ["--vae", "models/vae/minimax_h3_video_vae_fp16.safetensors",
                    "--name", "minimaxh3_gabrielle_aplin_refmod",
                    "--mode", "encode", "--concept-type", "identity",
                    "--resolution", "1024", "--max-tokens", "8192",
                    "--output", "models/refmods",
                    "--comfy-dir", str(self.comfy_root)]
            argv += sum([["--image", path] for path in self.images], [])
            mod = extract_mod.main(argv)
        finally:
            os.chdir(cwd)
        written = (self.comfy_root / "models" / "refmods"
                   / "minimaxh3_gabrielle_aplin_refmod.safetensors")
        self.assertTrue(written.is_file(), self.log.getvalue())
        self.assertEqual(mod.token_count, 8192)
        self.assertEqual(mod.concept_type, "identity")

    def test_bulk_mode_loads_the_vae_once_and_skips_existing(self):
        root = Path(self.temp.name)
        calls = []
        with patch.object(extract_mod, "load_h3_vae",
                          side_effect=lambda *a, **k: calls.append(a) or self.vae):
            rc = generate.main([str(root), "--each-subfolder", "--comfy-dir",
                                str(self.comfy_root), "--output", self.out.name])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)               # one VAE for every dataset
        first = self.output_path("minimaxh3_gabrielleaplin_v1_refmod")
        self.assertTrue(os.path.isfile(first))
        self.assertTrue(os.path.isfile(self.output_path("minimaxh3_secondpersona_v1_refmod")))
        mtime = os.path.getmtime(first)
        self.log.truncate(0)
        generate.main([str(root), "--each-subfolder", "--comfy-dir",
                       str(self.comfy_root), "--output", self.out.name])
        self.assertIn("exists (use --overwrite)", self.log.getvalue())
        self.assertEqual(os.path.getmtime(first), mtime)

    def test_portrait_dataset_keeps_its_aspect(self):
        rc = generate.main([str(self.other), "--comfy-dir",
                            str(self.comfy_root), "--output", self.out.name,
                            "--resolution", "768"])
        self.assertEqual(rc, 0)
        mod = H3RefMod.load(
            self.output_path("minimaxh3_secondpersona_v1_refmod")[:-len(".safetensors")])
        # 768x1024 source: short edge 768 -> 768x1024 canvas -> 48x64 latent
        self.assertEqual((mod.latent_h, mod.latent_w), (64, 48))
        self.assertEqual(mod.token_count, 3 * 32 * 24)

    # ── batched encode: verified, and safe when it can't be ────────────

    def test_batched_encode_matches_single_encodes(self):
        vae = StubH3VAE()
        sources = [torch.rand(1, 64, 96, 3) for _ in range(3)]
        batched = encode_ref_stills(vae, sources, max_batch=3)
        solo = StubH3VAE()
        expected = [solo.encode(s) for s in sources]
        self.assertEqual(len(batched), 3)
        for got, want in zip(batched, expected):
            self.assertTrue(torch.allclose(got.float(), want.float(), atol=1e-6))

    def test_unsafe_batch_dimension_falls_back_and_still_saves(self):
        vae = StubH3VAE(batch_unsafe=True)
        sources = [torch.rand(1, 64, 96, 3) for _ in range(3)]
        got = encode_ref_stills(vae, sources, max_batch=3)
        self.assertIn("disagrees with a single encode", self.log.getvalue())
        solo = StubH3VAE(batch_unsafe=True)
        for latent, src in zip(got, sources):
            self.assertTrue(torch.allclose(latent.float(),
                                           solo.encode(src).float(), atol=1e-6))

    def test_batching_is_skipped_when_sources_differ_in_size(self):
        vae = StubH3VAE()
        sources = [torch.rand(1, 64, 96, 3), torch.rand(1, 64, 64, 3)]
        got = encode_ref_stills(vae, sources, max_batch=4)
        self.assertEqual(len(got), 2)
        self.assertTrue(all(c[0] == 1 for c in vae.calls))   # one call per still

    def test_batch_is_capped_by_the_vaes_own_memory_estimate(self):
        vae = StubH3VAE(free_bytes=520_000_000)   # ~one 1024px still
        sources = [torch.rand(1, 1024, 1024, 3) for _ in range(4)]
        encode_ref_stills(vae, sources, max_batch=4)
        self.assertTrue(all(c[0] == 1 for c in vae.calls))


class NodeStubVAE(StubH3VAE):
    """Pretends to live on CUDA so the node's batched path engages on a CPU box.

    ``output_device`` is pinned to the CPU: the GPU-IO context manager only
    decides where an encode result is parked, and there is no GPU here.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cuda")

    @property
    def output_device(self):
        return torch.device("cpu")

    @output_device.setter
    def output_device(self, value):
        pass


class ExtractNodeTests(unittest.TestCase):
    """The Extract node must batch plain stills and produce the same mod."""

    @classmethod
    def setUpClass(cls):
        from comfy.cli_args import args
        args.cpu = True          # no GPU probe, same as tests/test_loaders.py
        try:
            cls.N = importlib.import_module(f"custom_nodes.{ROOT.name}.nodes")
        except Exception as exc:                      # pragma: no cover
            raise unittest.SkipTest(f"pack nodes not importable here: {exc}")

    def setUp(self):
        reset_batch_verdicts()
        self.addCleanup(reset_batch_verdicts)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.N._MOD_CACHE.clear()
        self.addCleanup(self.N._MOD_CACHE.clear)
        self.out = patch.object(self.N, "mod_output_path",
                                side_effect=lambda name, sub="":
                                os.path.join(self.temp.name, name))
        self.out.start()
        self.addCleanup(self.out.stop)
        self.log = io.StringIO()
        echo = patch("sys.stdout", self.log)
        echo.start()
        self.addCleanup(echo.stop)

    def run_extract(self, vae, count=3):
        refs = {f"ref_image_{i + 1}": torch.full((1, 64, 64, 3), 0.1 * (i + 1))
                for i in range(count)}
        return self.N.MiniMaxH3RefModExtract.execute(
            f"node_{id(vae)}", mode="encode", refs_image=refs, vae=vae,
            ref_resolution=64, save=True, concept_type="general")

    def test_batched_and_unbatched_runs_produce_the_same_latent(self):
        gpu = NodeStubVAE()
        bundled = self.run_extract(gpu)
        self.assertTrue(any(c[0] == 3 for c in gpu.calls), gpu.calls)
        self.assertIn("batched VAE calls", self.log.getvalue())

        cpu = StubH3VAE()
        solo = self.run_extract(cpu)
        self.assertTrue(all(c[0] == 1 for c in cpu.calls), cpu.calls)

        got, want = bundled[0][0][0], solo[0][0][0]
        self.assertEqual(tuple(got.latent.shape), tuple(want.latent.shape))
        self.assertTrue(torch.allclose(got.latent.float(), want.latent.float(),
                                       atol=1e-6))
        self.assertEqual(got.concept_type, "generic")   # alias canonicalized

    def test_masked_refs_stay_on_the_per_ref_path(self):
        # a mask has to be applied to each ref's own latent, so those refs can
        # never go through the shared batch
        vae = NodeStubVAE()
        refs = {f"ref_image_{i + 1}": torch.full((1, 64, 64, 3), 0.2)
                for i in range(3)}
        self.N.MiniMaxH3RefModExtract.execute(
            "masked", mode="encode", refs_image=refs, vae=vae,
            mask=torch.ones(64, 64), ref_resolution=64, save=True)
        self.assertEqual(vae.calls, [(1, 3, 1, 320, 320)] * 3, vae.calls)

    def test_training_mode_stays_on_the_per_ref_path(self):
        vae = NodeStubVAE()
        refs = {f"ref_image_{i + 1}": torch.full((1, 64, 64, 3), 0.3)
                for i in range(3)}
        self.N.MiniMaxH3RefModExtract.execute(
            "pooled", mode="training", refs_image=refs, vae=vae,
            ref_resolution=64, identity=0, save=True, pool_h=4, pool_w=4)
        self.assertEqual(vae.calls, [(1, 3, 1, 320, 320)] * 3, vae.calls)


def reset_batch_verdicts():
    """Drop the per-VAE batching verdict so each test starts undecided."""
    import common
    common._BATCH_VERDICTS.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
