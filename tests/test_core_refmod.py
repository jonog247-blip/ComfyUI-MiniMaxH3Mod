"""Unit regressions for the core/common helpers this pack adds.

Covers the concept-type aliases, the RefMod sanity checks, the memoized blur,
the CUDA-scoped context managers and the refinement early-exit. No weights, no
GPU: the CUDA paths are exercised through their pure decision logic.

Run with ComfyUI's Python:  python tests/test_core_refmod.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))

import common
from core import (CONCEPT_TYPES, H3RefMod, _blur_latent, blur_latent_cached,
                  clear_blur_cache, fit_token_budget, normalize_concept_type,
                  optimize_latent, select_frames_diverse)


class ConceptTypeTests(unittest.TestCase):
    def test_documented_guide_names_resolve(self):
        # the creation guide documents identity / style / general
        self.assertEqual(normalize_concept_type("identity"), "identity")
        self.assertEqual(normalize_concept_type("style"), "style")
        self.assertEqual(normalize_concept_type("general"), "generic")

    def test_aliases_and_spelling_variants(self):
        for raw, expected in (("Character", "identity"), ("  person ", "identity"),
                              ("pose-motion", "pose_motion"), ("Outfit", "clothing"),
                              ("SFX", "sound_fx"), ("art style", "style")):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_concept_type(raw), expected)

    def test_unknown_and_empty_stay_unresolved(self):
        for raw in ("", None, 3, "made_up_type"):
            with self.subTest(raw=raw):
                self.assertIsNone(normalize_concept_type(raw))

    def test_aliases_never_collide_with_canonical_names(self):
        from core import CONCEPT_TYPE_ALIASES
        for alias, target in CONCEPT_TYPE_ALIASES.items():
            self.assertIn(target, CONCEPT_TYPES)
            self.assertNotIn(alias, CONCEPT_TYPES)

    def test_saved_mods_read_back_canonically(self):
        raw = H3RefMod.from_metadata({"kind": "image", "concept_type": "general"},
                                     torch.zeros(1, 24, 1, 4, 4))
        self.assertEqual(raw.concept_type, "generic")
        custom = H3RefMod.from_metadata({"kind": "image", "concept_type": "my_label"},
                                        torch.zeros(1, 24, 1, 4, 4))
        self.assertEqual(custom.concept_type, "my_label")  # foreign labels survive


class RefModValidationTests(unittest.TestCase):
    def latent(self, shape, value=0.0):
        z = torch.zeros(*shape)
        if value:
            z += value
        return z

    def test_odd_latent_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not even"):
            H3RefMod(name="odd", kind="image", latent=self.latent((1, 24, 1, 5, 4)))

    def test_non_finite_latent_is_rejected(self):
        z = self.latent((1, 24, 1, 4, 4))
        z[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            H3RefMod(name="nan", kind="image", latent=z)

    def test_image_kind_with_several_frames_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kind='image'"):
            H3RefMod(name="frames", kind="image", latent=self.latent((1, 24, 3, 4, 4)))

    def test_wrong_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[1, 24, T, H, W\]"):
            H3RefMod(name="flat", kind="video", latent=self.latent((24, 4, 4)))

    def test_valid_mods_still_load(self):
        mod = H3RefMod(name="ok", kind="video", latent=self.latent((1, 24, 3, 6, 8)),
                       latent_h=6, latent_w=8, latent_t=3)
        self.assertEqual(mod.token_count, 3 * (6 // 2) * (8 // 2))


class BlurCacheTests(unittest.TestCase):
    def setUp(self):
        clear_blur_cache()
        self.addCleanup(clear_blur_cache)

    def test_matches_the_uncached_blur(self):
        z = torch.randn(1, 24, 3, 16, 16)
        self.assertTrue(torch.equal(blur_latent_cached(z), _blur_latent(z)))

    def test_second_call_is_a_hit(self):
        z = torch.randn(1, 24, 2, 16, 16)
        first = blur_latent_cached(z)
        self.assertIs(blur_latent_cached(z), first)

    def test_a_different_tensor_is_not_aliased(self):
        a = torch.randn(1, 24, 2, 16, 16)
        b = a.clone()
        self.assertIsNot(blur_latent_cached(a), blur_latent_cached(b))

    def test_cache_is_bounded(self):
        from core import _BLUR_CACHE, _BLUR_CACHE_MAX
        for _ in range(_BLUR_CACHE_MAX + 20):
            blur_latent_cached(torch.randn(1, 24, 1, 8, 8))
        self.assertLessEqual(len(_BLUR_CACHE), _BLUR_CACHE_MAX)


class RefinementTests(unittest.TestCase):
    def test_tolerance_reaches_the_same_objective_earlier(self):
        """Early exit must land on the same reconstruction error, sooner.

        Compared on the objective rather than the parameters: the loss is flat
        in directions the tiny grid cannot express, so two trajectories can
        differ element-wise while reconstructing identically.
        """
        import torch.nn.functional as F
        torch.manual_seed(7)
        full = torch.randn(1, 24, 1, 16, 16)
        small = torch.full((1, 24, 1, 4, 4), 0.1)

        def loss(p):
            up = F.interpolate(p, size=(1, 16, 16), mode="trilinear",
                               align_corners=False)
            return float(F.mse_loss(up, full))

        baseline = optimize_latent(small, full, steps=400, lr=0.05)
        log = io.StringIO()
        with patch("sys.stdout", log):
            fast = optimize_latent(small, full, steps=400, lr=0.05, tol=1e-4)
        self.assertIn("converged at step", log.getvalue())
        stopped = int(log.getvalue().split("converged at step ")[1].split("/")[0])
        self.assertLess(stopped, 400)
        self.assertGreater(loss(small), loss(baseline))       # the refine did work
        self.assertAlmostEqual(loss(fast), loss(baseline), delta=loss(baseline) * 1e-3)

    def test_zero_tolerance_runs_every_step(self):
        torch.manual_seed(7)
        full = torch.randn(1, 24, 1, 8, 8)
        small = torch.full((1, 24, 1, 4, 4), 0.1)
        log = io.StringIO()
        with patch("sys.stdout", log):
            optimize_latent(small, full, steps=40, lr=0.05, tol=0.0)
        self.assertNotIn("converged", log.getvalue())

    def test_no_steps_returns_the_input(self):
        small = torch.full((1, 24, 1, 4, 4), 0.25)
        self.assertIs(optimize_latent(small, small, steps=0), small)


class FrameSelectionTests(unittest.TestCase):
    def test_diverse_picks_are_at_least_as_spread_as_uniform(self):
        torch.manual_seed(3)
        z = torch.randn(1, 24, 12, 8, 8)
        frames = z[0].permute(1, 0, 2, 3).flatten(1)

        def spread(idx):
            sub = frames[torch.tensor(idx)]
            dist = torch.cdist(sub, sub)
            return float(dist[dist > 0].min())

        uniform = torch.linspace(0, 11, 6).round().long().tolist()
        self.assertGreaterEqual(spread(select_frames_diverse(z, 6)), spread(uniform))

    def test_edge_cases(self):
        z = torch.randn(1, 24, 5, 4, 4)
        self.assertEqual(select_frames_diverse(z, 1), [0])
        self.assertEqual(select_frames_diverse(z, 5), [0, 1, 2, 3, 4])
        self.assertEqual(select_frames_diverse(z, 99), [0, 1, 2, 3, 4])
        self.assertEqual(len(set(select_frames_diverse(z, 4))), 4)

    def test_unknown_selection_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown frame selection"):
            fit_token_budget(torch.randn(1, 24, 8, 64, 64), 1024, "x", selection="newest")


class CudaContextTests(unittest.TestCase):
    def test_fast_math_restores_the_previous_flags(self):
        before = (torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32,
                  torch.backends.cudnn.benchmark)
        with common.cuda_fast_math():
            pass
        self.assertEqual(before, (torch.backends.cuda.matmul.allow_tf32,
                                  torch.backends.cudnn.allow_tf32,
                                  torch.backends.cudnn.benchmark))

    def test_fast_math_enables_tf32_when_cuda_is_present(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.backends.cuda, "is_built", return_value=True), \
                patch.object(torch.cuda, "get_device_name", return_value="stub"):
            with common.cuda_fast_math() as engaged:
                self.assertTrue(engaged)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cudnn.benchmark)

    def test_env_vars_can_disable_each_half(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.backends.cuda, "is_built", return_value=True), \
                patch.object(torch.cuda, "get_device_name", return_value="stub"), \
                patch.dict(os.environ, {"REFMOD_TF32": "0", "REFMOD_CUDNN_BENCHMARK": "0"}):
            with common.cuda_fast_math():
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.benchmark)

    def test_vae_output_device_is_restored(self):
        class FakeVAE:
            output_device = torch.device("cpu")

        vae = FakeVAE()
        with common.vae_output_on(vae, torch.device("cuda:0")) as engaged:
            self.assertTrue(engaged)
            self.assertEqual(vae.output_device, torch.device("cuda:0"))
        self.assertEqual(vae.output_device, torch.device("cpu"))
        with common.vae_output_on(vae, torch.device("cpu")) as engaged:
            self.assertFalse(engaged)

    def test_vae_output_stays_on_the_cpu_when_vram_is_tight(self):
        # the tiled-encode fallback exists because VRAM ran out; don't make an
        # OOM more likely by moving the destination onto the GPU
        class FakeVAE:
            output_device = torch.device("cpu")

        vae = FakeVAE()
        tight = (512 * 1024 ** 2, 8 * 1024 ** 3)
        roomy = (6 * 1024 ** 3, 8 * 1024 ** 3)
        with patch.object(torch.cuda, "mem_get_info", return_value=tight):
            with common.vae_output_on(vae, torch.device("cuda:0")) as engaged:
                self.assertFalse(engaged)
                self.assertEqual(vae.output_device, torch.device("cpu"))
        with patch.object(torch.cuda, "mem_get_info", return_value=roomy):
            with common.vae_output_on(vae, torch.device("cuda:0")) as engaged:
                self.assertTrue(engaged)
                self.assertEqual(vae.output_device, torch.device("cuda:0"))


class OutputPathTests(unittest.TestCase):
    def test_relative_output_resolves_under_the_comfy_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            rooted = os.path.join(tmp, "models", "refmods")
            os.makedirs(rooted)
            self.assertEqual(common.resolve_output_dir("models/refmods", tmp), rooted)
            absolute = os.path.join(tmp, "elsewhere")
            self.assertEqual(common.resolve_output_dir(absolute, tmp), absolute)
            self.assertEqual(common.resolve_output_dir(None, tmp),
                             common.refmods_dir())

    def test_implausible_size_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            tiny = os.path.join(tmp, "tiny.safetensors")
            Path(tiny).write_bytes(b"x" * 1024)
            log = io.StringIO()
            self.assertFalse(common.warn_if_implausible_size(tiny, 1024, log=log.write))
            self.assertIn("100 KB", log.getvalue())
            big = os.path.join(tmp, "ok.safetensors")
            Path(big).write_bytes(b"x" * (200 * 1024))
            self.assertTrue(common.warn_if_implausible_size(big, 1024, log=lambda *_: None))
            self.assertFalse(common.warn_if_implausible_size(big, 0, log=log.write))


if __name__ == "__main__":
    unittest.main(verbosity=2)
