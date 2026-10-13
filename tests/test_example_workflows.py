"""The shipped example workflows must stay loadable and queue-able.

Every JSON in ``examples/workflows/`` is checked against the live node
schemas: node types, widget order, link slots and types. Each graph is then
converted to the API prompt format the frontend sends and pushed through
ComfyUI's own ``execution.validate_prompt`` — the same validator the queue
uses — with placeholder model files, so a renamed node input or a stale
widget default fails here instead of on a user's machine.

Run with ComfyUI's Python:  python tests/test_example_workflows.py
No models, no GPU, no server.
"""

import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))

from comfy.cli_args import args

args.cpu = True

import folder_paths

# Placeholder model names used by the example graphs. The validator only
# checks that a combo value is in the file list, so fake names are enough.
FAKE_FILES = [
    "minimax_h3_video_dit.safetensors",
    "minimax_h3_qwen3_vl.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae.safetensors",
    "example.png",
    "voice_clip.wav",
]

with tempfile.TemporaryDirectory() as import_dir:
    with patch.object(folder_paths, "models_dir", import_dir), \
            patch.object(folder_paths, "get_filename_list", lambda kind: list(FAKE_FILES)):
        N = importlib.import_module(f"custom_nodes.{ROOT.name}.nodes")

        import execution
        import nodes as comfy_nodes

        # comfy_extras (the native H3 nodes, VAEDecodeAudio, CreateVideo, ...)
        # are loaded by the server at startup; do the same here.
        asyncio.run(comfy_nodes.init_extra_nodes(init_custom_nodes=False,
                                                 init_api_nodes=False))
        comfy_nodes.NODE_CLASS_MAPPINGS.update(N.NODE_CLASS_MAPPINGS)

        # Input schemas are built with the fake file list in place, so the
        # loader widgets offer exactly the names the examples reference.
        SCHEMAS = {}
        OUTPUTS = {}
        for name, cls in comfy_nodes.NODE_CLASS_MAPPINGS.items():
            try:
                SCHEMAS[name] = cls.INPUT_TYPES()
            except Exception:
                continue
            OUTPUTS[name] = list(getattr(cls, "RETURN_TYPES", ()) or ())

WORKFLOWS = sorted((ROOT / "examples" / "workflows").glob("*.json"))

# Created by the frontend, never sent to the server, so they have no schema.
FRONTEND_ONLY = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}
# Resolved at runtime rather than by a fixed type name.
FLEXIBLE_TYPES = {"COMFY_MATCHTYPE_V3", "COMFY_MATCHTYPE", "*"}
# Types the frontend renders as a widget; anything else is a connection socket.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}


def raw_input(spec, name):
    for section in ("required", "optional"):
        if name in spec.get(section, {}):
            return spec[section][name]
    return ("*",)


def input_type(spec, name):
    value = raw_input(spec, name)
    return value[0] if isinstance(value, (list, tuple)) else value


def input_opts(spec, name):
    value = raw_input(spec, name)
    if isinstance(value, (list, tuple)) and len(value) > 1:
        return value[1] or {}
    return {}


def ordered_names(spec):
    return list(spec.get("required", {})) + list(spec.get("optional", {}))


def is_widget(spec, name):
    t = input_type(spec, name)
    if isinstance(t, list):
        return True
    return t in WIDGET_TYPES


def is_autogrow(spec, name):
    t = input_type(spec, name)
    return isinstance(t, str) and t.startswith("COMFY_AUTOGROW_V3")


def widget_names(spec, linked):
    """Widget values the frontend serializes, in order, for this node."""
    names = []
    for name in ordered_names(spec):
        opts = input_opts(spec, name)
        if opts.get("hidden") or opts.get("forceInput"):
            continue
        if is_autogrow(spec, name):
            continue
        if is_widget(spec, name):
            if name in linked:
                continue          # converted to an input socket
            names.append(name)
            if "control_after_generate" in opts:
                names.append(f"{name} (control after generate)")
    return names


def socket_names(spec):
    names = []
    for name in ordered_names(spec):
        opts = input_opts(spec, name)
        if opts.get("hidden") or is_autogrow(spec, name):
            continue
        if not is_widget(spec, name):
            names.append(name)
    return names


def to_api_prompt(workflow):
    """Mirror of the frontend's graph -> prompt conversion."""
    links = {link[0]: link for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        class_type = node["type"]
        if class_type in FRONTEND_ONLY:
            continue
        spec = SCHEMAS[class_type]
        linked = {i["name"]: i.get("link") for i in node.get("inputs", [])}
        values = iter(node.get("widgets_values") or [])
        inputs = {}
        for name in ordered_names(spec):
            opts = input_opts(spec, name)
            if opts.get("hidden") or is_autogrow(spec, name):
                continue
            if is_widget(spec, name):
                if name in linked:
                    continue
                inputs[name] = next(values)
                if "control_after_generate" in opts:
                    next(values)
            elif linked.get(name) is not None:
                link = links[linked[name]]
                inputs[name] = [str(link[1]), link[2]]
        prompt[str(node["id"])] = {"class_type": class_type, "inputs": inputs}
    return prompt


def write_tone(path):
    """A real audio file — LoadAudio's VALIDATE_INPUTS looks for it on disk."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


class ExampleWorkflowTests(unittest.TestCase):
    def test_example_workflows_shipped(self):
        self.assertTrue(WORKFLOWS, "examples/workflows/ holds no workflow JSON")

    def test_node_types_exist(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                for node in workflow["nodes"]:
                    if node["type"] in FRONTEND_ONLY:
                        continue
                    self.assertTrue(
                        node["type"] in SCHEMAS,
                        f"{path.name}: unknown node {node['type']}")

    def test_widget_values_line_up_with_the_schema(self):
        """A wrong widgets_values length silently shifts every later widget."""
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                for node in workflow["nodes"]:
                    if node["type"] in FRONTEND_ONLY:
                        continue
                    spec = SCHEMAS[node["type"]]
                    linked = {i["name"] for i in node.get("inputs", [])}
                    expected = widget_names(spec, linked)
                    got = node.get("widgets_values") or []
                    self.assertEqual(
                        len(got), len(expected),
                        f"{path.name} node {node['id']} ({node['type']}): "
                        f"{len(got)} widget values for {expected}")
                    for name, value in zip(expected, got):
                        self.assertIsNotNone(
                            value, f"{path.name} node {node['id']}: no value for {name}")

    def test_sockets_are_real_inputs(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                for node in workflow["nodes"]:
                    if node["type"] in FRONTEND_ONLY:
                        continue
                    spec = SCHEMAS[node["type"]]
                    allowed = set(socket_names(spec))
                    for entry in node.get("inputs", []):
                        self.assertTrue(
                            entry["name"] in allowed,
                            f"{path.name} node {node['id']}: {entry['name']} is not a "
                            f"connection input of {node['type']}")

    def test_links_resolve_both_ends(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                by_id = {n["id"]: n for n in workflow["nodes"]}
                used = set()
                for link in workflow["links"]:
                    link_id, src_id, src_slot, dst_id, dst_slot, ltype = link
                    src, dst = by_id[src_id], by_id[dst_id]
                    dst_spec = SCHEMAS[dst["type"]]
                    src_outputs = OUTPUTS[src["type"]]
                    self.assertLess(src_slot, len(src_outputs),
                                    f"{path.name}: link {link_id} output slot")
                    dst_name = ordered_names(dst_spec)[dst_slot]
                    dst_type = input_type(dst_spec, dst_name)
                    src_type = src_outputs[src_slot]
                    if src_type not in FLEXIBLE_TYPES and dst_type not in FLEXIBLE_TYPES:
                        self.assertEqual(
                            src_type,
                            dst_type if isinstance(dst_type, str) else "COMBO",
                            f"{path.name}: link {link_id} {src['type']}->{dst['type']}"
                            f".{dst_name} type mismatch")
                    self.assertEqual(ltype, src_type,
                                     f"{path.name}: link {link_id} label")
                    # both ends must list the link
                    out_links = src["outputs"][src_slot].get("links") or []
                    self.assertIn(link_id, out_links,
                                  f"{path.name}: link {link_id} missing from "
                                  f"{src['type']} output {src_slot}")
                    entry = [i for i in dst.get("inputs", []) if i["name"] == dst_name]
                    self.assertTrue(entry and entry[0].get("link") == link_id,
                                    f"{path.name}: link {link_id} missing from "
                                    f"{dst['type']}.{dst_name}")
                    used.add(link_id)
                self.assertEqual(used, {link[0] for link in workflow["links"]})

    def test_graph_passes_comfyui_prompt_validation(self):
        """The real queue validator, with placeholder model files."""
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                prompt = to_api_prompt(workflow)
                self.assertTrue(prompt, f"{path.name} converts to no executable node")
                with tempfile.TemporaryDirectory() as tmp:
                    # the folder widget is a combo of input subdirectories, so
                    # every folder a graph points at has to exist to validate
                    for node in prompt.values():
                        folder = node["inputs"].get("folder")
                        if isinstance(folder, str) and folder:
                            (Path(tmp) / folder).mkdir(parents=True, exist_ok=True)
                        audio = node["inputs"].get("audio")
                        if isinstance(audio, str) and audio:
                            write_tone(Path(tmp) / audio)
                    with patch.object(folder_paths, "get_input_directory", lambda: tmp), \
                            patch.object(folder_paths, "get_filename_list",
                                         lambda kind: list(FAKE_FILES)):
                        valid, error, _, _ = asyncio.run(
                            execution.validate_prompt("test", prompt, None))
                self.assertTrue(valid, f"{path.name}: {json.dumps(error)[:800]}")

    def test_graph_reaches_an_output_node(self):
        """Every graph ends in an OUTPUT_NODE, so Queue actually runs it."""
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                outputs = [n["type"] for n in workflow["nodes"]
                           if n["type"] not in FRONTEND_ONLY
                           and getattr(comfy_nodes.NODE_CLASS_MAPPINGS[n["type"]],
                                       "OUTPUT_NODE", False)]
                self.assertTrue(outputs, f"{path.name} has no output node")

def load(name):
    return json.loads((ROOT / "examples" / "workflows" / name).read_text())


def widgets_of(workflow, node):
    """Map a node's serialized widget values back to their schema names."""
    spec = SCHEMAS[node["type"]]
    linked = {i["name"] for i in node.get("inputs", [])}
    return dict(zip(widget_names(spec, linked), node.get("widgets_values") or []))


def feed(workflow, node, input_name):
    """(source node, slot) feeding `input_name`, or None when it is unlinked."""
    for link in workflow["links"]:
        if link[3] == node["id"]:
            spec = SCHEMAS[node["type"]]
            if ordered_names(spec)[link[4]] == input_name:
                return next(n for n in workflow["nodes"] if n["id"] == link[1]), link[2]
    return None


class LayoutTests(unittest.TestCase):
    """The graphs are the pack's front door, so the polish is part of the work."""

    def test_every_graph_is_grouped_and_has_instructions(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                self.assertTrue(workflow.get("groups"), f"{path.name}: no groups")
                notes = [n for n in workflow["nodes"] if n["type"] == "Note"]
                self.assertEqual(len(notes), 1, f"{path.name}: wants exactly one Note")
                self.assertGreater(len(notes[0]["widgets_values"][0]), 200,
                                   f"{path.name}: the Note is too terse to be useful")

    def test_every_node_sits_inside_a_group(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                boxes = [g["bounding"] for g in workflow["groups"]]
                for node in workflow["nodes"]:
                    if node["type"] == "Note":
                        continue
                    x, y = node["pos"]
                    w, h = node["size"]
                    inside = any(bx <= x and by <= y and x + w <= bx + bw
                                 and y + h <= by + bh for bx, by, bw, bh in boxes)
                    self.assertTrue(
                        inside, f"{path.name}: node {node['id']} ({node['type']}) "
                        f"at {node['pos']} {node['size']} escapes every group box")

    def test_groups_do_not_overlap(self):
        for path in WORKFLOWS:
            with self.subTest(workflow=path.name):
                boxes = json.loads(path.read_text())["groups"]
                for i, a in enumerate(boxes):
                    for b in boxes[i + 1:]:
                        overlap_x = a["bounding"][0] < b["bounding"][0] + b["bounding"][2] and \
                            b["bounding"][0] < a["bounding"][0] + a["bounding"][2]
                        overlap_y = a["bounding"][1] < b["bounding"][1] + b["bounding"][3] and \
                            b["bounding"][1] < a["bounding"][1] + a["bounding"][3]
                        self.assertFalse(
                            overlap_x and overlap_y,
                            f"{path.name}: groups {a['title']!r} and {b['title']!r} overlap")


class CharacterMasterWorkflowTests(unittest.TestCase):
    """refmod_character_master.json: stills + a motion clip + a voice clip,
    one graph, one RefMod bundle. This is the workflow the pack advertises,
    so its wiring is asserted rather than eyeballed."""

    NAME = "refmod_character_master.json"

    def setUp(self):
        self.workflow = load(self.NAME)
        self.by_type = {}
        for node in self.workflow["nodes"]:
            self.by_type.setdefault(node["type"], []).append(node)

    def test_master_reads_a_folder_that_can_hold_images_and_video(self):
        master = self.by_type["MiniMaxH3RefModMasterExtract"][0]
        source = feed(self.workflow, master, "refs_bundle")
        self.assertIsNotNone(source, "nothing feeds refs_bundle")
        folder_node, slot = source
        self.assertEqual(folder_node["type"], "MiniMaxH3RefModFolderLoader")
        self.assertEqual(slot, 0, "must take the reference list, not the count")

    def test_master_takes_audio_and_both_vaes(self):
        master = self.by_type["MiniMaxH3RefModMasterExtract"][0]
        for name in ("audio", "audio_vae", "vae"):
            self.assertIsNotNone(feed(self.workflow, master, name),
                                 f"nothing feeds {name}")

    def test_master_saves_one_bundle_at_the_guide_budget(self):
        values = widgets_of(self.workflow,
                            self.by_type["MiniMaxH3RefModMasterExtract"][0])
        self.assertEqual(values["concept_type"], "identity")
        self.assertEqual(values["ref_resolution"], 1024)
        self.assertEqual(values["max_tokens"], 8192,
                         "the creation guide budgets 8192 tokens for a persona")
        self.assertTrue(values["save"], "the example would not save anything")
        self.assertEqual(values["save_layout"], "bundle",
                         "images, motion and voice belong in one file")
        self.assertEqual(values["audio_concept_type"], "voice")

    def test_visual_and_audio_previews_are_both_wired(self):
        inspect = self.by_type["MiniMaxH3RefModInspect"][0]
        image_out = feed(self.workflow, self.by_type["PreviewImage"][0], "images")
        audio_out = feed(self.workflow, self.by_type["PreviewAudio"][0], "audio")
        self.assertEqual(image_out[0]["id"], inspect["id"])
        self.assertEqual(audio_out[0]["id"], inspect["id"])
        self.assertEqual(
            widgets_of(self.workflow, inspect)["preview"], "stored",
            "the example should show what was stored, not a re-encode")
        master = self.by_type["MiniMaxH3RefModMasterExtract"][0]
        self.assertEqual(feed(self.workflow, inspect, "mods")[0]["id"], master["id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
