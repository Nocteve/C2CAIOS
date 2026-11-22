"""Frozen C2C model loading and a shared greedy decoding path."""

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from .project import ROOT, configure_environment

configure_environment()

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_models(config, lock):
    upstream = ROOT / "third_party/C2C"
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != config["c2c_revision"]:
        raise ValueError(f"C2C revision mismatch: {revision}")
    changed = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if changed:
        raise ValueError("C2C has modified tracked files; document and pin integration changes before running")
    sys.path.insert(0, str(upstream))
    from rosetta.model.projector import load_projector
    from rosetta.model.wrapper import RosettaModel

    paths = {}
    for role, key in (("receiver", "receiver"), ("sender", "sender"), ("fuser", "fuser_repo")):
        entry = lock[role]
        if entry["repo_id"] != config[key]:
            raise ValueError(f"Pinned {role} does not match configuration")
        paths[role] = snapshot_download(**entry, cache_dir=ROOT / ".cache/huggingface/hub", local_files_only=True)
    dtype = getattr(torch, config["dtype"])
    models = []
    for role in ("receiver", "sender"):
        print(f"Loading {role}: {paths[role]}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            paths[role], torch_dtype=dtype, device_map={"": "cuda:0"},
            attn_implementation=config["attention_backend"], local_files_only=True,
        ).eval().requires_grad_(False)
        models.append(model)

    checkpoint = Path(paths["fuser"]) / config["fuser_subdir"] / "final"
    weights = sorted(checkpoint.glob("projector_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not weights:
        raise FileNotFoundError(f"No fuser weights in {checkpoint}")
    projectors = []
    for index, weight_path in enumerate(weights):
        if weight_path.stem != f"projector_{index}":
            raise ValueError("Non-contiguous projector checkpoint indices")
        projector = load_projector(str(weight_path.with_suffix(".json")))
        projector.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
        projector = projector.to(device="cuda:0", dtype=dtype).eval().requires_grad_(False)
        projectors.append(projector)
    combined = RosettaModel(model_list=models, base_model_idx=0, projector_list=projectors).eval()
    combined.load_projector_config(str(checkpoint / "projector_config.json"))
    tokenizers = {
        role: AutoTokenizer.from_pretrained(paths[role], local_files_only=True)
        for role in ("receiver", "sender")
    }
    return combined, tokenizers

