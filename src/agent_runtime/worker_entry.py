"""Isolated Worker child process: consumes a fused KV payload from disk and decodes.

The PPT fault-isolation design makes every Agent a real OS child process. This
module is the child entry point: the runtime kernel (Planner + fuser + GPU
fusion) publishes the fused receiver cache plus command metadata to files,
then spawns this entry under cgroup/rlimit limits via ``os_isolate``. The
child loads only the frozen 0.6B receiver, materializes the published cache,
and runs the verified ``generate_worker`` decode path unchanged.

Usage:
    python -m agent_runtime.worker_entry --meta meta.json --cache cache.pt \
        --out record.json

Exit code 0 + record.json  = task artifact delivered.
Any other exit (exception, signal, OOM kill, timeout kill) = contained
failure the parent maps to a FAILED Agent Control Block.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated C2C Worker child")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mmap", action="store_true",
                        help="zero-copy host load: map the published raw cache "
                             "bin instead of torch.load deserialization")
    args = parser.parse_args()

    from .project import configure_environment
    configure_environment()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    cmd_ids = [int(t) for t in meta["cmd_ids"]]
    max_new_tokens = int(meta.get("max_new_tokens", 320))

    model = AutoModelForCausalLM.from_pretrained(
        meta["receiver_path"], torch_dtype=getattr(torch, meta.get("dtype", "bfloat16")),
        device_map={"": "cuda:0"}, attn_implementation=meta.get("attention_backend", "eager"),
        local_files_only=True,
    ).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(meta["receiver_path"], local_files_only=True)

    from time import perf_counter
    load_started = perf_counter()
    if args.mmap:
        # Host-level zero copy: the child's address space maps the same page
        # cache pages the publisher wrote -- no deserialization copy.  GPU
        # device zero-copy stays honestly unimplemented (.to(cuda) copies).
        import numpy as np
        layout = sorted(meta["cache_layout"], key=lambda e: e["index"])
        buf = np.memmap(args.cache, dtype=np.uint8, mode="r")
        published = DynamicCache()
        offset = 0
        for entry in layout:
            k_numel = 1
            for dim in entry["k_shape"]:
                k_numel *= dim
            v_numel = 1
            for dim in entry["v_shape"]:
                v_numel *= dim
            k_bytes = k_numel * 2  # bfloat16 = 2 bytes
            v_bytes = v_numel * 2
            key = torch.from_numpy(buf[offset:offset + k_bytes]).view(torch.bfloat16) \
                       .reshape(entry["k_shape"])
            offset += k_bytes
            value = torch.from_numpy(buf[offset:offset + v_bytes]).view(torch.bfloat16) \
                         .reshape(entry["v_shape"])
            offset += v_bytes
            published.update(key.to(device="cuda:0", dtype=model.dtype),
                             value.to(device="cuda:0", dtype=model.dtype),
                             int(entry["index"]))
        cache_load_mode = "mmap_zero_copy_host"
    else:
        raw = torch.load(args.cache, map_location="cpu", weights_only=True)
        published = DynamicCache()
        for index in sorted(int(i) for i in raw.keys()):
            key, value = raw[str(index)] if str(index) in raw else raw[index]
            published.update(key.to(device="cuda:0", dtype=model.dtype),
                             value.to(device="cuda:0", dtype=model.dtype), int(index))
        del raw
        cache_load_mode = "torch_deserialize"
    cache_load_seconds = perf_counter() - load_started

    from .kv import PlanningPayload, generate_worker
    payload = PlanningPayload(
        tuple(cmd_ids[:-1]), published, fused=True,
        hidden_context_tokens=int(meta["hidden_context_tokens"]),
    )
    # generate_worker validates base_model_idx/model_list/eval/cuda and, for
    # the c2c_hidden decode path, only ever touches the receiver; the source
    # slot is satisfied by the same frozen receiver instance.
    shim = types.SimpleNamespace(
        model_list=[model, model], base_model_idx=0, training=False,
        multi_source_fusion_mode=meta.get("fusion_mode", "sequential"),
    )
    record = generate_worker(shim, {"receiver": tokenizer}, cmd_ids, payload,
                             "c2c_hidden", max_new_tokens)
    record.update({
        "worker_pid": os.getpid(),
        "worker_process": True,
        "worker_isolated_execution": True,
        "worker_role_boundary": "native_worker_system_command_template_only",
        "worker_consumed_fused_cache": True,
        "arm_wall_seconds": record.get("service_seconds"),
        "cache_load_mode": cache_load_mode,
        "cache_load_seconds": cache_load_seconds,
    })
    Path(args.out).write_text(
        json.dumps(_json_safe(record), ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
