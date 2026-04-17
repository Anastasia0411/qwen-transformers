from PIL import Image
import os
import re
import sys
import time
import gc
import copy
import importlib.util
from collections import defaultdict
import importlib.metadata as importlib_metadata

import torch

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "16"   # try 8, 16, 32


QUANT_LAYERS = [
    "kv_cache", "k_proj", "v_proj", "key", "value", "kv_proj",
    "kv_a_proj", "kv_b_proj", "kv_a_proj_with_mqa"
]


# def _as_bool_env(value: str) -> bool:
#     return str(value).strip().lower() in {"1", "true", "yes", "on"}


# def _guard_transformers_against_broken_kernels_pkg():
#     """
#     Work around broken `kernels` installations that crash during transformers import.

#     Set ALLOW_HF_KERNELS_IMPORT=1 to opt out of this guard.
#     """
#     if _as_bool_env(os.environ.get("ALLOW_HF_KERNELS_IMPORT", "0")):
#         return

#     os.environ["USE_HUB_KERNELS"] = "0"
#     sys.modules["kernels"] = None


# def _parse_version_triplet(version_str):
#     parts = []
#     for token in version_str.split("."):
#         digits = "".join(ch for ch in token if ch.isdigit())
#         if digits == "":
#             break
#         parts.append(int(digits))
#         if len(parts) == 3:
#             break
#     while len(parts) < 3:
#         parts.append(0)
#     return tuple(parts)


# def _validate_hf_hub_for_transformers_import():
#     if sys.version_info < (3, 10):
#         return

#     hub_version = importlib_metadata.version("huggingface_hub")
#     if _parse_version_triplet(hub_version) < (0, 24, 0):
#         raise RuntimeError(
#             f"Incompatible huggingface_hub=={hub_version} detected on Python {sys.version.split()[0]}. "
#             "Please upgrade before importing transformers:\n"
#             "  pip install -U 'huggingface_hub>=0.24.0' 'transformers>=4.57.0'"
#         )


def _has_modelopt() -> bool:
    return importlib.util.find_spec("modelopt") is not None


def _make_kv_cache_only_quant_cfg(full_quant_cfg):
    """
    Keep only KV-cache related quantizers enabled and disable all other quantizers.
    """
    def disable_cfg_preserving_shape(cfg):
        if not isinstance(cfg, dict):
            return cfg

        if "enable" in cfg:
            disabled = copy.deepcopy(cfg)
            disabled["enable"] = False
            return disabled

        disabled = {}
        for key, value in cfg.items():
            if isinstance(value, dict):
                disabled[key] = disable_cfg_preserving_shape(value)
            else:
                disabled[key] = value
        return disabled

    kv_only_cfg = copy.deepcopy(full_quant_cfg)
    quant_entries = kv_only_cfg.get("quant_cfg", {})

    filtered_entries = {}
    for pattern, cfg in quant_entries.items():
        key = str(pattern).lower()
        if any(token in key for token in QUANT_LAYERS):
            filtered_entries[pattern] = cfg
        else:
            filtered_entries[pattern] = disable_cfg_preserving_shape(cfg)

    kv_only_cfg["quant_cfg"] = filtered_entries
    return kv_only_cfg


def _set_quantizer_enabled(module, enabled: bool):
    if enabled and hasattr(module, "enable"):
        module.enable()
        return
    if (not enabled) and hasattr(module, "disable"):
        module.disable()
        return
    if hasattr(module, "enabled"):
        setattr(module, "enabled", bool(enabled))


def enforce_kv_cache_only_quantizers_enabled(model):
    """
    Force-enable only K/V projection output quantizers and disable everything else.
    """
    enabled = 0
    disabled = 0
    kv_name_tokens = QUANT_LAYERS
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "quantizer" not in cls_name:
            continue

        lname = module_name.lower()
        is_kv_output_quant = (
            ".self_attn." in lname
            and ".output_quantizer" in lname
            and any(token in lname for token in kv_name_tokens)
        )
        _set_quantizer_enabled(module, is_kv_output_quant)
        if is_kv_output_quant:
            enabled += 1
        else:
            disabled += 1

    print(f"[PTQ debug] forced quantizer state: enabled={enabled}, disabled={disabled}")
    return model


def quantize_with_modelopt_nvfp4_kv_cache(model, calibration_input):
    # Imported lazily so the script still runs when modelopt is unavailable.
    import modelopt.torch.quantization as mtq

    print("Applying Model Optimizer PTQ for KV cache quantization to NVFP4...")
    quant_cfg = _make_kv_cache_only_quant_cfg(mtq.NVFP4_DEFAULT_CFG)

    # Run a short calibration loop over the same prompt.
    def calibration_loop(m):
        with torch.no_grad():
            _ = m(**calibration_input, use_cache=True)

    model = mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)
    model = enforce_kv_cache_only_quantizers_enabled(model)
    print("ModelOpt PTQ done. KV cache quantizers are now inserted.")
    return model


def quantize_with_modelopt_fp8(model, calibration_input):
    # Imported lazily so the script still runs when modelopt is unavailable.
    import modelopt.torch.quantization as mtq

    print("Applying Model Optimizer PTQ for quantization to FP8...")

    # Run a short calibration loop over the same prompt.
    def calibration_loop(m):
        with torch.no_grad():
            _ = m(**calibration_input, use_cache=True)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=calibration_loop)
    print("ModelOpt PTQ done. Quantizers are now inserted.")
    return model


# def estimate_kv_cache_size_gib(model, batch_size, total_seq_len, dtype):
#     cfg = getattr(model, "config", None)
#     if cfg is None:
#         return None

#     num_key_value_heads = getattr(cfg, "num_key_value_heads", None)
#     if num_key_value_heads is None and hasattr(cfg, "text_config"):
#         num_key_value_heads = getattr(cfg.text_config, "num_key_value_heads", None)
#     if num_key_value_heads is None:
#         return None

#     hidden_size = getattr(cfg, "hidden_size", None)
#     num_attention_heads = getattr(cfg, "num_attention_heads", None)
#     if hidden_size is None and hasattr(cfg, "text_config"):
#         hidden_size = getattr(cfg.text_config, "hidden_size", None)
#     if num_attention_heads is None and hasattr(cfg, "text_config"):
#         num_attention_heads = getattr(cfg.text_config, "num_attention_heads", None)
#     if hidden_size is None or num_attention_heads is None:
#         return None

#     head_dim = hidden_size // num_attention_heads
#     kv_layers = set()
#     for name, _ in model.named_modules():
#         lname = name.lower()
#         if ".self_attn." in lname and any(token in lname for token in QUANT_LAYERS):
#             m = re.search(r"layers\.(\d+)\.", lname)
#             if m:
#                 kv_layers.add(int(m.group(1)))
#     num_kv_layers = len(kv_layers) if kv_layers else getattr(cfg, "num_hidden_layers", 0)
#     if num_kv_layers == 0 and hasattr(cfg, "text_config"):
#         num_kv_layers = getattr(cfg.text_config, "num_hidden_layers", 0)
#     if num_kv_layers == 0:
#         return None

#     dtype_size = torch.tensor([], dtype=dtype).element_size()
#     bytes_total = (
#         int(batch_size)
#         * int(total_seq_len)
#         * int(num_kv_layers)
#         * int(num_key_value_heads)
#         * int(head_dim)
#         * 2  # K and V
#         * int(dtype_size)
#     )
#     return bytes_total / (1024 ** 3), num_kv_layers, num_key_value_heads, head_dim, dtype_size


def print_kv_quantizer_debug_info(model):
    quantizer_like = []
    kv_quantizer_like = []
    for module_name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "quantizer" in cls_name:
            quantizer_like.append((module_name, module))
            lname = module_name.lower()
            if any(token in lname for token in QUANT_LAYERS):
                kv_quantizer_like.append((module_name, module))

    def is_enabled(mod):
        if hasattr(mod, "is_enabled"):
            attr = getattr(mod, "is_enabled")
            return bool(attr() if callable(attr) else attr)
        if hasattr(mod, "enabled"):
            return bool(getattr(mod, "enabled"))
        return True

    enabled_total = sum(1 for _, m in quantizer_like if is_enabled(m))
    enabled_kv = sum(1 for _, m in kv_quantizer_like if is_enabled(m))
    print(f"[PTQ debug] total quantizer-like modules: {len(quantizer_like)} (enabled={enabled_total})")
    print(f"[PTQ debug] kv-related quantizer-like modules: {len(kv_quantizer_like)} (enabled={enabled_kv})")

    kv_output = []
    for name, mod in kv_quantizer_like:
        lname = name.lower()
        if ".self_attn." in lname and ".output_quantizer" in lname and any(
            token in lname for token in QUANT_LAYERS
        ):
            kv_output.append((name, mod))

    all_layers = set()
    enabled_layers = set()
    for name, mod in kv_output:
        m = re.search(r"layers\.(\d+)\.", name)
        if not m:
            continue
        layer_id = int(m.group(1))
        all_layers.add(layer_id)
        if is_enabled(mod):
            enabled_layers.add(layer_id)

    model_cfg = getattr(model, "config", None)
    model_layers = getattr(model_cfg, "num_hidden_layers", None)
    if model_layers is None and hasattr(model_cfg, "text_config"):
        model_layers = getattr(model_cfg.text_config, "num_hidden_layers", None)
    if model_layers is None and hasattr(model_cfg, "language_config"):
        model_layers = getattr(model_cfg.language_config, "num_hidden_layers", None)
    if model_layers is None:
        model_layers = "unknown"
    print(
        f"[PTQ debug] kv output quantizers: {len(kv_output)} "
        f"across {len(all_layers)} layers (model num_hidden_layers={model_layers})."
    )
    print(f"[PTQ debug] layers with kv output quantizers enabled: {sorted(enabled_layers)}")

    for name, mod in kv_quantizer_like[:10]:
        print(f"[PTQ debug] kv quantizer: {name} ({mod.__class__.__name__}) enabled={is_enabled(mod)}")


def prepare_inputs(img_path, processor, first_device):
    image = Image.open(img_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Describe this image in details."},
            ],
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        return_dict=True,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        processor_kwargs=dict(
            padding=False
        )
    )

    # print(inputs)
    inputs = {k: v.to(first_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs


def reset_cuda_peak_memory_all_devices():
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        with torch.cuda.device(idx):
            torch.cuda.reset_peak_memory_stats()


def report_cuda_peak_memory_all_devices():
    if not torch.cuda.is_available():
        return
    per_device = []
    for idx in range(torch.cuda.device_count()):
        peak = torch.cuda.max_memory_allocated(idx)
        per_device.append(peak)
    per_device_gib = [v / (1024 ** 3) for v in per_device]
    total_gib = sum(per_device_gib)
    print(f"[memory] peak cuda allocated per device (GiB): {per_device_gib}")
    print(f"[memory] peak cuda allocated total across devices: {total_gib:.2f} GiB")


def capture_cuda_memory_snapshot_all_devices():
    if not torch.cuda.is_available():
        return []
    out = []
    for idx in range(torch.cuda.device_count()):
        out.append(
            {
                "device": idx,
                "allocated": torch.cuda.memory_allocated(idx),
                "reserved": torch.cuda.memory_reserved(idx),
                "max_allocated": torch.cuda.max_memory_allocated(idx),
                "max_reserved": torch.cuda.max_memory_reserved(idx),
            }
        )
    return out


def print_cuda_memory_delta(before, after):
    if not before or not after:
        return
    by_dev_before = {item["device"]: item for item in before}
    for item in after:
        idx = item["device"]
        b = by_dev_before[idx]
        alloc_delta = (item["allocated"] - b["allocated"]) / (1024 ** 3)
        reserv_delta = (item["reserved"] - b["reserved"]) / (1024 ** 3)
        peak_delta = (item["max_allocated"] - b["allocated"]) / (1024 ** 3)
        print(
            f"[memory][gpu{idx}] delta allocated={alloc_delta:.2f} GiB, "
            f"delta reserved={reserv_delta:.2f} GiB, "
            f"peak-over-start={peak_delta:.2f} GiB"
        )


# def profile_kv_cache_growth(model, inputs, max_new_tokens, step_stride=64, min_new_tokens=0):
#     if not torch.cuda.is_available():
#         print("[kv-profile] CUDA is not available; skipping KV cache growth profiling.")
#         return

#     from transformers import StoppingCriteria, StoppingCriteriaList

#     class MemoryProbeCriteria(StoppingCriteria):
#         def __init__(self, stride):
#             self.step = 0
#             self.stride = max(1, stride)

#         def __call__(self, input_ids, scores, **kwargs):
#             self.step += 1
#             if self.step == 1 or self.step % self.stride == 0 or self.step == max_new_tokens:
#                 alloc_gib = torch.cuda.memory_allocated() / (1024 ** 3)
#                 reserv_gib = torch.cuda.memory_reserved() / (1024 ** 3)
#                 print(f"[kv-profile] {self.step}, {alloc_gib:.2f}, {reserv_gib:.2f}, n/a")
#             return False

#     model.eval()
#     with torch.no_grad():
#         print("[kv-profile] token_step, allocated_GiB, reserved_GiB, kv_cache_GiB")
#         probe = MemoryProbeCriteria(step_stride)
#         generation_kwargs = {}
#         if min_new_tokens > 0:
#             generation_kwargs["min_new_tokens"] = min_new_tokens

#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             remove_invalid_values=True,
#             use_cache=True,
#             stopping_criteria=StoppingCriteriaList([probe]),
#             **generation_kwargs,
#         )
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):]
#             for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
#         ]
#         generated_lengths = [ids.shape[0] for ids in generated_ids_trimmed]
#         print(f"[kv-profile] generated lengths: {generated_lengths}")


def parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ptq-kv-cache-nvfp4",
        action="store_true",
        help="Apply NVIDIA Model Optimizer PTQ with NVFP4 config (only KV cache quantization).",
    )
    parser.add_argument(
        "--ptq-fp8",
        action="store_true",
        help="Apply NVIDIA Model Optimizer PTQ with FP8 config (whole model).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["flash_attention_3", "flash_attention_2", "sdpa", "eager"],
        help="Attention backend for model loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Number of generated tokens. Use a large value (e.g. 512+) to observe KV-cache memory differences.",
    )
    # parser.add_argument(
    #     "--profile-kv-cache-growth",
    #     action="store_true",
    #     help="Run a manual token-by-token decode loop and print memory growth during KV-cache expansion.",
    # )
    # parser.add_argument(
    #     "--profile-kv-stride",
    #     type=int,
    #     default=64,
    #     help="Print interval for --profile-kv-cache-growth.",
    # )
    args = parser.parse_args()
    return args


def main():
    # safe transfromers import 
    # _guard_transformers_against_broken_kernels_pkg()
    # _validate_hf_hub_for_transformers_import()
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

    # device check
    print("pid:", os.getpid())
    print("cuda device count:", torch.cuda.device_count())
    args = parser()

    # QWEN3_PATH = "Qwen/Qwen3.5-397B-A17B"
    # QWEN3_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
    QWEN3_PATH = "Qwen/Qwen3.5-35B-A3B"
    # QWEN3_PATH = "/home/jovyan/shares/SR008.fs2/sentsov_a/checkpoints/Qwen3.5-397B-A17B-FP8"
    img_path_1 = "/home/jovyan/aigkargapoltseva/qwen-3.5-transformers/img.png"
    img_path_2 = "/home/jovyan/aigkargapoltseva/qwen-3.5-transformers/img2.jpeg"
    # max_memory = {i: "78GiB" for i in range(8)}

    print(os.system('nvidia-smi'))
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        QWEN3_PATH, 
        dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_3",
        attn_implementation=args.attn_implementation,
        # experts_implementation="grouped_mm",
        experts_implementation="eager",
        low_cpu_mem_usage=True
    )
    print(os.system('nvidia-smi'))

    processor = AutoProcessor.from_pretrained(QWEN3_PATH)
    first_device = model.get_input_embeddings().weight.device

    # prepare inputs for warm up and calibration procedures
    inputs = prepare_inputs(img_path_1, processor, first_device)

    if args.ptq_kv_cache_nvfp4:
        if not _has_modelopt():
            raise RuntimeError(
                "You requested --ptq-kv-cache-nvfp4 but modelopt is not installed. "
                "Install with: pip install -U 'nvidia-modelopt[hf]'"
            )
        model = quantize_with_modelopt_nvfp4_kv_cache(model, inputs)
        print_kv_quantizer_debug_info(model)

    if args.ptq_fp8:
        if not _has_modelopt():
            raise RuntimeError(
                "You requested --ptq-fp8 but modelopt is not installed. "
                "Install with: pip install -U 'nvidia-modelopt[hf]'"
            )
        model = quantize_with_modelopt_fp8(model, inputs)
        print_kv_quantizer_debug_info(model)

    # if args.profile_kv_cache_growth:
    #     reset_cuda_peak_memory_all_devices()
    #     profile_kv_cache_growth(
    #         model=model,
    #         inputs=inputs,
    #         max_new_tokens=args.max_new_tokens,
    #         step_stride=max(1, args.profile_kv_stride),
    #     )
    #     return

    # Generate warmup
    # estimate cache size
    # total_seq_len = int(inputs["input_ids"].shape[1]) + int(args.max_new_tokens)
    # est = estimate_kv_cache_size_gib(
    #     model=model,
    #     batch_size=int(inputs["input_ids"].shape[0]),
    #     total_seq_len=total_seq_len,
    #     dtype=torch.bfloat16,
    # )
    # if est is not None:
    #     gib, layers, kv_heads, head_dim, dtype_size = est
    #     print(
    #         f"[generate] estimated static cache size: {gib:.2f} GiB "
    #         f"(layers={layers}, kv_heads={kv_heads}, head_dim={head_dim}, "
    #         f"seq={total_seq_len}, dtype_bytes={dtype_size})"
    #     )

    print(f'start generate warmup')
    st = time.time()
    with torch.no_grad():
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            remove_invalid_values=True,
            use_cache=True, 
        )
    print(f'stop generate warmup, time: {time.time() - st}')

    # clean mem
    del generated_ids, inputs
    torch.cuda.empty_cache()
    gc.collect()

    # prepare inputs for test
    inputs = prepare_inputs(img_path_2, processor, first_device)
    reset_cuda_peak_memory_all_devices()
    mem_before = capture_cuda_memory_snapshot_all_devices()

    # Generate test
    print(f'start generate test')
    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            remove_invalid_values=True,
            use_cache=True, 
        )
    print(f'stop generate test, time: {time.time() - st}')
    mem_after = capture_cuda_memory_snapshot_all_devices()
    report_cuda_peak_memory_all_devices()
    print_cuda_memory_delta(mem_before, mem_after)

    # decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    generated_lengths = [ids.shape[0] for ids in generated_ids_trimmed]
    print(f"[generate] new token lengths: {generated_lengths}")

    out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    print(out)


if __name__ == '__main__':
    main()
