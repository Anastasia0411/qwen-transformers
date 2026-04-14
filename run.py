# this script works with FP8 chckpt with transformers == 5.3.0 (with dummy output)
# doesn't work with transformers == 5.5.0 and 5.6.0
from PIL import Image
# import requests
# from io import BytesIO
import os
import time

import torch
# from accelerate import dispatch_model
# from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "16"   # try 8, 16, 32


# def remap_state_dict_keys(state_dict):
#     remapped = {}
#     for k, v in state_dict.items():
#         new_k = k

#         # Main fix: checkpoint has model.layers.*, loader expects model.language_model.layers.*
#         if new_k.startswith("model.layers."):
#             new_k = "model.language_model.layers." + new_k[len("model.layers."):]

#         # Keep everything else unchanged
#         remapped[new_k] = v
#     return remapped


# def load_all_safetensors_weights(model_dir):
#     state_dict = {}
#     for fn in sorted(os.listdir(model_dir)):
#         if fn.endswith(".safetensors"):
#             path = os.path.join(model_dir, fn)
#             shard = load_file(path)
#             state_dict.update(shard)
#     return state_dict


def main():
    # device check
    print("pid:", os.getpid())
    print("cuda device count:", torch.cuda.device_count())

    # QWEN3_PATH = "Qwen/Qwen3.5-397B-A17B"
    # QWEN3_PATH = "Qwen/Qwen3.5-397B-A17B-FP8"
    QWEN3_PATH = "Qwen/Qwen3.5-35B-A3B"
    # QWEN3_PATH = "/home/jovyan/shares/SR008.fs2/sentsov_a/checkpoints/Qwen3.5-397B-A17B-FP8"
    IMAGE_PATH = "/home/jovyan/aigkargapoltseva/qwen-3.5-transformers/img.png"
    max_memory = {i: "78GiB" for i in range(8)}

    # config = AutoConfig.from_pretrained(QWEN3_PATH)
    # # Build empty model first
    # model = Qwen3_5MoeForConditionalGeneration._from_config(config)

    # # Load checkpoint manually and rename keys
    # print(f"Loading weights from: {QWEN3_PATH}")
    # state_dict = load_all_safetensors_weights(QWEN3_PATH)
    # state_dict = remap_state_dict_keys(state_dict)

    # missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", len(missing))
    # print("Unexpected keys:", len(unexpected))

    # if missing:
    #     print("First missing keys:", missing[:20])
    # if unexpected:
    #     print("First unexpected keys:", unexpected[:20])

    # # Dispatch model across GPUs after weights are loaded
    # model = model.to(dtype=torch.float16)
    # model.eval()

    # # Optional: move whole model with HF dispatch if you have accelerate available
    # device_map = "balanced_low_0"
    # model = dispatch_model(
    #     model,
    #     device_map=device_map,
    #     offload_dir=None,
    # )
    # key_mapping = {
    #     r"^model\.layers\.": "model.language_model.layers.",
    #     r"^model\.embed_tokens\.": "model.language_model.embed_tokens.",
    #     r"^model\.norm\.": "model.language_model.norm.",
    # }
    print(os.system('nvidia-smi'))

    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        QWEN3_PATH, 
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_3",
        # attn_implementation="eager",
        # experts_implementation="grouped_mm",
        experts_implementation="eager",
        low_cpu_mem_usage=True
    )
    # print(type(model))
    print(os.system('nvidia-smi'))

    processor = AutoProcessor.from_pretrained(QWEN3_PATH)

    image = Image.open(IMAGE_PATH).convert("RGB")
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
    # inputs = inputs.to(model.device)
    first_device = model.get_input_embeddings().weight.device
    print(inputs)
    inputs = {k: v.to(first_device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # Generate
    print(f'start generate')
    st = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=32,
            do_sample=True,
            remove_invalid_values=True,
            use_cache=False, 
        )
    print(f'stop generate, time: {time.time() - st}')
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    print(out)


if __name__ == '__main__':
    main()
