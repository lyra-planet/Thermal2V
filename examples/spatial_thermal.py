#!/usr/bin/env python
# coding: utf-8

# ==================== 配置区 ====================
import os
import sys

# ============= 根路径配置 =============
ROOT = "/home/cunjian/kai/cache/T2V"
RUN_DIR = "/home/cunjian/kai/cache/runs/20251103-133606/ckpt/16000"
PYTHON_PACKAGES = "/mnt/sda/python_packages"


# ============= 模型和项目路径 =============
PROJECT_ROOT = f"{ROOT}/OminiControl"
CONFIG_PATH = f"{PROJECT_ROOT}/spatial_alignment_thermal.yaml"
FLUX_PATH = f"{ROOT}/requirements/FLUX.1-dev"

# ============= 权重路径 =============
LORA_WEIGHT_PATH = f"{RUN_DIR}/default.safetensors"
MOGLE_WEIGHT_PATH = f"{RUN_DIR}/mogle.pt"
ADAPTER_NAME = "ir_16000"

# ============= 推理参数配置 =============
IMAGE_SIZE = 256
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 2.0
IMAGE_GUIDANCE_SCALE = 1.0
USE_MOGLE = True
RESULTS_DIR = f"{ROOT}/output/gen_result/results_{GUIDANCE_SCALE}_{IMAGE_GUIDANCE_SCALE}"
# ============= 模型配置 =============
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
TORCH_DTYPE = "bfloat16"
DEVICE_MAP = "balanced"
USE_AUTH_TOKEN = True
COMPILE_MODE = "reduce-overhead"
LORA_DEVICE = "cuda:0"

# ============= 环境配置 =============
HF_CACHE_DIR = f"{ROOT}/hf_cache"
HTTP_PROXY = "http://127.0.0.1:7890"
HTTPS_PROXY = "http://127.0.0.1:7890"
ALL_PROXY = "socks5h://127.0.0.1:7890"
ENABLE_HF_TRANSFER = "1"
SUPPRESS_COMPILE_ERRORS = True

# ============= 创建输出目录 =============
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "concat"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "result"), exist_ok=True)

# ==================== 脚本开始 ====================
import time
import json
import torch
from PIL import Image
from tqdm import tqdm

print("=" * 60)
print("初始化 - Flux OminiControl 推理脚本")
print("=" * 60)
start_total = time.time()

# ============= 环境设置 =============
sys.path.insert(0, PYTHON_PACKAGES)
sys.path.insert(0, PROJECT_ROOT)

# ============= 逐步导入 =============
print("=" * 60)
print("开始逐步导入模块")
print("=" * 60)

start = time.time()
print(f"\n[0s] 导入基础库...")
import json
print(f"[{time.time()-start:.2f}s] json ✓")

print(f"\n[{time.time()-start:.2f}s] 导入 torch...")
import torch
print(f"[{time.time()-start:.2f}s] torch ✓")

print(f"\n[{time.time()-start:.2f}s] 导入 PIL...")
from PIL import Image
print(f"[{time.time()-start:.2f}s] PIL ✓")

print(f"\n[{time.time()-start:.2f}s] 导入 tqdm...")
from tqdm import tqdm
print(f"[{time.time()-start:.2f}s] tqdm ✓")

print(f"\n[{time.time()-start:.2f}s] 导入 huggingface_hub...")
from huggingface_hub import snapshot_download
print(f"[{time.time()-start:.2f}s] huggingface_hub ✓")

print(f"\n[{time.time()-start:.2f}s] 导入 diffusers...")
start_diffusers = time.time()
from diffusers.pipelines import FluxPipeline
diffusers_time = time.time() - start_diffusers
print(f"[{time.time()-start:.2f}s] diffusers ✓ (耗时: {diffusers_time:.2f}s)")

print(f"\n[{time.time()-start:.2f}s] 导入 OminiControl...")
start_omini = time.time()
from omini.pipeline.flux_omini import Condition, generate, seed_everything
from omini.moe.mogle_t2v_unet import MoGLE
omini_time = time.time() - start_omini
print(f"[{time.time()-start:.2f}s] OminiControl ✓ (耗时: {omini_time:.2f}s)")

print(f"\n{'=' * 60}")
print(f"总导入时间: {time.time()-start:.2f}s")
print(f"  - diffusers: {diffusers_time:.2f}s")
print(f"  - OminiControl: {omini_time:.2f}s")
print(f"{'=' * 60}")

# ============= 环境优化 =============
print(f"\n[{time.time()-start_total:.2f}s] 优化 GPU 环境...")
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
print(f"[{time.time()-start_total:.2f}s] 环境优化完成 ✓")

# ============= 加载基础模型 =============
print(f"\n[{time.time()-start_total:.2f}s] 加载 Flux.1-dev 模型...")
load_start = time.time()
from omini.train_flux.trainer_t2v import get_config
config = get_config(CONFIG_PATH)
training_config = config["train"]
DATA_ROOT = training_config["dataset"]["root"]
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
TESTA_DIR = f"{DATA_ROOT}/testA"
TESTB_DIR = f"{DATA_ROOT}/testB"
PROMPT_JSON = f"{DATA_ROOT}/test_descriptions.json"

pipe = FluxPipeline.from_pretrained(
    config["flux_path"],
    dtype=torch.bfloat16,
    device_map=DEVICE_MAP,
    use_auth_token=USE_AUTH_TOKEN
)

load_time = time.time() - load_start
print(f"[{time.time()-start_total:.2f}s] Flux 模型加载完成 ✓ (耗时: {load_time:.2f}s)")

# ============= 加载 LoRA =============
print(f"\n[{time.time()-start_total:.2f}s] 加载 LoRA 权重...")

if not os.path.exists(LORA_WEIGHT_PATH):
    print(f"❌ 错误: LoRA 文件不存在: {LORA_WEIGHT_PATH}")
    sys.exit(1)

lora_start = time.time()
pipe.unload_lora_weights()
pipe.load_lora_weights(
    LORA_WEIGHT_PATH,
    adapter_name=ADAPTER_NAME,
    device=LORA_DEVICE
)
pipe.set_adapters([ADAPTER_NAME])
lora_time = time.time() - lora_start
print(f"[{time.time()-start_total:.2f}s] LoRA 加载完成 ✓ (耗时: {lora_time:.2f}s)")

# ============= 编译模型 =============
print(f"\n[{time.time()-start_total:.2f}s] 编译模型 (首次较慢)...")
compile_start = time.time()

import torch._dynamo
torch._dynamo.config.suppress_errors = SUPPRESS_COMPILE_ERRORS
torch._dynamo.config.verbose = False

pipe.transformer = torch.compile(pipe.transformer, mode=COMPILE_MODE)
pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode=COMPILE_MODE)

compile_time = time.time() - compile_start
print(f"[{time.time()-start_total:.2f}s] 模型编译完成 ✓ (耗时: {compile_time:.2f}s)")

# ============= 加载 MoGLE =============
print(f"\n[{time.time()-start_total:.2f}s] 加载 MoGLE 权重...")
mogle_start = time.time()

mogle = MoGLE()
if not os.path.exists(MOGLE_WEIGHT_PATH):
    print(f"❌ 错误: MoGLE 权重文件不存在: {MOGLE_WEIGHT_PATH}")
    sys.exit(1)

moe_weight = torch.load(MOGLE_WEIGHT_PATH, map_location="cpu")
mogle.load_state_dict(moe_weight, strict=True)
mogle = mogle.to(device=LORA_DEVICE, dtype=torch.bfloat16)
mogle.eval()

mogle_time = time.time() - mogle_start
print(f"[{time.time()-start_total:.2f}s] MoGLE 加载完成 ✓ (耗时: {mogle_time:.2f}s)")

# ============= 加载 prompt 映射 =============
print(f"\n[{time.time()-start_total:.2f}s] 加载 prompt 映射...")
with open(PROMPT_JSON, "r") as f:
    prompt_map = json.load(f)
print(f"[{time.time()-start_total:.2f}s] 加载了 {len(prompt_map)} 个 prompt ✓")

print(f"\n初始化完成！总耗时: {time.time()-start_total:.2f}s")
print(f"输出目录: {RESULTS_DIR}\n")

# ============= 批量处理逻辑 =============
files = sorted(os.listdir(TESTA_DIR))
files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print("=" * 60)
print(f"找到 {len(files)} 张图片待处理")
print("=" * 60)

success_count = 0
skip_count = 0
error_count = 0
process_start = time.time()

for idx, filename in enumerate(tqdm(files, desc="处理进度", ncols=80), 1):
    try:
        key = filename
        if key not in prompt_map:
            print(f"\n⚠️  [{idx}/{len(files)}] 跳过: JSON中未找到 prompt - {filename}")
            skip_count += 1
            continue

        prompt = prompt_map[key]
        print(f"\nPrompt: {prompt}")

        # ========== 读取 testA 图片 ==========
        src_path = os.path.join(TESTA_DIR, filename)
        if not os.path.exists(src_path):
            print(f"⚠️  [{idx}/{len(files)}] 跳过: testA 文件不存在 - {filename}")
            skip_count += 1
            continue

        image = Image.open(src_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        # ========== 读取 testB 图片 ==========
        tgt_path = os.path.join(TESTB_DIR, filename)
        if not os.path.exists(tgt_path):
            print(f"⚠️  [{idx}/{len(files)}] 跳过: testB 文件不存在 - {filename}")
            skip_count += 1
            continue

        target_image = Image.open(tgt_path).convert("RGB")
        target_image = target_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        # ========== 生成结果图像 ==========
        condition = Condition(image, ADAPTER_NAME)
        seed_everything()
        result_img = generate(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            image_guidance_scale=IMAGE_GUIDANCE_SCALE,
            mogle=mogle,
            use_mogle=USE_MOGLE
        ).images[0]

        # ========== 拼接图像 (输入 | 目标 | 生成) ==========
        concat_image = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
        concat_image.paste(image, (0, 0))
        concat_image.paste(target_image, (IMAGE_SIZE, 0))
        concat_image.paste(result_img, (IMAGE_SIZE * 2, 0))

        # ========== 保存拼接图 ==========
        concat_name = filename.rsplit(".", 1)[0] + "_concat.png"
        concat_path = os.path.join(RESULTS_DIR, "concat", concat_name)
        concat_image.save(concat_path)

        # ========== 单独保存生成图 ==========
        result_path = os.path.join(RESULTS_DIR, "result", filename)
        result_img.save(result_path)

        success_count += 1

    except Exception as e:
        print(f"\n❌ [{idx}/{len(files)}] 处理失败 - {filename}: {str(e)}")
        error_count += 1

process_time = time.time() - process_start

# ============= 统计信息 =============
print("\n" + "=" * 60)
print("处理完成！")
print("=" * 60)
print(f"✓ 成功处理: {success_count} 张")
print(f"⚠️  跳过: {skip_count} 张")
print(f"❌ 错误: {error_count} 张")
print(f"\n处理耗时: {process_time:.2f}s ({process_time/60:.2f}min)")
if success_count > 0:
    print(f"平均每张耗时: {process_time/success_count:.2f}s")
print(f"\n总耗时 (含初始化): {time.time()-start_total:.2f}s")
print(f"结果保存至: {RESULTS_DIR}")
print("=" * 60)