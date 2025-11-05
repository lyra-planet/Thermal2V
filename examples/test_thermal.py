#!/usr/bin/env python
# coding: utf-8
"""
对比实验：找到最优的生成参数组合（DEBUG VERSION）
"""

import time
import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import pyiqa
from torchvision import transforms
import tempfile
import shutil

# ============================================================
# =============== 路径与环境配置 ==============================
# ============================================================

ROOT = "/home/cunjian/kai/cache/T2V"
RUN_DIR = "/home/cunjian/kai/cache/runs/20251103-133606/ckpt/16000"
RESULTS_DIR = f"{ROOT}/output/test_thermal/results_mogle_gate_1.0"
PYTHON_PACKAGES = "/mnt/sda/python_packages"
ADAPTER_NAME = "ir_16000"

PROJECT_ROOT = f"{ROOT}/OminiControl"
CONFIG_PATH = f"{PROJECT_ROOT}/spatial_alignment_thermal.yaml"
FLUX_PATH = f"{ROOT}/requirements/FLUX.1-dev"
LORA_WEIGHT_PATH = f"{RUN_DIR}/default.safetensors"
MOGLE_WEIGHT_PATH = f"{RUN_DIR}/mogle.pt"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# =============== 模型与依赖初始化 ============================
# ============================================================

os.chdir("..")
sys.path.insert(0, PYTHON_PACKAGES)
sys.path.insert(0, PROJECT_ROOT)

from omini.pipeline.flux_omini import Condition, generate, seed_everything
from omini.train_flux.trainer_t2v import get_config
from diffusers.pipelines import FluxPipeline

config = get_config(CONFIG_PATH)
training_config = config["train"]
DATA_ROOT = training_config["dataset"]["root"]
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
TESTA_DIR = f"{DATA_ROOT}/testA"
TESTB_DIR = f"{DATA_ROOT}/testB"
PROMPT_JSON = f"{DATA_ROOT}/test_descriptions.json"

print(f"TESTA_DIR: {TESTA_DIR}")
print(f"TESTB_DIR: {TESTB_DIR}")
print(f"PROMPT_JSON: {PROMPT_JSON}")
print("=" * 60)
print("初始化模型")
print("=" * 60)

pipe = FluxPipeline.from_pretrained(
    config["flux_path"],
    dtype=torch.bfloat16,
    device_map="balanced",
    use_auth_token=True
)

pipe.load_lora_weights(LORA_WEIGHT_PATH, adapter_name=ADAPTER_NAME, device="cuda:0")
pipe.set_adapters([ADAPTER_NAME])

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="reduce-overhead")

from omini.moe.mogle_t2v_unet import MoGLE
print("✓ 模型加载完成")

mogle = MoGLE()
moe_weight = torch.load(MOGLE_WEIGHT_PATH, map_location="cpu")
mogle.load_state_dict(moe_weight, strict=True)
mogle = mogle.to(device="cuda:0", dtype=torch.bfloat16)
mogle.eval()

# ============================================================
# =============== 初始化 PyIQA 指标 ==========================
# ============================================================

print("\n初始化 PyIQA 指标...")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
metric_psnr = pyiqa.create_metric('psnr', device=device)
metric_ssim = pyiqa.create_metric('ssim', device=device)
metric_lpips = pyiqa.create_metric('lpips', device=device)
metric_fid = pyiqa.create_metric('fid', device=device)
print("✓ PyIQA 模型加载完成")

# ============================================================
# =============== 读取数据与配置 =============================
# ============================================================

# 读取 prompt JSON
with open(PROMPT_JSON, "r") as f:
    prompt_map = json.load(f)

print(f"✓ Prompt JSON 已加载，包含 {len(prompt_map)} 个样本")

# 参数搜索表
param_configs = [
    {"steps": 28, "guidance": 0.5, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_0.5"},
    {"steps": 28, "guidance": 0.7, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_0.7"},
    {"steps": 28, "guidance": 1.0, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_1.0"},
    {"steps": 28, "guidance": 1.2, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_1.2"},
    {"steps": 28, "guidance": 1.5, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_1.5_baseline"},
    {"steps": 28, "guidance": 1.8, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_1.8"},
    {"steps": 28, "guidance": 2.0, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_2.0"},
    {"steps": 28, "guidance": 2.5, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_2.5"},
    {"steps": 28, "guidance": 3.0, "image_guidance": 1.0, "lora_weight": 1.0, "name": "guidance_3.0"},
    {"steps": 28, "guidance": 3.5, "image_guidance": 3.5, "lora_weight": 1.0, "name": "guidance_3.5"},
]

# 选择验证集文件
files = sorted(os.listdir(TESTA_DIR))
files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"✓ TESTA_DIR 中找到 {len(files)} 张图片")

num_val = 2
if len(files) > num_val:
    indices = np.linspace(0, len(files) - 1, num_val, dtype=int)
    val_files = [files[i] for i in indices]
else:
    val_files = files

print(f"✓ 使用 {len(val_files)} 张图片进行验证")
print("=" * 60)

# 图像预处理
to_tensor = transforms.ToTensor()
def preprocess_image(img_path, target_size=256):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return to_tensor(img).unsqueeze(0)

# ============================================================
# =============== 实验主循环 ================================
# ============================================================

all_results = []

for config_item in param_configs:
    config_name = config_item["name"]
    print(f"\n[{len(all_results)+1}/{len(param_configs)}] 测试配置: {config_name}")
    print(f"  参数: steps={config_item['steps']}, guidance={config_item['guidance']}, "
          f"image_guidance={config_item['image_guidance']}, lora_weight={config_item['lora_weight']}")

    config_output_dir = os.path.join(RESULTS_DIR, config_name)
    os.makedirs(config_output_dir, exist_ok=True)

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    config_results, success_count = [], 0
    skip_count = 0

    for filename in tqdm(val_files, desc=f"  处理中", ncols=60):
        try:
            # 🔧 prompt_map 的 key 包含扩展名，所以直接用 filename 作为 key
            key = filename
            if key not in prompt_map:
                skip_count += 1
                continue
            prompt = prompt_map[key]
            if not isinstance(prompt, str):
                skip_count += 1
                continue

            src_path = os.path.join(TESTA_DIR, filename)
            tgt_path = os.path.join(TESTB_DIR, filename)
            if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
                skip_count += 1
                continue

            image = Image.open(src_path).convert("RGB").resize((256, 256))
            target_image = Image.open(tgt_path).convert("RGB").resize((256, 256))

            condition = Condition(image, ADAPTER_NAME)
            seed_everything()
            result_img = generate(
                pipe,
                prompt=prompt,
                conditions=[condition],
                height=256,
                width=256,
                num_inference_steps=config_item["steps"],
                guidance_scale=config_item["guidance"],
                image_guidance_scale=config_item["image_guidance"],
                mogle=mogle,
                use_mogle=True
            ).images[0]

            result_tensor = to_tensor(result_img).unsqueeze(0).to(device)
            target_tensor = to_tensor(target_image).unsqueeze(0).to(device)

            with torch.no_grad():
                psnr = metric_psnr(result_tensor, target_tensor).item()
                ssim = metric_ssim(result_tensor, target_tensor).item()
                lpips = metric_lpips(result_tensor, target_tensor).item()

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            lpips_scores.append(lpips)
            success_count += 1

            config_results.append({"filename": filename, "PSNR": psnr, "SSIM": ssim, "LPIPS": lpips})

            compare = Image.new('RGB', (512, 256))
            compare.paste(target_image, (0, 0))
            compare.paste(result_img, (256, 0))
            compare.save(os.path.join(config_output_dir,
                        f"{os.path.splitext(filename)[0]}_P{psnr:.2f}_S{ssim:.4f}_L{lpips:.4f}.png"))

        except Exception as e:
            print(f"    ❌ 错误: {filename}: {e}")
            continue

    print(f"    跳过: {skip_count}, 成功: {success_count}/{len(val_files)}")

    # 计算统计指标
    if psnr_scores:
        avg_psnr, avg_ssim, avg_lpips = np.mean(psnr_scores), np.mean(ssim_scores), np.mean(lpips_scores)
        std_psnr, std_ssim, std_lpips = np.std(psnr_scores), np.std(ssim_scores), np.std(lpips_scores)

        print("  计算 FID...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_gen, tmp_ref = os.path.join(tmpdir, "gen"), os.path.join(tmpdir, "ref")
            os.makedirs(tmp_gen, exist_ok=True)
            os.makedirs(tmp_ref, exist_ok=True)

            for res in config_results:
                gen = Image.open(os.path.join(TESTA_DIR, res["filename"])).convert('RGB').resize((256, 256))
                ref = Image.open(os.path.join(TESTB_DIR, res["filename"])).convert('RGB').resize((256, 256))
                gen.save(os.path.join(tmp_gen, res["filename"]))
                ref.save(os.path.join(tmp_ref, res["filename"]))

            fid_score = metric_fid(tmp_gen, tmp_ref).item()

        result_entry = {
            "config": config_name,
            "params": config_item,
            "psnr": {"mean": float(avg_psnr), "std": float(std_psnr)},
            "ssim": {"mean": float(avg_ssim), "std": float(std_ssim)},
            "lpips": {"mean": float(avg_lpips), "std": float(std_lpips)},
            "fid": float(fid_score),
            "success_count": success_count
        }
        all_results.append(result_entry)

        df = pd.DataFrame(config_results)
        df.loc[len(df)] = {"filename": "Average", "PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}
        df.to_csv(os.path.join(config_output_dir, "metrics.csv"), index=False)

        print(f"  ✓ PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | FID: {fid_score:.4f}")
    else:
        print("  ❌ 无有效结果")

# ============================================================
# =============== 结果总结与推荐 =============================
# ============================================================

print("\n" + "=" * 60)
print("实验结果总结")
print("=" * 60)

# ⚠️ 添加空检查
if not all_results:
    print("❌ 错误：没有成功处理任何配置！")
    print("可能原因：")
    print("  1. prompt_map 为空或 JSON 文件不存在")
    print("  2. testA/testB 目录不存在或为空")
    print("  3. 文件名不匹配 (key not in prompt_map)")
    print("  4. 图像文件损坏")
    sys.exit(1)

results_by_psnr = sorted(all_results, key=lambda x: x["psnr"]["mean"], reverse=True)
results_by_ssim = sorted(all_results, key=lambda x: x["ssim"]["mean"], reverse=True)
results_by_lpips = sorted(all_results, key=lambda x: x["lpips"]["mean"])
results_by_fid = sorted(all_results, key=lambda x: x["fid"])

output_file = os.path.join(RESULTS_DIR, "tune_results.json")
with open(output_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✓ 结果已保存到: {output_file}")

def print_top(title, arr, key, reverse=True):
    print(f"\n【按 {title} 排序】")
    for i, r in enumerate(arr, 1):
        val = r[key]['mean'] if isinstance(r[key], dict) else r[key]
        print(f"{i}. {r['config']:25s} {key}: {val:.4f}")

print_top("PSNR（越高越好）", results_by_psnr, "psnr")
print_top("SSIM（越高越好）", results_by_ssim, "ssim")
print_top("LPIPS（越低越好）", results_by_lpips, "lpips", reverse=False)
print_top("FID（越低越好）", results_by_fid, "fid", reverse=False)

# 推荐配置
best_lpips, best_psnr, best_ssim, best_fid = results_by_lpips[0], results_by_psnr[0], results_by_ssim[0], results_by_fid[0]

print(f"\n{'='*60}\n推荐配置\n{'='*60}")
print(f"\n最佳 LPIPS: {best_lpips['config']}  LPIPS={best_lpips['lpips']['mean']:.4f}")
print(f"最佳 PSNR:  {best_psnr['config']}  PSNR={best_psnr['psnr']['mean']:.4f}")
print(f"最佳 SSIM:  {best_ssim['config']}  SSIM={best_ssim['ssim']['mean']:.4f}")
print(f"最佳 FID:   {best_fid['config']}   FID={best_fid['fid']:.4f}")

print(f"\n【综合推荐】建议使用 {best_lpips['config']} 的参数：")
print(f"  num_inference_steps={best_lpips['params']['steps']},")
print(f"  guidance_scale={best_lpips['params']['guidance']},")
print(f"  image_guidance_scale={best_lpips['params']['image_guidance']}")
if best_lpips['params']['lora_weight'] != 1.0:
    print(f"  adapter_weights=[{best_lpips['params']['lora_weight']}]")
print(f"{'='*60}")