import sys
sys.path.append("../../") 
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont

from datasets import load_dataset
from omini.train_flux.trainer_t2v import OminiModel, get_config, train
from omini.pipeline.flux_omini import Condition, convert_to_condition, generate

import json

# ==================== 加载配置和初始化 ====================
ROOT = "/home/cunjian/kai/cache/T2V"
PROJECT_ROOT = f"{ROOT}/OminiControl"
CONFIG_PATH = f"{PROJECT_ROOT}/spatial_alignment_thermal.yaml"
config = get_config(CONFIG_PATH)
training_config = config["train"]
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
description_file = os.path.join(training_config["dataset"]["root"], "train_descriptions.json")

# ==================== 数据集类 ====================

class ThermalToVisibleDataset(Dataset):
    def __init__(self, root_dir, description_file,
                 condition_size=(512, 512), target_size=(512, 512),
                 drop_text_prob=0.05, drop_image_prob=0.0):  # ← 改: 0.1 → 0.05
        self.trainA_dir = os.path.join(root_dir, "trainA")
        self.trainB_dir = os.path.join(root_dir, "trainB")
        self.condition_size = condition_size
        self.target_size = target_size
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob

        # 读取 JSON 文本描述
        with open(description_file, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)
        
        # ========== 新增：数据集质量统计 ==========
        self._print_dataset_stats()
        # =========================================
        
        # 创建热成像-可见光图像对
        self.pairs = []
        
        trainB_dict = {f: f for f in os.listdir(self.trainB_dir)}
        for fnameA in os.listdir(self.trainA_dir):
            keyA = fnameA
            if keyA in trainB_dict:
                thermal_path = os.path.join(self.trainA_dir, fnameA)
                visible_path = os.path.join(self.trainB_dir, trainB_dict[keyA])
                self.pairs.append((thermal_path, visible_path))
            else:
                print(f"[警告] 无对应可见光图像: {fnameA}")

        self.to_tensor = T.ToTensor()
        self._init_font()

    def _print_dataset_stats(self):
        """打印数据集质量统计信息"""
        print("\n" + "="*70)
        print("📊 数据集质量统计")
        print("="*70)
        
        empty_count = 0
        short_count = 0
        long_count = 0
        token_lengths = []
        
        for key, desc in self.descriptions.items():
            if not desc or desc.strip() == "":
                empty_count += 1
                token_lengths.append(0)
            else:
                # 粗略估计：1 token ≈ 1.3 characters
                token_est = len(desc) / 1.3
                token_lengths.append(token_est)
                
                if token_est < 5:
                    short_count += 1
                elif token_est > 77:
                    long_count += 1
        
        total = len(self.descriptions)
        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        avg_padding = max(0, 77 - avg_tokens)
        
        print(f"✓ 总样本数: {total}")
        print(f"✓ 空描述: {empty_count} ({100*empty_count/total:.1f}%)")
        print(f"✓ 短描述 (<5 tokens): {short_count}")
        print(f"✓ 长描述 (>77 tokens): {long_count}")
        print(f"✓ 平均描述长度: {avg_tokens:.1f} tokens")
        print(f"✓ 平均需要填充: {avg_padding:.1f} tokens (endoftext)")
        print(f"✓ drop_text_prob: {self.drop_text_prob} (额外空 prompt 比例)")
        
        estimated_empty = empty_count + int(total * self.drop_text_prob)
        estimated_padding_percent = (estimated_empty * 77 + (total - estimated_empty) * avg_padding) / (total * 77)
        print(f"✓ 预估填充比例: {100*estimated_padding_percent:.1f}% (包含 drop_text)")
        print("="*70 + "\n")

    def _init_font(self):
        """初始化字体，避免重复加载"""
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            try:
                self.font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 20)
                self.font_small = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 16)
            except:
                self.font = ImageFont.load_default()
                self.font_small = ImageFont.load_default()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        thermal_path, visible_path = self.pairs[idx]
    
        thermal_img = Image.open(thermal_path).convert("RGB").resize(self.condition_size, Image.Resampling.LANCZOS)
        visible_img = Image.open(visible_path).convert("RGB").resize(self.target_size, Image.Resampling.LANCZOS)
    
        rel_path = os.path.relpath(visible_path, self.trainB_dir)
        prompt = self.descriptions.get(rel_path, "")
        
        # ========== 新增：处理空描述 ==========
        # 如果描述为空，设置默认描述（除非故意 drop_text）
        if (not prompt or prompt.strip() == ""):
            prompt = "a detailed face"
        # ======================================
        
        # ========== 新增：截断过长描述 ==========
        # 防止信息丢失和过度填充
        # MAX_PROMPT_LENGTH = 75  # 留 2 个给 EOS/special tokens
        # if len(prompt) > MAX_PROMPT_LENGTH * 1.3:  # 粗略估计：1.3 chars per token
        #     # 在词边界截断，避免在单词中间断开
        #     prompt = prompt[:int(MAX_PROMPT_LENGTH * 1.3)].rsplit(' ', 1)[0]
        # # ======================================
        
        # 记录是否丢弃了 prompt，用于一致性处理
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        
        if drop_text:
            prompt = ""  # 只有在 drop_text 时才真正设为空（用于无条件训练）
        if drop_image:
            thermal_img = Image.new("RGB", self.condition_size, (0, 0, 0))
    
        return {
            "image": self.to_tensor(visible_img),
            "condition_0": self.to_tensor(thermal_img),
            "condition_type_0": "thermal",
            "position_delta_0": np.array([0, 0]),
            "description": prompt,
            "drop_text": drop_text,
            "drop_image": drop_image
        }


# ==================== 创建数据集 ====================

dataset = ThermalToVisibleDataset(
    root_dir=training_config["dataset"]["root"],
    description_file=description_file,
    condition_size=training_config["dataset"]["condition_size"],
    target_size=training_config["dataset"]["target_size"],
    drop_text_prob=training_config["dataset"]["drop_text_prob"],
    drop_image_prob=training_config["dataset"]["drop_image_prob"],
)

# 验证数据集
print("数据集大小:", len(dataset))
sample = dataset[0]
print("样本键:", sample.keys())
print("图像张量形状:", sample["image"].shape)
print("条件图像张量形状:", sample["condition_0"].shape)
print("条件类型:", sample["condition_type_0"])
print("文本描述:", sample["description"])


# ==================== 改进的测试函数 ====================

@torch.no_grad()
def test_function(model, dataset, save_path, file_name, num_samples=3):
    """
    使用数据集中随机选取的热成像图像 + prompt 进行测试
    并生成并排对比图像（热成像 | 目标图像 | 生成图像 + Prompt）
    
    Args:
        model: OminiModel 实例
        dataset: ThermalToVisibleDataset 实例
        save_path: 保存结果的目录
        file_name: 保存文件的前缀名
        num_samples: 测试样本数量，默认 3
    """
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]
    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    adapter = model.adapter_names[2]
    condition_type = model.training_config.get("condition_type", "thermal")
    
    # =============== 创建所有必需的子目录 ===============
    subdirs = ["comparison", "thermal", "target", "generated"]
    for subdir in subdirs:
        subdir_path = os.path.join(save_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    # 从数据集中随机选取样本
    dataset_size = len(dataset)
    sample_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))

    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()

    for i, idx in enumerate(sample_indices):
        try:
            # 从数据集获取样本
            sample = dataset[idx]
            thermal_tensor = sample["condition_0"]  # 热成像 tensor
            target_tensor = sample["image"]  # 目标可见光 tensor
            prompt = sample["description"]
            drop_text = sample.get("drop_text", False)

            # 如果 prompt 为空，使用默认值
            if not prompt or prompt.strip() == "":
                if drop_text:
                    prompt = ""
                else:
                    prompt = "a detailed face"

            # 转换 tensor 到 PIL Image
            thermal_img_pil = to_pil(thermal_tensor)
            target_img_pil = to_pil(target_tensor)

            # =============== 生成图像 ===============
            condition = Condition(
                thermal_img_pil,
                adapter,
                position_delta,
                position_scale
            )

            generator = torch.Generator(device=model.device)
            generator.manual_seed(42)  # 统一使用相同的随机种子

            res = generate(
                model.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=target_size[1],
                width=target_size[0],
                generator=generator,
                model_config=model.model_config,
                kv_cache=model.model_config.get("independent_condition", False),
                mogle=model.mogle,
                guidance_scale=1.8, 
                image_guidance_scale=1.0,   
            )
            generated_img = res.images[0]

            # =============== 生成并排对比图像 ===============
            img_width, img_height = target_size[0], target_size[1]
            
            # 调整热成像尺寸到目标尺寸
            thermal_img_resized = thermal_img_pil.resize((img_width, img_height), Image.Resampling.LANCZOS)
            
            # 创建水平排列的画布
            canvas_width = img_width * 3 + 20  # 三张图像 + 间隔
            canvas_height = img_height + 100  # 额外空间放 prompt 文字
            
            canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
            
            # 粘贴三张图像
            canvas.paste(thermal_img_resized, (0, 0))
            canvas.paste(target_img_pil, (img_width + 10, 0))
            canvas.paste(generated_img, (img_width * 2 + 20, 0))
            
            # =============== 添加文字标签和 Prompt ===============
            draw = ImageDraw.Draw(canvas)
            
            # 使用数据集中预加载的字体
            font = dataset.font
            font_small = dataset.font_small
            
            # 添加列标题
            draw.text((img_width // 2 - 30, img_height + 10), "Thermal", fill="black", font=font)
            draw.text((img_width + img_width // 2 - 20, img_height + 10), "Target", fill="black", font=font)
            draw.text((img_width * 2 + img_width // 2 - 20, img_height + 10), "Generated", fill="black", font=font)
            
            # 添加 Prompt（处理长文本换行）
            prompt_y = img_height + 45
            max_chars_per_line = 80
            if len(prompt) > max_chars_per_line:
                prompt_lines = []
                for line_idx in range(0, len(prompt), max_chars_per_line):
                    prompt_lines.append(prompt[line_idx:line_idx + max_chars_per_line])
                for line_idx, line in enumerate(prompt_lines[:2]):  # 最多显示两行
                    draw.text((10, prompt_y + line_idx * 25), f"Prompt: {line}", fill="black", font=font_small)
            else:
                draw.text((10, prompt_y), f"Prompt: {prompt}", fill="black", font=font_small)

            # =============== 保存对比图像 ===============
            comparison_file_path = os.path.join(
                save_path, "comparison", f"{file_name}_{condition_type}_comparison_{i}.jpg"
            )
            canvas.save(comparison_file_path)
            print(f"✅ 已保存对比图像: {comparison_file_path}")
            
            # =============== 分别保存三张原始图像 ===============
            thermal_file_path = os.path.join(
                save_path, "thermal", f"{file_name}_{condition_type}_thermal_{i}.jpg"
            )
            target_file_path = os.path.join(
                save_path, "target", f"{file_name}_{condition_type}_target_{i}.jpg"
            )
            generated_file_path = os.path.join(
                save_path, "generated", f"{file_name}_{condition_type}_generated_{i}.jpg"
            )
            
            thermal_img_resized.save(thermal_file_path)
            target_img_pil.save(target_file_path)
            generated_img.save(generated_file_path)
            
            print(f"   ├─ Thermal: {thermal_file_path}")
            print(f"   ├─ Target: {target_file_path}")
            print(f"   ├─ Generated: {generated_file_path}")
            print(f"   └─ Prompt: {prompt}\n")

            # 显式释放内存
            del thermal_img_pil, target_img_pil, generated_img, canvas, draw

        except FileNotFoundError as e:
            print(f"❌ 文件未找到 (样本 {i}): {e}\n")
        except RuntimeError as e:
            print(f"❌ 运行时错误 (样本 {i}): {e}\n")
        except Exception as e:
            print(f"❌ 处理第 {i} 个样本时出错: {type(e).__name__}: {e}\n")


# ==================== 创建模型 ====================

# ========== 关键变动 1: 从配置中读取 MoGLE 配置 ==========
use_mogle = config.get("use_mogle", False)
mogle_config = config.get("mogle_config", {}) if use_mogle else None

trainable_model = OminiModel(
    flux_pipe_id=config["flux_path"],
    lora_config=training_config["lora_config"],
    device=f"cuda",
    dtype=getattr(torch, config["dtype"]),
    optimizer_config=training_config["optimizer"],
    model_config=config.get("model", {}),
    gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    # ========== 关键变动 2: 传入 MoGLE 相关参数 ==========
    use_mogle=use_mogle,
    mogle_config=mogle_config,
    condition_type=config.get("condition_type", "thermal"), 
)

# ========== 关键变动 3: 如果提供了保存的 MoGLE checkpoint，则加载 ==========
mogle_checkpoint = config.get("mogle_checkpoint_path", None)
if use_mogle and mogle_checkpoint and os.path.exists(mogle_checkpoint):
    print(f"🔧 Loading MoGLE checkpoint from {mogle_checkpoint}")
    trainable_model.load_mogle_checkpoint(mogle_checkpoint)


# ==================== 开始训练 ====================

train(dataset, trainable_model, config, test_function)