import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
import requests
import time
import logging
from typing import List, Optional, Dict, Any
import prodigyopt
from torch.utils.data import DataLoader, DistributedSampler
from peft import LoraConfig, get_peft_model_state_dict
from ..pipeline.flux_omini import transformer_forward, encode_images
from ..moe.mogle_t2v_unet import MoGLE  # 导入MoGLE模块
import warnings
# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 辅助工具函数 ====================

def get_rank():
    """获取分布式训练的 LOCAL_RANK，单卡默认为 0"""
    try:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    except:
        rank = 0
    return rank


def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    支持从本地文件或 URL 读取配置。
    优先级：config_path > 环境变量 OMINI_CONFIG
    """
    if config_path is None:
        config_path = os.environ.get("OMINI_CONFIG")
        assert config_path is not None, (
            "Please provide config_path or set the OMINI_CONFIG environment variable"
        )

    if config_path.startswith("http://") or config_path.startswith("https://"):
        try:
            response = requests.get(config_path, timeout=20)
            response.raise_for_status()
            config = yaml.safe_load(response.text)
            logger.info(f"Loaded config from URL: {config_path}")
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load remote YAML: {e}")
    else:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from file: {config_path}")
        return config


def init_wandb(wandb_config: Dict[str, Any], run_name: str):
    """初始化 Weights & Biases，若失败则容错"""
    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
        logger.info("WanDB initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize WanDB: {e}")


def check_loss_validity(loss: torch.Tensor) -> bool:
    """检查 loss 是否为 NaN 或 Inf"""
    return not (torch.isnan(loss).any() or torch.isinf(loss).any())


def normalize_position_delta(p_delta: Any, default_value: tuple = (0, 0)) -> tuple:
    """
    统一处理 position_delta 格式
    支持：[0, 0], [[0, 0]], (0, 0), ((0, 0))
    """
    if isinstance(p_delta, (list, tuple)):
        if len(p_delta) == 2 and isinstance(p_delta[0], (int, float)):
            return tuple(p_delta)
        elif len(p_delta) > 0 and isinstance(p_delta[0], (list, tuple)):
            return tuple(p_delta[0])
    return default_value


# ==================== OminiModel 定义（核心训练模块） ====================

class OminiModel(L.LightningModule):
    """LoRA 微调 Flux Transformer 的训练模块，集成MoGLE特征处理"""
    
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = None,
        adapter_names: List[Optional[str]] = None,
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        max_sequence_length: int = 512,
        # MoGLE 相关参数
        use_mogle: bool = False,
        mogle_config: dict = None,
        condition_type: str = "thermal", 
    ):
        super().__init__()
        
        # 默认值处理
        if model_config is None:
            model_config = {}
        if adapter_names is None:
            adapter_names = [None, None, "default"]
        if mogle_config is None:
            mogle_config = {}
        
        self.model_config = model_config
        self.optimizer_config = optimizer_config or {}
        self.adapter_names = adapter_names
        self.max_sequence_length = max_sequence_length
        self.use_mogle = use_mogle
        self.mogle_config = mogle_config
        self.condition_type = condition_type
        logger.info(f"Initializing OminiModel with adapter_names: {adapter_names}")
        logger.info(f"MoGLE enabled: {use_mogle}")

        # 加载预训练的 FluxPipeline
        logger.info(f"Loading FluxPipeline from {flux_pipe_id}")
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # 冻结不需要训练的模块
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # 收集需要训练的 adapter 名称
        self.adapter_set = set([each for each in adapter_names if each is not None])
        logger.info(f"Adapter set: {self.adapter_set}")

        # 初始化 LoRA 层
        self.lora_layers = self.init_lora(lora_path, lora_config)
        logger.info(f"Initialized {len(self.lora_layers)} LoRA parameters")

        # 初始化 MoGLE 模块（如果启用）
        self.mogle = None
        self.mogle_adapter_map = {}  # 映射 adapter_name 到 MoGLE 模块
        if use_mogle:
            self.init_mogle(mogle_config)
            logger.info(f"Initialized MoGLE with config: {mogle_config}")

        # 迁移到设备和数据类型
        self.to(device).to(dtype)

        # 初始化训练指标
        self.log_loss = 0.0
        self.last_t = 0.0

    def init_mogle(self, mogle_config: dict):
        """初始化 MoGLE 模块，为每个 adapter 创建一个独立实例"""
        if not mogle_config:
            mogle_config = {
                "input_dim": 64,
                "hidden_dim": 256,
                "output_dim": 64,
                "has_expert": True,
                "has_gating": True,
                "weight_is_scale": False,
            }
        
        logger.info(f"🔧 MoGLE Configuration: {mogle_config}")
        
        # 为每个 adapter 创建一个 MoGLE 实例（或共用一个）
        # 这里采用共用策略以减少参数量
        self.mogle = MoGLE(
            input_dim=mogle_config.get("input_dim", 64),
            hidden_dim=mogle_config.get("hidden_dim", 256),
            output_dim=mogle_config.get("output_dim", 64),
        )
        self.mogle.train()
        
        logger.info(f"✓ Initialized shared MoGLE module")

    def init_lora(self, lora_path: str, lora_config: dict):
        # 确保至少提供了路径或配置之一
        assert lora_path or lora_config
        # 如果提供 lora_path，表示要加载已有权重（此处尚未实现）
        if lora_path:
            # TODO: 实现从 safetensors/目录加载 LoRA 权重的逻辑
            raise NotImplementedError
        else:
            # 如果没有权重路径，则为 adapter_set 中的每个 adapter 创建 LoRA 配置并注册到 transformer
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        """保存 LoRA 权重和 MoGLE 权重"""
        os.makedirs(path, exist_ok=True)
        
        # 保存 LoRA
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )
        
        # 保存 MoGLE（如果启用）
        if self.use_mogle and self.mogle is not None:
            torch.save(self.mogle.state_dict(), os.path.join(path, "mogle.pt"))
            logger.info(f"✓ Saved MoGLE checkpoint to {path}/mogle.pt")

    def load_mogle_checkpoint(self, mogle_path: str):
        """加载 MoGLE checkpoint"""
        if not os.path.exists(mogle_path):
            raise FileNotFoundError(f"MoGLE checkpoint not found at {mogle_path}")
        
        if self.mogle is None:
            raise RuntimeError("MoGLE is not initialized. Set use_mogle=True in __init__")
        
        state_dict = torch.load(mogle_path, map_location=self.device)
        self.mogle.load_state_dict(state_dict)
        logger.info(f"✓ Loaded MoGLE checkpoint from {mogle_path}")

    def configure_optimizers(self):
        """配置优化器"""
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        self.trainable_params = self.lora_layers.copy()
        
        # 添加 MoGLE 参数到可训练参数列表
        if self.use_mogle and self.mogle is not None:
            self.trainable_params.extend(list(self.mogle.parameters()))
            logger.info(f"Added {sum(p.numel() for p in self.mogle.parameters())} MoGLE parameters to training")

        for p in self.trainable_params:
            p.requires_grad_(True)

        if opt_config.get("type") == "AdamW":
            optimizer = torch.optim.AdamW(
                self.trainable_params, **opt_config.get("params", {})
            )
        elif opt_config.get("type") == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params, **opt_config.get("params", {})
            )
        elif opt_config.get("type") == "SGD":
            optimizer = torch.optim.SGD(
                self.trainable_params, **opt_config.get("params", {})
            )
        else:
            raise NotImplementedError(f"Optimizer {opt_config.get('type')} not implemented")
        
        logger.info(f"Initialized {opt_config.get('type')} optimizer")
        return optimizer

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """单个训练步骤"""
        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)

        # 收集所有条件
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(100):  # 改为更合理的上限
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            
            # 标准化 position_delta
            raw_delta = batch.get(f"position_delta_{i}", [0, 0])
            position_deltas.append(normalize_position_delta(raw_delta))
            
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        with torch.no_grad():
            # 编码图像
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
                warnings.filterwarnings("ignore", message=".*truncated because CLIP.*")
        
                # 编码文本
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    text_ids,
                ) = self.flux_pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=None,
                    prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    device=self.flux_pipe.device,
                    num_images_per_prompt=1,
                    max_sequence_length=self.max_sequence_length,
                    lora_scale=None,
                )

            # 采样时间步与噪音
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # 处理条件
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                
                # =============== MoGLE 处理特征（如果启用）===============
                if self.use_mogle and self.mogle is not None:
                    c_latents = self.mogle.forward(
                        c_latents,  # [bs, 256, 64]
                        noise_latent=x_t,  # [bs, 256, 64]
                        timestep=t  # [bs,]
                    )  # 输出: [bs, 256, 64]
                
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                
                # 应用位置偏移（已标准化为 tuple）
                c_ids[:, 1] += p_delta[0]
                c_ids[:, 2] += p_delta[1]
                
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # =============== 构建 group_mask ===============
        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # =============== 前向传播 ===============
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
        )
        pred = transformer_out[0]

        # =============== 计算损失 ===============
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        
        # 检查 loss 有效性
        if not check_loss_validity(step_loss):
            logger.warning(f"Invalid loss detected: {step_loss.item()}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        self.last_t = t.mean().item()

        # 指数平滑记录 loss
        self.log_loss = (
            step_loss.item()
            if self.log_loss == 0.0
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        
        return step_loss


# ==================== 训练回调类 ====================

class TrainingCallback(L.Callback):
    """训练回调：处理打印、保存、采样"""
    
    def __init__(
        self, 
        run_name: str, 
        training_config: dict = None, 
        test_function=None, 
        dataset=None
    ):
        if training_config is None:
            training_config = {}
            
        self.run_name = run_name
        self.training_config = training_config
        self.dataset = dataset

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Any, 
        batch: Dict[str, Any], 
        batch_idx: int
    ):
        """每个训练批次结束时调用"""
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        
        if hasattr(pl_module, 'trainable_params'):
            for param in pl_module.trainable_params:
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    gradient_size += grad_norm
                    max_gradient_size = max(max_gradient_size, grad_norm)
                    count += 1
        
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        if self.use_wandb:
            try:
                loss_value = outputs.item() if torch.is_tensor(outputs) else outputs.get("loss", 0).item()
                report_dict = {
                    "batch_idx": batch_idx,
                    "total_steps": self.total_steps,
                    "epoch": trainer.current_epoch,
                    "loss": loss_value,
                    "gradient_size": gradient_size,
                    "max_gradient_size": max_gradient_size,
                    "t": pl_module.last_t,
                }
                wandb.log(report_dict)
            except Exception as e:
                logger.warning(f"Failed to log to WanDB: {e}")

        if self.total_steps % self.print_every_n_steps == 0:
            logger.info(
                f"Epoch: {trainer.current_epoch} | Steps: {self.total_steps} | "
                f"Loss: {pl_module.log_loss:.4f} | Grad size: {gradient_size:.4f} | "
                f"Max grad: {max_gradient_size:.4f}"
            )

        if self.total_steps % self.save_interval == 0:
            logger.info(f"Saving LoRA weights at step {self.total_steps}")
            try:
                pl_module.save_lora(
                    f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
                )
            except Exception as e:
                logger.error(f"Failed to save LoRA: {e}")

        if self.total_steps % self.sample_interval == 0 and self.test_function:
            logger.info(f"Generating samples at step {self.total_steps}")
            try:
                pl_module.eval()
                self.test_function(
                    pl_module,
                    self.dataset,
                    f"{self.save_path}/{self.run_name}/output",
                    f"lora_{self.total_steps}",
                    num_samples=3,
                )
                pl_module.train()
            except Exception as e:
                logger.error(f"Failed to generate samples: {e}")
                pl_module.train()


# ==================== 训练主函数 ====================

def train(
    dataset,
    trainable_model: OminiModel,
    config: Dict[str, Any],
    test_function=None,
):
    """启动训练的主函数"""
    is_main_process = get_rank() == 0
    rank = get_rank()
    torch.cuda.set_device(rank)

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    if is_main_process:
        logger.info("=" * 80)
        logger.info(f"Training run: {run_name}")
        logger.info("=" * 80)

    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    if is_main_process:
        logger.info(f"Rank: {rank}")
        logger.info(f"Config: {config}")

    logger.info(f"Dataset length: {len(dataset)}")

    # =============== DataLoader 初始化 ===============
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset,
        batch_size=training_config.get("batch_size", 1),
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=training_config.get("dataloader_workers", 0),
        drop_last=True,
        pin_memory=True,
    )

    callbacks = []
    if is_main_process:
        callbacks.append(
            TrainingCallback(run_name, training_config, test_function, dataset)
        )

    # =============== Trainer 初始化 ===============
    trainer = L.Trainer(
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=is_main_process,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", 1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        accelerator="gpu",
        devices=training_config.get("devices", 1),
        strategy=(
            "ddp" 
            if (torch.distributed.is_available() and 
                training_config.get("devices", 1) > 1)
            else "auto"
        ),
        default_root_dir=training_config.get("default_root_dir", "/tmp/train"),
        num_sanity_val_steps=0,
        precision=training_config.get("precision", "bf16-true"),
    )

    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}", exist_ok=True)
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info(f"Config saved to {save_path}/{run_name}/config.yaml")

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    logger.info("Starting training...")
    trainer.fit(trainable_model, train_loader)
    
    if is_main_process:
        logger.info("Training completed!")