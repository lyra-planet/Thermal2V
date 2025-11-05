import sys
sys.path.append("../../") 
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
from omini.train_flux.model_t2v import OminiModel
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