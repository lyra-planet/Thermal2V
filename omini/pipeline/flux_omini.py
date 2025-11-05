# 导入 PyTorch，用于张量运算和深度学习模型构建
import torch

# 从 typing 模块中导入常用的类型注解工具，用于函数参数和返回值的类型提示，提高可读性和静态检查能力
from typing import List, Union, Optional, Dict, Any, Callable, Type, Tuple

# 从 diffusers 库（Hugging Face 出品）中导入 FLUX 扩散模型相关的管线类
from diffusers.pipelines import FluxPipeline

# 从 FLUX 的具体 pipeline 文件中导入关键组件，包括：
# - FluxPipelineOutput：推理输出结果的数据结构
# - FluxTransformer2DModel：核心 Transformer 去噪网络结构（用于图像生成）
# - calculate_shift：用于位置编码或潜空间偏移的辅助函数
# - retrieve_timesteps：调度时间步函数，用于扩散过程的控制
# - np：在该模块中直接暴露的 numpy 兼容接口
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    FluxTransformer2DModel,
    calculate_shift,
    retrieve_timesteps,
    np,
)

# 导入 Attention 和 F 模块，用于扩散模型中的注意力机制处理
from diffusers.models.attention_processor import Attention, F

# 导入旋转位置编码（RoPE）方法，用于在 Transformer 中注入二维空间位置感知能力
from diffusers.models.embeddings import apply_rotary_emb

# 从 transformers 库中导入 pipeline，用于快速加载如 CLIP、深度估计模型等
from transformers import pipeline

# 导入 PEFT（Parameter-Efficient Fine-Tuning）框架中的基类，用于 LoRA 等微调模块的识别
from peft.tuners.tuners_utils import BaseTunerLayer

# 导入 accelerate 工具中的函数，用于判断当前 PyTorch 版本是否兼容某些优化
from accelerate.utils import is_torch_version

# 导入 contextmanager 装饰器，用于构建上下文管理器（如 with 语句中使用）
from contextlib import contextmanager

# 导入 OpenCV，用于图像边缘检测、滤波等传统图像处理任务
import cv2

# 从 Pillow 导入图像类和滤镜模块，用于图像读取、转换和高斯模糊等处理
from PIL import Image, ImageFilter


# 定义一个设置随机种子的函数，用于确保模型可重复性
def seed_everything(seed: int = 42):
    # 设置 cuDNN 的随机算法为确定性模式，避免 GPU 上的随机性波动
    torch.backends.cudnn.deterministic = True
    # 设置 PyTorch 的全局随机种子
    torch.manual_seed(seed)
    # 设置 numpy 的随机种子，保证所有依赖 numpy 的随机操作一致
    np.random.seed(seed)


# 定义一个裁剪隐藏状态数值范围的函数，用于避免 float16 下的溢出问题
def clip_hidden_states(hidden_states: torch.FloatTensor) -> torch.FloatTensor:
    # 如果输入是半精度浮点数（float16），则限制其范围在 [-65504, 65504] 内，避免 NaN 或 Inf
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    # 返回安全范围内的张量
    return hidden_states


# 定义一个图像编码函数，将原始 RGB 图像编码为 FLUX 模型可识别的 latent tokens 和位置 ID
def encode_images(pipeline: FluxPipeline, images: torch.Tensor):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    将输入图像编码为 FLUX 扩散模型所需的潜在空间 tokens 和 对应的位置 id。
    """
    # 预处理图像（调整尺寸、归一化等），使其符合模型输入格式
    images = pipeline.image_processor.preprocess(images)
    # 将图像张量移动到模型所在设备，并转换为模型所需的数据类型（如 float16 或 bfloat16）
    images = images.to(pipeline.device).to(pipeline.dtype)
    # 使用 VAE 编码器将图像转换为潜在空间表示，并从分布中采样
    images = pipeline.vae.encode(images).latent_dist.sample()
    # 对编码结果进行缩放和偏移处理，使其符合扩散模型的输入规范
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    # 打包为 FLUX 格式的 latent tokens
    images_tokens = pipeline._pack_latents(images, *images.shape)
    # 生成与 tokens 对应的位置 ID，决定 RoPE 等位置编码如何注入
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    # 如果位置 ID 的数量与 tokens 数量不匹配，则重新生成 ID（通常处理奇数尺寸）
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    # 返回 tokens 和 对应的位置 id
    return images_tokens, images_ids


# 定义一个全局变量，用于缓存深度估计模型，避免重复加载造成性能损失
depth_pipe = None


# 定义一个图像条件生成函数，将输入图像转换为指定的 "条件图像"（如边缘图、深度图等）
def convert_to_condition(
    condition_type: str,
    raw_img: Union[Image.Image, torch.Tensor],
    blur_radius: Optional[int] = 5,
) -> Union[Image.Image, torch.Tensor]:
    # 如果条件类型是 depth（深度图），则调用深度估计模型
    if condition_type == "depth":
        # 使用全局变量缓存深度模型，如果还未初始化则加载
        global depth_pipe
        depth_pipe = depth_pipe or pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device="cpu",  # 指定为 CPU，以实现多任务并行推理，不占用 GPU 资源
        )
        # 将输入图像转换为 RGB 模式，以保证一致性
        source_image = raw_img.convert("RGB")
        # 执行深度估计并获取深度图，最终也转换为 RGB 便于后续处理
        condition_img = depth_pipe(source_image)["depth"].convert("RGB")
        return condition_img

    # Canny 边缘检测模式，将图像转换为边缘线稿
    elif condition_type == "canny":
        # 将 PIL 图像转换为 numpy 数组，方便 OpenCV 处理
        img = np.array(raw_img)
        # 执行 Canny 边缘检测，阈值范围为 [100, 200]
        edges = cv2.Canny(img, 100, 200)
        # 将二值边缘图转换回 PIL 并转为 RGB，保持 3 通道结构
        edges = Image.fromarray(edges).convert("RGB")
        return edges

    # 颜色模式，直接转为灰度图再回 RGB
    elif condition_type == "coloring":
        return raw_img.convert("L").convert("RGB")

    # 模糊模式，将图像做高斯模糊作为条件输入
    elif condition_type == "deblurring":
        condition_image = (
            raw_img.convert("RGB")
            .filter(ImageFilter.GaussianBlur(blur_radius))
            .convert("RGB")
        )
        return condition_image

    # 如果传入的类型不支持，则直接返回原图，并打印警告
    else:
        print("Warning: Returning the raw image.")
        return raw_img.convert("RGB")

# 定义一个 Condition（条件）类，用于封装单个控制条件（例如 Canny 图、深度图、模糊图等）
# 该类负责存储条件图像以及其在扩散模型中对应的位置/缩放等参数，并提供编码接口将其转换为 latent tokens。
class Condition(object):
    # 初始化方法，用于接收与条件相关的各种配置
    def __init__(
        self,
        condition: Union[Image.Image, torch.Tensor],  # 条件图像，可以是 PIL 或 torch 张量
        adapter_setting: Union[str, dict],           # 控制适配器设置（例如决定使用哪种控制模式，或 LoRA 等）
        position_delta=None,                         # 位置偏移量，用于实现论文中提到的「位置平移（Δ）」
        position_scale=1.0,                          # 位置缩放比例，用于控制空间大小/缩放
        latent_mask=None,                            # 可选掩码，用于选择某些 latent token 参与控制（用于局部控制）
        is_complement=False,                         # 是否作为补充控制信号（可能用于多条件融合）
    ) -> None:
        # 存储原始条件图像
        self.condition = condition
        # 存储控制适配器设置（可能是字符串或字典配置）
        self.adapter = adapter_setting
        # 存储位置偏移量（控制 RoPE 位置编码平移）
        self.position_delta = position_delta
        # 存储位置缩放比例（控制空间扩张或压缩）
        self.position_scale = position_scale
        # 若存在 latent 掩码，则将其转置并展平为一维索引，用于后续筛选 tokens
        self.latent_mask = (
            latent_mask.T.reshape(-1) if latent_mask is not None else None
        )
        # 存储是否为「补充控制」标志
        self.is_complement = is_complement

    # 定义一个编码方法，用于将 Condition 转换为模型可接受的 latent tokens 与 position ids
    def encode(
        self, pipe: FluxPipeline, empty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # 创建一个全黑的空图像，尺寸与原图一致，用于 "empty" 模式下输出无内容的占位编码
        condition_empty = Image.new("RGB", self.condition.size, (0, 0, 0))
        # 根据 empty 参数判断是否编码真实图像或空白图像
        tokens, ids = encode_images(pipe, condition_empty if empty else self.condition)

        # 若设置了位置偏移量（position_delta），即论文中的「位置平移 Δ」
        if self.position_delta is not None:
            # 针对所有 token 的第 1 列（通常对应 x 或行方向）加上偏移值
            ids[:, 1] += self.position_delta[0]
            # 针对第 2 列（通常对应 y 或列方向）加上偏移值
            ids[:, 2] += self.position_delta[1]

        # 若位置缩放比例不为 1.0，则对位置编码进行缩放变换
        if self.position_scale != 1.0:
            # 计算缩放引入的偏移基数，使得缩放后中心对齐
            scale_bias = (self.position_scale - 1.0) / 2
            # 对所有位置值进行缩放
            ids[:, 1:] *= self.position_scale
            # 再添加修正偏移，使其位移中心化
            ids[:, 1:] += scale_bias

        # 若 latent_mask 存在，则选择部分 token（如仅保留边缘区域的控制影响）
        if self.latent_mask is not None:
            # 在 token 维度进行索引裁剪
            tokens = tokens[:, self.latent_mask]
            # 在 position ids 上同样裁剪
            ids = ids[self.latent_mask]

        # 返回编码后的 token，位置 id，另一个 int（目前未用，可能为将来扩展）
        return tokens, ids

# 导入 contextmanager 装饰器创建一个上下文管理器，用于在 with 语句块内临时改变 LoRA 模块的行为并在退出时恢复

@contextmanager
# 定义一个名为 specify_lora 的上下文管理器函数，用于临时指定某个 LoRA adapter 生效（其余 adapter 被屏蔽）
def specify_lora(lora_modules: List[BaseTunerLayer], specified_lora):
    # 过滤出真正的 BaseTunerLayer（LoRA）实例，避免传入非 LoRA 对象导致异常
    # 背景：PEFT 框架中，LoRA 模块继承 BaseTunerLayer，我们只对这些模块调整 scaling
    valid_lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
    # 保存所有有效 LoRA 模块当前的 scaling 配置（每个 adapter 当前的缩放系数）
    # 目的：进入上下文后临时修改 scaling，退出时需要恢复为原来的值以避免副作用
    original_scales = [
        {
            adapter: module.scaling[adapter]
            for adapter in module.active_adapters
            if adapter in module.scaling
        }
        for module in valid_lora_modules
    ]
    # 进入上下文：调整每个有效 LoRA 模块中各 adapter 的 scaling，使得只有指定 adapter 的 scaling 为 1，其余为 0
    # 这一步实现了“激活指定 adapter，屏蔽其它 adapter”的语义控制
    for module in valid_lora_modules:
        for adapter in module.active_adapters:
            if adapter in module.scaling:
                module.scaling[adapter] = 1 if adapter == specified_lora else 0
    # 进入 with 块后将控制权交给 with 块体，with 块执行完毕后走 finally 分支恢复原始状态
    try:
        yield
    finally:
        # 退出上下文：按之前保存的 original_scales 恢复每个模块的 adapter scaling，避免永久改变模块状态
        for module, scales in zip(valid_lora_modules, original_scales):
            for adapter in module.active_adapters:
                if adapter in module.scaling:
                    module.scaling[adapter] = scales[adapter]

# 定义自定义的注意力前向函数 attn_forward，用于在多分支（文本/图像/条件）间计算注意力
def attn_forward(
    # attn：一个 Attention 实例（来自 diffusers 的注意力实现），封装了投影/归一化等
    attn: Attention,
    # hidden_states：图像分支的隐藏状态列表（每个元素是张量）
    hidden_states: List[torch.FloatTensor],
    
    # adapters：对应每个分支应使用的 LoRA adapter 名称（或 None）
    adapters: List[str],
    
    # hidden_states2：可选的第二类隐藏状态（通常用于文本/上下文分支）
    hidden_states2: Optional[List[torch.FloatTensor]] = [],
    
    # position_embs：可选的位置信息列表，用于 RoPE 等旋转位置编码的注入
    position_embs: Optional[List[torch.Tensor]] = None,
    # group_mask：分支间的注意力连接掩码（布尔矩阵），用于禁止某些分支间的交互
    group_mask: Optional[torch.Tensor] = None,
    # cache_mode：可选的缓存模式（"write" 或 "read"），用于 KV 缓存机制
    cache_mode: Optional[str] = None,
    # to_cache：布尔列表，指示哪些分支的 K/V 需要写入缓存（与 cache_storage 配合）
    # 作用：在多分支场景下可以选择只缓存某些分支（如条件分支），以复用 K/V 提速
    to_cache: Optional[List[torch.Tensor]] = None,
    # cache_storage：缓存容器（按 attention 层索引组织），用于读写 KV
    cache_storage: Optional[List[torch.Tensor]] = None,
    # kwargs：透传其他可选参数（保留扩展性）
    **kwargs: dict,
) -> torch.FloatTensor:
    # 读取 batch size 以及隐藏状态的 shape 信息（假设 hidden_states[0] 形如 [B, T, D]）
    bs, _, _ = hidden_states[0].shape
    # h2_n：文本/第二分支的个数（hidden_states2 的长度）
    h2_n = len(hidden_states2)

    # 初始化 queries/keys/values 存储列表（将按分支构造）
    queries, keys, values = [], [], []

    # 为每个 encoder（文本分支）隐藏状态准备 Q/K/V 投影
    # 背景：文本分支可能使用不同的投影（attn.add_q_proj 等），作为“额外”分支接入多模态注意力
    for i, hidden_state in enumerate(hidden_states2):
        # 通过 attn 提供的 add_* 投影获得 query/key/value（通常是针对附加分支的投影）
        query = attn.add_q_proj(hidden_state)
        key = attn.add_k_proj(hidden_state)
        value = attn.add_v_proj(hidden_state)

        # head_dim：单头注意力的维度 = 最后维度除以头数（attn.heads）
        head_dim = key.shape[-1] // attn.heads
        # reshape_fn：将 [B, T, heads*head_dim] 的张量reshape为 [B, heads, T, head_dim] 方便计算
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        # 对 query/key/value 做变形以匹配多头注意力的形状
        query, key, value = map(reshape_fn, (query, key, value))
        # 对 query/key 进行额外的归一化（attn.norm_added_q / attn.norm_added_k），通常用于稳定数值
        query, key = attn.norm_added_q(query), attn.norm_added_k(key)

        # 将该分支的 Q/K/V 添加到列表中，后续会与其他分支的 K/V 一起用于注意力计算
        queries.append(query)
        keys.append(key)
        values.append(value)

    # 为每个主分支（图像分支）准备 Q/K/V 投影
    for i, hidden_state in enumerate(hidden_states):
        # 使用指定的 LoRA adapter 临时激活对应的微调参数，确保 to_q/to_k/to_v 在不同 adapter 下行为不同
        with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapters[i + h2_n]):
            query = attn.to_q(hidden_state)
            key = attn.to_k(hidden_state)
            value = attn.to_v(hidden_state)

        # 计算单头维度并构造 reshape 函数（与上面相同）
        head_dim = key.shape[-1] // attn.heads
        reshape_fn = lambda x: x.view(bs, -1, attn.heads, head_dim).transpose(1, 2)

        # 对 q/k/v 执行变形
        query, key, value = map(reshape_fn, (query, key, value))
        # 对 q/k 做归一化（attn.norm_q / attn.norm_k），这通常是某种层归一化或缩放
        query, key = attn.norm_q(query), attn.norm_k(key)

        # 将构造好的 q/k/v 加入列表
        queries.append(query)
        keys.append(key)
        values.append(value)

    # 如果提供了位置嵌入，则对 queries 和 keys 应用旋转位置编码（RoPE）
    # 这样可以把二维空间位置信息注入到多头注意力的相位中，适用于图像 token 的空间感知
    if position_embs is not None:
        queries = [apply_rotary_emb(q, position_embs[i]) for i, q in enumerate(queries)]
        keys = [apply_rotary_emb(k, position_embs[i]) for i, k in enumerate(keys)]

    # 如果当前模式为 "write"，把需要缓存的 K/V 存入 cache_storage；用于后续步骤的 "read"
    if cache_mode == "write":
        for i, (k, v) in enumerate(zip(keys, values)):
            # to_cache 指示哪些分支要写入缓存（例如条件分支）
            if to_cache[i]:
                # 将每层的 key/value 追加到对应的缓存槽（按 attn.cache_idx 索引）
                cache_storage[attn.cache_idx][0].append(k)
                cache_storage[attn.cache_idx][1].append(v)

    # 用于存储每个 query 分支对应的注意力输出
    attn_outputs = []
    # 针对每个 query（每个分支）计算该 query 对所有允许的 keys/values 的注意力输出
    for i, query in enumerate(queries):
        # 临时收集允许参与计算的 keys 和 values（受 group_mask 控制）
        keys_, values_ = [], []
        # 将来自其他分支的 k/v 添加进 keys_/values_（依据 group_mask 控制是否允许跨分支注意力）
        for j, (k, v) in enumerate(zip(keys, values)):
            # 如果提供了 group_mask 并且当前 (i,j) 被禁用，则跳过
            if (group_mask is not None) and not (group_mask[i][j].item()):
                continue
            keys_.append(k)
            values_.append(v)
        # 若 cache_mode 为 "read"，则把之前写入的缓存扩展到 keys_/values_
        if cache_mode == "read":
            keys_.extend(cache_storage[attn.cache_idx][0])
            values_.extend(cache_storage[attn.cache_idx][1])
        # TODO: 这里注释了 "Add keys and values from cache TODO"；表明未来可能还会改进缓存合并策略

        # 使用 PyTorch 的 scaled_dot_product_attention（高效实现）计算注意力输出
        # 其中 query 形状 [B, heads, Tq, head_dim]，cat(keys_, dim=2) 会将所有 keys 在时间维拼接
        attn_output = F.scaled_dot_product_attention(
            query, torch.cat(keys_, dim=2), torch.cat(values_, dim=2)
        ).to(query.dtype)
        # 将注意力输出转换回 [B, Tq, heads*head_dim] 的形状，便于后续线性投影恢复
        attn_output = attn_output.transpose(1, 2).reshape(bs, -1, attn.heads * head_dim)
        # 将该分支的注意力输出追加到列表中
        attn_outputs.append(attn_output)

    # 重塑注意力输出以匹配原始隐藏状态分支的后续处理路径，分为 h_out（图像分支）和 h2_out（文本/额外分支）
    h_out, h2_out = [], []

    # 对文本 / 附加分支的注意力输出使用 attn.to_add_out 做投影（或其它后处理）
    for i, hidden_state in enumerate(hidden_states2):
        h2_out.append(attn.to_add_out(attn_outputs[i]))

    # 对图像分支的注意力输出进行 to_out 投影（若有的话），同时支持 LoRA 临时切换 to_out 的行为
    for i, hidden_state in enumerate(hidden_states):
        # 取得对应的注意力输出（注意索引偏移 i + h2_n，因为前面有文本分支的输出占位）
        h = attn_outputs[i + h2_n]
        # 如果 attn 对象具有 to_out（通常是一个列表/模块），则在该投影上临时激活对应 adapter 的 LoRA
        if getattr(attn, "to_out", None) is not None:
            with specify_lora((attn.to_out[0],), adapters[i + h2_n]):
                h = attn.to_out[0](h)
        # 将处理后的 h 添加到图像分支输出列表
        h_out.append(h)

    # 返回值：如果存在文本/第二分支（h2_n > 0），则返回 (h_out, h2_out) 二元组；否则只返回 h_out
    return (h_out, h2_out) if h2_n else h_out

# 定义 block_forward，用于处理 DiT 中的 dual-branch block（图像分支 + 文本分支）前向逻辑
def block_forward(
    # self：当前 transformer block（通常继承自 FluxTransformer2DModel 的 block 模块）
    self,
    # image_hidden_states：图像分支隐藏表示列表
    image_hidden_states: List[torch.FloatTensor],
    # text_hidden_states：文本分支隐藏表示列表
    text_hidden_states: List[torch.FloatTensor],
    # tembs：每个分支对应的时间/文本联合嵌入列表
    tembs: List[torch.FloatTensor],
    # adapters：每个分支的 LoRA adapter 指定，用于局部微调控制
    adapters: List[str],
    # position_embs：可选的位置嵌入（传入到 attn_forward）
    position_embs=None,
    # attn_forward：用于计算注意力的函数，默认为上面定义的 attn_forward（支持多分支与 LoRA）
    attn_forward=attn_forward,
    # kwargs：透传参数
    **kwargs: dict,
):
    # 文本分支数量
    txt_n = len(text_hidden_states)

    # 初始化用于存储 norm/ff 等中间变量的容器（文本与图像）
    img_variables, txt_variables = [], []

    # 对每个文本隐藏状态执行 norm1_context（带时间/文本嵌入），得到其 gate/shift/scale 等参数元组
    for i, text_h in enumerate(text_hidden_states):
        txt_variables.append(self.norm1_context(text_h, emb=tembs[i]))

    # 对每个图像隐藏状态执行 norm1（可能在内部用 LoRA 做适配），得到对应的参数元组
    for i, image_h in enumerate(image_hidden_states):
        # 在 norm1 的线性层上使用指定 adapter 的 LoRA（通过 specify_lora 临时激活）
        with specify_lora((self.norm1.linear,), adapters[i + txt_n]):
            img_variables.append(self.norm1(image_h, emb=tembs[i + txt_n]))

    # Attention：调用 attn_forward 计算图像与文本分支间的联合注意力输出
    img_attn_output, txt_attn_output = attn_forward(
        self.attn,
        # 传入 img_variables 与 txt_variables 的第一个元素（通常是归一化后的 hidden 表示）
        hidden_states=[each[0] for each in img_variables],
        hidden_states2=[each[0] for each in txt_variables],
        position_embs=position_embs,
        adapters=adapters,
        **kwargs,
    )

    # 文本分支的后续 MLP/残差流程：基于 attn 输出与 gate 等系数构建最终输出
    text_out = []
    for i in range(len(text_hidden_states)):
        # 解包 norm1 context 返回的元组：第 0 个元素通常是归一化后的 hidden（被忽略用 _ 占位）
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = txt_variables[i]
        # MSA（多头自注意）输出按 gate_msa 缩放并加入残差
        text_h = text_hidden_states[i] + txt_attn_output[i] * gate_msa.unsqueeze(1)
        # 应用第二层归一化并根据 MLP 的 scale/shift 做仿射变换
        norm_h = (
            self.norm2_context(text_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        # 通过上下文 MLP（ff_context）并用 gate_mlp 控制输出强度，再与残差相加
        text_h = self.ff_context(norm_h) * gate_mlp.unsqueeze(1) + text_h
        # 裁剪浮点数范围以防 float16 溢出，随后追加到文本输出列表
        text_out.append(clip_hidden_states(text_h))

    # 图像分支的后续 MLP/残差流程：与文本分支类似，但对 FF 层应用 LoRA（在指定位置）
    image_out = []
    for i in range(len(image_hidden_states)):
        # 同上，从 img_variables 中解包 gate/shift/scale 等参数
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = img_variables[i]
        # 将注意力输出按 gate_msa 缩放并与原始隐藏态做残差连接，保持 dtype 与输入一致
        image_h = (
            image_hidden_states[i] + img_attn_output[i] * gate_msa.unsqueeze(1)
        ).to(image_hidden_states[i].dtype)
        # 应用第二层归一化并做 scale/shift 仿射
        norm_h = self.norm2(image_h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        # 在 FF 层的第三个子模块上临时激活对应 adapter 的 LoRA，然后执行前向并与残差相加
        with specify_lora((self.ff.net[2],), adapters[i + txt_n]):
            image_h = image_h + self.ff(norm_h) * gate_mlp.unsqueeze(1)
        # 裁剪数值并加入图像输出列表
        image_out.append(clip_hidden_states(image_h))
    # 返回图像与文本分支的输出列表（后续 transformer 层会继续处理）
    return image_out, text_out

# 定义 single_block_forward，用于处理只含单一隐藏状态列表的 Transformer block（例如后续的单流 Transformer 层）
def single_block_forward(
    # self：Transformer 的单个块实例
    self,
    # hidden_states：一个包含所有分支（文本+图像）合并后的隐藏状态列表
    hidden_states: List[torch.FloatTensor],
    # tembs：每个位置对应的时间/文本嵌入
    tembs: List[torch.FloatTensor],
    # adapters：每个位置对应的 LoRA adapter 名称，用于局部微调控制
    adapters: List[str],
    # position_embs：可选的位置嵌入，用于 RoPE
    position_embs=None,
    # attn_forward：注入自定义的注意力前向函数
    attn_forward=attn_forward,
    # kwargs：透传的其他参数
    **kwargs: dict,
):
    # 为 mlp_hidden_states 和 gates 初始化结构，长度与 hidden_states 一致
    mlp_hidden_states, gates = [[None for _ in hidden_states] for _ in range(2)]

    # 用于保存归一化后的 hidden（用于后续 attn）
    hidden_state_norm = []
    for i, hidden_state in enumerate(hidden_states):
        # 备注：此处与原始 FLUX 实现有出入 —— 原实现可能基于合并后的隐藏态计算 gate；这里每分支独立计算
        # 在 norm 与 proj_mlp 上临时激活对应 adapter 的 LoRA（确保 proj_mlp 使用 adapter 权重）
        with specify_lora((self.norm.linear, self.proj_mlp), adapters[i]):
            # norm 返回归一化后的张量 h_norm 与 gate（用于与 MSA 输出融合）
            h_norm, gates[i] = self.norm(hidden_state, emb=tembs[i])
            # 将 h_norm 通过 proj_mlp + 激活函数得到 MLP 分支的输出
            mlp_hidden_states[i] = self.act_mlp(self.proj_mlp(h_norm))
        # 保存归一化结果供后续注意力使用
        hidden_state_norm.append(h_norm)

    # 调用 attn_forward 对归一化后的 hidden_state_norm 计算注意力输出
    attn_outputs = attn_forward(
        self.attn, hidden_state_norm, adapters, position_embs=position_embs, **kwargs
    )

    # 将注意力输出与 MLP 输出按 proj_out 做融合，并通过 gate 控制强度，最后与残差相加
    h_out = []
    for i in range(len(hidden_states)):
        # 在 proj_out 上临时激活 adapter 的 LoRA，以便投影使用当前 adapter 的微调权重
        with specify_lora((self.proj_out,), adapters[i]):
            # 将注意力输出与 mlp 输出在最后一维拼接（channel 维），以便通过 proj_out 恢复到 hidden size
            h = torch.cat([attn_outputs[i], mlp_hidden_states[i]], dim=2)
            # 用 gate 缩放 proj_out 的输出并添加残差，形成最终的块输出
            h = gates[i].unsqueeze(1) * self.proj_out(h) + hidden_states[i]
            # 裁剪数值并追加到输出列表
            h_out.append(clip_hidden_states(h))

    # 返回单流块的输出列表
    return h_out

# 定义 transformer_forward，用于把整个 FluxTransformer2DModel 的前向流程组织起来（支持多分支的 image/text/condition）
def transformer_forward(
    # transformer：FluxTransformer2DModel 实例（包含 transformer_blocks、single_transformer_blocks 等）
    transformer: FluxTransformer2DModel,
    # image_features：按顺序传入的图像（主 latent + condition latents）特征张量列表
    image_features: List[torch.Tensor],
    # text_features：文本特征张量列表（通常只有一个 prompt_embeds）
    text_features: List[torch.Tensor] = None,
    # img_ids：每个图像分支对应的位置 ids 列表（用于位置编码）
    img_ids: List[torch.Tensor] = None,
    # txt_ids：文本 token 的位置 id（用于位置嵌入）
    txt_ids: List[torch.Tensor] = None,
    # pooled_projections：每分支的 pooled 文本投影（time-text 联合嵌入需要）
    pooled_projections: List[torch.Tensor] = None,
    # timesteps：每分支对应的时间步张量列表（用于 time_text_embed）
    timesteps: List[torch.LongTensor] = None,
    # guidances：每分支的 guidance 标量列表（用于 guidance_embeds）
    guidances: List[torch.Tensor] = None,
    # adapters：每分支所使用的 LoRA adapter 名称列表
    adapters: List[str] = None,
    # 可注入的函数，允许用自定义的 single_block_forward / block_forward / attn_forward 替换默认实现
    single_block_forward=single_block_forward,
    block_forward=block_forward,
    attn_forward=attn_forward,
    # kwargs：透传额外参数
    **kwargs: dict,
):
    # 将 transformer 绑定为局部 self，便于后续直接引用属性/子模块
    self = transformer
    # 文本分支数量（可能为 0）
    txt_n = len(text_features) if text_features is not None else 0

    # 若未显式传入 adapters，则默认每个分支的 adapter 为 None（表示不使用 LoRA）
    adapters = adapters or [None] * (txt_n + len(image_features))
    # 断言 adapters 的数量应与 timesteps 对应（每个分支需要一个 timestep）
    assert len(adapters) == len(timesteps)

    # --------------- 预处理 image_features：通过 x_embedder 投影到模型隐藏维 ---------------
    image_hidden_states = []
    for i, image_feature in enumerate(image_features):
        # 在 x_embedder 上临时激活对应 adapter 的 LoRA（允许不同分支用不同 adapter）
        with specify_lora((self.x_embedder,), adapters[i + txt_n]):
            image_hidden_states.append(self.x_embedder(image_feature))

    # --------------- 预处理 text_features：通过 context_embedder 获得文本分支隐藏态 ---------------
    text_hidden_states = []
    for text_feature in text_features:
        text_hidden_states.append(self.context_embedder(text_feature))

    # 确保 timesteps 的长度等于 image_features + text_features（每分支都要有 timestep）
    assert len(timesteps) == len(image_features) + len(text_features)

    # 定义内部函数 get_temb：用于构造每个分支的 time-text 联合嵌入（time + guidance + pooled_projection）
    def get_temb(timestep, guidance, pooled_projection):
        # 将 timestep 转为与 image_hidden_states 相同的 dtype，并乘以 1000（与模型的尺度约定有关）
        timestep = timestep.to(image_hidden_states[0].dtype) * 1000
        # 如果存在 guidance，则同时将 guidance 转换为相同 dtype 并乘以 1000，然后调用 time_text_embed（带 guidance）
        if guidance is not None:
            guidance = guidance.to(image_hidden_states[0].dtype) * 1000
            return self.time_text_embed(timestep, guidance, pooled_projection)
        else:
            # 否则只用 timestep 与 pooled_projection 构造嵌入
            return self.time_text_embed(timestep, pooled_projection)

    # 为每个分支调用 get_temb 构建 tembs 列表（顺序与 timesteps/guidances/pooled_projections 一一对应）
    tembs = [get_temb(*each) for each in zip(timesteps, guidances, pooled_projections)]

    # 为每个 token（文本 + 图像）准备位置嵌入：pos_embed 接受 token id 并返回对应的位置信息
    position_embs = [self.pos_embed(each) for each in (*txt_ids, *img_ids)]

    # 为可能的梯度检查点（gradient checkpointing）准备参数：PyTorch 1.11+ 的 use_reentrant 选项处理不同
    gckpt_kwargs: Dict[str, Any] = (
        {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    )

    # --------------- 逐层执行 dual-branch transformer blocks（图像 + 文本） ---------------
    for block in self.transformer_blocks:
        # 为每个 block 准备传入参数字典
        block_kwargs = {
            "self": block,
            "image_hidden_states": image_hidden_states,
            "text_hidden_states": text_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        # 如果处于训练模式且启用了 gradient_checkpointing，则使用 torch.utils.checkpoint 保存内存
        if self.training and self.gradient_checkpointing:
            image_hidden_states, text_hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            # 正常前向调用 block_forward，得到更新后的图像与文本隐藏状态
            image_hidden_states, text_hidden_states = block_forward(**block_kwargs)

    # --------------- 将文本与图像隐藏状态合并后传入 single-stream 的 transformer blocks ---------------
    all_hidden_states = [*text_hidden_states, *image_hidden_states]
    for block in self.single_transformer_blocks:
        # 为 single block 准备参数
        block_kwargs = {
            "self": block,
            "hidden_states": all_hidden_states,
            "tembs": tembs,
            "position_embs": position_embs,
            "adapters": adapters,
            "attn_forward": attn_forward,
            **kwargs,
        }
        # 同样支持 gradient checkpointing 的分支化执行
        if self.training and self.gradient_checkpointing:
            all_hidden_states = torch.utils.checkpoint.checkpoint(
                single_block_forward, **block_kwargs, **gckpt_kwargs
            )
        else:
            # 普通前向：single_block_forward 会返回更新后的 all_hidden_states
            all_hidden_states = single_block_forward(**block_kwargs)

    # 对合并后的 image_hidden_states（在 all_hidden_states 中位置为 txt_n）做最后的归一化处理
    image_hidden_states = self.norm_out(all_hidden_states[txt_n], tembs[txt_n])
    # 通过 proj_out 投影得到最终的输出（通常是 decoder/后续模块的输入）
    output = self.proj_out(image_hidden_states)

    # 返回单元素元组 (output,) 保持与管线接口一致（便于上层统一解包）
    return (output,)


# 装饰器：在该函数内部禁用梯度计算，节省显存与计算；生成推理阶段通常不需要反向传播
@torch.no_grad()
# 定义图像生成主函数，封装了从文本编码、条件融合、时序调度到VAE解码的一整套流程
def generate(
    # 管道对象，来自 diffusers 的 FluxPipeline，封装了 tokenizer、文本编码器、VAE、Scheduler、Transformer 等
    pipeline: FluxPipeline,
    # 文本提示，可以是单个字符串或字符串列表（批量）
    prompt: Union[str, List[str]] = None,
    # 第二路文本提示（部分模型支持双编码器或双路prompt，用于风格/结构差异化条件）
    prompt_2: Optional[Union[str, List[str]]] = None,
    # 目标输出高度（像素）；若为 None，会根据默认采样尺寸与缩放因子推断
    height: Optional[int] = 512,
    # 目标输出宽度（像素）；同上
    width: Optional[int] = 512,
    # 采样迭代步数（扩散/流式变换的离散步数），步数越多通常越精细但更慢
    num_inference_steps: int = 28,
    # 可选：自定义时间步数组；若提供，则覆盖默认调度器的时间步生成
    timesteps: List[int] = None,
    # 文本指导强度（类似 classifier-free guidance，但本实现基于 transformer 的 guidance_embeds 设计）
    guidance_scale: float = 3.5,
    # 每个 prompt 生成的图像数量（批量展开）
    num_images_per_prompt: Optional[int] = 1,
    # 随机数生成器（可传入单个或列表以实现不同样本的可复现性）
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    # 初始潜变量（latents）；若提供则直接用它作为起点，实现重采样/延续生成
    latents: Optional[torch.FloatTensor] = None,
    # 预编码的文本嵌入（跳过encode_prompt的计算）；适合自定义文本编码器或复用缓存
    prompt_embeds: Optional[torch.FloatTensor] = None,
    # 池化后的文本嵌入（通常来自 CLS/pooled 输出），供时间-文本联合嵌入使用
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    # 输出类型："pil" 返回 PIL.Image；"pt" 返回 torch 张量；"numpy" 返回 NumPy；"latent" 返回潜空间
    output_type: Optional[str] = "pil",
    # 是否返回字典形式的管道输出（与 diffusers 统一接口保持一致）
    return_dict: bool = True,
    # 传入给注意力层的联合参数（例如 attention mask、scale 等），用于微调注意力行为
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    # 步进回调：用于在每个时间步结束时观察或修改中间状态（如可视化、日志、early stopping）
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    # 回调函数需要从局部变量带出的张量名列表（如 "latents"），以避免复制无关数据
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    # 文本序列最大长度，避免越界；超长会在 encode_prompt 内部做截断/滑窗
    max_sequence_length: int = 512,
    # -------------------- 条件控制相关（可选） --------------------
    # 主适配器名称列表（通常对应 LoRA adapter 名称），用于指定各分支使用哪个 LoRA
    main_adapter: Optional[List[str]] = None,
    # 条件列表（自定义 Condition 对象），支持多种图像/边缘/深度等条件分支
    conditions: List[Condition] = [],
    # 图像指导强度（条件分支的无条件预测与条件预测的插值系数，类似 CFG on image）
    image_guidance_scale: float = 1.0,
    # 传递给 transformer 前向的额外关键字参数（例如缓存、分组掩码等）
    transformer_kwargs: Optional[Dict[str, Any]] = {},
    # 是否启用 KV-Cache（仅在多步共享 K/V 时能加速后续步；需配合自定义 attn_forward）
    kv_cache=False,
    # 潜变量掩码：用于只生成/替换潜空间中的部分区域（如局部编辑、拼接）
    latent_mask=None,
    mogle=None,
    use_mogle=False,
    # 额外参数字典（向后兼容，接收未声明的可选项）
    **params: dict,
):
    # 为与 diffusers 的 Pipeline 调用风格保持一致，将 pipeline 绑定到局部 self
    self = pipeline

    # 若未显式指定高宽，则依据默认采样尺寸（token 网格尺寸）与 VAE 缩放因子推断像素尺寸
    height = height or self.default_sample_size * self.vae_scale_factor
    # 同上，推断宽度
    width = width or self.default_sample_size * self.vae_scale_factor

    # 输入检查：校验 prompt / prompt_2 / 尺寸 / 预编码向量 / 回调输入名 / 最大序列长度等是否合法
    # 不合法会抛出明确错误，避免在深层计算中才报错
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    # 记录 guidance scale 到管道实例上，供后续模块读取（如 transformer 的 guidance_embeds）
    self._guidance_scale = guidance_scale
    # 记录全局注意力参数（可能被下游 attention 层引用）
    self._joint_attention_kwargs = joint_attention_kwargs

    # -------------------- 计算 batch 大小 --------------------
    # 若传入了字符串 prompt，则 batch 为 1
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    # 若传入了 prompt 列表，则 batch 为列表长度
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    # 否则走 embedding 模式（已给 prompt_embeds），以其第一维作为 batch
    else:
        batch_size = prompt_embeds.shape[0]

    # 选择执行设备（CPU/GPU/MPS），由 Pipeline 事先设置好
    device = self._execution_device

    # -------------------- 文本编码：获取 prompt 的三类输出 --------------------
    # prompt_embeds: token 级别的文本特征，供 transformer 上下文分支使用
    # pooled_prompt_embeds: 句级/池化后的文本特征，常用于与时间步/指导联合编码
    # text_ids: 文本 token 的位置/索引信息，供位置编码或对齐使用
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # -------------------- 准备潜变量（latents）与其对应的图像 token id --------------------
    # 模型 in_channels 表示 latent 通道数的四倍（因为每个 token 可能打包多个通道），此处除以 4 得到实际 latent 通道
    num_channels_latents = self.transformer.config.in_channels // 4
    # 组装（或采样）初始潜变量，并返回每个 latent token 的二维网格 id（行/列）+ batch 索引等
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 若提供了 latent_mask（布尔/索引掩码），只保留被掩码选中的潜 token，实现"局部生成/编辑"
    if latent_mask is not None:
        # 将 HxW 掩码转为一维 token 掩码（与打包后的 token 排列一致）
        latent_mask = latent_mask.T.reshape(-1)
        # 选择被掩码的位置的潜变量
        latents = latents[:, latent_mask]
        # 同步筛选对应 token 的 id（空间坐标）
        latent_image_ids = latent_image_ids[latent_mask]

    # -------------------- 处理条件分支（如深度、Canny、灰度等） --------------------
    # c_latents：条件分支的潜 tokens；uc_latents：无条件（空条件）潜 tokens（用于 image CFG）
    # c_ids：条件分支的 token 空间索引；c_timesteps：每个分支对应时间步（通常为 0）
    c_latents, uc_latents, c_ids, c_timesteps = ([], [], [], [])
    # c_projections：每分支的 pooled 文本投影；c_guidances：每分支的 guidance 值；c_adapters：每分支使用的 LoRA adapter
    c_projections, c_guidances, c_adapters = ([], [], [])
    # complement_cond：可选的"补片"条件（is_complement=True），用于在末尾把两者潜空间拼接合成
    complement_cond = None
    # 遍历用户传入的 Condition 列表，将其编码为潜 tokens 与空间 ids
    for condition in conditions:
        # 编码当前条件（encode 内部会用 VAE 将图像编码为 latent tokens，并生成对应的空间 ids）
        tokens, ids = condition.encode(self)
        # 保存条件 tokens（形状：[B, token数, token维度]）
        c_latents.append(tokens)  # [batch_size, token_n, token_dim]
        # 若启用了图像 guidance（image CFG），则还需要一个"空条件"的对照分支用于插值
        # empty=True 时 encode 会用纯黑图像作为条件，得到"无条件"版本
        if image_guidance_scale != 1.0:
            uc_latents.append(condition.encode(self, empty=True)[0])
        # 保存空间 ids（形状：[token数, 3]，通常为 [层索引/批次?, 行, 列]）
        c_ids.append(ids)  # [token_n, id_dim(3)]
        # 条件分支时间步默认置 0（与主图像分支的时间步不同，这里主要用于统一接口）
        c_timesteps.append(torch.zeros([1], device=device))
        # 每个条件分支共享相同的 pooled 文本嵌入（也可扩展为不同文本控制）
        c_projections.append(pooled_prompt_embeds)
        # guidance 默认为 1（若模型支持可单独调参）
        c_guidances.append(torch.ones([1], device=device))
        # 指定该条件分支使用的 LoRA 适配器（adapter 名称或配置）
        c_adapters.append(condition.adapter)
        # 若该条件被标记为补片（complement），在后处理阶段会将其与主结果在潜空间维度进行覆盖/合成
        # 参考 OminiControl2 文章提出的 token-level 融合策略
        # See the token integration of OminiControl2 [https://arxiv.org/abs/2503.08280]
        if condition.is_complement:
            complement_cond = (tokens, ids)
    thermal_latents_packed = c_latents[0]  # [1, 256, 64]
    thermal_ids = c_ids[0]  # [256, 3]
    # -------------------- 准备时间步（调度） --------------------
    # 构建 sigma 序列（这里用线性从 1.0 到 1/步数），具体用法由 retrieve_timesteps 结合 mu 做校正
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    # 图像 token 序列长度（H*W after pack）；用于计算 shift（mu），实现长序列平移策略
    image_seq_len = latents.shape[1]
    # 根据当前图像 token 序列长度与调度器配置，计算 shift 值 mu，以适配可变分辨率/序列长度
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    # 结合调度器、期望步数、设备、可选自定义时间步、sigma、以及平移 mu，最终得到实际的时间步序列
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
    )
    # 计算 warmup 步数：在部分多阶调度器中，前若干步不进行图像更新（或按 order 聚合更新）
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    # 记录总步数，供外部或回调读取
    self._num_timesteps = len(timesteps)

    # -------------------- KV-Cache 初始化（可选加速） --------------------
    # 当启用 kv_cache 时，会为每个 Attention 层分配一个 cache_idx，并分别维护条件/无条件两套缓存
    if kv_cache:
        # 统计 transformer 中 Attention 层的数量，以便为每层分配缓存槽位
        attn_counter = 0
        for module in self.transformer.modules():
            if isinstance(module, Attention):
                # 为每个注意力层打上 cache_idx，便于在自定义 attn_forward 中索引缓存
                setattr(module, "cache_idx", attn_counter)
                attn_counter += 1
        # 条件分支的 KV 缓存（每层两列表：keys 列表与 values 列表）
        kv_cond = [[[], []] for _ in range(attn_counter)]
        # 无条件分支的 KV 缓存
        kv_uncond = [[[], []] for _ in range(attn_counter)]

        # 清空缓存的帮助函数：在"写入"模式开始前调用，确保没有脏数据
        def clear_cache():
            for storage in [kv_cond, kv_uncond]:
                for keys, values in storage:
                    keys.clear()
                    values.clear()

    # -------------------- 构建注意力分组掩码（控制跨分支注意力） --------------------
    # 分支数量 = 文本分支(1) + 图像主分支(1) + 条件分支数
    branch_n = len(conditions) + 2
    # 初始化为全 True（允许所有分支互相注意）
    group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool)
    # 禁止不同条件分支之间互相注意：仅允许每个条件分支自注意（对角线为 True，其余为 False）
    group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
    # 若启用了 KV 缓存，则禁止条件分支去注意文本/主图像分支（减少干扰，便于 cache 复用）
    if kv_cache:
        group_mask[2:, :2] = False

    # -------------------- 获取各模型的dtype和device --------------------
    mogle_dtype = next(mogle.parameters()).dtype
    mogle_device = next(mogle.parameters()).device
    transformer_dtype = next(self.transformer.parameters()).dtype
    transformer_device = next(self.transformer.parameters()).device

    # -------------------- 主去噪循环（推理核心） --------------------
    # 使用进度条显示生成进度
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        # 遍历每一个调度时间步
        for i, t in enumerate(timesteps):
            # ===== 核心修复：分别为mogle和transformer准备张量 =====
            # 保存原始dtype用于后续恢复（用于scheduler）
            backup_type = latents.dtype
            
            # ===== 第一步：为mogle准备张量（转到mogle的dtype和device） =====
            thermal_latents_mogle = thermal_latents_packed.to(device=mogle_device, dtype=mogle_dtype)
            latents_mogle = latents.to(device=mogle_device, dtype=mogle_dtype)
            t_expanded = t.expand(thermal_latents_mogle.shape[0]).to(device=mogle_device, dtype=mogle_dtype)
            
            
            thermal_condition_processed = mogle.forward(
                thermal_latents_mogle,
                noise_latent=latents_mogle,
                timestep=t_expanded
            )
            # ===== 第二步：为transformer准备张量（转到transformer的dtype和device） =====
            # 将所有张量转换到transformer的dtype和device
            latents = latents.to(device=transformer_device, dtype=transformer_dtype)
            cur_condition_latents = thermal_condition_processed.to(device=transformer_device, dtype=transformer_dtype)
            prompt_embeds = prompt_embeds.to(device=transformer_device, dtype=transformer_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device=transformer_device, dtype=transformer_dtype)
            latent_image_ids = latent_image_ids.to(device=transformer_device)
            text_ids = text_ids.to(device=transformer_device)
            thermal_ids = thermal_ids.to(device=transformer_device)
            
            # 转换时间步到transformer的dtype和device
            timestep = t.expand(latents.shape[0]).to(device=transformer_device, dtype=transformer_dtype) / 1000
            # 处理 guidance：若模型配置支持 guidance_embeds，则构造 guidance 张量并广播到 batch
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=transformer_device, dtype=transformer_dtype)
                guidance = guidance.expand(latents.shape[0])
            else:
                # 否则关闭 guidance，同时对条件分支的 guidance 也统一置 None（与接口对齐）
                guidance, c_guidances = None, [None for _ in c_guidances]

            # 若启用 KV 缓存：第一步写入缓存，其余步骤读取缓存
            if kv_cache:
                mode = "write" if i == 0 else "read"
                if mode == "write":
                    clear_cache()
            # 仅在写入阶段实际将条件分支并入 transformer；读取阶段复用缓存，不再重复算条件分支
            use_cond = not (kv_cache) or mode == "write"

            # -------------------- 计算条件版噪声预测（含文本 + 主图像 + 条件分支） --------------------
            noise_pred = transformer_forward(
                # 传入 transformer 主体
                self.transformer,
                # image_features：第一个是主 latents，其后是条件分支的潜 tokens（写入阶段才带入）
                image_features=[latents] + ([cur_condition_latents] if use_cond else []),
                # text_features：文本嵌入（通常只有一条文本分支）
                text_features=[prompt_embeds],
                # 图像 token 的空间 id：主 + 条件（写入阶段才带入）
                img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                # 文本 token 的 id（用于位置编码/对齐）
                txt_ids=[text_ids],
                # 为每个分支准备对应的时间步（文本与图像主分支共享相同时间步；条件分支通常传 0 或同值）
                timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                # 每个分支的 pooled 文本特征（文本/主图像各一份，条件分支也各一份）
                pooled_projections=[pooled_prompt_embeds] * 2 + (c_projections if use_cond else []),
                # guidance 向量（若启用 guidance_embeds，则传相同标量扩展到各分支）
                guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                # 与 diffusers 兼容：返回 tuple 而非 dict
                return_dict=False,
                # 为各分支指定其 LoRA 适配器（文本与主图像分支用 main_adapter，条件分支用各自 adapter）
                adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                # KV 缓存模式（写入或读取），仅在启用 kv_cache 时有效
                cache_mode=mode if kv_cache else None,
                # 对应的缓存存储（条件版缓存）
                cache_storage=kv_cond if kv_cache else None,
                # 指示哪些分支需要写入缓存：只缓存条件分支（主图像/文本不缓存）
                to_cache=[False, False, *[True] * len(c_latents)],
                # 跨分支注意力掩码，控制条件分支的注意力连通性
                group_mask=group_mask,
                # 透传额外 transformer 参数（如自定义 attn_forward 行为）
                **transformer_kwargs,
            )[0]

            # -------------------- 图像级 Guidance（Image CFG）：用无条件条件分支做插值 --------------------
            if image_guidance_scale != 1.0:
                # 计算无条件条件分支（uc_latents）对应的噪声预测
                unc_pred = transformer_forward(
                    self.transformer,
                    image_features=[latents] + (uc_latents if use_cond else []),
                    text_features=[prompt_embeds],
                    img_ids=[latent_image_ids] + (c_ids if use_cond else []),
                    txt_ids=[text_ids],
                    timesteps=[timestep, timestep] + (c_timesteps if use_cond else []),
                    pooled_projections=[pooled_prompt_embeds] * 2 + (c_projections if use_cond else []),
                    guidances=[guidance] * 2 + (c_guidances if use_cond else []),
                    return_dict=False,
                    adapters=[main_adapter] * 2 + (c_adapters if use_cond else []),
                    cache_mode=mode if kv_cache else None,
                    cache_storage=kv_uncond if kv_cache else None,
                    to_cache=[False, False, *[True] * len(c_latents)],
                    **transformer_kwargs,
                )[0]

                # 按 image_guidance_scale 做插值：unc_pred + s*(cond_pred - unc_pred)
                # s>1 增强条件约束，s=1 等效关闭，s<1 弱化条件
                noise_pred = unc_pred + image_guidance_scale * (noise_pred - unc_pred)
            # -------------------- 调度器一步更新：从 x_t 预测到 x_{t-1} --------------------
            # 将latents转回原始dtype以供scheduler处理
            latents = latents.to(dtype=backup_type)
            noise_pred = noise_pred.to(dtype=backup_type)
            
            # 保存当前数据类型，部分设备在 step 后 dtype 可能变化
            latents_dtype = latents.dtype
            # 调度器根据预测噪声与当前时间步对潜变量进行更新（核心去噪/流式推进步骤）
            latents = self.scheduler.step(noise_pred, t, latents)[0]

            # 针对特定平台（如 Apple MPS）存在 dtype 漂移问题，必要时将其拉回原 dtype
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # 参考已知 PyTorch issue：https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            # -------------------- 步进回调（可选）：提供对中间结果的观察与修改通道 --------------------
            if callback_on_step_end is not None:
                # 准备传给回调的关键张量，避免传递整个 locals 造成不必要的开销
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                # 执行回调，可能返回更新后的 latents / prompt_embeds 等，以支持交互式或自适应采样
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                # 允许回调覆盖 latents 或 prompt_embeds（如果回调没有提供，则用原值）
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # -------------------- 更新进度条（按调度器阶数控制可见步） --------------------
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    # -------------------- 潜变量掩码后处理（补片条件拼接） --------------------
    if latent_mask is not None:
        # 若开启了 latent_mask，通常意味着只生成了局部；此处将生成结果与 complement 条件进行合并
        # 需要 complement_cond 存在（包含补片 tokens 与其 ids）
        assert complement_cond is not None
        comp_latent, comp_ids = complement_cond
        # 汇总主潜 ids 与补片 ids，用于计算完整画布的网格尺寸
        all_ids = torch.cat([latent_image_ids, comp_ids], dim=0)  # (Ta+Tc,3)
        # 取各维最大索引 + 1 得到网格尺寸（包含行列尺寸），转为 long
        shape = (all_ids.max(dim=0).values + 1).to(torch.long)  # (3,)
        # H, W 分别为网格的高宽
        H, W = shape[1].item(), shape[2].item()
        # B 为 batch，C 为每个 token 的通道维（embedding 维）
        B, _, C = latents.shape
        # 创建一个空画布（B, H*W, C），用以就地写入主潜与补片潜
        canvas = latents.new_zeros(B, H * W, C)  # (B,H*W,C)

        # 内部函数：根据 ids 的（行、列）位置，将 tokens 写入到画布对应的扁平索引位置
        def _stash(canvas, tokens, ids, H, W) -> None:
            B, T, C = tokens.shape
            ids = ids.to(torch.long)
            # 将二维网格 (row, col) 转为一维下标 flat_idx = row*W + col
            flat_idx = (ids[:, 1] * W + ids[:, 2]).to(torch.long)
            # 采用 index_copy_ 按索引写入，提高效率
            canvas.view(B, -1, C).index_copy_(1, flat_idx, tokens)

        # 先写入主潜变量，再写入补片；后写入的会覆盖相同位置，实现"局部替换/融合"
        _stash(canvas, latents, latent_image_ids, H, W)
        _stash(canvas, comp_latent, comp_ids, H, W)
        # 合并后的潜变量作为新的 latents 输出
        latents = canvas.view(B, H * W, C)

    # -------------------- 输出后处理：决定返回潜变量还是解码为图像 --------------------
    if output_type == "latent":
        # 直接返回潜空间结果（供外部自定义解码或跨步骤处理）
        image = latents
    else:
        # 将 token 打包的潜变量还原为 (B, C, H, W) 的空间布局
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        # 反向应用 VAE 的缩放与偏移，复原到 VAE 解码输入的标度
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        # 通过 VAE 解码得到图像张量（范围与类型由 VAE 与 image_processor 决定）
        image = self.vae.decode(latents, return_dict=False)[0]
        # 统一后处理：转为 PIL / numpy / torch，以及可能的去归一化、裁切等
        image = self.image_processor.postprocess(image, output_type=output_type)

    # 资源回收：根据 hooks 策略卸载/挪走部分模型到 CPU 或释放显存，降低峰值占用
    self.maybe_free_model_hooks()

    # 与 diffusers 习惯一致：当不需要字典包装时，直接返回 tuple
    if not return_dict:
        return (image,)

    # 默认返回标准化的 Pipeline 输出对象，包含 images 字段（可扩展别的元信息）
    return FluxPipelineOutput(images=image)