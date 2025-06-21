import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.guided_diffusion_modules.unet import UNet

from models.guided_diffusion_modules.gaussian_diffusion import GaussianDiffusion

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载配置（以 Celeba-HQ 图像修复为例）
config = {
    "diffusion": {
        "timesteps": 1000,
        "beta_schedule": "linear",
        "linear_start": 1e-4,
        "linear_end": 2e-2,
        "objective": "pred_x0"
    },
    "model": {
        "in_channels": 6,  # 输入通道 (3 图像 + 3 掩码)
        "out_channels": 3,  # 输出通道
        "model_channels": 128,
        "channel_mult": [1, 2, 2, 4],
        "attention_resolutions": [8, 16],
        "num_res_blocks": 2,
        "dropout": 0.1,
        "resamp_with_conv": True
    }
}

# 创建模型
model = UNet(**config["model"]).to(device)
diffusion = GaussianDiffusion(**config["diffusion"])

# 加载预训练权重
checkpoint = torch.load("/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/save_models/49_Network.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("模型加载成功！")

# 准备输入图像和掩码（示例：中心掩码）
def prepare_input(image_path, mask=None, image_size=256):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # 创建中心掩码（示例）
    if mask is None:
        mask = torch.ones_like(image)
        h, w = image_size, image_size
        mask[:, :, h//4:3*h//4, w//4:3*w//4] = 0  # 中心区域为 0（待修复）
    
    # 合并图像和掩码作为模型输入
    model_input = torch.cat([image, mask], dim=1).to(device)
    return image.to(device), mask.to(device), model_input

# 采样函数
@torch.no_grad()
def sample(model, diffusion, model_input, mask, steps=250):
    # 使用 DDIM 采样（速度更快）
    sample_fn = diffusion.ddim_sample_loop
    samples = sample_fn(
        model=model,
        shape=(1, 3, 256, 256),
        noise=None,
        clip_denoised=True,
        model_kwargs={"x_mod": model_input},
        progress=True,
        eta=0.0,  # 0 表示确定性采样
        steps=steps  # 减少采样步数以加速
    )
    
    # 结合原始图像和生成结果（仅修复掩码区域）
    reconstructed = samples * (1 - mask) + model_input[:, :3] * mask
    return reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy()

# 运行采样
image_path = "path/to/your/image.jpg"  # 替换为你的图像路径
image, mask, model_input = prepare_input(image_path)
result = sample(model, diffusion, model_input, mask)

# 保存结果
plt.imsave("reconstructed.png", result)
print("采样完成，结果已保存！")