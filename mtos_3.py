
# 实现第一个配置
# 先用 64x64的模型进行推理
# 然后用256x256的模型进行推理
# 具体流程：
    # 1. 读取输入图像，将输入图像下采样为64x64的大小
    # 2. 从图片中读取掩码，将掩码转换为64×64和256x256的大小
#    3. 使用64x64的模型进行推理，生成图像
#    4. 对图像进行上采样，将图像从64x64上采样到256x256
#    5. 使用256x256的模型进行推理，生成最终图像
# 6. 保存最终生成的图像

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import numpy as np
import core.praser as Praser
import torch
from core.util import set_device, tensor2img
from data.util.mask import get_irregular_mask
from models.network import Network
from PIL import Image
from torchvision import transforms
import random
def set_deterministic(seed=42):
    """设置所有随机种子和确定性配置"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic(42)  
model_pth_64 = "/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/save_models/200_Network.pth"
model_pth_256 = "/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/save_models/49_Network.pth"

input_image_pth = "/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/g_image/GT_682_34.png"


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/config/inpainting_places2.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str,
                        choices=['train', 'test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int,
                        default=16, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)
    return opt

def generate_heatmap(H, W, x_norm, y_norm, sigma):  #生成高斯热图.....
    # 生成网格坐标
    y_indices, x_indices = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='xy'  #这里要用xy,不能用ij....
    )
    heatmap = torch.exp(
        -((x_indices - x_norm)**2 + (y_indices - y_norm)**2) / (2 * sigma**2)
    )
    # 归一化到 [0, 1]
    heatmap = heatmap / heatmap.max()
    return heatmap  # 形状 (H, W)
# config arg
opt = parse_config()
model_args = opt["model"]["which_networks"][0]["args"]

# 1. initializa model
model_64 = Network(**model_args)
state_dict_64 = torch.load(model_pth_64)
model_64.load_state_dict(state_dict_64, strict=False)


model_256 = Network(**model_args)
state_dict_256 = torch.load(model_pth_256)
model_256.load_state_dict(state_dict_256, strict=False)


# 转换回可显示的格式
def tensor_to_img(tensor, is_mask=False):
    if is_mask:
        return tensor.squeeze().numpy()
    else:
        # 反归一化并转换为numpy数组
        tensor = tensor * 0.5 + 0.5  # 从[-1,1]转回[0,1]
        return tensor.permute(1, 2, 0).numpy()
## 2. 图像生成
# read input and create random mask
def preprocess_image(path,size=64,gaussian_sigma=5):
    img_pillow = Image.open(path).convert('RGB')
    img_pillow = img_pillow.resize((size, size), Image.BICUBIC)  # 下采样
    tfs = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = tfs(img_pillow)  # 转换为张量并预处理
    mask = np.array(img_pillow.convert("L"))  # 将掩码图像转换为灰度图
    antenna_mask = mask.copy()
    max_value = antenna_mask.max()
    mask[mask==max_value] =0  
    mask[mask > 0] = 1 #1为掩码区
    mask= torch.tensor(np.expand_dims(mask, axis=0))  # 添加新维度，变为 (1, H, W)  #mask 是图片的掩码。掩码区域为1.。。...
    gaussain_sigma = gaussian_sigma
    indicies = np.argwhere(antenna_mask==max_value)[0]
    print(indicies)
    x_norm, y_norm = indicies[0] , indicies[1]
    heatmap = generate_heatmap(size, size, x_norm, y_norm, gaussain_sigma) #添加高斯热图
    cond_image_0 = img_tensor*(1. - mask) + mask*torch.randn_like(img_tensor)  
    cond_image = cond_image_0 + heatmap   # 采样时必须要加heatmap,否则基站位置会漂移

    save_debug_images(img_tensor, mask, cond_image, size)
    return img_tensor, mask, cond_image
def denoise_image(model, cond_image,y_t=None,y_0=None, mask=None, sample_num=8, sampling_method="ddim"):
    """
    使用模型对图像进行去噪
    :param model: 扩散模型
    :param cond_image: 条件图像
    :param gt_image: 真实图像
    :param mask: 掩码
    :param sample_num: 采样数量
    :param sampling_method: 采样方法
    :return: 去噪后的图像
    """
    device = torch.device('cuda:0')
    model.to(device)
    model.set_new_noise_schedule(phase='test')
    model.eval()
    # set device
    cond_image = set_device(cond_image)
    y_0 = set_device(y_0)
    mask = set_device(mask)
    y_t = set_device(y_t) if y_t is not None else None
    # unsqueeze
    cond_image = cond_image.unsqueeze(0).to(device)
    y_0 = y_0.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    y_t = y_t.unsqueeze(0).to(device)


    with torch.no_grad():
        output, visuals = model.restoration(y_cond=cond_image, y_t=y_t,
                                           y_0=y_0, mask=mask, sample_num=sample_num,
                                           sampling_method=sampling_method)
    
    return output, visuals

def save_debug_images(img_tensor, mask_tensor, cond_image, size):
    """
    保存调试图像
    :param img_tensor: 原始图像张量
    :param mask_tensor: 掩码张量
    :param cond_image: 条件图像张量
    :param size: 图像尺寸
    """
    # 创建结果目录
    import os
    os.makedirs("./result_m2s", exist_ok=True)
    
    # 保存原始图像
    save_tensor_as_image(img_tensor, f"./result_m2s/original_{size}.jpg")
    
    # 保存掩码
    save_mask_as_image(mask_tensor, f"./result_m2s/mask_{size}.jpg")
    
    # 保存条件图像
    save_tensor_as_image(cond_image, f"./result_m2s/cond_image_{size}.jpg")

def save_tensor_as_image(tensor, path):
    """
    保存张量为图像文件
    :param tensor: 输入张量 (C, H, W)
    :param path: 保存路径
    """
    # 复制张量并移动到CPU
    tensor = tensor.detach().clone().cpu()
    
    # 反归一化: [-1, 1] -> [0, 1]
    tensor = tensor * 0.5 + 0.5
    
    # 转换为numpy数组并调整通道顺序
    img_np = tensor.permute(1, 2, 0).numpy()
    
    # 转换为0-255范围
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    
    # 保存图像
    Image.fromarray(img_np).save(path, quality=100)

def save_mask_as_image(mask, path):
    """
    保存掩码为图像文件
    :param mask: 掩码张量 (1, H, W)
    :param path: 保存路径
    """
    # 复制张量并移动到CPU
    mask = mask.detach().clone().cpu()
    
    # 转换为numpy数组
    mask_np = mask.squeeze(0).numpy()  # 移除通道维度
    
    # 转换为0-255范围
    mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
    
    # 保存图像
    Image.fromarray(mask_np).save(path, quality=100)

def add_denoise(model,loop = 10,step=10,fir_step=15,denoise_steps=10,
                y_cond=None,y_t=None,y_0=None, mask=None, 
                sample_num=8, sampling_method="m2s"):
    """denoise block
    添加去噪音
    :param model: 扩散模型
    :param loop: 去噪音次数
    :return: 去噪后的图像
    """
    if y_cond.shape[-1]==64:
        T = 250
    elif y_cond.shape[-1]==256:
        T=75
    device = torch.device('cuda:0')
    model.to(device)
    model.set_new_noise_schedule(phase='test')
    model.eval()
    # set device
    y_cond = set_device(y_cond)
    y_0 = set_device(y_0)
    mask = set_device(mask)
    y_t = set_device(y_t) if y_t is not None else None
    # unsqueeze
    y_cond = y_cond.unsqueeze(0).to(device)
    y_0 = y_0.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    y_t = y_t.unsqueeze(0).to(device)
    output = y_t  
    for k in range(int(T/fir_step)):
        with torch.no_grad():
            output = model.restoration(y_cond=y_cond, y_t=output,
                                            y_0=y_0, mask=mask, sample_num=sample_num,
                                            sampling_method=sampling_method,m2s_steps=fir_step,
                                            ) # # m2s_steps是第一次去噪的步数
        
        for j in range(loop):
            with torch.no_grad():
                y_t = model.m2s_add_noise(output, noise=None, t=step)  # 添加噪声

                output = model.restoration(y_cond=y_cond, y_t=y_t,
                                                y_0=y_0, mask=mask, sample_num=sample_num,
                                                sampling_method=sampling_method,
                                                m2s_steps=step,)

    return output.squeeze()  # 去掉batch维度
img_64, mask_64, cond_image_64 = preprocess_image(input_image_pth, size=64,gaussian_sigma=5)

# 3. 使用64x64的模型进行推理
# 改变一下去噪音步数
output, visuals = denoise_image(model_64, cond_image=cond_image_64,y_t=cond_image_64, y_0=img_64, mask=mask_64, sample_num=8, sampling_method="ddim")

# output = add_denoise(model_64,loop=8,step=10,fir_step=200,y_cond=cond_image_64,
#                      y_t=cond_image_64, y_0=img_64, mask=mask_64, sample_num=1, 
#                      sampling_method="m2s")
print("output.shape:", output.shape)  # 输出形状应为 (1, 3, 64, 64)
output = output.squeeze()   # 去掉batch维度
save_tensor_as_image(output, "./result_m2s/output_64.jpg")  # 保存64x64的输出图像

# 4. 对图像进行上采样，将图像从64x64上采样到256x256
upsample = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)
upsampled_output = upsample(output)  # 上采样到256x256，tensor
#保存上采样后的图像
print("upsampled output.shape",upsampled_output.shape)  # 输出形状应为 (3, 256, 256)
save_tensor_as_image(upsampled_output, "./result_m2s/upsampled_output.jpg")  # 保存上采样后的图像

img_256,mask_256, cond_image_256 = preprocess_image(input_image_pth, size=256)  # 读取256x256的图像和掩码

output_256 = add_denoise(model_256,loop=10,step=10,y_cond=cond_image_256,
                     y_t=upsampled_output, y_0=img_256, mask=mask_256, sample_num=1, 
                     sampling_method="m2s",fir_step=10,)
# 保存256x256的输出图像
save_tensor_as_image(output_256, "./result_m2s/output_256.jpg")  # 保存256x256的输出图像