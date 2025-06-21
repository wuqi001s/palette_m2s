"""
1. Download model and save the model to git_root/model/celebahq/200_Network.pth
2. Modify inpainting_celebahq.json
    ["path"]["resume_state"]: "model/celebahq/200"
    ["datasets"]["test"]["args"]["data_root"]: "<Folder Constains Inference Images>"

    (optinally) change ["model"]["which_networks"]["args"]["beta_schedule"]["test"]["n_timestep"] value to reduce # steps inference should take
                more steps yields better results
3. Modify in your particular case in this code:
    model_pth = "<PATH-TO-MODEL>/200_Network.pth"
    input_image_pth = "<PATH-TO-DATASET_PARENT_DIT>/02323.jpg"
5. Run inpainting code (assume save this code to git_root/inference/inpainting.py)
    cd inference
    python inpainting.py -c ../config/inpainting_celebahq.json -p test
"""
# 实现第一个配置
# 先用 64x64的模型进行推理
# 然后用64x256的模型进行推理
# 具体流程：
    # 1. 读取输入图像，将输入图像下采样为64x64的大小
    # 2. 从图片中读取掩码，将掩码转换为64×64和256x256的大小
#    3. 使用64x64的模型进行推理，生成图像
#    4. 对图像进行上采样，将图像从64x64上采样到256x256
#    5. 使用256x256的模型进行推理，生成最终图像
# 6. 保存最终生成的图像
import os
import sys

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换工作目录
os.chdir(script_dir)

# # 将项目目录加入PATH（可选）
# sys.path.insert(0, script_dir)

print(f"当前工作目录: {os.getcwd()}")

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

set_deterministic(42)  # 在代码最开头调用
model_pth_64 = "/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/save_models/200_Network_ema.pth"
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

# initializa model
model_64 = Network(**model_args)
state_dict_64 = torch.load(model_pth_64)
model_64.load_state_dict(state_dict_64, strict=False)


model_256 = Network(**model_args)
state_dict_256 = torch.load(model_pth_256)
model_256.load_state_dict(state_dict_256, strict=False)

# 1, 首先使用64x64的模型进行推理
device = torch.device('cuda:0')
model_64.to(device)
model_64.set_new_noise_schedule(phase='test')
model_64.eval()
print(model_64)

# 读取输入图像

def preprocess_image(path,size=64):
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
    gaussain_sigma = 5 
    indicies = np.argwhere(antenna_mask==max_value)[0]
    print(indicies)
    x_norm, y_norm = indicies[0] , indicies[1]
    heatmap = generate_heatmap(size, size, x_norm, y_norm, gaussain_sigma) #添加高斯热图
    cond_image_0 = img_tensor*(1. - mask) + mask*torch.randn_like(img_tensor)  
    cond_image = cond_image_0 + heatmap   # 采样时必须要加heatmap,否则基站位置会漂移
    return img_tensor, mask, cond_image
img, mask, cond_image = preprocess_image(input_image_pth, size=64)


# save conditional image used a inference input
cond_image_np = cond_image.detach().float().cpu().numpy()
cond_image_np = (cond_image_np* 255).astype(np.uint8)  # 从[-1,1]转回[0,255]
cond_image_np = np.transpose(cond_image_np, (1, 2, 0))  # 转换为(H, W, C)格式
Image.fromarray(cond_image_np).save("./result/cond_image.jpg",quality=100)

mask_np = mask.detach().float().cpu().numpy()
mask_np = mask_np.squeeze()  # 去掉batch维度
mask_np = (mask_np * 255).astype(np.uint8)  # 从[0,1]转回[0,255]
Image.fromarray(mask_np, mode='L').save("./result/mask.jpg",quality=100)
# set device
cond_image = set_device(cond_image)
gt_image = set_device(img)
mask = set_device(mask)

# unsqueeze
cond_image = cond_image.unsqueeze(0).to(device)
gt_image = gt_image.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

# inference
with torch.no_grad():
    output, visuals = model_64.restoration(cond_image, y_t=cond_image,
                                        y_0=gt_image, mask=mask, sample_num=8,sampling_method="ddim")

# save intermediate processes
output_img = output.detach().float().cpu()
for i in range(visuals.shape[0]):
    img = tensor2img(visuals[i].detach().float().cpu())
    Image.fromarray(img).save(f"./result/process_{i}.jpg")

# save output (output should be the same as last process_{i}.jpg)
img = tensor2img(output_img)
Image.fromarray(img).save("./result/output.jpg")
