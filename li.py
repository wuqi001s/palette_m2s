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
model_pth = "/home/y/project/dm/Palette-Image-to-Image-Diffusion-Models/save_models/49_Network.pth"
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
    
    # 计算高斯响应
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
model = Network(**model_args)
state_dict = torch.load(model_pth)
model.load_state_dict(state_dict, strict=False)

# # 添加保存代码
# checkpoint_path = "full_model.ckpt"
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'model_args': model_args,
#     'network_class': Network,
# }, checkpoint_path)


device = torch.device('cuda:0')
model.to(device)
model.set_new_noise_schedule(phase='test')
model.eval()
print(model)
tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# read input and create random mask
img_pillow = Image.open(input_image_pth).convert('RGB')
mask = Image.open(input_image_pth).convert("L")
img = tfs(img_pillow)
# mask = get_irregular_mask([256, 256])
mask = np.array(mask)
antenna_mask = mask.copy()
print(antenna_mask)
print(np.max(antenna_mask))
print(antenna_mask==255)
mask[mask==255] =0  #3....don't set antenna to mask ,so comment this line

mask[mask > 0] = 1 #1为掩码区
mask= torch.tensor(np.expand_dims(mask, axis=0))  # 添加新维度，变为 (1, H, W)  #mask 是图片的掩码。掩码区域为1.。。...

gaussain_sigma = 5 #60-10-5-2-1-1.5...
print(np.argwhere(antenna_mask==255))
indicies = np.argwhere(antenna_mask==255)[0]
x_norm, y_norm = indicies[0] , indicies[1]
heatmap = generate_heatmap(256, 256, x_norm, y_norm, gaussain_sigma) #添加高斯热图
cond_image_0 = img*(1. - mask) + mask*torch.randn_like(img)   #  *0.999   # 3.....  # 去掉随机噪声看看
cond_image = cond_image_0 + heatmap
mask_img = img*(1. - mask) + mask


# save conditional image used a inference input
cond_image_np = tensor2img(cond_image)
Image.fromarray(cond_image_np).save("./result/cond_image.jpg")

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
    output, visuals = model.restoration(cond_image, y_t=cond_image,
                                        y_0=gt_image, mask=mask, sample_num=8)

# save intermediate processes
output_img = output.detach().float().cpu()
for i in range(visuals.shape[0]):
    img = tensor2img(visuals[i].detach().float().cpu())
    Image.fromarray(img).save(f"./result/process_{i}.jpg")

# save output (output should be the same as last process_{i}.jpg)
img = tensor2img(output_img)
Image.fromarray(img).save("./result/output.jpg")
