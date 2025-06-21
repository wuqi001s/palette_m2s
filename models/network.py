import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork

import torch.nn.functional as F
from pytorch_msssim import SSIM

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet) #**unet,对unet字典(来自配置文件）进行解包操作...
        self.beta_schedule = beta_schedule
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3) #添加

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('gammas_prev', to_torch(gammas_prev))
        self.register_buffer('sqrt_gammas', to_torch(np.sqrt(gammas)))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):  #计算y0?...yt-noise
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):   #往y0中加噪
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )
    
    # def q_sample_reverse(self, y_noisy, sample_gammas, noise=None):  #根据y_noisy和预测的噪声，求得y0^
    #     # 默认噪声
    #     noise = default(noise, lambda: torch.randn_like(y_noisy))
        
    #     # 计算恢复的 y_0
    #     sqrt_gamma_t = sample_gammas.sqrt()
    #     sqrt_1_minus_gamma_t = (1 - sample_gammas).sqrt()
        
    #     y_0_recovered = (y_noisy - sqrt_1_minus_gamma_t * noise) / sqrt_gamma_t
        
    #     return y_0_recovered

    def q_sample_reverse(self, y_noisy, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_noisy))
        
        # Ensure gradient tracking if needed
        if not y_noisy.requires_grad:
            y_noisy = y_noisy.clone().detach().requires_grad_(True)  # Force gradients
        
        sqrt_gamma_t = sample_gammas.sqrt()
        sqrt_1_minus_gamma_t = (1 - sample_gammas).sqrt()
        
        y_0_recovered = (y_noisy - sqrt_1_minus_gamma_t * noise) / sqrt_gamma_t
        return y_0_recovered

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):  #explain:根据当前噪声图像 y_t 和时间步 t，预测下一步的更清晰图像（即去噪一步）....
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp() 


# repaint implement
    # @torch.no_grad()
    # def restoration(self, y_cond, y_t=None, y_0=None, mask=None,  sample_num=8):
    #     b, *_ = y_cond.shape
    #     print('-----------------')
    #     t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
    #     gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
    #     sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
    #     sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
    #     sample_gammas = sample_gammas.view(b, -1)
    #     print(t.shape)                # 应为 (4,)
    #     print(gamma_t1.shape)         # 应为 (4, 1, 1)
    #     print(sqrt_gamma_t2.shape)    # 应为 (4, 1, 1)
    #     print(torch.rand((b,1)).shape) # 应为 (4, 1)
    #     print(sample_gammas.shape)    # 应为 (4, 1)
    #     print('-----------------')
    #     assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
    #     sample_inter = (self.num_timesteps//sample_num)
        
    #     y_t = default(y_t, lambda: torch.randn_like(y_cond))
    #     ret_arr = y_t
    #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
    #         t = torch.full((b,), i, device=y_cond.device, dtype=torch.long) #创建一个一维张量t,长度为 b,每个元素都为 i

    #         for u in range(4):
    #             noise_u = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
    #             sqrt_noise_level = extract(self.sqrt_gammas, t, x_shape=(1, 1)).to(y_t.device).view(-1, 1, 1, 1)
    #             print("sqrt_noise_level",sqrt_noise_level.shape)
    #             print("y_0",y_0.shape)
    #             print("noise_u",noise_u.shape)
    #             print("self.gammas",self.gammas.shape)
    #             x_t_k = sqrt_noise_level*y_0 + (1-sqrt_noise_level)* noise_u
    #             y_t = self.p_sample(y_t, t, y_cond=y_cond)   #==x_t-1 unknown #根据当前噪声图像 y_t 和时间步 t，预测下一步的更清晰图像（即去噪一步）。#不过他是重参数化得到y_t
    #             if mask is not None:
    #                 y_t = x_t_k*(1.-mask) + mask*y_t
                
    #         if i % sample_inter == 0:
    #             ret_arr = torch.cat([ret_arr, y_t], dim=0)
    #     return y_t, ret_arr

#ddim sampling
#使用run.py进行测试时，若使用ddim,则图像中始终存在噪点。而ddpm则不会。ddpm 运行5min
#使用mtos_1.py进行测试时，ddim和ddpm都能正常运行，且图像中没有噪点。但图像都比较模糊。
    def restoration(
        self, 
        y_cond,
        sample_num,
        y_t=None, 
        y_0=None, 
        mask=None, 
        m2s_steps=10, #m2s的步数
        ddim_num_steps=250, #微调 #ddim的步数
        ddim_eta=0,
        sampling_method='ddpm'
        ):
        
        if sampling_method == 'ddpm':
            return self.restoration_ddpm(y_cond, sample_num=sample_num, y_t=y_t, y_0=y_0, mask=mask)

        elif sampling_method == 'ddim':
            return self.restoration_ddim(y_cond, sample_num=sample_num, ddim_num_steps=ddim_num_steps, ddim_eta=ddim_eta, y_t=y_t, y_0=y_0, mask=mask)
        elif sampling_method =='m2s':
            return self.restoration_m2s(y_cond, m2s_steps=m2s_steps,sample_num=sample_num,  ddim_eta=ddim_eta, y_t=y_t, y_0=y_0, mask=mask)

    def restoration_ddim(self, y_cond, ddim_num_steps, ddim_eta, sample_num, y_t=None, y_0=None, mask=None):
        print('--------ddim sampling---------')
        b, *_ = y_cond.shape
        assert ddim_num_steps > sample_num, 'ddim_num_steps must be greater than sample_num'
        sample_inter = (ddim_num_steps // sample_num)

        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        # linear step schedule
        tseq = list(
            np.linspace(0, self.num_timesteps - 1, ddim_num_steps).astype(int)
        )
        print("sample_num: ", sample_num)
        print("ddim_num_steps: ", ddim_num_steps)
        print("ddim_eta: ", ddim_eta)

        tlist = torch.zeros([y_t.shape[0]], device=y_t.device, dtype=torch.long)

        for i in tqdm(range(ddim_num_steps), desc='sampling loop time step', total=ddim_num_steps):
            tlist = tlist * 0 + tseq[-1 - i]

            if i != ddim_num_steps - 1:
                prevt = torch.ones_like(tlist, device=y_cond.device) * tseq[-2 - i]
            else:
                prevt = - torch.ones_like(tlist, device=y_cond.device)
            
            y_t = self.p_sample_ddim(y_t, tlist, prevt, ddim_num_steps, ddim_eta, y_cond=y_cond) 

            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            if (i + 1) % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
            
        return y_t, ret_arr

    @torch.no_grad()
    def restoration_ddpm(self, y_cond, y_t=None, y_0=None, mask=None,  sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long) #创建一个一维张量t,长度为 b,每个元素都为 i
            y_t = self.p_sample(y_t, t, y_cond=y_cond)   #根据当前噪声图像 y_t 和时间步 t，预测下一步的更清晰图像（即去噪一步）。#不过他是重参数化得到y_t
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr
    def p_sample_ddim(self, y_t, t, prevt, ddim_numsteps, ddim_eta, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance_ddim(y_t=y_t, t=t, prevt=prevt, clip_denoised=clip_denoised, y_cond=y_cond, 
        ddim_numsteps=ddim_numsteps, ddim_eta=ddim_eta)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)

        return model_mean + noise * (0.5 * model_log_variance).exp()

    def restoration_m2s(self, y_cond,m2s_steps=10 , ddim_eta=0, sample_num=8, y_t=None, y_0=None, mask=None):
            print('--------m2s sampling---------')
            b, *_ = y_cond.shape
            

            y_t = default(y_t, lambda: torch.randn_like(y_cond))
            ret_arr = y_t
            self.num_timesteps = m2s_steps
            ddim_num_steps = int(self.num_timesteps/5)    
            # linear step schedule
            tseq = list(
                np.linspace(0, self.num_timesteps - 1, ddim_num_steps).astype(int)
            )
            print("sample_num: ", sample_num)
            print("ddim_num_steps: ", ddim_num_steps)
            print("ddim_eta: ", ddim_eta)
            assert ddim_num_steps > sample_num, 'ddim_num_steps must be greater than sample_num'
            sample_inter = (ddim_num_steps // sample_num)
            tlist = torch.zeros([y_t.shape[0]], device=y_t.device, dtype=torch.long)

            for i in tqdm(range(ddim_num_steps), desc='sampling loop time step', total=ddim_num_steps):
                tlist = tlist * 0 + tseq[-1 - i]

                if i != ddim_num_steps - 1:
                    prevt = torch.ones_like(tlist, device=y_cond.device) * tseq[-2 - i]
                else:
                    prevt = - torch.ones_like(tlist, device=y_cond.device)
                
                y_t = self.p_sample_ddim(y_t, tlist, prevt, ddim_num_steps, ddim_eta, y_cond=y_cond) 

                if mask is not None:
                    y_t = y_0 * (1. - mask) + mask * y_t

                
            return y_t
    def p_mean_variance_ddim(self, y_t, t, prevt, ddim_numsteps, ddim_eta, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        noise_hat = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
        y_0_hat = self.predict_start_from_noise(y_t, t=t, noise=noise_hat)
        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)
        
        ## denoising formula in DDIM
        gamma_t = extract(self.gammas, t, y_t.shape)
        gamma_prev_t = extract(self.gammas_prev, prevt + 1, y_t.shape)

        sigma = ddim_eta * torch.sqrt((1. - gamma_prev_t) / (1. - gamma_t) * (1. - gamma_t / gamma_prev_t))
        p_var = sigma ** 2
        posterior_log_variance = torch.log(p_var)

        coef_eps = 1. - gamma_prev_t - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)

        model_mean = (
            # torch.sqrt(gamma_prev_t) * (y_t - torch.sqrt(1.0 - gamma_t) * noise_hat) / torch.sqrt(gamma_t) + coef_eps * noise_hat
            torch.sqrt(gamma_prev_t) * y_0_hat + coef_eps * (y_t - torch.sqrt(gamma_t) * y_0_hat) / (torch.sqrt(1. - gamma_t))
        )
        
        return model_mean, posterior_log_variance

#origing samplining
    # @torch.no_grad()
    # def restoration(self, y_cond, y_t=None, y_0=None, mask=None,  sample_num=8):
    #     b, *_ = y_cond.shape

    #     assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
    #     sample_inter = (self.num_timesteps//sample_num)
        
    #     y_t = default(y_t, lambda: torch.randn_like(y_cond))
    #     ret_arr = y_t
    #     for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
    #         t = torch.full((b,), i, device=y_cond.device, dtype=torch.long) #创建一个一维张量t,长度为 b,每个元素都为 i
    #         y_t = self.p_sample(y_t, t, y_cond=y_cond)   #根据当前噪声图像 y_t 和时间步 t，预测下一步的更清晰图像（即去噪一步）。#不过他是重参数化得到y_t
    #         if mask is not None:
    #             y_t = y_0*(1.-mask) + mask*y_t
            
    #         if i % sample_inter == 0:
    #             ret_arr = torch.cat([ret_arr, y_t], dim=0)
    #     return y_t, ret_arr

 


    # def forward(self, y_0, y_cond=None, mask=None, noise=None):
    #     # sampling from p(gammas)
    #     b, *_ = y_0.shape
    #     t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
    #     gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
    #     sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
    #     sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
    #     sample_gammas = sample_gammas.view(b, -1)

    #     noise = default(noise, lambda: torch.randn_like(y_0))
    #     y_noisy = self.q_sample(
    #         y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

    #     if mask is not None:
    #         noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
    #         loss = F.l1_loss(mask*noise, mask*noise_hat)
    #     else:
    #         noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
    #         loss = F.l1_loss(noise, noise_hat)
    #     return loss

#新定义了一个加噪函数，用在minutes to second 中给上采样的图片加噪
    def m2s_add_noise(self, y_0 , noise=None,t=75):
        b, *_ = y_0.shape
        
        print(y_0.device)
        y_0 = y_0.to(self.gammas.device)
        # t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        t = torch.full((b,), t, device=self.gammas.device, dtype=torch.long)
        print(t.device)
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        m2s_y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
        return m2s_y_noisy        

    def forward(self, y_0, y_cond=None, mask=None, noise=None, antenna_image = None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
# #对高斯热图加噪音进行去噪(batch)
#         lambda_weight = 1
#         noise_new = noise.clamp(-1, 1) * 0.5 + 0.5 
#         heatmap = 
#         heatmap_1 = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())




        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
        # print('-----------------------')
        print(y_cond.shape)  #torch.Size([batch_size, 3, 256, 256])
        if mask is not None:  
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas) #前面输入的是y_cond +y_t.这里输入y_cond,ynoisy.(接下来看一下y_cond,y_t是什么，然后模型对他们的处理是否一样呢？最后在考虑能否多输入一张图片)
            loss =  F.l1_loss(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss =  F.l1_loss(noise, noise_hat)

# image_loss ....
#         # y_pred = y_noisy - noise_hat #这里得到的还只是y^t-1 ,而不是最终的预测结果吗？？？....
#         y_pred = self.q_sample_reverse(y_noisy=y_noisy, sample_gammas=sample_gammas.view(-1,1,1,1), noise=noise)
#         image_loss = 0.05 *  self.loss_fn(y_pred,y_0)# change coefients..
#         total_loss = loss + image_loss  

#         print('------mse  loss----:    ',loss)
#         print('-----  image losss:',image_loss)
#         # print('----- maxpixeleloss:',max_pixel_loss)
#         print('zzzzzz  tottlaloss   ',total_loss)
# ....image_loss

## 尝试添加 最大像素值损失 (最大为1)
        # y_pred = self.q_sample_reverse(y_noisy=y_noisy, sample_gammas=sample_gammas.view(-1,1,1,1), noise=noise)
        # o_max = torch.max(y_0)
        # g_max = torch.max(y_pred)
        # print("----------g_max-----------")
        # print(g_max)
        # max_pixel_loss = F.mse_loss(o_max,g_max)*1e9
##

        k = 5  # 取 Top-5 像素
        y_pred = self.q_sample_reverse(y_noisy=y_noisy, sample_gammas=sample_gammas.view(-1,1,1,1), noise=noise)

        # 方法 1：仅约束最大值（更明确目标）
        # o_max = y_0.max()
        # g_max = y_pred.max()
        # max_pixel_loss = F.mse_loss(g_max, o_max) * 1e13  # 权重需实验调整
        print("y0's pixel range : ",y_0.min())

        
        # method 3:计算SSIM损失（值越大越好，因此用1 - SSIM转为损失）
        y_pred_s = (y_pred+1)/2  #befor using ssim loss ,you need to transform pixel range to 0,1
        y_0_s = (y_0+1)/2
        ssim_weight = 1e5  # 权重需实验调整
        ssim_loss = (1 - self.ssim_loss(y_pred_s, y_0_s))*ssim_weight

        # 方法 2：添加空间约束（固定基站位置）
        # 3. add antenna region mask .mask .将基站周围的区域设为1,远离的为0.用于计算基站周围的Mse
        bound = 200
        antenna_region_mask = y_0_s.clone()
        antenna_region_mask = antenna_region_mask * 255
        antenna_region_mask[(antenna_region_mask >=0) & (antenna_region_mask <= bound)] = 0
        antenna_region_mask[antenna_region_mask > bound] = 1
        print('------antennna region mask -----')
        print(antenna_region_mask)
        base_station_loss = F.mse_loss(y_pred * antenna_region_mask, y_0 * antenna_region_mask)*1e15

        # 总损失 = 原始损失 + 最大值约束损失 + 基站区域损失 + SSIM损失
        total_loss = loss + base_station_loss + ssim_loss
        
        # 打印SSIM损失监控
        print('ssim_loss:', ssim_loss.item())
        # 确保 y_pred 可梯度传播
        print("y_pred.requires_grad:", y_pred.requires_grad)  # 应为 True
        print('base_station_loss:',base_station_loss)
        # print('max_pixel_loss',max_pixel_loss)
        print('loss:----',loss)
        print('----tottlaloss -----  ',total_loss)
        return total_loss     #3........
    
    
#add noise....
    # def forward(self, y_0, y_cond=None, mask=None, noise=None):
    #     num_inner_loops =5
    #     # sampling from p(gammas)
    #     b, *_ = y_0.shape
    #     t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
    #     gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
    #     sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
    #     sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
    #     sample_gammas = sample_gammas.view(b, -1)

    #     noise = default(noise, lambda: torch.randn_like(y_0))
    #     y_noisy = self.q_sample(
    #         y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
    
    #     for _ in range(num_inner_loops):  #inner loop update y_noisy using noise_hat.????.....
    #         # Masked denoising
    #         noise_hat = self.denoise_fn(
    #             torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), 
    #             sample_gammas
    #         )
    #         # Update the noisy data based on the predicted noise
    #         y_noisy = self.q_sample(
    #             y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise_hat)
            

    #     loss = self.loss_fn(mask * noise, mask * noise_hat)
    #                 #4....最大值的位置越靠近越好，最大值越接近255越好。
    #     y_pred = noise - noise_hat
    #     o_max = torch.max(y_0)
    #     g_max = torch.max(y_pred)
    #     print('gggggg   gamax',g_max)
    #     print('ooooo omax',o_max)
    #     # max_pixel_loss = (o_max-g_max)**2  #.....

    #     o_location = torch.where(y_0[0,0,:,:]==o_max)  #original_antenna_location
    #     g_location = torch.where(y_pred[0,:,:,:]==g_max)[1:] #generated_antenna_location   # transform 255 to g_max
    #     # 检查是否找到目标位置
    #     if o_location[0].numel() == 0 or g_location[0].numel() == 0:
    #         # 如果任何一个位置为空张量，设定一个默认的 location_loss
    #         location_loss = 1e3  # 或其他合理的默认值
    #     else:
    #         location_loss = torch.sum(
    #         (o_location[0].float() - g_location[0].float()) ** 2 +
    #         (o_location[1].float() - g_location[1].float()) ** 2
    # )

    #     total_loss = loss + location_loss 
    #             # total_loss = loss + location_loss +  1e3* max_pixel_loss  #....

    #     print('------mse  loss----:    ',loss)
    #     print('-----  location losss:',location_loss)
    #     # print('----- maxpixeleloss:',max_pixel_loss)
    #     print('zzzzzz  tottlaloss   ',total_loss)
    #     return total_loss     #3........

    #     return loss

#add noise....




        # if mask is not None:
        #     noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
        #     loss = self.loss_fn(mask*noise, mask*noise_hat)
        # else:
        #     noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
        #     loss = self.loss_fn(noise, noise_hat)
        # return loss

        # # Calculate the maximum pixel value difference loss....2....
        # y_pred = y_noisy - noise_hat  #noise-noise_hat,y_0-noise_hat is the same.
        # _, max_pos_y_0 = y_0.view(b, -1).max(dim=1)
        # _, max_pos_y_hat = y_pred.view(b, -1).max(dim=1)
        # position_loss = 1e3 * (max_pos_y_0 != max_pos_y_hat).float().mean()
        # # Total loss
        # print('hhhhhh  loss:    ',loss)
        # total_loss = loss + position_loss
        # print('tttttt   losss:',total_loss)
        # return total_loss     #2........


    #     #3....location loss
    #     y_pred = y_noisy - noise_hat
    #     o_location = torch.where(y_0[0,:,:]==255)  #original_antenna_location
    #     g_location = torch.where(y_pred[0,:,:]==255) #generated_antenna_location
    #     # 检查是否找到目标位置
    #     if o_location[0].numel() == 0 or g_location[0].numel() == 0:
    #         # 如果任何一个位置为空张量，设定一个默认的 location_loss
    #         location_loss = 1e9  # 或其他合理的默认值
    #     else:
    #         location_loss = torch.sum(
    #         (o_location[0].float() - g_location[0].float()) ** 2 +
    #         (o_location[1].float() - g_location[1].float()) ** 2
    # )

    #     total_loss = loss + location_loss
    #     print('hhhhhh  loss:    ',loss)
    #     print('tttttt   total losss:',total_loss)
    #     return total_loss     #3........
    
    #         #4....最大值的位置越靠近越好，最大值越接近255越好。
    #     y_pred = noise - noise_hat
    #     o_max = torch.max(y_0)
    #     g_max = torch.max(y_pred)
    #     print('gggggg   gamax',g_max)
    #     print('ooooo omax',o_max)
    #     # max_pixel_loss = (o_max-g_max)**2  #.....

    #     o_location = torch.where(y_0[0,0,:,:]==o_max)  #original_antenna_location
    #     g_location = torch.where(y_pred[0,:,:,:]==g_max)[1:] #generated_antenna_location   # transform 255 to g_max
    #     # 检查是否找到目标位置
    #     if o_location[0].numel() == 0 or g_location[0].numel() == 0:
    #         # 如果任何一个位置为空张量，设定一个默认的 location_loss
    #         location_loss = 1e3  # 或其他合理的默认值
    #     else:
    #         location_loss = torch.sum(
    #         (o_location[0].float() - g_location[0].float()) ** 2 +
    #         (o_location[1].float() - g_location[1].float()) ** 2
    # )

    #     total_loss = loss + location_loss 
    #             # total_loss = loss + location_loss +  1e3* max_pixel_loss  #....

    #     print('------mse  loss----:    ',loss)
    #     print('-----  location losss:',location_loss)
    #     # print('----- maxpixeleloss:',max_pixel_loss)
    #     print('zzzzzz  tottlaloss   ',total_loss)
    #     return total_loss     #3........


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):   
    b, *_ = t.shape         ##通过 Python 的解包语法获取张量 t 的第一个维度大小，并将其赋值给 b
    out = a.gather(-1, t)   #gather 是 PyTorch 中的一个操作，用于从张量 a 中提取元素。a.gather(dim, index)
                            # 从张量 a 中根据 index 张量指定的索引提取元素，dim 是提取元素的维度。
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


