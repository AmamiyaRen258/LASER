import copy
import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from torchvision import transforms
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from laser_utils import *
import torch.nn.functional as F

# suppress partial model loading warning
logging.set_verbosity_error()


class LASER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        sd_version = config["sd_version"]

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float32).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        print('SD model loaded')

        self.inversion_func = self.ddim_inversion




        seed_everything(config['seed'])

        save_path = os.path.join(config['save_dir'],
                                 os.path.splitext(os.path.basename(config['data_path']))[0])

        os.makedirs(save_path, exist_ok=True)

        self.toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.toy_scheduler.set_timesteps(config['save_steps'])

        self.scheduler.set_timesteps(config["n_timesteps"])



        self.target_prompt_list = config["target_prompt_list"]


    def get_timesteps(self, scheduler, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to('cuda')
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, guidance_scale=7.5, save_latents=False, timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)

        text_embeddings = cond

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion")):

                register_time(self, t.item())
                if guidance_scale > 1.:
                    latent_model_input = torch.cat([latent] * 2)
                else:
                    latent_model_input = latent
                # predict the noise
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                if guidance_scale > 1.:
                    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

                # compute the previous noise sample x_t-1 -> x_t
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                pred_x0 = (latent - sigma_prev * noise_pred) / mu_prev
                latent = mu * pred_x0 + sigma * noise_pred

                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))

            torch.save(latent, os.path.join(save_path, f'origin.pt'))

        print("size of latent: " + str(latent.shape))

        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_path, guidance_scale=7.5, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                if guidance_scale > 1.:
                    latent_model_input = torch.cat([x] * 2)
                else:
                    latent_model_input = x
                cond_batch = cond
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent_model_input, t, encoder_hidden_states=cond_batch).sample

                if guidance_scale > 1.:
                    noise_pred_uncon, noise_pred_con = eps.chunk(2, dim=0)
                    eps = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps

            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x

    @torch.no_grad()
    def extract_latents(self, num_steps, img_path, save_path, timesteps_to_save,
                        inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, inversion_prompt)

        image = self.load_img(img_path)
        latent = self.encode_imgs(image)

        inverted_x = self.inversion_func(cond, latent, save_path, guidance_scale=self.config["guidance_scale"],
                                         timesteps_to_save=timesteps_to_save)

        return inverted_x

    @torch.no_grad()
    def load_inverted_latents(self, i):
        latents_path = "latents/" + str(self.config['target_prompt_list'][i])
        latents_files = sorted(
            glob.glob(os.path.join(latents_path, 'noisy_latents_*.pt')),
            key=lambda x: int(x.split('_')[-1].replace('.pt', ''))
        )
        self.inverted_latents = [torch.load(f).to(self.device) for f in latents_files]
        print(f"Loaded {len(self.inverted_latents)} inverted latents from {latents_path}")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self):
        # load image
        image = Image.open(self.config["img_path"]).convert('RGB')
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        # get noise
        latents_path = os.path.join(self.config["latents_path"],
                                    os.path.splitext(os.path.basename(self.config["img_path"]))[0],
                                    f'origin.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        return image, noisy_latent

    @torch.no_grad()
    def denoise_step(self, x, t, guidance_scale=7.5):


        latent_model_input = torch.cat([x] * 2)

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = self.text_embeds

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist

            latents = posterior.mean * 0.18215
        return latents


    def init_trans(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []

        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def init_act(self, kv_injection_t):
        self.kv_injection_timesteps = self.scheduler.timesteps[kv_injection_t:] if kv_injection_t >= 0 else []
        register_kv_control_efficient(self, self.kv_injection_timesteps)

    def init_mix(self, kv_injection_t):
        self.kv_injection_timesteps = self.scheduler.timesteps[kv_injection_t:] if kv_injection_t >= 0 else []
        register_mix_control_efficient(self, self.kv_injection_timesteps)

    def init_attn(self):
        unregister_attention_control_efficient(self)
        unregister_kv_control_efficient(self)
        unregister_conv_control_efficient(self)

    def run_laser(self, num_frames):
        self.qk_t = int(self.config["n_timesteps"] * self.config["qk_t"])
        self.kv_t = int(self.config["n_timesteps"] * self.config["kv_t"])
        self.f_t = int(self.config["n_timesteps"] * self.config["f_t"])

        self.save_sign = 0



        st = 0
        all_img = []
        self.origin_path = self.config["img_path"]
        self.path = self.config["img_path"]

        totol_len = len(self.config["control_list"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        timesteps_to_save, num_inference_steps = self.get_timesteps(self.toy_scheduler,
                                                                    num_inference_steps=self.config['save_steps'],
                                                                    strength=1.0,
                                                                    device=device)


        for i in range(0, totol_len):

            # load image
            save_path = os.path.join(self.config['save_dir'],
                                     os.path.splitext(os.path.basename(self.config['data_path']))[0])

            print("save_path = "+str(save_path))
            print("The inversion_prompt = "+str(self.config["inversion_prompt"]))
            print("img_path = "+str(self.config["img_path"]))

            os.makedirs(save_path, exist_ok=True)


            register_save(self, True)




            sign = self.config["control_list"][i]
            print("sign = " + str(sign))

            edited_img = []

            alpha_list = torch.linspace(0, 1, num_frames)

            if sign == "2":
                alpha_list *= 0.8

            print(alpha_list)

            register_alpha(self, 1)

            if (sign != "3"):

                current_text_embeds = self.get_text_embeds(self.target_prompt_list[i], self.target_prompt_list[i])
                text_embeds_target = self.get_text_embeds(self.target_prompt_list[i+1], self.target_prompt_list[i])

                self.text_embeds = text_embeds_target.clone().detach()

                if (sign == "0"):
                    print("get sign 0")
                    self.init_trans(conv_injection_t=self.f_t, qk_injection_t=self.qk_t)
                elif (sign == "1"):
                    print("get sign 1")
                    self.init_act(kv_injection_t=self.kv_t)
                else:
                    print("get sign 2")
                    self.init_mix(kv_injection_t=self.kv_t)
                self.save_sign = i + 1

            self.eps = self.extract_latents(img_path=self.config['img_path'],
                                            num_steps=self.config['steps'],
                                            save_path=save_path,
                                            timesteps_to_save=timesteps_to_save,
                                            inversion_prompt=self.config["inversion_prompt"])


            self.inv_embeds = self.get_text_embeds(self.target_prompt_list[i], self.target_prompt_list[i])

            register_save(self, False)

            for j in range(0, num_frames):

                register_alpha(self, alpha_list[j])


                st = st + 1
                self.cur_ratio = 1 - alpha_list[j]
                self.text_embeds = current_text_embeds * (1 - alpha_list[j]) + text_embeds_target * alpha_list[j]

                if (j < num_frames-1):
                    edited_img.append(self.sample_loop(self.eps, st))
                else:
                    edited_img.append(self.sample_loop(self.eps, st, get_path=True))

                all_img.append(edited_img[j])

            if (i < totol_len - 1):
                self.config["inversion_prompt"] = self.target_prompt_list[i + 1]
                self.init_attn()

        all_img[0].save(f'{self.config["output_path"]}/output.gif', save_all=True,
                        append_images=all_img[1:], duration=100, loop=0)

    def sample_loop(self, x, st, get_path=False):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):

                x = self.denoise_step(x, t, guidance_scale=self.config["guidance_scale"])

            decoded_latent = self.decode_latent(x)
            edited_img = T.ToPILImage()(decoded_latent[0])
            T.ToPILImage()(decoded_latent[0]).save(f'{self.config["output_path"]}/output-{st}.png')
            if get_path:
                T.ToPILImage()(decoded_latent[0]).save(
                    f'data/{self.config["quest"]}/{self.target_prompt_list[self.save_sign]}.png')
                self.path = f'data/{self.config["quest"]}/{self.target_prompt_list[self.save_sign]}.png'
                self.config['img_path']= self.path
                print("The saved img_path = "+str(self.path))
                self.config['data_path'] = self.path

        return edited_img


def cal_LASER(config):
    os.makedirs(config["output_path"], exist_ok=True)

    seed_everything(config["seed"])

    laser = LASER(config)
    laser.run_laser(num_frames=config["num_frames"])

