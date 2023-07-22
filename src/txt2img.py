import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

import util
import prompts
import shared
import resources
import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image


import sys
import types

# create new modules for 'modules' and 'modules.devices'
modules = types.ModuleType("modules")
devices = types.ModuleType("modules.devices")

# now define your functions
def get_device_for(name):
    return shared.device

# and add them to your module
devices.get_device_for = get_device_for

# add 'devices' as an attribute of 'modules'
setattr(modules, 'devices', devices)

# add the new modules to sys.modules under the appropriate name
sys.modules['modules'] = modules
sys.modules['modules.devices'] = devices

#import processor

# from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def randn(seed, shape):
    torch.manual_seed(seed)
    return torch.randn(shape, device=shared.device, dtype=shared.dtype)


def randn_without_seed(shape):
    return torch.randn(shape, device=shared.device, dtype=shared.dtype)

def create_random_tensors(shape, seeds, seed_delta, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
    xs = []

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)        
        
        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        xs.append(noise)

        if seed_delta > 0:
            torch.manual_seed(seed + seed_delta)

    return torch.stack(xs)

def txt2img(
    checkpoint, 
    positive, 
    negative, 
    steps, 
    seed,
    width = 512,
    height = 512,
    eta_noise_seed_delta = 31337,
    batch_size = 1,
    cfg_scale = 7,
    subseed_strength = 0.5,
    sub_seed = 1,
    seed_resize_from_h = 0,
    seed_resize_from_w = 0,
):
    pipe = StableDiffusionPipeline.from_single_file( #StableDiffusionControlNetPipeline
        resources.checkpoints[checkpoint], 
        #use_safetensors = True,
        local_files_only = True,
        controlnet = False,
        torch_dtype = shared.dtype,
        #controlnet = MultiControlNetModel([
            #ControlNetModel.from_single_file(resources.controlnets["control_v11p_sd15_canny"], local_files_only=True),
        #]),
    )

    pipe.to(shared.device)
    pipe.safety_checker = None
    pipe.enable_vae_slicing()

    from diffusers import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    """
    canny_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
    )
   
    (canny_image, success) = processor.zoe_depth(np.asarray(canny_image))
    canny_image = Image.fromarray(canny_image)

    canny_image.save("../output/zoe.png")
    """


    [positive_conditioning, negative_conditioning] = prompts.apply_prompts(pipe, positive, negative)

    opt_f = 8
    opt_C = 4

    images = pipe(
        #image = [canny_image],
        prompt_embeds = positive_conditioning,
        negative_prompt_embeds = negative_conditioning,
        num_inference_steps = steps,
        guidance_scale = cfg_scale,
        num_images_per_prompt = batch_size,
        generator = torch.Generator(device=shared.device).manual_seed(seed + eta_noise_seed_delta),
        latents = create_random_tensors(
            (opt_C, height // opt_f, width // opt_f), 
            [seed + i for i in range(batch_size)], eta_noise_seed_delta, 
            subseeds=[sub_seed + i for i in range(batch_size)], 
            subseed_strength=subseed_strength, 
            seed_resize_from_h=seed_resize_from_h, 
            seed_resize_from_w=seed_resize_from_w
        ),
        #controlnet_conditioning_scale=[1.0]
    ).images


    def image_grid(imgs, rows, cols):
        if len(imgs) == 1:
            return imgs[0]

        assert len(imgs) == rows*cols

        grid = Image.new('RGB', size=(cols*width, rows*height))
        
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*width, i//cols*height))

        return grid

    rows = max(batch_size // 2, 1)
    cols = max(batch_size // 2, 1)

    return image_grid(images, rows, cols)
