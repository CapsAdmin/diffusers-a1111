import torch
import shared
import latent_noise

import k_diffusion
sampling = k_diffusion.sampling

samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"uses_ensd": True}),
    ('Euler', 'sample_euler', ['k_euler'], {}),
    ('LMS', 'sample_lms', ['k_lms'], {}),
    ('Heun', 'sample_heun', ['k_heun'], {"second_order": True}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'discard_next_to_last_sigma': True}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'discard_next_to_last_sigma': True, "uses_ensd": True}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {"uses_ensd": True, "second_order": True}),
    ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {}),
    ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {"second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {"brownian_noise": True}),
    ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {"uses_ensd": True}),
    ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"uses_ensd": True}),
    ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a_ka'], {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    ('DPM++ 2M Karras', 'sample_dpmpp_2m', ['k_dpmpp_2m_ka'], {'scheduler': 'karras'}),
    ('DPM++ SDE Karras', 'sample_dpmpp_sde', ['k_dpmpp_sde_ka'], {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_ka'], {'scheduler': 'karras', "brownian_noise": True}),
]

k_diffusion_scheduler = {
    'Automatic': None,
    'karras': k_diffusion.sampling.get_sigmas_karras,
    'exponential': k_diffusion.sampling.get_sigmas_exponential,
    'polyexponential': k_diffusion.sampling.get_sigmas_polyexponential
}

class ModelWrapper:
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        return self.model(*args, encoder_hidden_states=encoder_hidden_states, **kwargs).sample

def create_denoiser(self, device, enable_quantization = True):
    model = ModelWrapper(self.unet, self.scheduler.alphas_cumprod)

    if self.scheduler.config.prediction_type == "v_prediction":
        denoiser = k_diffusion.external.CompVisVDenoiser(model, quantize=enable_quantization)
    else:
        denoiser = k_diffusion.external.CompVisDenoiser(model, quantize=enable_quantization)
    
    denoiser.sigmas = denoiser.sigmas.to(device)
    denoiser.log_sigmas = denoiser.log_sigmas.to(device)
    return denoiser

def get_sigmas(
        steps, 
        denoiser,
        dtype,
        discard_next_to_last_sigma = False, 
        second_order = False, 
        k_sched_type = 'Automatic',
        default_scheduler = None,
        rho = 1.0,
        custom_sigma_min = 0,
        custom_sigma_max = 0,
        use_old_karras_scheduler_sigmas = False
    ):

    if second_order:
        steps *= 2

    steps += 1 if discard_next_to_last_sigma else 0

    if k_sched_type != "Automatic":
        m_sigma_min, m_sigma_max = (denoiser.sigmas[0].item(), denoiser.sigmas[-1].item())
        sigma_min, sigma_max = (0.1, 10) if use_old_karras_scheduler_sigmas else (m_sigma_min, m_sigma_max)
        sigmas_kwargs = {
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
        }

        sigmas_func = k_diffusion_scheduler[k_sched_type]

        if custom_sigma_min != m_sigma_min and custom_sigma_min != 0:
            sigmas_kwargs['sigma_min'] = custom_sigma_min
        if custom_sigma_max != m_sigma_max and custom_sigma_max != 0:
            sigmas_kwargs['sigma_max'] = custom_sigma_max

        default_rho = 1. if k_sched_type == "polyexponential" else 7.

        if k_sched_type != 'exponential' and rho != 0 and rho != default_rho:
            sigmas_kwargs['rho'] = rho

        sigmas = sigmas_func(n=steps, **sigmas_kwargs, device=shared.device)
    elif default_scheduler:
        sigma_min, sigma_max = (0.1, 10) if use_old_karras_scheduler_sigmas else (denoiser.sigmas[0].item(), denoiser.sigmas[-1].item())

        sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=shared.device)
    else:
        sigmas = denoiser.get_sigmas(steps)

    if discard_next_to_last_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

    # cast sigmas to same dtype
    sigmas = sigmas.to(dtype)
    return sigmas

def kdiffusion_sampler(
    self, 
    latents,
    num_inference_steps, 
    guidance_scale, 
    prompt_embeds, 
    callback, 
    callback_steps, 
    seed, 
    eta_noise_seed_delta, 
    width, 
    height, 
    seed_resize_from_h, 
    seed_resize_from_w,
    randn_source = "gpu",

    enable_quantization = True,
    use_old_karras_scheduler_sigmas = False,
    always_discard_next_to_last_sigma = True,
    sampler_name = "DPM++ 2M SDE",
    k_sched_type = "Automatic",
    rho = 1.0,
    sigma_min = 0,
    sigma_max = 0,
    eta_ancestral = 1,
    s_churn = 0.0,
    s_tmin = 0.0,
    s_tmax = 0.0,
    s_noise = 1.0,
):
    sampler_info = next((sampler for sampler in samplers_k_diffusion if sampler[0] == sampler_name), None)
    if sampler_info is None:
        raise ValueError(f"Sampler {sampler_name} not found")
    
    sampler_func = getattr(sampling, sampler_info[1])
    sampler_info = sampler_info[3]

    denoiser = create_denoiser(self, latents.device, enable_quantization)
    
    sigmas = get_sigmas(
        steps = num_inference_steps,
        denoiser=denoiser,
        dtype=latents.dtype,
        discard_next_to_last_sigma = always_discard_next_to_last_sigma or sampler_info.get('discard_next_to_last_sigma', False),
        second_order = sampler_info.get('second_order', False),
        k_sched_type = k_sched_type,
        default_scheduler = sampler_info.get('scheduler', None),
        rho = rho,
        custom_sigma_min = sigma_min,
        custom_sigma_max = sigma_max,
        use_old_karras_scheduler_sigmas = use_old_karras_scheduler_sigmas,
    )

    progress_bar = self.progress_bar(total=len(sigmas))
    
    def sampler_step(x, t):
        progress_bar.update()

        latent_model_input = torch.cat([x] * 2)

        noise_pred = denoiser(latent_model_input, t, cond=prompt_embeds)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def sampler_callback(data):
        # call the callback, if provided
        if callback is not None and i % callback_steps == 0:
            callback(data["i"], data["sigma"], data["x"])

    # https://github.com/crowsonkb/k-diffusion/issues/25
    # the original a1111 implementation doesn't use this apart from brownian noise
    gen = latent_noise.create_a1111_sampler_generator(seed, eta_noise_seed_delta)
    noise_shape = latent_noise.get_noise_shape(width, height, seed_resize_from_h, seed_resize_from_w)

    def create_noise_sampler():
        if sampler_info.get('brownian_noise', False):
            sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
            #current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            return k_diffusion.sampling.BrownianTreeNoiseSampler(latents, sigma_min, sigma_max, seed=[seed])

        return lambda sigma, sigma_next: torch.randn(
            noise_shape, 
            layout=latents.layout, 
            device=shared.gpu if randn_source == "gpu" else shared.cpu, 
            dtype=shared.dtype, 
            generator=gen
        ).to(shared.device)
    
    # this depends on kdiffusion not changing arguments around too much
    sampler_args = {
        "model":  sampler_step,
        "x": latents,
        "callback": sampler_callback,
        "sigmas": sigmas,
        "rho": rho,
        "sigma_min": denoiser.sigmas[0].item(),
        "sigma_max": denoiser.sigmas[-1].item(),
        "eta": eta_ancestral,
        "noise_sampler": create_noise_sampler(),
        "n": num_inference_steps,
        "s_churn": s_churn,
        "s_tmin": s_tmin,
        "s_tmax": s_tmax,
        "s_noise": s_noise,
    }

    import inspect
    parameters = inspect.signature(sampler_func).parameters
    for param_name in sampler_args.copy().keys():
        if param_name not in parameters:
            del sampler_args[param_name]

    latents = sampler_func(**sampler_args)

    return latents


from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, StableDiffusionPipelineOutput

@torch.no_grad()
def StableDiffusionPipeline__call__WithCustomDenoising(
    self,

    # extra
    seed,
    eta_noise_seed_delta,
    seed_resize_from_h,
    seed_resize_from_w,
    enable_quantization = True,
    use_old_karras_scheduler_sigmas = False,
    always_discard_next_to_last_sigma = True,
    sampler_name = "DPM++ 2M SDE",
    k_sched_type = "Automatic",
    rho = 1.0,
    sigma_min = 0,
    sigma_max = 0,
    randn_source = "gpu",

    s_churn = 0.0,
    s_tmin = 0.0,
    s_tmax = 0.0,
    s_noise = 1.0,

    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta_ancestral: float = 0.75,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
):
    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
    )

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    ## self.scheduler.set_timesteps(num_inference_steps, device=device)
    ## timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    ## extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop

    latents = kdiffusion_sampler(
        self = self,
        latents = latents,
        num_inference_steps = num_inference_steps, 
        guidance_scale = guidance_scale, 
        prompt_embeds = prompt_embeds, 
        callback = callback, 
        callback_steps = callback_steps, 
        seed = seed, 
        eta_noise_seed_delta = eta_noise_seed_delta, 
        width = width, 
        height = height, 
        seed_resize_from_h = seed_resize_from_h, 
        seed_resize_from_w = seed_resize_from_w,
        enable_quantization = enable_quantization,
        use_old_karras_scheduler_sigmas = use_old_karras_scheduler_sigmas,
        always_discard_next_to_last_sigma = always_discard_next_to_last_sigma,
        sampler_name = sampler_name,
        k_sched_type = k_sched_type,
        rho = rho,
        sigma_min = sigma_min,
        sigma_max = sigma_max,
        eta_ancestral = eta_ancestral,
        randn_source = randn_source,
        s_churn = s_churn,
        s_tmin = s_tmin,
        s_tmax = s_tmax,
        s_noise = s_noise,
    )

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)