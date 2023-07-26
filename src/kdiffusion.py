import torch
import shared
import latent_noise
import inspect

import k_diffusion
sampling = k_diffusion.sampling

# TODO: how to handle use_ensd?
samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', {"uses_ensd": True}),
    ('Euler', 'sample_euler', {}),
    ('LMS', 'sample_lms', {}),
    ('Heun', 'sample_heun', {"second_order": True}),
    ('DPM2', 'sample_dpm_2', {'discard_next_to_last_sigma': True}),
    ('DPM2 a', 'sample_dpm_2_ancestral', {'discard_next_to_last_sigma': True, "uses_ensd": True}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', {"uses_ensd": True, "second_order": True}),
    ('DPM++ 2M', 'sample_dpmpp_2m', {}),
    ('DPM++ SDE', 'sample_dpmpp_sde', {"second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', {"brownian_noise": True}),
    ('DPM fast', 'sample_dpm_fast', {"uses_ensd": True, "default_eta": 1.0}), # a1111 has a default_eta of 1.0 from the ui, but in kdiffusion the eta is 0.0 by default for this sampler
    ('DPM adaptive', 'sample_dpm_adaptive', {"uses_ensd": True, "default_eta": 1.0}), # a1111 has a default_eta of 1.0 from the ui, but in kdiffusion the eta is 0.0 by default for this sampler
    ('LMS Karras', 'sample_lms', {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    ('DPM++ 2M Karras', 'sample_dpmpp_2m', {'scheduler': 'karras'}),
    ('DPM++ SDE Karras', 'sample_dpmpp_sde', {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', {'scheduler': 'karras', "brownian_noise": True}),
]

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

def create_denoiser(unet, alphas_cumprod, prediction_type, device, dtype, denoiser_enable_quantization):
    model = ModelWrapper(unet, alphas_cumprod)

    if prediction_type == "v_prediction":
        denoiser = k_diffusion.external.CompVisVDenoiser(model, quantize=denoiser_enable_quantization)
    else:
        denoiser = k_diffusion.external.CompVisDenoiser(model, quantize=denoiser_enable_quantization)
    
    denoiser.sigmas = denoiser.sigmas.to(device, dtype)
    denoiser.log_sigmas = denoiser.log_sigmas.to(device, dtype)
    return denoiser

def build_sigmas(
        steps, 
        denoiser,
        discard_next_to_last_sigma, 
        scheduler_name,
        custom_rho,
        custom_sigma_min,
        custom_sigma_max
    ):
    print("discard_next_to_last_sigma", discard_next_to_last_sigma)
    steps += 1 if discard_next_to_last_sigma else 0

    if scheduler_name is None:
        print(f"Building sigmas with default scheduler and arguments steps={steps}")
        sigmas = denoiser.get_sigmas(steps)
    else:
        sigma_min, sigma_max = (denoiser.sigmas[0].item(), denoiser.sigmas[-1].item())
        rho = None

        if scheduler_name == "polyexponential":
            rho = 1

        if scheduler_name == "karras":
            rho = 7

        if custom_sigma_min is not None:
            sigma_min = custom_sigma_min

        if custom_sigma_max is not None:
            sigma_max = custom_sigma_max

        if custom_rho is not None:
            rho = custom_rho

        arguments = {
            "n": steps, 
            "sigma_min": sigma_min, 
            "sigma_max": sigma_max, 
            "rho": rho,
            #"device": shared.device,
        }

        if rho is None:
            del arguments["rho"]

        print(f"Building sigmas with scheduler {scheduler_name} and arguments {arguments}")

        sigmas = getattr(k_diffusion.sampling, "get_sigmas_" + scheduler_name)(**arguments)

    if discard_next_to_last_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

    return sigmas

def kdiffusion_sampler(
    unet,
    alphas_cumprod,
    prediction_type,
    progress_bar,

    latents,
    num_inference_steps, 
    guidance_scale, 
    prompt_embeds, 
    callback, 
    callback_steps, 
    seed, 
    width, 
    height,

    eta_noise_seed_delta, 
    seed_resize_from_h, 
    seed_resize_from_w,
    randn_source,

    # passed to denoiser
    denoiser_enable_quantization,

    # passed to sigma builder
    sigma_scheduler_name,
    sigma_use_old_karras_scheduler,
    sigma_always_discard_next_to_last,
    sigma_rho,
    sigma_min,
    sigma_max,

    # passed to sampler function
    sampler_name,
    sampler_eta,
    sampler_churn,
    sampler_tmin,
    sampler_tmax,
    sampler_noise,
):
    sampler_tuple = next((sampler for sampler in samplers_k_diffusion if sampler[0] == sampler_name), None)
    if sampler_tuple is None:
        raise ValueError(f"Sampler {sampler_name} not found")
    
    sampler_func = getattr(sampling, sampler_tuple[1])
    sampler_info = sampler_tuple[2]

    denoiser = create_denoiser(
        unet = unet, 
        alphas_cumprod = alphas_cumprod, 
        prediction_type = prediction_type, 
        denoiser_enable_quantization = denoiser_enable_quantization,
        device = latents.device,
        dtype = latents.dtype,
    )


    scheduler_name = sampler_info.get('scheduler', sigma_scheduler_name)

    if scheduler_name == "karras" and sigma_use_old_karras_scheduler:
        sigma_min = 0.1
        sigma_max = 10

    print("sampler: ", sampler_tuple)
    
    sigmas = build_sigmas(
        # TODO: second_order is not simply multiplying steps by 2, but it sort of works. in a111 this is done in prompt_parser.py somehow
        steps = num_inference_steps,# * 2 if sampler_info.get('second_order', False) else num_inference_steps,
        denoiser=denoiser,
        discard_next_to_last_sigma = sampler_info.get('discard_next_to_last_sigma', sigma_always_discard_next_to_last),
        scheduler_name = scheduler_name,
        
        custom_rho = sigma_rho,
        custom_sigma_min = sigma_min,
        custom_sigma_max = sigma_max,
    )

    sigmas = sigmas.to(latents.device, latents.dtype)

    progress_bar = progress_bar(total=len(sigmas))
    
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

    def create_noise_sampler():
        if sampler_info.get('brownian_noise', False):
            # TODO current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
            return k_diffusion.sampling.BrownianTreeNoiseSampler(latents, sigmas[sigmas > 0].min(), sigmas.max(), seed=[seed])

        # https://github.com/crowsonkb/k-diffusion/issues/25
        # the original a1111 implementation overrides torch.randn_like but we can use the generator argument instead
        if eta_noise_seed_delta > 0: # TODO: or len(seeds) > 1 and opts.enable_batch_seeds
            gen = latent_noise.create_a1111_sampler_generator(seed, eta_noise_seed_delta)
            noise_shape = latent_noise.get_noise_shape(width, height, seed_resize_from_h, seed_resize_from_w)

            def noise(sigma, sigma_next): 
                t = torch.randn(
                    noise_shape, 
                    layout=latents.layout, 
                    device=shared.gpu if randn_source == "gpu" else shared.cpu, 
                    dtype=shared.dtype, 
                    generator=gen
                ).to(shared.device)


                print("NOISE", t.sum())
                return t

            return noise
    
    # in a1111 DPM Fast and DPM adaptive gets the default eta of 1.0 from the ui, but in kdiffusion the eta is 0.0 by default
    # this is a subtle bug in a1111, but we need to replicate it
    if sampler_eta is None:
        sampler_eta = sampler_info.get('default_eta', None)

    # this depends on kdiffusion not changing arguments around too much
    sampler_args = {
        "n": num_inference_steps,
        "model":  sampler_step,
        "x": latents,
        "callback": sampler_callback,
        "sigmas": sigmas,
        "sigma_min": denoiser.sigmas[0].item(),
        "sigma_max": denoiser.sigmas[-1].item(),
        "noise_sampler": create_noise_sampler(),

        # extra arguments
        "eta": sampler_eta,
        "s_churn": sampler_churn,
        "s_tmin": sampler_tmin,
        "s_tmax": sampler_tmax,
        "s_noise": sampler_noise,
    }
    
    # remove all arguments that are not in the sampler function
    parameters = inspect.signature(sampler_func).parameters
    for key in sampler_args.copy().keys():
        if key not in parameters or sampler_args[key] is None:
            del sampler_args[key]

    print(f"Running sampler {sampler_name} with arguments:")

    for key, val in sampler_args.items():
        if key == "eta" or key == "s_noise" or key == "s_churn" or key == "s_tmin" or key == "s_tmax" or key == "sigmas" or key == "sigma_min" or key == "sigma_max":
            print(f"\t{key} = {val}")
    
    return sampler_func(**sampler_args)

