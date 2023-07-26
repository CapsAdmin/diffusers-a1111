from diffusers import StableDiffusionPipeline, PNDMScheduler, UniPCMultistepScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler
import kdiffusion
import prompts
import shared
import resources
from PIL import Image
import latent_noise
from stable_diffusion_pipeline_custom_call import StableDiffusionPipeline__call__WithCustomDenoising

def txt2img(
    checkpoint = "juggernaut_final", 
    width = 512,
    height = 512,
    batch_size = 1,

    ## Conditioning
        positive = "", 
        negative = "", 
        cfg_scale = 7,

        # Clip skip (ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer) https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip
        CLIP_stop_at_last_layers = 1,
    ##

    # Which algorithm is used to produce the image
    sampler_name = "UniPC",

    steps = 20,
    
    ## Denoiser arguments
        # Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply.
        enable_quantization = True,
    ##

    ## Seed
        seed = 0,

        # Random number generator source. (changes seeds drastically; use CPU to produce the same picture across different videocard vendors)
        randn_source = "gpu",

        # Eta noise seed delta (ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images)
        eta_noise_seed_delta = 0,

        subseed_strength = 0,
        sub_seed = 0,
        seed_resize_from_h = 0,
        seed_resize_from_w = 0,
    ##

    ## Other Sampler arguments
        s_churn = None,
        s_tmin = None,
        s_tmax = None, # this doesn't actually exist in the a1111 ui, but it's in the code
        s_noise = None,
        # Eta for ancestral samplers, noise multiplier; applies to Euler a and other samplers that have a in them
        eta_ancestral = None,
    ##
    
    ## Other Sampler sigma arguments
        # scheduler type (lets you override the noise schedule for k-diffusion samplers; choosing Automatic disables the three parameters below)
        k_sched_type = None,
        
        # sigma min (0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler)
        sigma_min = None,

        # sigma max (0 = default (~14.6); maximum noise strength for k-diffusion noise schedule)
        sigma_max = None,

        #  rho (0 = default (7 for karras, 1 for polyexponential); higher values result in a more steep noise schedule (decreases faster))
        rho = None,

        # Always discard next-to-last sigma https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044
        always_discard_next_to_last_sigma = False,

        # Use old karras scheduler sigmas (0.1 to 10).
        use_old_karras_scheduler_sigmas = False,
    ##

    # Do not make DPM++ SDE deterministic across different batch sizes.
    no_dpmpp_sde_batch_determinism = False,

    # Do not fix prompt schedule for second order samplers.
    dont_fix_second_order_samplers_schedule = False,

    # Make K-diffusion samplers produce same images in a batch as when making a single image
    k_diffusion_batch = True,


    # DDIM Sampler
        # Eta for DDIM, noise multiplier; higher = more unperdictable results
        eta_ddim = 0.0,
        ddim_discretize = "uniform", # quad
    ##
    
    ## UniPC Sampler
        # UniPC variant
        uni_pc_variant = "bh1", # bh2, vary_coef

        # UniPC skip type
        uni_pc_skip_type = "time_uniform", # time_quadratic logSNR

        # UniPC order (must be < sampling steps)
        uni_pc_order = 3,

        # UniPC lower order final
        uni_pc_lower_order_final = True,
    ##
):
    pipe = StableDiffusionPipeline.from_single_file(
        resources.checkpoints[checkpoint], 
        local_files_only = True,
        controlnet = False, # not sure why this is needed. Loading the safetensors from disk makes diffusers think it's a controlnet module
        torch_dtype = shared.dtype, 
    )

    pipe.to(shared.device)
    pipe.safety_checker = None
    pipe.enable_vae_slicing()

    default_scheduler = pipe.scheduler

    if sampler_name == "PLMS":
        pipe.scheduler.config.skip_prk_steps = True # https://github.com/huggingface/diffusers/issues/960
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "UniPC":

        pipe.scheduler.config.solver_type = uni_pc_variant
        pipe.scheduler.config.lower_order_final = uni_pc_lower_order_final

        # some of these may not be correct
        # a1111: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/c76a30af41e50932847230631d26bfa9635ebd62/modules/models/diffusion/uni_pc/uni_pc.py#L462
        # diffusers: https://github.com/huggingface/diffusers/blob/5e80827369272f4b24af51573466aba24454b068/src/diffusers/schedulers/scheduling_pndm.py#L181
        if uni_pc_skip_type == "time_uniform":
            pipe.scheduler.config.timestep_spacing = "time_uniform"
        elif uni_pc_skip_type == "time_quadratic":
            pipe.scheduler.config.timestep_spacing = "time_quadratic"
        elif uni_pc_skip_type == "logSNR":
            pipe.scheduler.config.timestep_spacing = "linspace"
        else:
            raise Exception("Unknown skip type")

        pipe.scheduler.config.solver_order = uni_pc_order

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "DDIM":
        pipe.scheduler.config.eta = eta_ddim

        # some of these may not be correct
        # a1111: https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/util.py#L47C1-L47C1
        # diffusers: https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/schedulers/scheduling_ddim.py#L321
        if ddim_discretize == "uniform":
            pipe.scheduler.config.timestep_spacing = "leading"
        elif ddim_discretize == "quad":
            pipe.scheduler.config.timestep_spacing = "linspace"

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif False:
        if sampler_name == "Euler a":
            #pipe.scheduler.config.eta = eta_ancestral # NYI
            #pipe.scheduler.config.s_noise = s_noise # NYI
            #pipe.scheduler.config.s_noise = s_noise # NYI

            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sampler_name == "Euler":
            #pipe.scheduler.config.s_churn = s_churn # NYI
            #pipe.scheduler.config.s_tmin = s_tmin # NYI
            #pipe.scheduler.config.s_noise = s_noise # NYI
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif sampler_name == "Heun":
            #pipe.scheduler.config.s_churn = s_churn # NYI
            #pipe.scheduler.config.s_tmin = s_tmin # NYI
            #pipe.scheduler.config.s_noise = s_noise # NYI
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)        

    if default_scheduler is not pipe.scheduler:
        images = pipe(
            num_inference_steps = steps,
            guidance_scale = cfg_scale,
            num_images_per_prompt = batch_size,
            width = width,
            height = height,

            **prompts.create_a1111_conditioning(pipe, positive, negative, CLIP_stop_at_last_layers),
            generator = latent_noise.create_a1111_sampler_generator(seed, eta_noise_seed_delta, randn_source),
            latents = latent_noise.create_a1111_latent_noise(
                randn_source = randn_source,
                seed = seed, 
                width = width, 
                height = height, 
                eta_noise_seed_delta = eta_noise_seed_delta, 
                batch_size = batch_size, 
                sub_seed = sub_seed, 
                subseed_strength = subseed_strength, 
                seed_resize_from_h = seed_resize_from_h, 
                seed_resize_from_w = seed_resize_from_w
            ),
        ).images
    else:
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

        def custom_denoise(latents, prompt_embeds, callback, callback_steps, width, height):
            return kdiffusion.kdiffusion_sampler(
                unet = pipe.unet,
                alphas_cumprod = pipe.scheduler.alphas_cumprod,
                prediction_type = pipe.scheduler.config.prediction_type,
                progress_bar = pipe.progress_bar,
                latents = latents,
                num_inference_steps = steps, 
                guidance_scale = cfg_scale, 
                prompt_embeds = prompt_embeds, 
                callback = callback, 
                callback_steps = callback_steps, 
                seed = seed, 
                width = width, 
                height = height,
                eta_noise_seed_delta = eta_noise_seed_delta, 
                seed_resize_from_h = seed_resize_from_h, 
                seed_resize_from_w = seed_resize_from_w,
                randn_source = randn_source,
                
                # passed to denoiser
                denoiser_enable_quantization = enable_quantization,

                # passed to sigma builder
                sigma_scheduler_name = k_sched_type,
                sigma_use_old_karras_scheduler = use_old_karras_scheduler_sigmas,
                sigma_always_discard_next_to_last = always_discard_next_to_last_sigma,
                sigma_rho = rho,
                sigma_min = sigma_min,
                sigma_max = sigma_max,

                # passed to sampler function
                sampler_name = sampler_name,
                sampler_eta = eta_ancestral,
                sampler_churn = s_churn,
                sampler_tmin = s_tmin,
                sampler_tmax = s_tmax,
                sampler_noise = s_noise,
            )

        images = StableDiffusionPipeline__call__WithCustomDenoising(
            pipe,
            denoise = custom_denoise,
            num_inference_steps = steps,
            guidance_scale = cfg_scale,
            num_images_per_prompt = batch_size,
            width = width,
            height = height,
            **prompts.create_a1111_conditioning(pipe, positive, negative, CLIP_stop_at_last_layers),
            generator = latent_noise.create_a1111_sampler_generator(seed, eta_noise_seed_delta, randn_source),
            latents = latent_noise.create_a1111_latent_noise(
                randn_source = randn_source,
                seed = seed, 
                width = width, 
                height = height, 
                eta_noise_seed_delta = eta_noise_seed_delta, 
                batch_size = batch_size, 
                sub_seed = sub_seed, 
                subseed_strength = subseed_strength, 
                seed_resize_from_h = seed_resize_from_h, 
                seed_resize_from_w = seed_resize_from_w
            ),
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
