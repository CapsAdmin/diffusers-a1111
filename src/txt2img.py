from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler, ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import kdiffusion
import prompts
import shared
import resources
from PIL import Image
import latent_noise

def txt2img(
    checkpoint = "juggernaut_final", 
    positive = "", 
    negative = "", 
    steps = 20, 
    seed = 0,
    width = 512,
    height = 512,
    
    batch_size = 1,
    cfg_scale = 7,
    subseed_strength = 0,
    sub_seed = 1,
    seed_resize_from_h = 0,
    seed_resize_from_w = 0,
    clip_skip = 1,
    
    # Which algorithm is used to produce the image
    sampler_name = "DPM++ 2M SDE",

    # Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply.
    enable_quantization = True,

    # Use old karras scheduler sigmas (0.1 to 10).
    use_old_karras_scheduler_sigmas = False,

    # Do not make DPM++ SDE deterministic across different batch sizes.
    dont_fix_dpmpp_sde_batch_size = False,

    # Do not fix prompt schedule for second order samplers.
    dont_fix_prompt_schedule = False,

    # Make K-diffusion samplers produce same images in a batch as when making a single image
    k_diffusion_batch = True,

    # Random number generator source. (changes seeds drastically; use CPU to produce the same picture across different videocard vendors)
    random_source = "gpu",

    # sampler parameters from a1111
    sigma_churn = 0.0,
    sigma_tmin = 0.0,
    sigma_noise = 1.0,

    # scheduler type (lets you override the noise schedule for k-diffusion samplers; choosing Automatic disables the three parameters below)
    k_sched_type = "Automatic",
    
    # sigma min (0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler)
    sigma_min = 0,

    # sigma max (0 = default (~14.6); maximum noise strength for k-diffusion noise schedule)
    sigma_max = 0,

    #  rho (0 = default (7 for karras, 1 for polyexponential); higher values result in a more steep noise schedule (decreases faster))
    rho = 1.0,

    # Eta noise seed delta (ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images)
    eta_noise_seed_delta = 31337,

    # Always discard next-to-last sigma https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044
    always_discard_next_to_last_sigma = True,

    unipc_variant = "bh1", # bh2, vary_coef
    unipc_skip_type = "time_uniform", # time_quadratic logSNR

    # UniPC order (must be < sampling steps)
    unipc_order = 3,

    unipc_lower_order_final = True,
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

    from diffusers import LMSDiscreteScheduler
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    images = kdiffusion.StableDiffusionPipeline__call__WithCustomDenoising(
        pipe,
        
        seed,
        eta_noise_seed_delta,
        seed_resize_from_h,
        seed_resize_from_w,
        enable_quantization,
        use_old_karras_scheduler_sigmas,
        always_discard_next_to_last_sigma,
        sampler_name,
        k_sched_type,
        rho,
        sigma_min,
        sigma_max,

        num_inference_steps = steps,
        guidance_scale = cfg_scale,
        num_images_per_prompt = batch_size,
        width = width,
        height = height,
        **prompts.create_a1111_conditioning(pipe, positive, negative, clip_skip),
        generator = latent_noise.create_a1111_sampler_generator(seed, eta_noise_seed_delta),
        latents = latent_noise.create_a1111_latent_noise(
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
