# Setup

    python3.10 -m venv .venv
    python3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.5
    python3.10 -m pip install -r requirements.txt

# Currently implemented

This implements a lot of the txt2img features from a1111 with diffusers. The goal is to be able to reproduce results from a1111 in diffusers

- seed resize
- variation seed
- k-diffusion samplers
- advanced sampler parameters
- clip skip
- prompt weighing
- lora, lokr, ia3, hada and full

# Not yet implemented

- hypernetworks
- face upsacling
- tile
- highres fix
- img2img and inpaint
- controlnet extension (some work)
- AND, and other prompt features

```python
from txt2img import txt2img

image = txt2img(
    checkpoint = "juggernaut_final",

    positive = """(old man:1.3)  <lora:studioGhibliStyle_offset:1>""",
    negative = """""",

    steps = 20,
    seed = 1337,

    width = 512,
    height = 512,

    batch_size = 1, # doesn't work yet
    cfg_scale = 7.5,

    subseed_strength = 0,
    sub_seed = 1,
    seed_resize_from_h = 0,
    seed_resize_from_w = 0,

    # Clip skip (ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer) https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip
    CLIP_stop_at_last_layers = 1,

    # Which algorithm is used to produce the image
    sampler_name = "UniPC",

    # Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply.
    enable_quantization = True,

    # Use old karras scheduler sigmas (0.1 to 10).
    use_old_karras_scheduler_sigmas = False,

    # Do not make DPM++ SDE deterministic across different batch sizes.
    no_dpmpp_sde_batch_determinism = False,

    # Do not fix prompt schedule for second order samplers.
    dont_fix_second_order_samplers_schedule = False,

    # Make K-diffusion samplers produce same images in a batch as when making a single image
    k_diffusion_batch = True,

    # Random number generator source. (changes seeds drastically; use CPU to produce the same picture across different videocard vendors)
    randn_source = "gpu",

    # sampler parameters from a1111

    # these affect sample_euler, sample_heun and sample_dpm_2
    s_churn = 0.0,
    s_tmin = 0.0,
    s_tmax = 0.0, # this doesn't actually exist in the a1111 ui, but it's in the code
    s_noise = 1.0,

    # scheduler type (lets you override the noise schedule for k-diffusion samplers; choosing Automatic disables the three parameters below)
    k_sched_type = "Automatic",

    # sigma min (0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler)
    sigma_min = 0,

    # sigma max (0 = default (~14.6); maximum noise strength for k-diffusion noise schedule)
    sigma_max = 0,

    #  rho (0 = default (7 for karras, 1 for polyexponential); higher values result in a more steep noise schedule (decreases faster))
    rho = 0.0,

    # Eta noise seed delta (ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images)
    eta_noise_seed_delta = 0,

    # Eta for ancestral samplers, noise multiplier; applies to Euler a and other samplers that have a in them
    eta_ancestral = 1.0,

    # Eta for DDIM, noise multiplier; higher = more unperdictable results
    eta_ddim = 0.0,

    ddim_discretize = "uniform", # quad

    # Always discard next-to-last sigma https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044
    always_discard_next_to_last_sigma = False,

    # UniPC variant
    uni_pc_variant = "bh1", # bh2, vary_coef

    # UniPC skip type
    uni_pc_skip_type = "time_uniform", # time_quadratic logSNR

    # UniPC order (must be < sampling steps)
    uni_pc_order = 3,

    # UniPC lower order final
    uni_pc_lower_order_final = True,
)

image.save("../output/output.png")
```
