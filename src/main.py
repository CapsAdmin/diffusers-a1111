import hacks
import os
from txt2img import txt2img

image = txt2img(
    checkpoint = "juggernaut_final",
    positive = """old man""",
    negative = """""",
    steps = 20,
    seed = 2781143491,
    enable_quantization = True,
    use_old_karras_scheduler_sigmas = False,
    always_discard_next_to_last_sigma = True,
    sampler_name = "DPM++ 2M SDE",
    k_sched_type = "Automatic",
    rho = 1.0,
    sigma_min = 0,
    sigma_max = 0,
)

image.save("../output/output.png")