from txt2img import txt2img
from kdiffusion import samplers_k_diffusion

samplers = [
    "Euler a",
    "Euler",
    "LMS",
    "Heun", # very similar
    "DPM2",
    "DPM2 a", # very similar
    "DPM++ 2S a",
    "DPM++ 2M",
    "DPM++ SDE", 
    "DPM++ 2M SDE",
    "DPM fast", # very similar
    "DPM adaptive",
    "LMS Karras",
    "DPM2 Karras",
    "DPM2 a Karras", # TODO: completely different
    "DPM++ 2S a Karras",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras", # very similar
    "DPM++ 2M SDE Karras",
]

for sampler_name in samplers:
    image = txt2img(
        checkpoint = "juggernaut_final",
        
        positive = """old man""",
        negative = """""",
        
        steps = 20,
        seed = 1337,  

        width = 512,
        height = 512,
        cfg_scale = 7.5,
        sampler_name = sampler_name,
        randn_source = "gpu",
    )

    image.save(f"../output/{sampler_name}.png")