import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import utils
import shared


def txt2img(checkpoint, positive, negative, steps, seed):
    pipe = utils.load_checkpoint(StableDiffusionPipeline, checkpoint)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    [positive_conditioning, negative_conditioning] = utils.apply_prompts(pipe, positive, negative)

    image = pipe(
        prompt_embeds = positive_conditioning,
        negative_prompt_embeds = negative_conditioning,
        num_inference_steps = steps,
        generator = torch.Generator(device=shared.device).manual_seed(seed)
    ).images[0]

    return image
