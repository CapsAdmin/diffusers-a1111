# this approximates the seed implementation from a1111, but since diffusers samplers are a bit different, it's not exactly the same but it's close enough
import torch
import shared

opt_f = 8
opt_C = 4

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

def randn(shape, seed):
    torch.manual_seed(seed)
    return torch.randn(shape, device=shared.device, dtype=shared.dtype)

def get_noise_shape(width, height, seed_resize_from_h=0, seed_resize_from_w=0):
    shape = (opt_C, height // opt_f, width // opt_f)
    return shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//opt_f, seed_resize_from_w//opt_f)  

def create_random_tensors(shape, seeds, seed_delta, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
    xs = []

    do_subseed = subseeds is not None and subseed_strength != 0.0

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//opt_f, seed_resize_from_w//opt_f)        
                
        if do_subseed:
            subseed = 0 if i >= len(subseeds) else subseeds[i]
            subnoise = randn(noise_shape, subseed)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = randn(noise_shape, seed)

        if do_subseed:
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

def create_a1111_latent_noise(seed, width, height, eta_noise_seed_delta=31337, batch_size=1, sub_seed=0, subseed_strength=0, seed_resize_from_h=0, seed_resize_from_w=0):
    return create_random_tensors(
        (opt_C, height // opt_f, width // opt_f), 
        [seed + i for i in range(batch_size)], eta_noise_seed_delta, 
        subseeds=[sub_seed + i for i in range(batch_size)], 
        subseed_strength=subseed_strength, 
        seed_resize_from_h=seed_resize_from_h, 
        seed_resize_from_w=seed_resize_from_w
    )


# some samplers require noise for each step such as the euler sampler
# in a1111 this was done with create_random_tensors but in diffusers 
# this is done internally using a generator. 
# so with euler this doesn't look exactly the same but it's close enough
def create_a1111_sampler_generator(seed, eta_noise_seed_delta=31337):
    return torch.Generator(device=shared.device).manual_seed(seed + eta_noise_seed_delta)


    