import torch
from safetensors.torch import load_file
from collections import namedtuple
import re
import shared

def make_weight_cp(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)

def rebuild_conventional(up, down, shape, dyn_dim=None):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    if dyn_dim is not None:
        up = up[:, :dyn_dim]
        down = down[:dyn_dim, :]
    return (up @ down).reshape(shape)


def rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)

NetworkWeights = namedtuple('NetworkWeights', ['network_key', 'sd_key', 'w', 'sd_module'])
class NetworkModuleBase:
    dyn_dim = None
    unet_multiplier = 1.0
    te_multiplier = 1.0

    def __init__(self, weights: NetworkWeights):
        self.network_key = weights.network_key
        self.sd_key = weights.sd_key
        self.sd_module = weights.sd_module

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.dim = None
        self.bias = weights.w.get("bias")
        self.alpha = weights.w["alpha"].item() if "alpha" in weights.w else None
        self.scale = weights.w["scale"].item() if "scale" in weights.w else None

    def multiplier(self):
        if 'transformer' in self.sd_key[:20]:
            return self.te_multiplier
        else:
            return self.unet_multiplier

    def calc_scale(self):
        if self.scale is not None:
            return self.scale
        if self.dim is not None and self.alpha is not None:
            return self.alpha / self.dim

        return 1.0

    def finalize_updown(self, updown, orig_weight, output_shape):
        if self.bias is not None:
            updown = updown.reshape(self.bias.shape)
            updown += self.bias.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = updown.reshape(output_shape)

        if len(output_shape) == 4:
            updown = updown.reshape(output_shape)

        if orig_weight.size().numel() == updown.size().numel():
            updown = updown.reshape(orig_weight.shape)

        return updown * self.calc_scale() * self.multiplier()

    def calc_updown(self, target):
        raise NotImplementedError()

    def forward(self, x, y):
        raise NotImplementedError()

class NetworkModuleLora(NetworkModuleBase):
    @staticmethod
    def from_weights(weights: NetworkWeights):
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(weights)

        return None

    def __init__(self, weights: NetworkWeights):
        super().__init__(weights)

        self.up_model = self.create_module(weights.w, "lora_up.weight")
        self.down_model = self.create_module(weights.w, "lora_down.weight")
        self.mid_model = self.create_module(weights.w, "lora_mid.weight", none_ok=True)

        self.dim = weights.w["lora_down.weight"].shape[0]

    def create_module(self, weights, key, none_ok=False):
        weight = weights.get(key)

        if weight is None and none_ok:
            return None

        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention]
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]

        if is_linear:
            weight = weight.reshape(weight.shape[0], -1)
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif is_conv and key == "lora_down.weight" or key == "dyn_up":
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        elif is_conv and key == "lora_mid.weight":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        elif is_conv and key == "lora_up.weight" or key == "dyn_down":
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            raise AssertionError(f'Lora layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')

        with torch.no_grad():
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            module.weight.copy_(weight)

        module.to(device=shared.cpu, dtype=shared.dtype)
        module.weight.requires_grad_(False)

        return module

    def calc_updown(self, orig_weight):
        up = self.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = self.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [up.size(0), down.size(1)]
        if self.mid_model is not None:
            # cp-decomposition
            mid = self.mid_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = rebuild_conventional(up, down, output_shape, self.dyn_dim)

        return self.finalize_updown(updown, orig_weight, output_shape)

    def forward(self, x, y):
        self.up_model.to(device=shared.device)
        self.down_model.to(device=shared.device)

        return y + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()


def make_kron(orig_shape, w1, w2):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(orig_shape)

class NetworkModuleLokr(NetworkModuleBase):
    @staticmethod
    def from_weights(weights: NetworkWeights):
        has_1 = "lokr_w1" in weights.w or ("lokr_w1_a" in weights.w and "lokr_w1_b" in weights.w)
        has_2 = "lokr_w2" in weights.w or ("lokr_w2_a" in weights.w and "lokr_w2_b" in weights.w)
        if has_1 and has_2:
            return NetworkModuleLokr(weights)

        return None

    def __init__(self, weights: NetworkWeights):
        super().__init__(weights)

        self.w1 = weights.w.get("lokr_w1")
        self.w1a = weights.w.get("lokr_w1_a")
        self.w1b = weights.w.get("lokr_w1_b")
        self.dim = self.w1b.shape[0] if self.w1b is not None else self.dim
        self.w2 = weights.w.get("lokr_w2")
        self.w2a = weights.w.get("lokr_w2_a")
        self.w2b = weights.w.get("lokr_w2_b")
        self.dim = self.w2b.shape[0] if self.w2b is not None else self.dim
        self.t2 = weights.w.get("lokr_t2")

    def calc_updown(self, orig_weight):
        if self.w1 is not None:
            w1 = self.w1.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
            w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
            w1 = w1a @ w1b

        if self.w2 is not None:
            w2 = self.w2.to(orig_weight.device, dtype=orig_weight.dtype)
        elif self.t2 is None:
            w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = w2a @ w2b
        else:
            t2 = self.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = make_weight_cp(t2, w2a, w2b)

        output_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape

        updown = make_kron(output_shape, w1, w2)

        return self.finalize_updown(updown, orig_weight, output_shape)
class NetworkModuleIa3(NetworkModuleBase):
    @staticmethod 
    def from_weights(weights: NetworkWeights):
        if all(x in weights.w for x in ["weight"]):
            return NetworkModuleIa3(weights)

        return None
    def __init__(self, weights: NetworkWeights):
        super().__init__(weights)
        self.w = weights.w["weight"]
        self.on_input = weights.w["on_input"].item()

    def calc_updown(self, orig_weight):
        w = self.w.to(orig_weight.device, dtype=orig_weight.dtype)
        
        output_shape = [w.size(0), orig_weight.size(1)]
        if self.on_input:
            output_shape.reverse()
        else:
            w = w.reshape(-1, 1)

        updown = orig_weight * w

        return self.finalize_updown(updown, orig_weight, output_shape)

class NetworkModuleHada(NetworkModuleBase):
    @staticmethod
    def from_weights(weights: NetworkWeights):
        if all(x in weights.w for x in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"]):
            return NetworkModuleHada(weights)

        return None

    def __init__(self, weights: NetworkWeights):
        super().__init__(weights)

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.w1a = weights.w["hada_w1_a"]
        self.w1b = weights.w["hada_w1_b"]
        self.dim = self.w1b.shape[0]
        self.w2a = weights.w["hada_w2_a"]
        self.w2b = weights.w["hada_w2_b"]

        self.t1 = weights.w.get("hada_t1")
        self.t2 = weights.w.get("hada_t2")

    def calc_updown(self, orig_weight):
        w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [w1a.size(0), w1b.size(1)]

        if self.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = self.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = make_weight_cp(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = rebuild_conventional(w1a, w1b, output_shape)

        if self.t2 is not None:
            t2 = self.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = make_weight_cp(t2, w2a, w2b)
        else:
            updown2 = rebuild_conventional(w2a, w2b, output_shape)

        updown = updown1 * updown2

        return self.finalize_updown(updown, orig_weight, output_shape)
class NetworkModuleFull(NetworkModuleBase):
    @staticmethod
    def from_weights(weights: NetworkWeights):
        if all(x in weights.w for x in ["diff"]):
            return NetworkModuleFull(weights)

        return None

    def __init__(self, weights: NetworkWeights):
        super().__init__(weights)
        self.weight = weights.w.get("diff")

    def calc_updown(self, orig_weight):
        output_shape = self.weight.shape
        updown = self.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        return self.finalize_updown(updown, orig_weight, output_shape)

module_types = [
    NetworkModuleLora,
    NetworkModuleHada,
    NetworkModuleIa3,
    NetworkModuleLokr,
    NetworkModuleFull,
]

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key

def merge_lora_to_pipeline(pipeline, checkpoint_path, alpha, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # load LoRA weight from .safetensors

    state_dict = None
    if checkpoint_path.endswith(".safetensors"):
       state_dict = load_file(checkpoint_path, device=device)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)

    visited = []
    matched_networks = {}
    # directly update weight in diffusers model
    for network_key, weight in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if network_key in visited:
            continue

        if "text" in network_key:
            layer_infos = network_key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = network_key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
            
        key_network_without_network_parts, network_part = network_key.split(".", 1)
        compvis_key = convert_diffusers_name_to_compvis(key_network_without_network_parts, False)

        if compvis_key not in matched_networks:
            matched_networks[compvis_key] = NetworkWeights(network_key=network_key, sd_key=compvis_key, w={}, sd_module=curr_layer)

        matched_networks[compvis_key].w[network_part] = weight

    for key, weights in matched_networks.items():
        updated = False
        for module in module_types:
            m = module.from_weights(weights)
            if m is not None:
                m.te_multiplier = alpha
                m.unet_multiplier = alpha
                m.dyn_dim = None
                updated = True
                
                try:
                    weights.sd_module.weight.data += m.calc_updown(weights.sd_module.weight.data)
                except Exception as e:
                    print("FAIL", network_part, curr_layer.weight.data.shape, weight.shape)
                    print("\t", network_key)
                    print("\t", compvis_key)
                    for k, v in weights.w.items():
                        print("\t", k, v.shape)

        if not updated:
            print(f'{network_key} matched no layer')
            pass