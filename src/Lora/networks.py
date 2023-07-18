import shared
import os
import re

import Lora.network
import Lora.network_lora
import Lora.network_hada
import Lora.network_ia3
import Lora.network_lokr
import Lora.network_full

import torch
from safetensors.torch import load_file

from typing import Union




module_types = [
    Lora.network_lora.ModuleTypeLora(),
    Lora.network_hada.ModuleTypeHada(),
    Lora.network_ia3.ModuleTypeIa3(),
    Lora.network_lokr.ModuleTypeLokr(),
    Lora.network_full.ModuleTypeFull(),
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

def convert_sd_lora_to_diffusers(lora_state_dict, diffuser_text_encoder, diffuser_unet):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    map = {}
    visited = []

    # directly update weight in diffusers model
    for key in lora_state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        temp_key = ""

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = diffuser_text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = diffuser_unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)

                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                    temp_key += temp_name + "_"

                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
                    temp_key += temp_name + "_"
        
        visited.append(key)

        key_network_without_network_parts, network_part = key.split(".", 1)
        key = convert_diffusers_name_to_compvis(key_network_without_network_parts, False)
        map[key] = curr_layer
        curr_layer.network_layer_name = key

    return map

def load_network(pipe, name, file_path):
    net = Lora.network.Network(name)
    net.mtime = os.path.getmtime(file_path)

    lora = load_file(file_path, device=shared.device)
    # this should not be needed but is here as an emergency fix for an unknown error people are experiencing in 1.2.0
    if not hasattr(pipe, 'network_layer_mapping'):
        pipe.network_layer_mapping = convert_sd_lora_to_diffusers(lora, pipe.text_encoder, pipe.unet)


    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in pipe.network_layer_mapping

    matched_networks = {}

    for key_network, weight in lora.items():
        key_network_without_network_parts, network_part = key_network.split(".", 1)

        key = convert_diffusers_name_to_compvis(key_network_without_network_parts, is_sd2)
        pipe_module = pipe.network_layer_mapping.get(key, None)

        if pipe_module is None:
            m = re_x_proj.match(key)
            if m:
                pipe_module = pipe.network_layer_mapping.get(m.group(1), None)

        # SDXL loras seem to already have correct compvis keys, so only need to replace "lora_unet" with "diffusion_model"
        if pipe_module is None and "lora_unet" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("lora_unet", "diffusion_model")
            pipe_module = pipe.network_layer_mapping.get(key, None)
        elif pipe_module is None and "lora_te1_text_model" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("lora_te1_text_model", "0_transformer_text_model")
            pipe_module = pipe.network_layer_mapping.get(key, None)

        if pipe_module is None:
            keys_failed_to_match[key_network] = key
            continue

        if key not in matched_networks:
            matched_networks[key] = Lora.network.NetworkWeights(network_key=key_network, sd_key=key, w={}, sd_module=pipe_module)

        matched_networks[key].w[network_part] = weight

    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                break

        if net_module is None:
            raise AssertionError(f"Could not find a module type (out of {', '.join([x.__class__.__name__ for x in module_types])}) that would accept those keys: {', '.join(weights.w)}")

        net.modules[key] = net_module

    if keys_failed_to_match:
        print(f"Failed to match keys when loading network {file_path}: {keys_failed_to_match}")

    return net


def load_networks(pipe, names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    already_loaded = {}

    for net in loaded_networks:
        if net.name in names:
            already_loaded[net.name] = net

    loaded_networks.clear()

    networks_on_disk = [available_network_aliases.get(name, None) for name in names]
    failed_to_load_networks = []


    for i, name in enumerate(names):
        net = already_loaded.get(name, None)

        network_on_disk = networks_on_disk[i]

        if network_on_disk is not None:
            if net is None or os.path.getmtime(network_on_disk) > net.mtime:
                try:
                    net = load_network(pipe, name, network_on_disk)
                except Exception as e:
                    print(e, f"loading network {network_on_disk}")
                    continue

            net.mentioned_name = name
            
        if net is None:
            failed_to_load_networks.append(name)
            print(f"Couldn't find network with name {name}")
            continue

        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        loaded_networks.append(net)

    if failed_to_load_networks:
        print("Failed to find networks: " + ", ".join(failed_to_load_networks))


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)

    if weights_backup is None:
        return

    if isinstance(self, torch.nn.MultiheadAttention):
        self.in_proj_weight.copy_(weights_backup[0])
        self.out_proj.weight.copy_(weights_backup[1])
    else:
        with torch.no_grad():
            self.weight.copy_(weights_backup)


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(torch.device("cpu"), copy=True), self.out_proj.weight.to(torch.device("cpu"), copy=True))
        else:
            weights_backup = self.weight.to(torch.device("cpu"), copy=True)

        self.network_weights_backup = weights_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)

        for net in loaded_networks:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                with torch.no_grad():
                    updown = module.calc_updown(self.weight)

                    if len(self.weight.shape) == 4 and self.weight.shape[1] == 9:
                        # inpainting model. zero pad updown to make channel[1]  4 to 9
                        updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
                    
                    self.weight += updown
                    continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                with torch.no_grad():
                    print("yes")
                    updown_q = module_q.calc_updown(self.in_proj_weight)
                    updown_k = module_k.calc_updown(self.in_proj_weight)
                    updown_v = module_v.calc_updown(self.in_proj_weight)
                    updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                    updown_out = module_out.calc_updown(self.out_proj.weight)

                    self.in_proj_weight += updown_qkv
                    self.out_proj.weight += updown_out
                    continue

            if module is None:
                continue

            print(f'failed to calculate network weights for layer {network_layer_name}')

        self.network_current_names = wanted_names


def network_forward(module, input, original_forward):
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """

    if len(loaded_networks) == 0:
        return original_forward(module, input)

    input = input.to("gpu")#devices.cond_cast_unet(input)

    network_restore_weights_from_backup(module)
    network_reset_cached_weight(module)

    y = original_forward(module, input)

    network_layer_name = getattr(module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue

        y = module.forward(y, input)

    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None


def network_Linear_forward(self, input):
    network_apply_weights(self)

    return torch.nn.Linear_forward_before_network(self, input)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_network(self, *args, **kwargs)


def network_Conv2d_forward(self, input):
    network_apply_weights(self)

    return torch.nn.Conv2d_forward_before_network(self, input)


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_network(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_network(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_network(self, *args, **kwargs)

re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")

available_network_aliases = {}
loaded_networks = []

def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_network
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_network
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_network
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_network
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_network
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_network

if not hasattr(torch.nn, 'Linear_forward_before_network'):
    torch.nn.Linear_forward_before_network = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_network'):
    torch.nn.Linear_load_state_dict_before_network = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_network'):
    torch.nn.Conv2d_forward_before_network = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_network'):
    torch.nn.Conv2d_load_state_dict_before_network = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_network'):
    torch.nn.MultiheadAttention_forward_before_network = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_network'):
    torch.nn.MultiheadAttention_load_state_dict_before_network = torch.nn.MultiheadAttention._load_from_state_dict

torch.nn.Linear.forward = network_Linear_forward
torch.nn.Linear._load_from_state_dict = network_Linear_load_state_dict
torch.nn.Conv2d.forward = network_Conv2d_forward
torch.nn.Conv2d._load_from_state_dict = network_Conv2d_load_state_dict
torch.nn.MultiheadAttention.forward = network_MultiheadAttention_forward
torch.nn.MultiheadAttention._load_from_state_dict = network_MultiheadAttention_load_state_dict