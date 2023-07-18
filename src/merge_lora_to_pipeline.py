import torch
from safetensors.torch import load_file
import re
import Lora.network_lora
import Lora.network_hada
import Lora.network_ia3
import Lora.network_lokr
import Lora.network_full
import Lora.network

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

def merge_lora_to_pipeline(pipeline, checkpoint_path, alpha, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    visited = []

    # directly update weight in diffusers model
    for key, weight in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
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

        pair_keys = []
        UP = None
        DOWN = None
        
        if "lora_down" in key or "lora_up" in key:
            UP = "lora_up"
            DOWN = "lora_down"
        
        if "lokr_w1" in key or "lokr_w2" in key:
            UP = "lokr_w2"
            DOWN = "lokr_w1"

        if UP and DOWN:
            if DOWN in key:
                pair_keys.append(key.replace(DOWN, UP))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace(UP, DOWN))

        key_network_without_network_parts, network_part = key.split(".", 1)
        compvis_key = convert_diffusers_name_to_compvis(key_network_without_network_parts, False)
        net = Lora.network.Network("test")
        net.te_multiplier = alpha
        net.unet_multiplier = alpha
        #net.dyn_dim = 2 not sure what this is
        weights = Lora.network.NetworkWeights(network_key=key, sd_key=compvis_key, w={}, sd_module=curr_layer)

        if UP and DOWN:
            weights.w[network_part.replace(DOWN, UP)] = state_dict[pair_keys[0]]
            weights.w[network_part] = state_dict[pair_keys[1]]
        else:
            weights.w[network_part] = state_dict[key]

        updated = False
        for module_type in module_types:
            m = module_type.create_module(net, weights)
            if m is not None:
                updated = True
                curr_layer.weight.data += m.calc_updown(weight)

        if not updated:
            #print(network_part, network_part.replace(DOWN, UP))
            print(f'{key} matched no layer')

        # update weight
        """
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(dtype)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(dtype)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(dtype)
            weight_down = state_dict[pair_keys[1]].to(dtype)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        """
        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline

