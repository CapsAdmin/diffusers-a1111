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

    state_dict = None
    if checkpoint_path.endswith(".safetensors"):
       state_dict = load_file(checkpoint_path, device=device)
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
    
    net = Lora.network.Network("test")
    net.te_multiplier = alpha
    net.unet_multiplier = alpha
    #net.dyn_dim = 2 # not sure what this is

    matching_keys = {
        "lora": ["lora_up.weight", "lora_down.weight", "lora_mid.weight"],
        "lokr": ["lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2"],
        "ia3": ["weight", "on_input"],
        "hada": ["alpha", "hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2"],
        "full": ["diff"]
    }

    visited = []
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

        matched_keys = []
        for module_type, keys in matching_keys.items():
            for a in keys:
                if a in network_key:
                    for b in keys:
                        replaced_key = network_key.replace(a, b)
                        if replaced_key in state_dict and not replaced_key in matched_keys:
                            matched_keys.append(replaced_key)

        key_network_without_network_parts, network_part = network_key.split(".", 1)   

        compvis_key = convert_diffusers_name_to_compvis(key_network_without_network_parts, False)

        weights = Lora.network.NetworkWeights(network_key=network_key, sd_key=compvis_key, w={}, sd_module=curr_layer)

        if matched_keys:
            for item in matched_keys:
                key_network_without_network_parts, network_part2 = item.split(".", 1)
                weights.w[network_part2] = state_dict[item]
        else:
            weights.w[network_part] = state_dict[network_key]

        updated = False
        for module_type in module_types:
            m = module_type.create_module(net, weights)
            if m is not None:
                updated = True
                
                try:
                    updown = m.calc_updown(curr_layer.weight.data)
                    curr_layer.weight.data += updown
                    print(compvis_key, weights.w.keys())
                    #print("SUCCESS", network_part, curr_layer.weight.data.shape, weight.shape)
                    #for k, v in weights.w.items():
                        #print("\t", k, v.shape)
                except Exception as e:
                    print("FAIL", network_part, curr_layer.weight.data.shape, weight.shape)
                    print("\t", network_key)
                    print("\t", compvis_key)
                    for k, v in weights.w.items():
                        print("\t", k, v.shape)

        if not updated:
            print(f'{network_key} matched no layer')
            pass

        for item in matched_keys:
            visited.append(item)

    return pipeline

