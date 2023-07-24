import re
import resources
import conditioning
import util

def apply_embeddings(pipe, input_str):
    for name, path in resources.embeddings.items():
        new_string = re.sub("(^|\W)(" + re.escape(name) + ")(\W|$)", r"\1<\2>\3", input_str)

        if new_string != input_str:
            pipe.load_textual_inversion(path)
            input_str = new_string

    return input_str

def apply_tag_weight(pipe, tag_name, path_dict, input_str, found_callback):
    for name, path in path_dict.items():
        pattern = "<"+tag_name+":(" + re.escape(name) + "):([0-9.]+)>"
        matched = re.search(pattern, input_str)
        while matched:
            weight = float(matched.group(2))

            found_callback(pipe, path, weight)
            
            input_str = re.sub(pattern, "", input_str, count=1)
            matched = re.search(pattern, input_str)
    return input_str

def apply_lora(pipe, path, weight):
    from merge_lora_to_pipeline import merge_lora_to_pipeline
    merge_lora_to_pipeline(pipe.text_encoder, pipe.unet, util.load_state_dict(path), weight)

def apply_hypernetwork(pipe, path, weight):
    from merge_hypernetwork_to_pipeline import merge_hypernetwork_to_pipeline
    merge_hypernetwork_to_pipeline(pipe.text_encoder, pipe.unet, util.load_state_dict(path), weight)

def create_a1111_conditioning(pipe, positive: str, negative: str, clip_skip = 1):

    positive = apply_embeddings(pipe, positive)
    negative = apply_embeddings(pipe, negative)
 
    positive = apply_tag_weight(pipe, "lora", resources.loras, positive, apply_lora)
    positive = apply_tag_weight(pipe, "lycoris", resources.loras, positive, apply_lora)

    negative = apply_tag_weight(pipe, "lora", resources.loras, negative, apply_lora)
    negative = apply_tag_weight(pipe, "lycoris", resources.loras, negative, apply_lora)

    positive = apply_tag_weight(pipe, "hypernetwork", resources.hypernetworks, positive, apply_hypernetwork)
    negative = apply_tag_weight(pipe, "hypernetwork", resources.hypernetworks, negative, apply_hypernetwork)

    positive_cond, negative_cond = conditioning.text_embeddings(pipe, positive, negative, clip_skip)
    
    return {
        'prompt_embeds': positive_cond,
        'negative_prompt_embeds': negative_cond,
    }
