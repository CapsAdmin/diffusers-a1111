import shared
from compel import Compel
import os
import re
import Lora.networks

WEBUI_MODELS = os.getenv("WEBUI_MODELS")

checkpoints = {}
for root, dirs, files in os.walk(WEBUI_MODELS + "Stable-diffusion"):
    for file in files:
        if file.endswith(".safetensors"):
            filname_without_extension = os.path.splitext(file)[0]
            full_path = os.path.join(root, file)
            checkpoints[filname_without_extension] = full_path

loras = {}
for root, dirs, files in os.walk(WEBUI_MODELS + "Lora"):
    for file in files:
        if file.endswith(".safetensors") or file.endswith(".pt"):
            filname_without_extension = os.path.splitext(file)[0]
            full_path = os.path.join(root, file)
            loras[filname_without_extension] = full_path


embeddings = {}
for root, dirs, files in os.walk(WEBUI_MODELS + "embeddings"):
    for file in files:
        if file.endswith(".pt"):
            filname_without_extension = os.path.splitext(file)[0]
            full_path = os.path.join(root, file)
            embeddings[filname_without_extension] = full_path

def apply_embeddings(pipe, input_str):
    for embedding in embeddings.keys():
        new_string = re.sub("(^|\W)(" + re.escape(embedding) + ")(\W|$)", r"\1<\2>\3", input_str)

        if new_string != input_str:
            pipe.load_textual_inversion(embeddings[embedding])
            input_str = new_string

    return input_str

def apply_tag_weight(pipe, tag_name, dict, input_str):
    for key in dict.keys():
        pattern = "<"+tag_name+":(" + re.escape(key) + "):([0-9.]+)>"
        matched = re.search(pattern, input_str)
        while matched:
            weight = float(matched.group(2))

            #import Lora.networks
            #Lora.networks.available_network_aliases = loras
            #Lora.networks.load_networks(pipe, [key], [weight], [weight], [int(1)])
            
            from merge_lora_to_pipeline import merge_lora_to_pipeline
            merge_lora_to_pipeline(pipe, dict[key], weight, shared.device, shared.dtype)
            
            input_str = re.sub(pattern, "", input_str, count=1)
            matched = re.search(pattern, input_str)
    return input_str

def apply_prompts(pipe, positive: str, negative: str):

    positive = apply_embeddings(pipe, positive)
    negative = apply_embeddings(pipe, negative)
    
    positive = apply_tag_weight(pipe, "lora", loras, positive)
    negative = apply_tag_weight(pipe, "lora", loras, negative)

    print(positive, negative)

    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts=False)
    positive_cond = compel.build_conditioning_tensor(positive)
    negative_cond = compel.build_conditioning_tensor(negative)
    [positive_cond, negative_cond] = compel.pad_conditioning_tensors_to_same_length([positive_cond, negative_cond])
    
    return [positive_cond, negative_cond]
    

def load_checkpoint(pipeline, name):
    pipe = pipeline.from_single_file(
        checkpoints[name], 
        use_safetensors = True,
        torch_dtype=shared.dtype,
    )
    pipe.to(shared.device)
    pipe.safety_checker = None
    pipe.enable_vae_slicing()
    return pipe