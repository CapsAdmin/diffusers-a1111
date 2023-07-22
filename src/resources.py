import os

WEBUI_MODELS = os.getenv("WEBUI_MODELS")

def valid_extension(filename):
    return filename.endswith(".safetensors") or filename.endswith(".pt")  or filename.endswith(".pth") or filename.endswith(".ckpt")

def crawl_directory(directory):
    output = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if valid_extension(file):
                filname_without_extension = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                output[filname_without_extension] = full_path
    return output

checkpoints = crawl_directory(WEBUI_MODELS + "Stable-diffusion")
embeddings = crawl_directory(WEBUI_MODELS + "embeddings")
hypernetworks = crawl_directory(WEBUI_MODELS + "hypernetworks")
controlnets = crawl_directory(WEBUI_MODELS + "ControlNet")

loras = crawl_directory(WEBUI_MODELS + "Lora")
loras.update(crawl_directory(WEBUI_MODELS + "LyCORIS"))