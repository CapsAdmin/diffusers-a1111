export DISABLE_TELEMETRY=YES
export HF_HOME="../.diffusers"
export WEBUI_MODELS="/home/caps/projects/stable-diffusion/models/"
source .venv/bin/activate
cd src
python3.10 main.py