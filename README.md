python3.10 -m venv .venv

# install torch somehow, i use rocm on linux

python3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.5

python3.10 -m pip install -r requirements.txt
