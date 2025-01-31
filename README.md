python3 -m venv llm_env
source llm_env/bin/activate  # Activate the environment

pip install --upgrade pip  # Ensure pip is up to date
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft tqdm bitsandbytes


accelerate config
