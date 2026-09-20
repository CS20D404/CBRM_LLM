"""
source myenv/bin/activate
pip install -U mlx-lm mlx-vlm pyyaml
python3 Models/Codes/Temperature_Analysis.py > Models/Codes/Temperature_Analysis.txt
"""

import yaml
from pathlib import Path
from tqdm import tqdm
from mlx_lm import load as lm_load, generate as lm_generate
from mlx_lm.sample_utils import make_sampler
from mlx_vlm import load as vlm_load, generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "../Models"
models = yaml.safe_load(open(SCRIPT_DIR / "models.yaml"))

PROMPT = "Choose between green and yellow. Reply with only one word. Do not explain your choice."
N_RUNS = 10

for name, info in models.items():
    path = str(MODELS_DIR / name)
    outputs = []

    if info["loader"] == "mlx-vlm":
        model, processor = vlm_load(path)
        config = load_config(path)
        formatted = apply_chat_template(processor, config, PROMPT, num_images=0)
        for _ in tqdm(range(N_RUNS), desc=name):
            out = vlm_generate(model, processor, formatted, max_tokens=10, temperature=0, verbose=False)
            outputs.append((out if isinstance(out, str) else out.text).strip())

    else:  # mlx-lm
        model, tokenizer = lm_load(path)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}], add_generation_prompt=True
        )
        sampler = make_sampler(temp=0)
        for _ in tqdm(range(N_RUNS), desc=name):
            out = lm_generate(model, tokenizer, prompt=formatted, sampler=sampler, max_tokens=10, verbose=False)
            outputs.append(out.strip())

    deterministic = len(set(outputs)) == 1
    print(f"{name}: deterministic={deterministic} outputs={outputs}\n")