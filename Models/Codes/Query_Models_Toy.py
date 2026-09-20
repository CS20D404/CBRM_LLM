"""
source myenv/bin/activate
pip install -U mlx-lm mlx-vlm pyyaml
python3 Models/Codes/Query_Models_Toy.py > Models/Codes/Query_Models_Toy.txt
"""

import yaml
from pathlib import Path
from mlx_lm import load as lm_load, generate as lm_generate
from mlx_lm.sample_utils import make_sampler
from mlx_vlm import load as vlm_load, generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
 
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "../Models"
models = yaml.safe_load(open(SCRIPT_DIR / "models.yaml"))
 
PROMPT = "Tell me a joke."
 
for name, info in models.items():
    path = str(MODELS_DIR / name)
    temperature = info["temperature"]
    thinking = info["thinking"]          # True / False / None (unsupported)
    print(f"\n=== {name} (temp={temperature}, thinking={thinking}) ===")
 
    if info["loader"] == "mlx-vlm":
        model, processor = vlm_load(path)
        config = load_config(path)
        template_kwargs = {} if thinking is None else {"enable_thinking": thinking}
        formatted = apply_chat_template(
            processor, config, PROMPT, num_images=0, **template_kwargs
        )
        output = vlm_generate(
            model, processor, formatted,
            max_tokens=200, temperature=temperature, verbose=False,
        )
        print(output)
    else:  # mlx-lm
        model, tokenizer = lm_load(path)
        template_kwargs = {} if thinking is None else {"enable_thinking": thinking}
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            add_generation_prompt=True,
            **template_kwargs,
        )
        sampler = make_sampler(temp=temperature)
        output = lm_generate(
            model, tokenizer, prompt=formatted, sampler=sampler, max_tokens=200, verbose=False,
        )
        print(output)