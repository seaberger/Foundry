---
base_model: unsloth/gemma-3-27b-it-unsloth-bnb-4bit
library_name: peft
model_name: madison-orpo-v3b-lr2e5
tags:
- base_model:adapter:unsloth/gemma-3-27b-it-unsloth-bnb-4bit
- lora
- orpo
- transformers
- trl
- unsloth
licence: license
pipeline_tag: text-generation
---

# Model Card for madison-orpo-v3b-lr2e5

This model is a fine-tuned version of [unsloth/gemma-3-27b-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/sbergman/foundry/runs/s84fcrt0) 


This model was trained with ORPO, a method introduced in [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691).

### Framework versions

- PEFT 0.16.0
- TRL: 0.19.1
- Transformers: 4.54.0
- Pytorch: 2.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.4

## Citations

Cite ORPO as:

```bibtex
@article{hong2024orpo,
    title        = {{ORPO: Monolithic Preference Optimization without Reference Model}},
    author       = {Jiwoo Hong and Noah Lee and James Thorne},
    year         = 2024,
    eprint       = {arXiv:2403.07691}
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```