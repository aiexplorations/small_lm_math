# small_lm_math
Small language model training using publicly available datasets, for performing mathematical computations such as basic math, going up to calculus and ODE/PDE solutions.

```bash
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/lora

```

```bash
# Prepare your dataset as JSONL (e.g., {"text": "Question: ... Answer: ..."})
python lora.py --model mlx-community/Mistral-7B-v0.2 --data path/to/gsm8k.jsonl --train
```

