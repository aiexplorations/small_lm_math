import logging
from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import GPT2Tokenizer
import numpy as np
import os
from configparser import ConfigParser
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(config_file: str) -> Tuple[List[mx.array], List[mx.array], int, GPT2Tokenizer]:
    logging.info("Loading and preprocessing data")
    config = ConfigParser()
    config.read(config_file)

    dataset_name = config.get("DATA", "dataset_name")
    config_name = config.get("DATA", "config_name")
    split = config.get("DATA", "split")
    tokenizer_name = config.get("TOKENIZER", "tokenizer_name")

    dataset = load_dataset(dataset_name, config_name, split=split)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    seq_len = int(config.get("TRAINING", "seq_len"))
    
    texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in dataset]
    tokens = [tokenizer.encode(
        text, 
        max_length=seq_len,
        truncation=True,
        padding="max_length"
    ) for text in texts]
    tokens = [mx.array(t) for t in tokens]

    split_ratio = config.getfloat("DATA", "split_ratio")
    split_idx = int(split_ratio * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]

    logging.info("Data loaded and preprocessed successfully")
    return train_data, val_data, vocab_size, tokenizer

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer_blocks = [nn.TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)]
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = min(x.shape[1], self.max_seq_len)
        positions = mx.arange(seq_len)
        mask = mx.tril(mx.ones((seq_len, seq_len))) == 0
        x = self.embedding(x[:, :seq_len]) + self.pos_embedding(positions)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        x = self.dropout(x)
        x = self.ln(x)
        return self.fc_out(x)

    def print_model_summary(self) -> None:
        total_params = sum(p.size for p in self.parameters())
        print(f"Model Architecture: {self.__class__.__name__}")
        print(f"Total Parameters: {total_params}")
        print(f"Number of Layers: {len(self.transformer_blocks)}")
        print(f"Embedding Dimension: {self.embedding.embedding_dim}")
        print(f"Number of Heads: {self.transformer_blocks[0].n_heads}")
        print(f"Max Sequence Length: {self.max_seq_len}")

def get_batch(data: List[mx.array], batch_size: int, seq_len: int) -> Tuple[mx.array, mx.array]:
    idx = np.random.randint(0, len(data), batch_size)
    x = mx.stack([data[i][:seq_len] for i in idx])
    y = mx.stack([mx.pad(data[i][1:], (0, max(0, seq_len - data[i][1:].shape[0])))[:seq_len] for i in idx])
    return x, y

def loss_fn(model: nn.Module, x: mx.array, y: mx.array, vocab_size: int) -> mx.array:
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)))

def train_step(model: nn.Module, optimizer: optim.Optimizer, x: mx.array, y: mx.array, step: int, vocab_size: int) -> mx.array:
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, x, y, vocab_size)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    
    if step % 100 == 0:
        save_checkpoint(model, optimizer, step)
    
    return loss

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, step: int, checkpoint_dir: str = "checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    def flatten_params(params, prefix=""):
        flat = {}
        for key, value in params.items():
            if isinstance(value, mx.array):
                flat[f"{prefix}{key}"] = value
            elif isinstance(value, dict):
                flat.update(flatten_params(value, f"{prefix}{key}."))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(flatten_params(item, f"{prefix}{key}.{i}."))
                    else:
                        logging.warning(f"Skipping non-dict list item: {key}[{i}] = {item}")
            else:
                logging.warning(f"Skipping non-array parameter: {key} = {value}")
        return flat
    
    model_params = model.parameters()
    flat_params = flatten_params(model_params)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.npz")
    mx.savez(checkpoint_path, **flat_params)
    
    flat_opt_state = {}
    for i, param_state in enumerate(optimizer.state):
        if isinstance(param_state, dict):
            for key, value in param_state.items():
                if isinstance(value, mx.array):
                    flat_opt_state[f"state_{i}_{key}"] = value
                else:
                    logging.warning(f"Skipping non-array optimizer state: state_{i}_{key} = {value}")
        else:
            logging.warning(f"Skipping non-dict optimizer state: state_{i} = {param_state}")
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_step_{step}.npz")
    if flat_opt_state:
        mx.savez(optimizer_path, **flat_opt_state)
    
    logging.info(f"Checkpoint saved at step {step}")

def train_model(num_steps: int, print_every: int, train_data: List[mx.array], val_data: List[mx.array], model: nn.Module, optimizer: optim.Optimizer, batch_size: int, seq_len: int, vocab_size: int) -> Tuple[List[float], List[float]]:
    logging.info("Starting training")
    train_losses = []
    val_losses = []
    steps = []
    
    for step in range(num_steps):
        x, y = get_batch(train_data, batch_size, seq_len)
        loss = train_step(model, optimizer, x, y, step, vocab_size)
        train_losses.append(loss.item())
        
        if step % print_every == 0:
            val_x, val_y = get_batch(val_data, batch_size, seq_len)
            val_loss = loss_fn(model, val_x, val_y, vocab_size)
            val_losses.append(val_loss.item())
            steps.append(step)
            logging.info(f"Step {step}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), train_losses, label="Training Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.close()
    logging.info("Loss plot saved as 'loss_plot.png'")
    
    return train_losses, val_losses

def coach(model: nn.Module, tokenizer: GPT2Tokenizer, prompt: str, max_tokens: int, seq_len: int, temperature: float = 0.7) -> str:
    logging.info("Generating response using coach interface")
    input_tokens = mx.array(tokenizer.encode(prompt))
    output = mx.expand_dims(input_tokens, 0)
    for _ in range(max_tokens):
        seq = output[:, -seq_len:]
        logits = model(seq) / temperature
        probs = mx.softmax(logits[:, -1, :], axis=-1)
        next_token = mx.argmax(probs, axis=-1)
        output = mx.concatenate([output, mx.expand_dims(next_token, 1)], axis=1)
        if tokenizer.decode(next_token.tolist()[0]) == tokenizer.eos_token:
            break
    return tokenizer.decode(output[0].tolist())

def run_tests(model: nn.Module, tokenizer: GPT2Tokenizer, seq_len: int) -> None:
    logging.info("Running standard tests")
    tests = [
        {"prompt": "Question: If I have 5 books and borrow 3 more, how many do I have?", "expected": "8"},
        {"prompt": "Question: If a shirt costs $20 and is on sale for 25% off, what is the sale price?", "expected": "15"},
        {"prompt": "Question: A car travels 60 miles in 2 hours. What is its average speed in miles per hour?", "expected": "30"},
        {"prompt": "Question: I have 12 apples and give 4 to my friend. How many do I have left?", "expected": "8"},
        {"prompt": "Question: If 3 pencils cost $0.75, how much does 1 pencil cost?", "expected": "0.25"},
        {"prompt": "Question: A rectangle has a length of 10 and a width of 4. What is its area?", "expected": "40"},
        {"prompt": "Question: If I save $5 each week for 8 weeks, how much will I have?", "expected": "40"},
        {"prompt": "Question: A store sells 2 dozen eggs for $6. How much is each egg?", "expected": "0.25"},
        {"prompt": "Question: If a train leaves at 2 PM and travels 100 miles in 2 hours, what time does it arrive?", "expected": "4 PM"},
        {"prompt": "Question: If I buy 3 items at $4 each and 2 at $5 each, what is the total cost?", "expected": "22"}
    ]
    
    for i, test in enumerate(tests):
        response = coach(model, tokenizer, test["prompt"], max_tokens=200, seq_len=seq_len)
        print(f"Test {i+1}:")
        print(f"Prompt: {test['prompt']}")
        print(f"Response: {response}")
        print(f"Expected: {test['expected']}")
        print("-" * 50)

if __name__ == "__main__":
    logging.info("Starting main process")
    config_file = "config.ini"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found")

    config = ConfigParser()
    config.read(config_file)
    
    def safe_get(section, option, fallback=None):
        if not config.has_section(section):
            raise ValueError(f"Missing required section [{section}] in config")
        if not config.has_option(section, option):
            raise ValueError(f"Missing required option '{option}' in section [{section}]")
        return config.get(section, option, fallback=fallback)
    
    dataset_name = safe_get("DATA", "dataset_name")
    config_name = safe_get("DATA", "config_name")
    split = safe_get("DATA", "split")
    split_ratio = float(safe_get("DATA", "split_ratio"))
    
    tokenizer_name = safe_get("TOKENIZER", "tokenizer_name")
    
    num_steps = int(safe_get("TRAINING", "num_steps"))
    print_every = int(safe_get("TRAINING", "print_every"))
    batch_size = int(safe_get("TRAINING", "batch_size"))
    seq_len = int(safe_get("TRAINING", "seq_len"))
    learning_rate = float(safe_get("TRAINING", "learning_rate"))
    
    checkpoint_dir = safe_get("CHECKPOINT", "checkpoint_dir")
    checkpoint_interval = int(safe_get("CHECKPOINT", "checkpoint_interval"))
    
    train_data, val_data, vocab_size, tokenizer = load_and_preprocess_data(config_file)
    
    d_model = int(safe_get("MODEL", "d_model"))
    n_heads = int(safe_get("MODEL", "n_heads"))
    n_layers = int(safe_get("MODEL", "n_layers"))
    max_seq_len = int(safe_get("MODEL", "max_seq_len"))
    
    model = SmallTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_seq_len=max_seq_len)
    mx.eval(model.parameters())
    
    logging.info("Model summary before training:")
    model.print_model_summary()
    
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    train_losses, val_losses = train_model(
        num_steps=num_steps,
        print_every=print_every,
        train_data=train_data,
        val_data=val_data,
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size
    )

    prompt = "Question: If I have 5 books and borrow 3 more, how many do I have?"
    response = coach(model, tokenizer, prompt, max_tokens=200, seq_len=seq_len)
    logging.info("Coach response generated")
    print("Coach says:", response)
    
    run_tests(model, tokenizer, seq_len)