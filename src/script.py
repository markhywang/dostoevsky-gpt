#!/usr/bin/env python
# coding: utf-8
"""
Dostoevsky-GPT: A character-level GPT model trained on Dostoevsky's work.

This script handles data loading, model definition, training, and text generation.
It can be configured and run from the command line.
"""

import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from pathlib import Path

# ---------------------------------------------------------------------------
# Argument Parsing & Global Config
# ---------------------------------------------------------------------------


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a character-level GPT on Dostoevsky's writings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument("--pretrain-data-path", type=Path, nargs='+', default=[Path("data/the-brothers-karamazov.txt"), Path("data/netochka-nezvanova.txt")],
                        help="Path to the pre-training text file(s).")
    parser.add_argument("--finetune-data-path", type=Path, default=Path("data/netochka-nezvanova.txt"),
                        help="Path to the fine-tuning text file. If provided, a fine-tuning phase will run.")
    parser.add_argument("--save-model-path", type=Path, default=Path("models/dostoevsky-transformer.pth"),
                        help="Path to save the trained model.")
    parser.add_argument("--output-text-path", type=Path, default=Path("output/text-generation.txt"),
                        help="Path to write the generated text.")
    parser.add_argument("--plot-loss", action="store_true", help="If set, plots the training loss.")

    # Model Hyperparameters
    parser.add_argument("--block-size", type=int, default=256, help="Max context length for predictions.")
    parser.add_argument("--n-embed", type=int, default=384, help="Embedding dimension.")
    parser.add_argument("--n-head", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")

    # Training Hyperparameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of pre-training epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Number of fine-tuning epochs. Set to 0 to disable.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for Adam optimizer during pre-training.")
    parser.add_argument("--finetune-lr", type=float, default=1e-5, help="Learning rate for the fine-tuning phase.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Clip gradients at this value. Set to 0 to disable.")
    parser.add_argument("--use-amp", action="store_true", help="Use Automatic Mixed Precision for training.")

    # Generation & Logging
    parser.add_argument("--num-generate", type=int, default=1000, help="Number of characters to generate.")
    parser.add_argument("--print-steps", type=int, default=250, help="How often to print training loss (in epochs).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------


class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embed, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    """The full Transformer model."""
    def __init__(self, vocab_size, n_embed, block_size, n_head, n_layers, dropout, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Data Handling
# ---------------------------------------------------------------------------


class DostoevskyDataset(Dataset):
    """PyTorch Dataset for character-level data."""
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def read_text_from_files(paths):
    """Reads and concatenates text from a list of files."""
    print(f"Reading text from: {', '.join(map(str, paths))}")
    full_text = ""
    for path in paths:
        if not path.is_file():
            print(f"Error: Data file not found at {path}")
            exit(1)
        with open(path, 'r', encoding='utf-8') as f:
            full_text += f.read()
    return full_text


def create_char_mappings(text):
    """Creates character-to-integer mappings from text."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return vocab_size, encode, decode


def create_dataloaders(text, encode_fn, block_size, batch_size, device):
    """Creates a DataLoader for training."""
    data = torch.tensor(encode_fn(text), dtype=torch.long, device=device)
    dataset = DostoevskyDataset(data, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# ---------------------------------------------------------------------------
# Training & Utility Functions
# ---------------------------------------------------------------------------


def train_model(model, dataloader, optimizer, epochs, print_steps, grad_clip, use_amp, device):
    """Main training loop with support for AMP and gradient clipping."""
    print("Starting training...")
    start_time = timer()
    all_losses = []
    
    use_amp = use_amp and "cuda" in device
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X, y in dataloader:
            # Data is already on the correct device from create_dataloaders
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16, enabled=use_amp):
                _, loss = model(X, y)
            
            epoch_loss += loss.item()

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()

        avg_epoch_loss = epoch_loss / len(dataloader)
        all_losses.append(avg_epoch_loss)

        if (epoch + 1) % print_steps == 0:
            print(f"Epoch: {epoch + 1}/{epochs} | Train Loss: {avg_epoch_loss:.4f}")

    end_time = timer()
    total_time = end_time - start_time
    print(f"\nTraining finished in {total_time:.2f} seconds.")
    return all_losses


def plot_loss_curve(losses):
    """Plots the training loss curve."""
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training Loss"])
    plt.show()


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


def main():
    """Main function to run the script."""
    args = get_args()

    # Setup
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Ensure output directories exist
    args.save_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_text_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Loading and Vocabulary ---
    # The vocabulary is built from all text sources combined to ensure consistency.
    print("\n--- Loading Data & Building Vocabulary ---")
    all_paths = list(set(args.pretrain_data_path + ([args.finetune_data_path] if args.finetune_data_path else [])))
    combined_text = read_text_from_files(all_paths)
    vocab_size, encode, decode = create_char_mappings(combined_text)
    print(f"Vocabulary size: {vocab_size}")

    # --- 2. Model Initialization ---
    model = Transformer(
        vocab_size=vocab_size,
        n_embed=args.n_embed,
        block_size=args.block_size,
        n_head=args.n_head,
        n_layers=args.n_layers,
        dropout=args.dropout,
        device=device
    ).to(device)

    # Attempt to compile the model for a speed-up (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled successfully (PyTorch 2.0+).")
    except Exception:
        print("Could not compile model (requires PyTorch 2.0+).")

    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    all_losses = []

    # --- 3. Pre-training Phase ---
    print("\n--- Starting Pre-training Phase ---")
    pretrain_text = read_text_from_files(args.pretrain_data_path)
    pretrain_dataloader = create_dataloaders(pretrain_text, encode, args.block_size, args.batch_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pretrain_losses = train_model(model, pretrain_dataloader, optimizer, args.epochs, args.print_steps, args.grad_clip, args.use_amp, device)
    all_losses.extend(pretrain_losses)

    # --- 4. Fine-tuning Phase ---
    if args.finetune_data_path and args.finetune_epochs > 0:
        print("\n--- Starting Fine-tuning Phase ---")
        finetune_text = read_text_from_files([args.finetune_data_path])
        finetune_dataloader = create_dataloaders(finetune_text, encode, args.block_size, args.batch_size, device)
        
        # Use a new optimizer with a lower learning rate for fine-tuning
        finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
        
        finetune_losses = train_model(model, finetune_dataloader, finetune_optimizer, args.finetune_epochs, args.print_steps, args.grad_clip, args.use_amp, device)
        all_losses.extend(finetune_losses)


    # --- 5. Save, Generate, and Plot ---
    print(f"\nSaving final model to {args.save_model_path}")
    torch.save(model.state_dict(), args.save_model_path)

    # Generate text
    print("\n--- Generated Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context, max_new_tokens=args.num_generate)[0].tolist())
    print(generated_chars)
    print("------------------------")

    # Write to file
    with open(args.output_text_path, 'w', encoding='utf-8') as f:
        f.write(generated_chars)
    print(f"\nGenerated text saved to {args.output_text_path}")

    # Plot loss if requested
    if args.plot_loss:
        plot_loss_curve(all_losses)


if __name__ == '__main__':
    main()