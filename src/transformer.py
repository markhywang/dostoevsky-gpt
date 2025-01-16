"""
A Multi-Head Self-Attention Decoder Transformer model using the architecture from the paper "Attention Is All You Need".
The paper can be found here: https://arxiv.org/abs/1706.03762
"""

import torch
from torch import nn
from script import BLOCK_SIZE, N_EMBED, N_HEAD, N_LAYERS, DROPOUT, vocab_size, device

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        # Key, query, and value vectors
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        # A lower triangular matrix used to perform autoregressive masking
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        
        # Dropout layer to prevent overfitting the transformer
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        # Get key, query, and values vectors
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Perform self-attention mechanism then dropout
        affinity = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        affinity = torch.softmax(affinity, dim=-1) # (B, T, T)
        affinity = self.dropout(affinity)

        # Return the product of all token affinities and the value matrix
        out = affinity @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention run parallel to capture more complex language, structures, and semantics """

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Get list of all attention heads that will be run in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # Add projection layer and dropout
        self.proj = nn.Linear(head_size * num_heads, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Perform self-attention for each head independently
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedFoward(nn.Module):
    """ A simple rectified multilayer perceptron where the vectors are propagated through after self-attention """

    def __init__(self):
        super().__init__()

        # The multilayer perceptron network
        # Note that, confirming to "Attention Is All You Need" paper, the hidden units are size 4 * N_EMBED
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(), # Rectified activation function for non-linearity
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT), # Dropout to prevent overfitting
        )

    def forward(self, x):
        return self.mlp(x)

class Block(nn.Module):
    """ 
    A single transformer block that performs, in order:
        1. Multi-head self-attention where token vectors communicate with each other
        2. Forward propagation through a multilayer perceptron network 
    """

    def __init__(self, n_head):
        # N_EMBED: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = N_EMBED // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(N_EMBED)
        
        # Add layer norms to re-scale data into range (0, 1) for faster model convergence and training
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        # Note that transformers use residual connections between each step
        x_residual_1 = self.sa(self.ln1(x))
        x = x + x_residual_1

        x_residual_2 = self.ffwd(self.ln2(x))
        x = x + x_residual_2

        return x

class Transformer(nn.Module):
    """ The complete Multi-Head Self-Attention Decoder Transformer architecture """

    def __init__(self):
        super().__init__()

        # Token and position embedding tables to convert tokenized indices and positions to vectors
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        
        # The sequence containing the number of transformer blocks to process
        self.blocks = nn.Sequential(*[Block(n_head=N_HEAD) for _ in range(N_LAYERS)])
        
        # Add layer norm for faster model convergence and training
        self.norm = nn.LayerNorm(N_EMBED)
        
        # Linear layer that takes embedding vectors to prediction logits (to be processed)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        
        # A better way to initialize ways
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ A better and more stable way to initialize weights for transformer model """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, embedding_vectors):
        B, T = embedding_vectors.shape
        
        # Get position and token embeddings
        token_embed = self.token_embedding_table(embedding_vectors).to(device)
        position_embed = self.position_embedding_table(torch.arange(T, device=device))
        
        # Add the two embeddings then traverse through the transformer
        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.norm(x)

        # Convert embedding vectors to prediction logits and return them
        logits = self.lm_head(x)
        return logits

    def generate(self, embedding, max_new_tokens):
        """ Generate text using Transformer inference """
        for _ in range(max_new_tokens):
            
            # Only consider the last BLOCK_SIZE tokens (maximum scope of the transformer)
            embedding_cond = embedding[:, -BLOCK_SIZE:]
            
            # Pass through transformer and get values for the last time step
            logits = self(embedding_cond)
            logits = logits[:, -1, :]

            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # sample from the distribution
            embedding_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            embedding = torch.cat((embedding, embedding_next), dim=1) # Add 1 to last dimension

        return embedding
