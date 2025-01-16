"""
A Multi-Head Self-Attention Decoder Transformer model using the architecture from the paper "Attention Is All You Need".
The paper can be found here: https://arxiv.org/abs/1706.03762
"""

import torch
from torch import nn

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()

        # Key, query, and value vectors
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # A lower triangular matrix used to perform autoregressive masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # Dropout layer to prevent overfitting the transformer
        self.dropout = nn.Dropout(dropout)

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

    def __init__(self, n_embed, n_head, head_size, block_size, dropout):
        super().__init__()

        # Get list of all attention heads that will be run in parallel
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(n_head)])
        
        # Add projection layer and dropout
        self.proj = nn.Linear(head_size * n_head, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Perform self-attention for each head independently
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedForward(nn.Module):
    """ A simple rectified multilayer perceptron where the vectors are propagated through after self-attention """

    def __init__(self, n_embed, dropout):
        super().__init__()

        # The multilayer perceptron network
        # Note that, following the "Attention Is All You Need" paper architecture, the hidden units are size 4 * n_embed
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(), # Rectified activation function for non-linearity
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout), # Dropout to prevent overfitting
        )

    def forward(self, x):
        return self.mlp(x)

class Block(nn.Module):
    """ 
    A single transformer block that performs, in order:
        1. Multi-head self-attention where token vectors communicate with each other
        2. Forward propagation through a rectified multilayer perceptron network 
    """

    def __init__(self, n_embed, n_head, block_size, dropout):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        
        # Get size of each attention head based on embedding dimension and number of heads requested
        self.head_size = n_embed // n_head

        self.sa = MultiHeadAttention(n_embed, n_head, self.head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        
        # Add layer norms to re-scale data into range (0, 1) for faster model convergence and training
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Note that transformers use residual connections between each step
        x_residual_1 = self.sa(self.ln1(x))
        x = x + x_residual_1

        x_residual_2 = self.ffwd(self.ln2(x))
        x = x + x_residual_2

        return x

class Transformer(nn.Module):
    """ The complete Multi-Head Self-Attention Decoder Transformer architecture """

    def __init__(self, n_layers, n_embed, block_size, vocab_size, n_head, dropout):
        super().__init__()

        # Token and position embedding tables to convert tokenized indices and positions to vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        
        # The sequence containing the number of transformer blocks to process
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layers)])
        
        # Add layer norm for faster model convergence and training
        self.norm = nn.LayerNorm(n_embed)
        
        # Linear layer that takes embedding vectors to prediction logits (to be processed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        # Store block size
        self.block_size = block_size
        
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
        # Get current device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get batch and time dimension
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
            
            # Only consider the last block_size tokens (maximum scope of the transformer)
            embedding_cond = embedding[:, -self.block_size:]
            
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
