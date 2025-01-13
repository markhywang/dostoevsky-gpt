class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        affinity = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        affinity = affinity.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        affinity = torch.softmax(affinity, dim=-1) # (B, T, T)
        affinity = self.dropout(affinity)

        out = affinity @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedFoward(nn.Module):
    """ A simple rectified multilayer perceptron """

    def __init__(self, n_embd):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.ReLU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.mlp(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_head):
        # N_EMBED: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = N_EMBED // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(N_EMBED)
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x_residual_1 = self.sa(self.ln1(x))
        x = x + x_residual_1

        x_residual_2 = self.ffwd(self.ln2(x))
        x = x + x_residual_2

        return x

class Transformer(nn.Module):
    """ The Transformer Architecture """

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(n_head=N_HEAD) for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, embedding_vectors):
        B, T = embedding_vectors.shape
        token_embed = self.token_embedding_table(embedding_vectors).to(device)
        position_embed = self.position_embedding_table(torch.arange(T, device=device))

        x = token_embed + position_embed
        x = self.blocks(x)
        x = self.norm(x)

        logits = self.lm_head(x)
        return logits

    def generate(self, embedding, max_new_tokens):
        for _ in range(max_new_tokens):
            embedding_cond = embedding[:, -BLOCK_SIZE:]
            logits = self(embedding_cond)
            logits = logits[:, -1, :]

            # apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # sample from the distribution
            embedding_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            embedding = torch.cat((embedding, embedding_next), dim=1) # Add 1 to last dimension

        return embedding
