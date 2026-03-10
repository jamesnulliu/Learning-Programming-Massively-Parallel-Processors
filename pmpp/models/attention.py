from typing import Optional
import torch
from torch import nn
from .mha_kernels import MHAKernel


class MHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads

        self.Wq = nn.Linear(embed_dim, hidden_dim)
        self.Wk = nn.Linear(embed_dim, hidden_dim)
        self.Wv = nn.Linear(embed_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, embed_dim)

        self.attn_kernel = MHAKernel(hidden_dim, num_heads)

    def forward(
        self,
        seq: torch.Tensor,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        seq : torch.Tensor; Shape: (seq_len, embed_dim)
            Input sequnce, containing `seq_len` tokens, and each token have
            been embedded to a `(embed_dim,)` tensor.
        k_cache : torch.Tensor; Shape: (kv_len, embed_dim)
            Cached K.
        v_cache : torch.Tensor; Shape: (kv_len, embed_dim)
            Cached V.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Attention output, cached K and cached V.

        Note
        ----
            When decoing, the input seq only has ONE newly generated token.
        """

        # q -> (seq_len, hidden_dim)
        q = self.Wq(seq)
        # k -> (seq_len, hidden_dim)
        k = self.Wk(seq)
        # v -> (seq_len, hidden_dim)
        v = self.Wv(seq)

        # k_cache -> (kv_len + seq_len, hidden_dim)
        k = k if k_cache is None else torch.cat([k_cache, k.detach()], dim=0)
        # v_cache -> (kv_len + seq_len, hidden_dim)
        v = v if v_cache is None else torch.cat([v_cache, v.detach()], dim=0)

        # out -> (seq_len, embed_dim)
        out = self.Wo(self.attn_kernel(q, k, v, mask))

        return out, k, v


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MHA(embed_dim, num_heads, hidden_dim)
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),  # Or nn.GELU()
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

        # Initialize K/V caches
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_prefilling: bool = True,
    ):
        """
        Forward pass of the Transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (seq_len, embed_dim) when prefilling,
            or (1, embed_dim) when decoding.

        mask : Optional[torch.Tensor], optional
            Attention mask of shape (seq_len, seq_len) when prefilling,
            or None when decoding. Defaults to None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (seq_len, embed_dim) when prefilling,
            or (1, embed_dim) when decoding.
        """

        # Multi-Head Attention
        if is_prefilling:
            # For prefilling, k_cache and v_cache are None.
            # The mask provided should be for the current input sequence x.
            # e.g., for causal mask, (seq_len, seq_len)
            attn_output, k_updated, v_updated = self.attention(
                seq=x, k_cache=None, v_cache=None, mask=mask
            )
        else:
            # For decoding, x is typically the new token (q_len=1).
            # We use the existing k_cache and v_cache.
            # The mask (if any) should be for the single query token against all keys (cached + current).
            # Often, `mask` can be None here if the setup implicitly handles causality.
            attn_output, k_updated, v_updated = self.attention(
                seq=x,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                mask=mask,  # or mask=None if appropriate for decoding
            )

        # Add & Norm (Residual connection)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward Network
        mlp_output = self.mlp(x)

        # Add & Norm (Residual connection)
        x = self.norm2(x + self.dropout(mlp_output))

        # Update k_cache and v_cache for the next step
        self.k_cache = k_updated.detach()
        self.v_cache = v_updated.detach()

        return x


class SimpleLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, num_heads, hidden_dim, mlp_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        input_seq: torch.Tensor,
        is_prefilling: bool = True,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the SimpleLM model.

        Parameters
        ----------
        input_seq : torch.Tensor
            Input sequence of shape (seq_len,) when prefilling,
            or (1,) when decoding.
        is_prefilling : bool, optional
            Whether the model is in prefilling mode. Defaults to True.
        mask : Optional[torch.Tensor], optional
            Attention mask of shape (seq_len, seq_len) when prefilling,
            or None when decoding. Defaults to None.
        Returns
        -------
        torch.Tensor
            Output tensor of shape (seq_len, vocab_size).
        """

        # embedded_seq: (seq_len, embed_dim)
        embedded_seq = self.embed(input_seq)

        # Pass through the transformer blocks
        for block in self.transformer_blocks:
            embedded_seq = block(
                embedded_seq, mask=mask, is_prefilling=is_prefilling
            )
        # embedded_seq: (seq_len, embed_dim)

        # logits: (seq_len, vocab_size)
        logits = self.lm_head(embedded_seq)

        return logits


if __name__ == "__main__":
    seq_len = 4
    vocab_size = 1024
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    hidden_dim = 256
    mlp_dim = 512
    dropout = 0.1
    n_generate = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lm = SimpleLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
    ).to(device)

    # Create a random prompt (for prefilling), containing `seq_len` tokens;
    # Each token is a random integer from 0 to `vocab_size`.
    # For example, prompt of "fly me to the" may be [0, 1023, 512, 100].
    prompt = torch.randint(0, vocab_size, (seq_len,), device=device)
    print(f"Original prompt shape: {prompt.shape}")  # (seq_len, )

    # Prefilling ==============================================================
    # At prefilling stage, we feed the model with a complete prompt of shape
    # (seq_len, ), and ask the model to predict the next token (word).
    # The model will returns a tensor of shape (vocab_size, ), representing the
    # probability of the next token being each word in the vocabulary bank.
    # For example, `probs` can be [0.1, 0.5, 0.05, ...], which means the
    # probability of the next token being the 1st word is 0.1 and 2nd is 0.5.
    probs = lm(prompt, is_prefilling=True)

    # We pick the word with the highest probability as the next token.
    # Here `keepdim` is set to `True` to keep the shape of `token` as (1, ),
    # which is consistent with the shape of the input prompt (seq_len, ). If
    # set to `False`, the shape of `token` will be (), i.e., scalar, and you
    # have to reshape it to (1, ) manually before the next round of generation.
    token = torch.argmax(probs[-1, :], dim=-1, keepdim=True)
    print(f"The 1th predicted token: {token}")
    print(f"|- Token Shape: {token.shape}")

    # Decoding ================================================================
    # At decoding stage, we feed the model with a token generated from the last
    # round, and since the shape of `token` is (1, ), you can also consider it
    # "an input sequnce with only one token".
    for i in range(1, n_generate):
        probs = lm(token, is_prefilling=False)
        token = torch.argmax(probs[-1, :], dim=-1, keepdim=True)
        print(f"The {i + 1}th predicted token: {token}")
        print(f"|- Token Shape: {token.shape}")
