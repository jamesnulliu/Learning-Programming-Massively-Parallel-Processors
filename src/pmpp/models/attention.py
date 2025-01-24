import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def set_random_seed(
    seed: int, rank: int = 0, force_deterministic: bool = False
) -> None:
    """
    Set the random seed for numpy and torch.
    """
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if force_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MultiHeadAttentionKernel(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_size: int = embed_dim // num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Calculates softmax(Q @ KT / sqrt(dk)) @ V .

        Parameters
        ----------
        q : torch.Tensor; Shape: (q_len, embed_dim)

        k : torch.Tensor; Shape: (kv_len, embed_dim)

        v : torch.Tensor; Shape: (kv_len, embed_dim)

        Note
        ----
        When prefilling, q_len equals to seq_len (number of tokens in the input
        seq);
        When decoding, q_len equals to 1, refering to the newly generated
        token. (Based on different sampling strategies, q_len could be larger
        than 1.)
        """

        q_len, kv_len = q.size(0), k.size(0)
        # q -> (num_heads, q_len, head_size)
        q = q.view(q_len, self.num_heads, self.head_size).transpose(0, 1)
        # k -> (num_heads, kv_len, head_size)
        k = k.view(kv_len, self.num_heads, self.head_size).transpose(0, 1)
        # v -> (num_heads, kv_len, head_size)
        v = v.view(kv_len, self.num_heads, self.head_size).transpose(0, 1)
        # scores -> (num_heads, q_len, kv_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.head_size
        )
        # scores -> (num_heads, q_len, kv_len)
        scores = scores + mask if mask is not None else scores
        # scores -> (num_heads, q_len, kv_len)
        scores = F.softmax(scores, dim=-1)
        # out -> (num_heads, q_len, head_size)
        out = torch.matmul(scores, v)
        # out -> (q_len, num_heads, head_size)
        out = out.transpose(0, 1).reshape(q_len, self.embed_dim)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.attn_kernel = MultiHeadAttentionKernel(embed_dim, num_heads)

    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        seq : torch.Tensor; Shape: (seq_len, embed_dim)
            Input sequnce, containing `seq_len` tokens, and each token have
            been embedded to a `(embed_dim,)` tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Attention output, cached K and cached V.
        """

        # q -> (seq_len, embed_dim)
        q = self.Wq(seq)
        # k -> (seq_len, embed_dim)
        k = self.Wk(seq)
        # v -> (seq_len, embed_dim)
        v = self.Wv(seq)

        # out -> (seq_len, embed_dim)
        out = self.Wo(self.attn_kernel(q, k, v, mask))

        return out, k, v


class CachedMultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.attn_kernel = MultiHeadAttentionKernel(embed_dim, num_heads)

    def forward(
        self,
        seq: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        seq : torch.Tensor; Shape: (1, embed_dim)
            Input sequnce, containing only ONE newly generated token.
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

        # q -> (1, embed_dim)
        q = self.Wq(seq)
        # k -> (1, embed_dim)
        k = self.Wk(seq)
        # v -> (1, embed_dim)
        v = self.Wv(seq)

        # k_cache -> (kv_len + 1, embed_dim)
        k_cache = torch.cat([k_cache, k.detach()], dim=0)
        # v_cache -> (kv_len + 1, embed_dim)
        v_cache = torch.cat([v_cache, v.detach()], dim=0)

        # out -> (seq_len, embed_dim)
        out = self.Wo(self.attn_kernel(q, k_cache, v_cache))

        return out, k_cache, v_cache


class SimpleLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.mha = MultiHeadAttention(embed_dim, num_heads)

        self.cached_mha = CachedMultiHeadAttention(embed_dim, num_heads)

        self.proj_to_vocab = nn.Linear(embed_dim, vocab_size)

        self.k_cache = nn.Buffer(torch.zeros(size=(0, embed_dim)))
        self.v_cache = nn.Buffer(torch.zeros(size=(0, embed_dim)))

    def forward(
        self,
        prompt: torch.Tensor,
        is_prefilling: bool = True,
    ):
        """
        Parameters
        ----------
        prompt : torch.Tensor; Shape: (seq_len,)
            Input prompt.

            When prefilling, a prompt is an input sequence, containing
            `seq_len` tokens.

            When decoding, a prompt is a single token generated from the last
            step, which means `seq_len` should equal to `1`.
        """

        # embedded_prompt -> (seq_len, embed_dim)
        embedded_prompt = self.embed(prompt)

        if is_prefilling:
            seq_len = prompt.size(0)
            mask = None
            if seq_len > 1:
                mask = torch.full(
                    (seq_len, seq_len), -float("Inf"), device=prompt.device
                )
                mask = torch.triu(mask, diagonal=1)
            # out -> (seq_len, embed_dim)
            # k -> (seq_len, embed_dim)
            # v -> (seq_len, embed_dim)
            out, k, v = self.mha(embedded_prompt, mask)
        else:
            assert prompt.size(0) == 1
            # out -> (seq_len, embed_dim)
            # k -> (kv_len, embed_dim)
            # v -> (kv_len, embed_dim)
            out, k, v = self.cached_mha(
                embedded_prompt, self.k_cache, self.v_cache
            )

        # Update k cache and v cache
        # [NOTE]
        # | We use `detach()` frist to detach k and v from computation graph,
        # | and then assign them to `self.k_cache` and `self.v_cache`.
        self.k_cache = k.detach()
        self.v_cache = v.detach()

        # Use the last token to calculate the probability of each word in the
        # vocabulary bank:
        # probs -> (vocab_size,)
        probs = torch.softmax(self.proj_to_vocab(out[-1]), dim=-1)

        return probs


if __name__ == "__main__":
    set_random_seed(114514)

    seq_len = 4
    vocab_size = 1024
    embed_dim = 128
    num_heads = 4
    n_generate = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lm = SimpleLM(vocab_size, embed_dim, num_heads).to(device)

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
    token = torch.argmax(probs, dim=-1, keepdim=True)
    print(f"The 1th predicted token: {token}")
    print(f"|- Token Shape: {token.shape}")
    print(f"|- K Cache Shape: {lm.k_cache.shape}")
    print(f"|- V Cache Shape: {lm.v_cache.shape}")

    # Decoding ================================================================
    # At decoding stage, we feed the model with a token generated from the last
    # round, and since the shape of `token` is (1, ), you can also consider it
    # "an input sequnce with only one token".
    for i in range(1, n_generate):
        probs = lm(token, is_prefilling=False)
        token = torch.argmax(probs, dim=-1, keepdim=True)
        print(f"The {i+1}th predicted token: {token}")
        print(f"|- Token Shape: {token.shape}")
        print(f"|- K Cache Shape: {lm.k_cache.shape}")
        print(f"|- V Cache Shape: {lm.v_cache.shape}")
