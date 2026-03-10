import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class MHAKernel(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.head_size: int = hidden_dim // num_heads

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
        q : torch.Tensor; Shape: (q_len, hidden_dim)

        k : torch.Tensor; Shape: (kv_len, hidden_dim)

        v : torch.Tensor; Shape: (kv_len, hidden_dim)

        mask: torch.Tensor; Shape: (q_len, kv_len), optional

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
        q = q.reshape(q_len, self.num_heads, self.head_size).transpose(0, 1)
        # k -> (num_heads, kv_len, head_size)
        k = k.reshape(kv_len, self.num_heads, self.head_size).transpose(0, 1)
        # v -> (num_heads, kv_len, head_size)
        v = v.reshape(kv_len, self.num_heads, self.head_size).transpose(0, 1)
        # scores -> (num_heads, q_len, kv_len)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_size**0.5)
        scores = (
            scores.masked_fill(mask == 0, float("-inf"))
            if mask is not None
            else scores
        )
        # scores -> (num_heads, q_len, kv_len)
        attn_probs = F.softmax(scores.to(torch.float32), dim=-1).type_as(
            scores
        )
        # out -> (num_heads, q_len, head_size)
        out = torch.matmul(attn_probs, v)
        # out -> (q_len, num_heads, head_size)
        out = out.transpose(0, 1).reshape(q_len, self.hidden_dim)

        return out
