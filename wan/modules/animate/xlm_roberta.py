# Modified from transformers.models.xlm_roberta.modeling_xlm_roberta
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['XLMRoberta', 'XLMRobertaEncoder', 'xlm_roberta_large']


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1, eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, n, d).permute(0, 2, 1, 3)

        # compute attention
        p = self.dropout.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, mask, p)
        x = x.permute(0, 2, 1, 3).reshape(b, s, c)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, post_norm, dropout=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.eps = eps

        # layers
        self.attn = SelfAttention(dim, num_heads, dropout, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
            nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, mask):
        if self.post_norm:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.ffn(x))
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
        return x


class XLMRoberta(nn.Module):
    """
    XLMRobertaModel with no pooler and no LM head.
    """

    def __init__(self,
                 vocab_size=250002,
                 max_seq_len=514,
                 type_size=1,
                 pad_id=1,
                 dim=1024,
                 num_heads=16,
                 num_layers=24,
                 post_norm=True,
                 dropout=0.1,
                 eps=1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.eps = eps

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.type_embedding = nn.Embedding(type_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        # blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, num_heads, post_norm, dropout, eps)
            for _ in range(num_layers)
        ])

        # norm layer
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, ids):
        """
        ids: [B, L] of torch.LongTensor.
        """
        b, s = ids.shape
        mask = ids.ne(self.pad_id).long()

        # embeddings
        x = self.token_embedding(ids) + \
            self.type_embedding(torch.zeros_like(ids)) + \
            self.pos_embedding(self.pad_id + torch.cumsum(mask, dim=1) * mask)
        if self.post_norm:
            x = self.norm(x)
        x = self.dropout(x)

        # blocks
        mask = torch.where(
            mask.view(b, 1, 1, s).gt(0), 0.0,
            torch.finfo(x.dtype).min)
        for block in self.blocks:
            x = block(x, mask)

        # output
        if not self.post_norm:
            x = self.norm(x)
        return x


class XLMRobertaEncoder(nn.Module):
    """Wrapper around XLMRoberta that initializes the large model on device."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.model = xlm_roberta_large(device=device)
        self.model.eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def forward(self, ids):
        return self.model(ids)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def xlm_roberta_large(pretrained=False,
                      return_tokenizer=False,
                      device='cpu',
                      **kwargs):
    """
    XLMRobertaLarge adapted from Huggingface.
    """
    # params
    cfg = dict(
        vocab_size=250002,
        max_seq_len=514,
        type_size=1,
        pad_id=1,
        dim=1024,
        num_heads=16,
        num_layers=24,
        post_norm=True,
        dropout=0.1,
        eps=1e-5)
    cfg.update(**kwargs)

    # init a model on device
    with torch.device(device):
        model = XLMRoberta(**cfg)
    return model