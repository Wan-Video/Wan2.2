# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from ..attention import flash_attention
from ..tokenizers import HuggingfaceTokenizer
from .xlm_roberta import XLMRoberta

__all__ = [
    'XLMRobertaCLIP',
    'clip_xlm_roberta_vit_h_14',
    'CLIPModel',
]


def pos_interpolate(pos, seq_len):
    if pos.size(1) == seq_len:
        return pos
    else:
        src_grid = int(math.sqrt(pos.size(1)))
        tar_grid = int(math.sqrt(seq_len))
        n = pos.size(1) - src_grid * src_grid
        return torch.cat([
            pos[:, :n],
            F.interpolate(
                pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(
                    0, 3, 1, 2),
                size=(tar_grid, tar_grid),
                mode='bicubic',
                align_corners=False).flatten(2).transpose(1, 2)
        ],
                         dim=1)


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)

        # compute attention
        p = self.attn_dropout if self.training else 0.0
        x = flash_attention(q, k, v, dropout_p=p, causal=self.causal, version=2)
        x = x.reshape(b, s, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Module):

    def __init__(self, dim, mid_dim):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def forward(self, x):
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 post_norm=False,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert activation in ['quick_gelu', 'gelu', 'swi_glu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == 'swi_glu':
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 activation='gelu',
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        """
        x:  [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.to_q(self.cls_embedding).view(1, 1, n, d).expand(b, -1, -1, -1)
        k, v = self.to_kv(x).view(b, s, 2, n, d).unbind(2)

        # compute attention
        x = flash_attention(q, k, v, version=2)
        x = x.reshape(b, 1, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)

        # mlp
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 pool_type='token',
                 pre_norm=True,
                 post_norm=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        if image_size % patch_size != 0:
            print(
                '[WARNING] image_size is not divisible by patch_size',
                flush=True)
        assert pool_type in ('token', 'token_fc', 'attn_pool')
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm)
        if pool_type in ('token', 'token_fc'):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(
            1, self.num_patches +
            (1 if pool_type in ('token', 'token_fc') else 0), dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ])
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # head
        if pool_type == 'token':
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == 'token_fc':
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == 'attn_pool':
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation,
                                      proj_dropout, norm_eps)

    def forward(self, x, interpolation=False, use_31_block=False):
        b = x.size(0)

        # embeddings
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in ('token', 'token_fc'):
            x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.size(1))
        else:
            e = self.pos_embedding
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            x = self.transformer[:-1](x)
            return x
        else:
            x = self.transformer(x)
            return x


class XLMRobertaWithHead(XLMRoberta):

    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop('out_dim')
        super().__init__(**kwargs)

        # head
        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(
            nn.Linear(self.dim, mid_dim, bias=False), nn.GELU(),
            nn.Linear(mid_dim, self.out_dim, bias=False))

    def forward(self, ids):
        # xlm-roberta
        x = super().forward(ids)

        # average pooling
        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        # head
        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):

    def __init__(self,
                 embed_dim=1024,
                 image_size=224,
                 patch_size=14,
                 vision_dim=1280,
                 vision_mlp_ratio=4,
                 vision_heads=16,
                 vision_layers=32,
                 vision_pool='token',
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 activation='gelu',
                 vocab_size=250002,
                 max_text_len=514,
                 type_size=1,
                 pad_id=1,
                 text_dim=1024,
                 text_heads=16,
                 text_layers=24,
                 text_post_norm=True,
                 text_dropout=0.1,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,
            max_seq_len=max_text_len,
            type_size=type_size,
            pad_id=pad_id,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            post_norm=text_post_norm,
            dropout=text_dropout)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(self, imgs, txt_ids):
        """
        imgs:       [B, 3, H, W] of torch.float32.
        - mean:     [0.48145466, 0.4578275, 0.40821073]
        - std:      [0.26862954, 0.26130258, 0.27577711]
        txt_ids:    [B, L] of torch.long.
                    Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay': 0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups


def _clip(pretrained=False,
          pretrained_name=None,
          model_cls=XLMRobertaCLIP,
          return_transforms=False,
          return_tokenizer=False,
          tokenizer_padding='eos',
          dtype=torch.float32,
          device='cpu',
          **kwargs):
    # init a model on device
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)
    output = (model,)

    # init transforms
    if return_transforms:
        # mean and std
        if 'siglip' in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        # transforms
        transforms = T.Compose([
            T.Resize((model.image_size, model.image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        output += (transforms,)
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(
        pretrained=False,
        pretrained_name='open-clip-xlm-roberta-large-vit-huge-14',
        **kwargs):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool='token',
        activation='gelu',
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0)
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, **cfg)


class CLIPModel:

    def __init__(self, dtype, device, checkpoint_path, tokenizer_path):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device)
        self.model = self.model.eval().requires_grad_(False)
        logging.info(f'loading {checkpoint_path}')
        self.model.load_state_dict(
            torch.load(
                checkpoint_path, map_location='cpu', weights_only=False))

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=self.model.max_text_len - 2,
            clean='whitespace')

    def visual(self, videos):
        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(
                u.transpose(0, 1),
                size=size,
                mode='bicubic',
                align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out