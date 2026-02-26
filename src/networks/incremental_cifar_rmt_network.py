from functools import partial
from typing import Callable

import torch
from torch import nn
from torchvision.utils import _log_api_usage_once

from .torchvision_modified_vit import Encoder


class MemoryVisionTransformer(nn.Module):
    """
    ViT-style encoder with external memory tokens for incremental CIFAR-100.
    """

    def __init__(
            self,
            image_size: int = 32,
            patch_size: int = 4,
            num_layers: int = 8,
            num_heads: int = 12,
            hidden_dim: int = 384,
            mlp_dim: int = 1536,
            num_classes: int = 100,
            n_mem: int = 2,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        torch._assert(n_mem > 0, "n_mem must be positive.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.n_mem = n_mem

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_length = self.num_patches + n_mem

        self.encoder = Encoder(
            seq_length=self.seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(c == 3, f"Wrong number of channels! Expected 3 but got {c}.")
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}.")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}.")
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

    def forward(
            self,
            x: torch.Tensor,
            memory_tokens: torch.Tensor,
            return_encoded_memory: bool = False,
            activations: list = None
    ):
        if memory_tokens.ndim != 2:
            raise ValueError(f"Expected memory_tokens with shape (n_mem, hidden_dim), got {tuple(memory_tokens.shape)}")
        if memory_tokens.shape != (self.n_mem, self.hidden_dim):
            raise ValueError(
                f"Expected memory_tokens shape ({self.n_mem}, {self.hidden_dim}), got {tuple(memory_tokens.shape)}"
            )

        x = self._process_input(x)
        batch_size = x.shape[0]
        batch_memory = memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat((batch_memory, x), dim=1)

        encoded = self.encoder(x, activations=activations)
        logits = self.head(encoded[:, 0, :])

        if return_encoded_memory:
            return logits, encoded[:, :self.n_mem, :]
        return logits
