import torch
from torch import nn


class MemoryTransformer(nn.Module):
    """
    A compact Transformer for PMNIST with recurrent memory tokens.
    """

    def __init__(
            self,
            patch_size: int = 4,
            d_model: int = 64,
            n_layers: int = 2,
            n_heads: int = 4,
            mlp_ratio: float = 2.0,
            n_mem: int = 2,
            num_classes: int = 10,
            img_size: int = 28
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")

        self.patch_size = patch_size
        self.d_model = d_model
        self.n_mem = n_mem
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + n_mem, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[1] != self.img_size * self.img_size:
                raise ValueError(f"Expected flattened input with {self.img_size * self.img_size} features, got {x.shape[1]}.")
            x = x.reshape(x.shape[0], 1, self.img_size, self.img_size)
        elif x.ndim == 3:
            if x.shape[1:] != (self.img_size, self.img_size):
                raise ValueError(f"Expected (B, {self.img_size}, {self.img_size}) input, got {tuple(x.shape)}.")
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            if x.shape[1:] == (1, self.img_size, self.img_size):
                pass
            elif x.shape[1:] == (self.img_size, self.img_size, 1):
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"Expected (B, 1, {self.img_size}, {self.img_size}) or (B, {self.img_size}, {self.img_size}, 1), got {tuple(x.shape)}."
                )
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")
        return x.to(torch.float32)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, 1, -1, self.patch_dim)
        return patches.squeeze(1)

    def forward(
            self,
            x: torch.Tensor,
            memory_tokens: torch.Tensor,
            return_encoded_memory: bool = False
    ):
        x = self._normalize_input(x)
        if memory_tokens.ndim != 2:
            raise ValueError(f"memory_tokens must be 2D (n_mem, d_model), got shape {tuple(memory_tokens.shape)}.")
        if memory_tokens.shape[0] != self.n_mem or memory_tokens.shape[1] != self.d_model:
            raise ValueError(
                f"memory_tokens shape must be ({self.n_mem}, {self.d_model}), got {tuple(memory_tokens.shape)}."
            )

        batch_size = x.shape[0]
        patch_embeddings = self.patch_embed(self._patchify(x))
        mem_tokens = memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        sequence = torch.cat([mem_tokens, patch_embeddings], dim=1)
        sequence = sequence + self.pos_embed[:, :sequence.shape[1], :]

        encoded = self.transformer(sequence)
        logits = self.head(encoded[:, 0, :])

        if not return_encoded_memory:
            return logits
        encoded_memory = encoded[:, :self.n_mem, :]
        return logits, encoded_memory
