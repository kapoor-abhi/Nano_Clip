import torch
import torch.nn as nn

from config import Config


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = Config.EMBED_DIM
        self.patch_size = Config.PATCH_SIZE
        self.num_patches = (Config.IMG_SIZE // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.NUM_LAYERS,
        )

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.encoder(x)

        x = x[:, 0]
        return self.projection(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_dim = Config.EMBED_DIM

        self.token_embed = nn.Embedding(
            Config.VOCAB_SIZE,
            self.embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, Config.MAX_LEN, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=Config.NUM_LAYERS,
        )

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)

        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len]

        x = self.encoder(
            x,
            src_key_padding_mask=(attention_mask == 0)
            if attention_mask is not None
            else None,
        )

        x = x[:, 0]
        return self.projection(x)


class MiniCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()

        self.logit_scale = nn.Parameter(
            torch.ones([]) * 2.6592
        )

    def forward(
        self,
        image: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        image_embeds = self.vision_encoder(image)
        text_embeds = self.text_encoder(text_input_ids, text_mask)

        image_embeds = image_embeds / image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_embeds = text_embeds / text_embeds.norm(
            dim=-1, keepdim=True
        )

        return image_embeds, text_embeds