import torch
import torch.nn as nn
import torchvision.models as models


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=7, patch_size=1, in_channels=2048, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class HybridResNetViT(nn.Module):
    def __init__(self, num_classes=7, img_size=224, vit_embed_dim=256, vit_heads=8, vit_depth=4, dropout=0.1):
        super().__init__()
        # CNN Backbone (ResNet50)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # keep conv feature map
        self.cnn_features = 2048
        feature_map_size = img_size // 32  # 224 -> 7
        
        # ViT from scratch
        self.patch_embed = PatchEmbedding(img_size=feature_map_size, patch_size=1,
                                          in_channels=self.cnn_features, embed_dim=vit_embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerEncoder(vit_embed_dim, vit_heads, dropout=dropout) for _ in range(vit_depth)]
        )
        # Classifier
        self.fc = nn.Sequential(
            nn.LayerNorm(vit_embed_dim),
            nn.Dropout(dropout),
            nn.Linear(vit_embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)           
        x = self.patch_embed(x)   
        x = self.transformer(x)
        cls_token = x[:, 0]       
        out = self.fc(cls_token)
        return out
