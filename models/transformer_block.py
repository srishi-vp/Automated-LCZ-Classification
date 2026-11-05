
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
        num_tokens=64
    ):
        super().__init__()

        # Learnable spatial positional embeddings
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim) * 0.02
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        # Add spatial positional information
        x = x + self.positional_encoding

        # Multi-head self-attention
        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized
        )

        x = x + attention_output

        # Feed-forward network
        x = x + self.mlp(self.norm2(x))

        return x


if __name__ == "__main__":

    model = TransformerBlock(
        embed_dim=128,
        num_heads=4,
        num_tokens=64
    )

    sample = torch.randn(2, 64, 128)

    output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Positional encoding:", model.positional_encoding.shape)
