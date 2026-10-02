"""
models/model_transformer.py
----------------------------
Attention-based Transformer model for sentiment classification.

Based on: Vaswani, A., et al. (2017). Attention Is All You Need.
Advances in Neural Information Processing Systems (NeurIPS).

Architecture:
    Embedding + Positional Encoding
    -> Transformer Encoder (multi-head self-attention + FFN)
    -> CLS Token Pooling
    -> Dropout -> FC (output)

Author : Aryan Sorout
Project : ICT 4442 - Deep Learning Mini Project
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Injects positional information into token embeddings using
    fixed sinusoidal encodings (Vaswani et al., 2017).

    Args:
        embed_dim : Embedding dimensionality.
        max_len   : Maximum sequence length supported. Default 512.
        dropout   : Dropout applied after adding positional encoding. Default 0.1.
    """

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute sinusoidal positional encodings once
        pe       = torch.zeros(max_len, embed_dim)             # (max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)                           # (1, max_len, embed_dim)

        # Register as buffer so it's saved with the model but not trained
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (batch, seq_len, embed_dim).
        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SentimentTransformer(nn.Module):
    """
    Transformer Encoder model for text sentiment classification.

    A learnable [CLS] token is prepended to the input sequence.
    After passing through the Transformer encoder layers, the [CLS]
    token representation is used for final classification — similar
    to the approach used in BERT (Devlin et al., 2019).

    Args:
        vocab_size   : Total vocabulary size.
        embed_dim    : Embedding dimensionality. Default 128.
        num_heads    : Number of attention heads. embed_dim must be
                       divisible by num_heads. Default 4.
        num_layers   : Number of Transformer encoder layers. Default 2.
        ffn_dim      : Feedforward network hidden dimension. Default 256.
        num_classes  : Number of output classes. Default 3.
        dropout_rate : Dropout probability. Default 0.3.
        max_seq_len  : Maximum input sequence length. Default 201 (200 + CLS).
        padding_idx  : Index of the <PAD> token. Default 0.
    """

    def __init__(
        self,
        vocab_size   : int,
        embed_dim    : int   = 128,
        num_heads    : int   = 4,
        num_layers   : int   = 2,
        ffn_dim      : int   = 256,
        num_classes  : int   = 3,
        dropout_rate : float = 0.3,
        max_seq_len  : int   = 201,
        padding_idx  : int   = 0,
    ):
        super(SentimentTransformer, self).__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        # ── Token embedding ─────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = padding_idx,
        )

        # ── Learnable [CLS] token ────────────────────────────────────
        # Prepended to each sequence; its final hidden state is used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding ──────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            embed_dim = embed_dim,
            max_len   = max_seq_len,
            dropout   = dropout_rate,
        )

        # ── Transformer Encoder ──────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ffn_dim,
            dropout         = dropout_rate,
            activation      = "relu",
            batch_first     = True,   # (batch, seq, feature) convention
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = num_layers,
        )

        # ── Classification head ──────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc      = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialise embedding and FC layer weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : LongTensor of shape (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        batch_size = x.size(0)

        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Prepend [CLS] token to each sequence
        # cls_token: (1, 1, embed_dim) -> (batch, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # (batch, seq_len+1, embed_dim)
        embedded = torch.cat([cls_tokens, embedded], dim=1)

        # Add positional encoding
        embedded = self.pos_encoding(embedded)

        # Pass through Transformer encoder layers
        # (batch, seq_len+1, embed_dim)
        encoded = self.transformer_encoder(embedded)

        # Extract [CLS] token representation (position 0)
        # (batch, embed_dim)
        cls_output = encoded[:, 0, :]

        # Dropout + classification
        out    = self.dropout(cls_output)
        logits = self.fc(out)
        return logits


# ─────────────────────────────────────────────
# MODEL SUMMARY UTILITY
# ─────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB_SIZE  = 10000
    EMBED_DIM   = 128
    NUM_HEADS   = 4
    NUM_LAYERS  = 2
    FFN_DIM     = 256
    NUM_CLASSES = 3
    BATCH_SIZE  = 32
    SEQ_LEN     = 200

    model = SentimentTransformer(
        vocab_size   = VOCAB_SIZE,
        embed_dim    = EMBED_DIM,
        num_heads    = NUM_HEADS,
        num_layers   = NUM_LAYERS,
        ffn_dim      = FFN_DIM,
        num_classes  = NUM_CLASSES,
        dropout_rate = 0.3,
    )

    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    output      = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}  (expected: [{BATCH_SIZE}, {NUM_CLASSES}])")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), "Shape mismatch!"
    print("\nSelf-test passed!")
