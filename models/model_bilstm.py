"""
models/model_bilstm.py
-----------------------
Bidirectional LSTM (BiLSTM) model for sentiment classification.

Based on: Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term
Memory. Neural Computation, 9(8), 1735-1780.

Architecture:
    Embedding → BiLSTM (stacked) → Attention Pooling
    → Dropout → FC (output)

Author : Shivam Lahoty
Project : ICT 4442 - Deep Learning Mini Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for text sentiment classification.

    Processes token sequences in both forward and backward directions
    to capture long-range dependencies. An attention mechanism is applied
    over the hidden states to produce a context-aware sentence representation
    before the final classification layer.

    Args:
        vocab_size   : Total vocabulary size.
        embed_dim    : Embedding dimensionality. Default 128.
        hidden_dim   : Number of units per LSTM direction. Default 128.
        num_layers   : Number of stacked BiLSTM layers. Default 2.
        num_classes  : Number of output classes. Default 3.
        dropout_rate : Dropout applied between LSTM layers and before FC. Default 0.4.
        padding_idx  : Index of the <PAD> token. Default 0.
    """

    def __init__(
        self,
        vocab_size   : int,
        embed_dim    : int   = 128,
        hidden_dim   : int   = 128,
        num_layers   : int   = 2,
        num_classes  : int   = 3,
        dropout_rate : float = 0.4,
        padding_idx  : int   = 0,
    ):
        super(BiLSTM, self).__init__()

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        # ── Embedding layer ─────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = padding_idx,
        )

        # ── BiLSTM ──────────────────────────────────────────────────
        # bidirectional=True doubles the output hidden size to 2*hidden_dim
        self.bilstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout_rate if num_layers > 1 else 0.0,
        )

        # ── Attention layer ─────────────────────────────────────────
        # Projects the bidirectional hidden states to a scalar score per timestep
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # ── Dropout & output ────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialise FC and attention weights."""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def attention_pool(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Soft attention over BiLSTM hidden states.

        Computes a weighted sum of all hidden states, where weights are
        learned attention scores (softmax-normalised).

        Args:
            lstm_out : Tensor of shape (batch, seq_len, 2*hidden_dim).

        Returns:
            context  : Tensor of shape (batch, 2*hidden_dim).
        """
        # (batch, seq_len, 1)
        scores = self.attention(lstm_out)

        # Normalise across the sequence dimension
        weights = F.softmax(scores, dim=1)           # (batch, seq_len, 1)

        # Weighted sum → (batch, 2*hidden_dim)
        context = (lstm_out * weights).sum(dim=1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : LongTensor of shape (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # (batch, seq_len, 2*hidden_dim)
        lstm_out, _ = self.bilstm(embedded)

        # Attention pooling → (batch, 2*hidden_dim)
        context = self.attention_pool(lstm_out)

        # Dropout + classification
        out    = self.dropout(context)
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
    HIDDEN_DIM  = 128
    NUM_LAYERS  = 2
    NUM_CLASSES = 3
    BATCH_SIZE  = 32
    SEQ_LEN     = 200

    model = BiLSTM(
        vocab_size   = VOCAB_SIZE,
        embed_dim    = EMBED_DIM,
        hidden_dim   = HIDDEN_DIM,
        num_layers   = NUM_LAYERS,
        num_classes  = NUM_CLASSES,
        dropout_rate = 0.4,
    )

    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    output      = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}  (expected: [{BATCH_SIZE}, {NUM_CLASSES}])")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), "Shape mismatch!"
    print("\nSelf-test passed!")
