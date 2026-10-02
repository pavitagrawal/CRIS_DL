"""
models/model_mlp.py
-------------------
Multilayer Perceptron (MLP) model for sentiment classification.

Architecture:
    Embedding → Flatten → FC1 → BatchNorm → ReLU → Dropout
    → FC2 → BatchNorm → ReLU → Dropout → FC3 (output)

Author : Pavit Agrawal
Project : ICT 4442 - Deep Learning Mini Project
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer Perceptron for text sentiment classification.

    The model embeds each token, averages the embeddings across the
    sequence (mean pooling), then passes the result through three
    fully connected layers with BatchNorm, ReLU activations, and
    Dropout for regularization.

    Args:
        vocab_size   : Total number of tokens in the vocabulary.
        embed_dim    : Dimensionality of each token embedding. Default 128.
        hidden_dim   : Number of units in the two hidden FC layers. Default 256.
        num_classes  : Number of output classes (3: neg / neu / pos). Default 3.
        dropout_rate : Dropout probability applied after each hidden layer. Default 0.4.
        padding_idx  : Vocabulary index used for <PAD> token. Default 0.
    """

    def __init__(
        self,
        vocab_size   : int,
        embed_dim    : int = 128,
        hidden_dim   : int = 256,
        num_classes  : int = 3,
        dropout_rate : float = 0.4,
        padding_idx  : int = 0,
    ):
        super(MLP, self).__init__()

        # ── Embedding layer ─────────────────────────────────────────
        # padding_idx=0 ensures <PAD> tokens contribute zero gradient
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # ── Hidden layer 1 ──────────────────────────────────────────
        self.fc1      = nn.Linear(embed_dim, hidden_dim)
        self.bn1      = nn.BatchNorm1d(hidden_dim)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # ── Hidden layer 2 ──────────────────────────────────────────
        self.fc2      = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2      = nn.BatchNorm1d(hidden_dim // 2)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # ── Output layer ────────────────────────────────────────────
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Initialise FC layer weights using He (Kaiming) initialisation."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : LongTensor of shape (batch_size, seq_len) — tokenized input.

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # Mean pooling over the sequence dimension → (batch, embed_dim)
        # Ignores <PAD> positions implicitly because their embeddings are zero
        pooled = embedded.mean(dim=1)

        # Hidden layer 1
        out = self.dropout1(self.relu1(self.bn1(self.fc1(pooled))))

        # Hidden layer 2
        out = self.dropout2(self.relu2(self.bn2(self.fc2(out))))

        # Output logits (no softmax — CrossEntropyLoss handles that)
        logits = self.fc3(out)
        return logits


# ─────────────────────────────────────────────
# MODEL SUMMARY UTILITY
# ─────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    VOCAB_SIZE  = 10000
    EMBED_DIM   = 128
    HIDDEN_DIM  = 256
    NUM_CLASSES = 3
    BATCH_SIZE  = 32
    SEQ_LEN     = 200

    model = MLP(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=0.4,
    )

    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    # Dummy forward pass
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    output = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}  (expected: [{BATCH_SIZE}, {NUM_CLASSES}])")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), "Shape mismatch!"
    print("\nSelf-test passed!")
