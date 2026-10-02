"""
models/model_textcnn.py
-----------------------
TextCNN model for sentiment classification.

Based on: Kim, Y. (2014). Convolutional Neural Networks for Sentence
Classification. Proceedings of EMNLP.

Architecture:
    Embedding → Parallel Conv1D filters (multiple kernel sizes)
    → ReLU → Global Max Pooling → Concat → Dropout → FC (output)

Author : Priyanshu Sharma
Project : ICT 4442 - Deep Learning Mini Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN for text sentiment classification.

    Uses multiple parallel convolutional filters of different kernel
    sizes (n-grams) to capture local patterns in the text. Each filter
    bank is followed by global max pooling, and all pooled features are
    concatenated before the final classification layer.

    Args:
        vocab_size    : Total vocabulary size.
        embed_dim     : Embedding dimensionality. Default 128.
        num_filters   : Number of filters per kernel size. Default 128.
        kernel_sizes  : List of kernel sizes (n-gram windows). Default [2, 3, 4].
        num_classes   : Number of output classes. Default 3.
        dropout_rate  : Dropout probability before the output layer. Default 0.5.
        padding_idx   : Index of the <PAD> token. Default 0.
    """

    def __init__(
        self,
        vocab_size   : int,
        embed_dim    : int = 128,
        num_filters  : int = 128,
        kernel_sizes : list = None,
        num_classes  : int = 3,
        dropout_rate : float = 0.5,
        padding_idx  : int = 0,
    ):
        super(TextCNN, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]

        self.kernel_sizes = kernel_sizes

        # ── Embedding layer ─────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        # ── Parallel convolutional layers ────────────────────────────
        # One Conv1d per kernel size — each captures different n-gram windows
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels  = embed_dim,
                out_channels = num_filters,
                kernel_size  = k,
            )
            for k in kernel_sizes
        ])

        # ── Dropout ─────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout_rate)

        # ── Output layer ─────────────────────────────────────────────
        # Input size = num_filters * number of kernel sizes (after concat)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """Initialise weights using Kaiming uniform for conv layers."""
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

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

        # Conv1d expects (batch, channels, length)
        # so permute to (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)

        # Apply each conv filter, ReLU, then global max pooling
        pooled_outputs = []
        for conv in self.convs:
            # (batch, num_filters, seq_len - kernel_size + 1)
            conv_out = F.relu(conv(embedded))

            # Global max pooling → (batch, num_filters)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)

        # Concatenate all pooled feature maps → (batch, num_filters * len(kernels))
        cat = torch.cat(pooled_outputs, dim=1)

        # Dropout + classification
        out    = self.dropout(cat)
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
    VOCAB_SIZE   = 10000
    EMBED_DIM    = 128
    NUM_FILTERS  = 128
    KERNEL_SIZES = [2, 3, 4]
    NUM_CLASSES  = 3
    BATCH_SIZE   = 32
    SEQ_LEN      = 200

    model = TextCNN(
        vocab_size   = VOCAB_SIZE,
        embed_dim    = EMBED_DIM,
        num_filters  = NUM_FILTERS,
        kernel_sizes = KERNEL_SIZES,
        num_classes  = NUM_CLASSES,
        dropout_rate = 0.5,
    )

    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    output      = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}  (expected: [{BATCH_SIZE}, {NUM_CLASSES}])")
    assert output.shape == (BATCH_SIZE, NUM_CLASSES), "Shape mismatch!"
    print("\nSelf-test passed!")
