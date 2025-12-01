import torch
import torch.nn as nn


class ComplaintClassifier(nn.Module):
    """
    Simple feed-forward classifier on top of Sentence-BERT embeddings.
    """

    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_labels),
        )

    def forward(self, embeddings):
        return self.net(embeddings)
