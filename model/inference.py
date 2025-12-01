import json
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pathlib import Path

from model.classifier import ComplaintClassifier
from model.config import MODEL_PATH, LABELS_PATH


class ComplaintInferencePipeline:
    """
    Full inference pipeline:
    - loads Sentence-BERT encoder
    - loads trained PyTorch classifier
    - runs prediction on a single text sample
    """

    def __init__(self):
        # Load label list
        with open(LABELS_PATH, "r") as f:
            self.labels = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Sentence-BERT encoder
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder.to(self.device)

        # Load PyTorch classifier
        embedding_dim = self.encoder.get_sentence_embedding_dimension()
        num_labels = len(self.labels)

        self.model = ComplaintClassifier(
            embedding_dim=embedding_dim,
            num_labels=num_labels
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        """
        Runs a forward pass:
        1) SBERT encode → embedding
        2) PyTorch classifier → logits
        3) softmax → probabilities
        """

        # Step 1: Encode text
        embedding = self.encoder.encode(
            [text],  # <<< wrap in list to enforce batch size 1
            convert_to_tensor=True,
            device=self.device
        )

        # Step 2: Classifier forward pass
        with torch.no_grad():
            logits = self.model(embedding)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        # Step 3: Convert to label + probability dict
        label_idx = int(probs.argmax())
        label = self.labels[label_idx]

        probabilities = {
            self.labels[i]: float(probs[i]) for i in range(len(self.labels))
        }

        return label, probabilities
