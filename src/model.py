import torch
import torch.nn as nn
from torchcrf import CRF


class LinguistBiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim=32, hidden_dim=64):
        super(LinguistBiLSTM_CRF, self).__init__()

        # 1. The standard layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Maps LSTM features to the number of tags (Emissions)
        self.fc = nn.Linear(in_features=hidden_dim * 2, out_features=num_tags)

        # 2. THE NEW UPGRADE: The Conditional Random Field Layer
        # batch_first=True tells it our data looks like [batch_size, window_size, features]
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, x, tags=None):
        """
        If tags are provided, it's TRAINING mode (calculates Loss).
        If tags are None, it's INFERENCE mode (calculates the Viterbi path).
        """
        # Get the rich context features from the BiLSTM
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # Get the raw tag guesses (emissions) for the ENTIRE window
        emissions = self.fc(lstm_out)

        if tags is not None:
            # --- TRAINING MODE ---
            # The CRF calculates the Negative Log Likelihood loss for us!
            # We return the negative value because PyTorch optimizers want to minimize loss
            loss = -self.crf(emissions, tags, reduction='mean')
            return loss
        else:
            # --- INFERENCE MODE ---
            # The CRF uses the Viterbi algorithm to find the mathematically perfect sequence
            predicted_sequences = self.crf.decode(emissions)
            return predicted_sequences


# --- Execution Block (For Testing) ---
if __name__ == "__main__":
    # Mock parameters matching our dataset
    mock_vocab_size = 30  # e.g., 26 letters + PAD + etc.
    mock_num_tags = 5  # B, I, O, E, S
    window_size = 5
    batch_size = 4

    # Create the model instance
    model = LinguistBiLSTM_CRF(vocab_size=mock_vocab_size, num_tags=mock_num_tags)

    # Create a fake batch of data (4 sliding windows, each with 5 random character IDs)
    dummy_input = torch.randint(low=0, high=mock_vocab_size, size=(batch_size, window_size))

    print("--- Model Architecture ---")
    print(model)
    print("\n--- Testing Forward Pass (Inference Mode) ---")
    print(f"Input Shape: {dummy_input.shape} (Batch Size, Window Size)")

    # Pass data through the model (returns a list of lists)
    output = model(dummy_input)

    # We use len() instead of .shape because the CRF returns standard Python lists!
    print(f"\nOutput Type: {type(output)}")
    print(f"Output Dimensions: {len(output)} (Batch Size) x {len(output[0])} (Window Size)")
    print("Output Data (Decoded Viterbi Paths):")
    for i, path in enumerate(output):
        print(f" Window {i+1}: {path}")