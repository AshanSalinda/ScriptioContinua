import torch
import torch.nn as nn


class LinguistBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim=32, hidden_dim=64):
        """
        Initializes the layers of the Neural Network.
        """
        super(LinguistBiLSTM, self).__init__()

        # 1. Embedding Layer: Converts integer IDs into dense floating-point vectors.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 2. BiLSTM Layer: Reads the character window both forwards and backwards.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # 3. Fully Connected Layer: Maps the LSTM features to our BIOES tags.
        # We multiply hidden_dim by 2 because it is Bidirectional (forward + backward states combined).
        self.fc = nn.Linear(in_features=hidden_dim * 2, out_features=num_tags)

    def forward(self, x):
        """
        Defines the forward pass: how data flows through the network.
        'x' is the batch of sliding windows from dataset.py.
        """
        # Step 1: Pass characters through the embedding layer
        # Input shape: (batch_size, window_size) --> Output shape: (batch_size, window_size, embedding_dim)
        embedded = self.embedding(x)

        # Step 2: Pass embeddings through the BiLSTM
        # Output shape: (batch_size, window_size, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)

        # Step 3: Extract the center character's context
        # Since our sliding window is designed to predict the tag of the CENTER character,
        # we isolate that character's output from the sequence.
        window_size = x.size(1)
        center_idx = window_size // 2  # For a window of 5, the center is index 2

        center_features = lstm_out[:, center_idx, :]  # Shape: (batch_size, hidden_dim * 2)

        # Step 4: Calculate final predictions (logits)
        # Output shape: (batch_size, num_tags)
        logits = self.fc(center_features)

        return logits


# --- Execution Block (For Testing) ---
if __name__ == "__main__":
    # Let's run a dummy test to ensure the math works without errors

    # Mock parameters matching our dataset
    mock_vocab_size = 30  # e.g., 26 letters + PAD + etc.
    mock_num_tags = 5  # B, I, O, E, S
    window_size = 5
    batch_size = 4

    # Create the model instance
    model = LinguistBiLSTM(vocab_size=mock_vocab_size, num_tags=mock_num_tags)

    # Create a fake batch of data (4 sliding windows, each with 5 random character IDs)
    dummy_input = torch.randint(low=0, high=mock_vocab_size, size=(batch_size, window_size))

    print("--- Model Architecture ---")
    print(model)
    print("\n--- Testing Forward Pass ---")
    print(f"Input Shape: {dummy_input.shape} (Batch Size, Window Size)")

    # Pass data through the model
    output = model(dummy_input)

    print(f"Output Shape: {output.shape} (Batch Size, Num Tags)")
    print("Output Data (Raw Probabilities):\n", output)