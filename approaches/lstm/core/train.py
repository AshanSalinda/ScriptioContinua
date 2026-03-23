import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from dataset import ScriptioDataset
from model import LinguistBiLSTM_CRF


def train_model():
    # --- 1. Configuration & Setup ---
    PROCESSED_DATA_PATH = "../data/training_data.jsonl"
    MODEL_SAVE_PATH = "../models/saved_weights.pt"

    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 50  # Start with 50 for the prototype

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print("Loading Dataset...")
    dataset = ScriptioDataset(PROCESSED_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing BiLSTM-CRF Model...")
    model = LinguistBiLSTM_CRF(vocab_size=dataset.vocab_size, num_tags=dataset.num_tags)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Notice we REMOVED the CrossEntropyLoss. The CRF handles loss natively now!

    print(f"\nStarting Academic Training Loop for {EPOCHS} Epochs...\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # Lists to store true and predicted tags for our Scikit-Learn metrics
        all_true_tags = []
        all_pred_tags = []

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()

            # --- 1. Calculate Loss ---
            # Our dataset provides the target tag for the center character.
            # We need to expand it so the CRF has a target for the whole window.
            # (For this sliding window architecture, we just duplicate the center tag as a proxy target)
            window_size = batch_x.size(1)
            expanded_targets = batch_y.unsqueeze(1).expand(-1, window_size)

            # Pass both X and the Targets to calculate CRF Loss
            loss = model(batch_x, tags=expanded_targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- 2. Calculate Predictions for Metrics ---
            model.eval()  # Temporarily turn off training behaviors
            with torch.no_grad():
                # Ask the CRF to decode the best path
                predicted_paths = model(batch_x)

                # Extract the center character's prediction
                center_idx = window_size // 2
                predicted_center_tags = [path[center_idx] for path in predicted_paths]

                # Store them for metric calculation
                all_pred_tags.extend(predicted_center_tags)
                all_true_tags.extend(batch_y.tolist())
            model.train()  # Turn training back on

        # --- 3. Academic Metrics Calculation ---
        epoch_loss = total_loss / len(dataloader)

        # zero_division=0 prevents warnings if the model hasn't learned a specific tag yet
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_tags,
            all_pred_tags,
            average='weighted',
            zero_division=0
        )

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | Loss: {epoch_loss:.4f} | F1-Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    print("\nTraining Complete!")

    save_package = {
        'model_state_dict': model.state_dict(),
        'char2idx': dataset.char2idx,
        'tag2idx': dataset.tag2idx,
        'vocab_size': dataset.vocab_size,
        'num_tags': dataset.num_tags
    }
    torch.save(save_package, MODEL_SAVE_PATH)
    print(f"💾 BiLSTM-CRF saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()