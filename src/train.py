import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom dataset and model
from dataset import ScriptioDataset
from model import LinguistBiLSTM


def train_model():
    # --- 1. Configuration & Setup ---
    PROCESSED_DATA_PATH = "../data/processed/training_data.jsonl"
    MODEL_SAVE_PATH = "../models/saved_weights.pt"

    # Hyperparameters (The "settings" for our learning process)
    BATCH_SIZE = 16  # How many windows to look at before updating weights
    LEARNING_RATE = 0.001  # How aggressively to tweak the weights (too high = erratic, too low = slow)
    EPOCHS = 10  # How many times to loop through the ENTIRE dataset

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print("Loading Dataset...")
    dataset = ScriptioDataset(PROCESSED_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing Model...")
    model = LinguistBiLSTM(vocab_size=dataset.vocab_size, num_tags=dataset.num_tags)

    # --- 2. The Engine: Loss and Optimizer ---
    # CrossEntropyLoss is the standard way to calculate error when predicting categories (like BIOES tags)
    criterion = nn.CrossEntropyLoss()

    # Adam is our optimizer. It acts like a smart search algorithm, finding the best weights
    # to minimize the error calculated by the criterion.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. The Training Loop ---
    print(f"\nStarting Training for {EPOCHS} Epochs...")
    model.train()  # Set the model to training mode

    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_x, batch_y in dataloader:
            # Step A: Clear the old gradients (reset the adjustment calculations from the last batch)
            optimizer.zero_grad()

            # Step B: Forward Pass (Make a guess)
            predictions = model(batch_x)

            # Step C: Calculate Loss (How wrong were we?)
            loss = criterion(predictions, batch_y)

            # Step D: Backward Pass (Calculate the adjustments needed)
            loss.backward()

            # Step E: Optimize (Apply the adjustments to the model's weights)
            optimizer.step()

            # --- Tracking Metrics (Optional but helpful for visibility) ---
            total_loss += loss.item()

            # Figure out which tag got the highest probability score
            predicted_tags = torch.argmax(predictions, dim=1)
            correct_predictions += (predicted_tags == batch_y).sum().item()
            total_predictions += batch_y.size(0)

        # Print progress at the end of each epoch
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = (correct_predictions / total_predictions) * 100
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    # --- 4. Save the Model ---
    print("\nTraining Complete!")

    # We save both the weights AND the vocab/tag mappings so we can use them later in inference
    save_package = {
        'model_state_dict': model.state_dict(),
        'char2idx': dataset.char2idx,
        'tag2idx': dataset.tag2idx,
        'vocab_size': dataset.vocab_size,
        'num_tags': dataset.num_tags
    }

    torch.save(save_package, MODEL_SAVE_PATH)
    print(f"💾 Model and vocabularies saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()