import torch
from src.model import LinguistBiLSTM

# --- Configuration ---
MODEL_PATH = "models/saved_weights.pt"
WINDOW_SIZE = 5
HALF_WINDOW = WINDOW_SIZE // 2
PAD_CHAR = "<PAD>"


def load_system():
    """Loads the trained model weights and vocabularies."""
    print("Loading AI Model...")
    checkpoint = torch.load(MODEL_PATH)

    char2idx = checkpoint['char2idx']
    tag2idx = checkpoint['tag2idx']
    idx2tag = {v: k for k, v in tag2idx.items()}  # Reverse lookup

    model = LinguistBiLSTM(
        vocab_size=checkpoint['vocab_size'],
        num_tags=checkpoint['num_tags']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation (testing) mode

    return model, char2idx, idx2tag


def predict_spaces(text, model, char2idx, idx2tag):
    """Passes a spaceless string through the AI to predict spacing."""
    # 1. Pad the input text exactly like we did in dataset.py
    padded_text = [PAD_CHAR] * HALF_WINDOW + list(text) + [PAD_CHAR] * HALF_WINDOW

    predicted_tags = []

    # 2. Slide the window across the text
    with torch.no_grad():  # Turn off gradient tracking to save memory
        for i in range(len(text)):
            window_chars = padded_text[i: i + WINDOW_SIZE]

            # Convert characters to IDs (use a default '0' or PAD if character is unknown)
            window_ids = [char2idx.get(c, char2idx[PAD_CHAR]) for c in window_chars]

            # Convert to PyTorch Tensor and add a batch dimension of 1
            x = torch.tensor([window_ids], dtype=torch.long)

            # Get AI prediction
            logits = model(x)
            predicted_id = torch.argmax(logits, dim=1).item()

            predicted_tags.append(idx2tag[predicted_id])

    # 3. Reconstruct the string based on BIOES tags
    segmented_text = ""
    for char, tag in zip(text, predicted_tags):
        segmented_text += char
        # If the AI says this character is the End of a word, or a Single-letter word, add a space!
        if tag in ['E', 'S']:
            segmented_text += " "

    return segmented_text.strip(), predicted_tags


if __name__ == "__main__":
    model, char2idx, idx2tag = load_system()

    # Test strings from our training data (it should get these perfect)
    test_strings = [
        "thequickbrownfoxjumpsoverthelazydog",
        "thisisatestofthescriptiocontinuasystem",
        "machinelearningjumpsthetestsystemisagoodchallenge"
    ]

    print("\n--- Epigrascan Linguist Module Test ---")
    for text in test_strings:
        print(f"\nInput (Scriptio Continua): {text}")

        segmented, tags = predict_spaces(text, model, char2idx, idx2tag)

        print(f"Predicted Tags: {' '.join(tags)}")
        print(f"AI Segmented Output:     {segmented}")