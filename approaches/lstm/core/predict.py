import torch
from approaches.lstm.core.model import LinguistBiLSTM_CRF

# --- Configuration ---
MODEL_PATH = "approaches/lstm/models/saved_weights.pt"
WINDOW_SIZE = 5
HALF_WINDOW = WINDOW_SIZE // 2
PAD_CHAR = "<PAD>"


def load_system():
    """Loads the trained BiLSTM-CRF model and vocabularies."""
    print("Loading AI Model...")
    checkpoint = torch.load(MODEL_PATH)

    char2idx = checkpoint['char2idx']
    tag2idx = checkpoint['tag2idx']
    idx2tag = {v: k for k, v in tag2idx.items()}  # Reverse lookup

    # Initialize the new CRF architecture
    model = LinguistBiLSTM_CRF(
        vocab_size=checkpoint['vocab_size'],
        num_tags=checkpoint['num_tags']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to inference mode

    return model, char2idx, idx2tag


def predict_viterbi_spaces(text, model, char2idx, idx2tag):
    """
    Uses the CRF's native Viterbi decoding to find the single
    mathematically perfect sequence of tags.
    """
    padded_text = [PAD_CHAR] * HALF_WINDOW + list(text) + [PAD_CHAR] * HALF_WINDOW

    predicted_tags = []

    with torch.no_grad():
        for i in range(len(text)):
            window_chars = padded_text[i: i + WINDOW_SIZE]
            window_ids = [char2idx.get(c, char2idx[PAD_CHAR]) for c in window_chars]
            x = torch.tensor([window_ids], dtype=torch.long)

            # The model now returns the decoded path directly!
            # Output format: [[tag_id_1, tag_id_2, ..., tag_id_window_size]]
            predicted_path = model(x)

            # Extract the prediction for the center character of our sliding window
            center_idx = WINDOW_SIZE // 2
            predicted_id = predicted_path[0][center_idx]

            predicted_tags.append(idx2tag[predicted_id])

    # Reconstruct the string based on BIOES tags
    segmented_text = ""
    for char, tag in zip(text, predicted_tags):
        segmented_text += char
        if tag in ['E', 'S']:
            segmented_text += " "

    return segmented_text.strip(), predicted_tags


if __name__ == "__main__":
    model, char2idx, idx2tag = load_system()

    # Our complex test string
    test_string = "peoplethinkthatgodisnowhere"

    print(f"\nInput (Scriptio Continua): {test_string}")

    segmented_text, tags = predict_viterbi_spaces(
        test_string,
        model,
        char2idx,
        idx2tag
    )

    print(f"\n--- Viterbi Optimal Path ---")
    print(f"Text: {segmented_text}")
    print(f"Tags: {' '.join(tags)}")
