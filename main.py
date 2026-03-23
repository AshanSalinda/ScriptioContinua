import torch
import torch.nn.functional as F
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


def predict_multiple_spaces(text, model, char2idx, idx2tag, internal_beams=15, return_top=3):
    """
    Uses Beam Search to find the best paths, then deduplicates by text
    to ensure the user sees distinct segmentations.
    """
    padded_text = [PAD_CHAR] * HALF_WINDOW + list(text) + [PAD_CHAR] * HALF_WINDOW
    beams = [(0.0, "", [])]

    with torch.no_grad():
        for i in range(len(text)):
            window_chars = padded_text[i: i + WINDOW_SIZE]
            window_ids = [char2idx.get(c, char2idx[PAD_CHAR]) for c in window_chars]
            x = torch.tensor([window_ids], dtype=torch.long)

            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1).squeeze()

            # --- THE FIX ---
            # Find out how many tags the model actually has (e.g., 5 or 6)
            available_tags = log_probs.size(0)

            # We can only branch out by the number of tags that actually exist!
            branch_factor = min(internal_beams, available_tags)

            # Get the top K valid tags for this character
            topk_log_probs, topk_indices = torch.topk(log_probs, branch_factor)

            new_beams = []

            for score, segmented_text, tags in beams:
                # Loop through our valid tag branches (e.g., top 5)
                for j in range(branch_factor):
                    tag_prob = topk_log_probs[j].item()
                    tag_id = topk_indices[j].item()
                    tag_str = idx2tag[tag_id]

                    new_score = score + tag_prob

                    new_char = text[i]
                    if tag_str in ['E', 'S']:
                        new_char += " "

                    new_beams.append((new_score, segmented_text + new_char, tags + [tag_str]))

            # Sort all the newly branched realities by score
            new_beams.sort(key=lambda x: x[0], reverse=True)

            # Prune the tree: NOW we enforce the internal_beams limit (keep top 15 paths overall)
            beams = new_beams[:internal_beams]

    # --- Text-Based Deduplication ---
    unique_texts = {}

    for score, segmented_text, tags in beams:
        # Clean up any trailing spaces or accidental double spaces
        clean_text = " ".join(segmented_text.strip().split())

        # If we haven't seen this text variation, OR this mathematical path has a higher score
        if clean_text not in unique_texts or score > unique_texts[clean_text]["score"]:
            unique_texts[clean_text] = {
                "score": round(score, 4),
                "text": clean_text,
                "tags": tags
            }

    # Convert dictionary back to a list, sort it, and return only the exact number requested
    final_results = list(unique_texts.values())
    final_results.sort(key=lambda x: x["score"], reverse=True)

    return final_results[:return_top]


if __name__ == "__main__":
    model, char2idx, idx2tag = load_system()

    test_string = "peoplethinkthatgodisnowhere"

    print(f"\nInput (Scriptio Continua): {test_string}")

    results = predict_multiple_spaces(
        test_string,
        model,
        char2idx,
        idx2tag,
        internal_beams=15,  # Track 15 possibilities in memory
        return_top=3  # Only show the top 3 UNIQUE strings to the user
    )

    for i, result in enumerate(results):
        print(f"\nOption {i + 1} (Confidence Score: {result['score']})")
        print(f"Text: {result['text']}")
        print(f"Tags: {' '.join(result['tags'])}")