import torch
from src.model import LinguistBiLSTM_CRF

# --- Configuration ---
MODEL_PATH = "models/saved_weights.pt"
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


# from src.model import LinguistBiLSTM_CRF
#
# # --- Configuration ---
# MODEL_PATH = "models/saved_weights.pt"
# WINDOW_SIZE = 5
# HALF_WINDOW = WINDOW_SIZE // 2
# PAD_CHAR = "<PAD>"
#
#
# def load_system():
#     print("Loading AI Model...")
#     checkpoint = torch.load(MODEL_PATH)
#
#     char2idx = checkpoint['char2idx']
#     tag2idx = checkpoint['tag2idx']
#     idx2tag = {v: k for k, v in tag2idx.items()}
#
#     model = LinguistBiLSTM_CRF(
#         vocab_size=checkpoint['vocab_size'],
#         num_tags=checkpoint['num_tags']
#     )
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     return model, char2idx, idx2tag
#
#
# def predict_crf_beam_search(text, model, char2idx, idx2tag, internal_beams=15, return_top=3):
#     """
#     Marries Beam Search with the CRF Transition Matrix to provide
#     multiple grammatically flawless segmentation options.
#     """
#     padded_text = [PAD_CHAR] * HALF_WINDOW + list(text) + [PAD_CHAR] * HALF_WINDOW
#
#     # Extract the learned grammatical rules from the CRF layer!
#     transitions = model.crf.transitions.detach()
#     start_transitions = model.crf.start_transitions.detach()
#
#     # Beams format: (cumulative_score, segmented_text, tags_list, last_tag_id)
#     # We now need to track the 'last_tag_id' to calculate the CRF transition score
#     beams = [(0.0, "", [], None)]
#
#     with torch.no_grad():
#         for i in range(len(text)):
#             window_chars = padded_text[i: i + WINDOW_SIZE]
#             window_ids = [char2idx.get(c, char2idx[PAD_CHAR]) for c in window_chars]
#             x = torch.tensor([window_ids], dtype=torch.long)
#
#             # 1. Bypass the strict Viterbi decode and get the raw BiLSTM features
#             embedded = model.embedding(x)
#             lstm_out, _ = model.lstm(embedded)
#             emissions = model.fc(lstm_out).squeeze(0)  # Shape: (window_size, num_tags)
#
#             # 2. Isolate the center character's raw scores
#             center_idx = WINDOW_SIZE // 2
#             center_emissions = emissions[center_idx]
#
#             new_beams = []
#
#             # 3. Branch out using the CRF's logic
#             for score, segmented_text, tags, last_tag_id in beams:
#                 # We loop through EVERY possible tag for this letter
#                 for tag_id in range(model.crf.num_tags):
#                     tag_str = idx2tag[tag_id]
#
#                     # Score Part A: How much does the BiLSTM like this letter here? (Emission)
#                     emission_score = center_emissions[tag_id].item()
#
#                     # Score Part B: Is this a grammatically legal move? (CRF Transition)
#                     transition_score = 0.0
#                     if last_tag_id is not None:
#                         # e.g., Transition score from 'E' to 'I' will be a massive negative number
#                         transition_score = transitions[last_tag_id, tag_id].item()
#                     else:
#                         # Score for the very first letter of the string
#                         transition_score = start_transitions[tag_id].item()
#
#                     # Add them all up!
#                     new_score = score + emission_score + transition_score
#
#                     # Apply physical spacing logic
#                     new_char = text[i]
#                     if tag_str in ['E', 'S']:
#                         new_char += " "
#
#                     new_beams.append((new_score, segmented_text + new_char, tags + [tag_str], tag_id))
#
#             # Sort and Prune the beams
#             new_beams.sort(key=lambda x: x[0], reverse=True)
#             beams = new_beams[:internal_beams]
#
#     # --- Text-Based Deduplication ---
#     unique_texts = {}
#
#     for score, segmented_text, tags, _ in beams:
#         clean_text = " ".join(segmented_text.strip().split())
#         if clean_text not in unique_texts or score > unique_texts[clean_text]["score"]:
#             unique_texts[clean_text] = {
#                 "score": round(score, 4),
#                 "text": clean_text,
#                 "tags": tags
#             }
#
#     final_results = list(unique_texts.values())
#     final_results.sort(key=lambda x: x["score"], reverse=True)
#
#     return final_results[:return_top]
#
#
# if __name__ == "__main__":
#     model, char2idx, idx2tag = load_system()
#
#     test_string = "peoplethinkthatgodisnowhere"
#
#     print(f"\nInput (Scriptio Continua): {test_string}")
#
#     # Now we call our custom CRF Beam Search!
#     results = predict_crf_beam_search(
#         test_string,
#         model,
#         char2idx,
#         idx2tag,
#         internal_beams=50,  # Keeping this high for maximum exploration
#         return_top=4
#     )
#
#     for i, result in enumerate(results):
#         print(f"\nOption {i + 1} (CRF Path Score: {result['score']})")
#         print(f"Text: {result['text']}")
#         print(f"Tags: {' '.join(result['tags'])}")