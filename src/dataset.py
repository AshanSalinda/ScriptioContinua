import json
import torch
from torch.utils.data import Dataset, DataLoader

# --- Configuration ---
# A window size of 5 means: 2 characters left + center character + 2 characters right
WINDOW_SIZE = 5
HALF_WINDOW = WINDOW_SIZE // 2
PAD_CHAR = "<PAD>"  # Used when the window goes past the edge of the text


class ScriptioDataset(Dataset):
    def __init__(self, jsonl_filepath):
        """Loads data, builds vocabularies, and generates sliding windows."""
        self.windows = []  # The input chunks (X)
        self.target_tags = []  # The tag for the center character (y)

        # 1. Load the processed data
        self.data = []
        with open(jsonl_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        # 2. Build the Vocabulary (Char -> ID) and Tag Map (Tag -> ID)
        self.chars = set([PAD_CHAR])
        self.tags = set()

        for item in self.data:
            self.chars.update(list(item["spaceless"]))
            self.tags.update(item["tags"])

        # Create fast lookup dictionaries
        self.char2idx = {char: idx for idx, char in enumerate(sorted(self.chars))}
        self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(self.tags))}

        self.vocab_size = len(self.char2idx)
        self.num_tags = len(self.tag2idx)

        # 3. Build the Sliding Windows
        self._build_windows()

    def _build_windows(self):
        """Chops the strings into fixed-size windows."""
        for item in self.data:
            text = item["spaceless"]
            tags = item["tags"]

            # Pad the edges of the text so we can create a window for the very first and last characters
            padded_text = [PAD_CHAR] * HALF_WINDOW + list(text) + [PAD_CHAR] * HALF_WINDOW

            for i in range(len(text)):
                # Extract a window of characters
                window_chars = padded_text[i: i + WINDOW_SIZE]

                # Convert characters to their integer IDs
                window_ids = [self.char2idx[c] for c in window_chars]
                target_id = self.tag2idx[tags[i]]

                self.windows.append(window_ids)
                self.target_tags.append(target_id)

    def __len__(self):
        """Tells PyTorch how many total windows we have."""
        return len(self.windows)

    def __getitem__(self, idx):
        """Fetches a single window and its target tag as PyTorch Tensors."""
        x = torch.tensor(self.windows[idx], dtype=torch.long)
        y = torch.tensor(self.target_tags[idx], dtype=torch.long)
        return x, y


# --- Execution Block (For Testing) ---
if __name__ == "__main__":
    PROCESSED_DATA_PATH = "../data/processed/training_data.jsonl"

    print("Loading Dataset...")
    dataset = ScriptioDataset(PROCESSED_DATA_PATH)

    print(f"Vocabulary Size: {dataset.vocab_size} unique characters")
    print(f"Number of Tags: {dataset.num_tags} unique tags {list(dataset.tag2idx.keys())}")
    print(f"Total Sliding Windows Generated: {len(dataset)}")

    # Wrap it in a DataLoader (simulates how the training loop will fetch data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Grab the first batch to inspect
    batch_x, batch_y = next(iter(dataloader))
    print("\n--- Sample Batch (Size 4) ---")
    print("Input Windows (X) Shape:", batch_x.shape)
    print("Input Windows (X) Data:\n", batch_x)
    print("Target Tags (y) Shape:", batch_y.shape)
    print("Target Tags (y) Data:", batch_y)