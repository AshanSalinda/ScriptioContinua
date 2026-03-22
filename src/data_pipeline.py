import os
import json


def generate_bioes_tags(sentence):
    """
    Takes a spaced sentence and returns a spaceless string
    along with its corresponding BIOES character tags.
    """
    words = sentence.strip().split()

    continuous_string = ""
    tags = []

    for word in words:
        word_len = len(word)

        # Rule 1: Single character words get the 'S' tag
        if word_len == 1:
            continuous_string += word[0]
            tags.append('S')

        # Rule 2: Multi-character words get B, I, and E tags
        else:
            for i, char in enumerate(word):
                continuous_string += char
                if i == 0:
                    tags.append('B')  # Beginning
                elif i == word_len - 1:
                    tags.append('E')  # End
                else:
                    tags.append('I')  # Inside

    return continuous_string, tags


def process_raw_data(input_filepath, output_filepath):
    """
    Reads a file of raw sentences and outputs a JSONL file
    with the processed training pairs.
    """
    processed_data = []

    # 1. Read the raw sentences
    with open(input_filepath, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    # 2. Transform the data
    for sentence in sentences:
        if not sentence.strip():
            continue  # Skip empty lines

        spaceless_text, char_tags = generate_bioes_tags(sentence)

        # We store this as a dictionary to easily save it as JSON
        processed_data.append({
            "original": sentence.strip(),
            "spaceless": spaceless_text,
            "tags": char_tags
        })

    # 3. Ensure the output directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # 4. Save to processed folder (using JSON Lines format - standard for NLP)
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for item in processed_data:
            outfile.write(json.dumps(item) + '\n')

    print(f"✅ Successfully processed {len(processed_data)} sentences!")
    print(f"💾 Saved to: {output_filepath}")


# --- Execution Block ---
if __name__ == "__main__":
    # Define our paths based on the project structure
    RAW_DATA_PATH = "../data/raw/english_sentences.txt"
    PROCESSED_DATA_PATH = "../data/processed/training_data.jsonl"

    # For testing right now, let's just create a dummy raw file if it doesn't exist
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    if not os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, 'w') as f:
            f.write("the quick brown fox jumps over the lazy dog\n")
            f.write("this is a test of the scriptio continua system\n")
            f.write("machine learning is fascinating\n")
            f.write("i love a good challenge\n")

    print("Starting data pipeline...")
    process_raw_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
