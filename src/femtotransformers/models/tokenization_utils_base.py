from pydantic import BaseModel
from data_model import EncodeInput, EncodeOutput, DecodeInput, DecodeOutput


class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}

    def fit(self, text):
        # Create a vocabulary of unique characters
        unique_chars = list(set(text))
        unique_chars.sort()

        # Assign an index to each character
        for idx, char in enumerate(unique_chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx]= char

    def encode(self, text: EncodeInput) -> EncodeOutput:
        # Convert a string to a list of character indices
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices: DecodeInput) -> DecodeOutput:
        # Convert a list of character indices back to a string
        return ''.join([self.idx_to_char[idx] for idx in indices])

