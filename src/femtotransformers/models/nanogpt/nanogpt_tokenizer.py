from ..tokenization_utils_base import (CharTokenizer, DecodeInput,
                                              DecodeOutput, EncodeInput,
                                              EncodeOutput)

class NanoGptTokenizer(CharTokenizer):
    def __init__(self):
        super().__init__()
    