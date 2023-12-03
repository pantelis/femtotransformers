import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()
files = ["/workspaces/artificial_intelligence/artificial_intelligence/aiml-common/lectures/nlp/transformers/transformers-from-scratch/nano_gpt/input.txt"]  # Replace with the path to your dataset file
tokenizer.train(files, trainer)
tokenizer.save("tokenizer.json")

from transformers import PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")

@pytest.fixture(scope="module")
def transformer_bpe_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    files = ["/workspaces/artificial_intelligence/artificial_intelligence/aiml-common/lectures/nlp/transformers/transformers-from-scratch/nano_gpt/input.txt"]  # Replace with the path to your dataset file
    tokenizer.train(files, trainer)
    tokenizer.save("/workspaces/artificial_intelligence/artificial_intelligence/aiml-common/lectures/nlp/transformers/transformers-from-scratch/nano_gpt/transformer_bpe_tokenizer.json")
    
    return tokenizer

def test_tokenization(transformer_bpe_tokenizer):
    # Test tokenization
    input_text = "Hello, how are you?"
    expected_tokens = ['He', 'llo', ',', 'how', 'are', 'you', '?']
    tokens = tokenizer.encode(input_text)

    assert tokens.tokens == expected_tokens