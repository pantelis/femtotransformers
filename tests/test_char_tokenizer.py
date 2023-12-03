# test_encoding_decoding.py

# Import the functions you want to test
from numpy import char
from sympy import Idx
from nano_gpt.char_tokenizer import CharTokenizer

filename = '/workspaces/artificial_intelligence/artificial_intelligence/aiml-common/lectures/nlp/transformers/transformers-from-scratch/nano_gpt/input.txt'
text = open(filename, 'r', encoding='utf-8').read()

char_tokenizer = CharTokenizer.CharTokenizer()
char_tokenizer.fit(text)
print(char_tokenizer.char_to_idx)

# Define test functions using the "test_" prefix
def test_encoding():
    
    result = char_tokenizer.encode("Hello AI Class")
    print(result)
    assert result == [20, 43, 50, 50, 53, 1, 13, 21, 1, 15, 50, 39, 57, 57]

def test_decoding():
    
    result = char_tokenizer.decode([20, 43, 50, 50, 53, 1, 13, 21, 1, 15, 50, 39, 57, 57])
    assert result == "Hello AI Class"

def test_encoding_decoding_round_trip():
    
    original_string = "Testing encoding and decoding round trip"
    encoded = char_tokenizer.encode(original_string)
    decoded = char_tokenizer.decode(encoded)
    assert decoded == original_string
