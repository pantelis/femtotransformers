import pickle
import torch
import femtotransformers.models.nanogpt.nanogpt_tokenizer as nanogpt_tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

block_size = 256

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('examples/nanogpt/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# char tokenization
nanogpt_tokenizer = nanogpt_tokenizer.NanoGptTokenizer()
nanogpt_tokenizer.fit(chars)

with open('examples/nanogpt/nanogpt-model.pkl', 'rb') as f:
    m = pickle.load(f)
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(nanogpt_tokenizer.decode(m.generate(context, block_size=block_size, max_new_tokens=500)[0].tolist()))
with open('examples/nanogpt/generated-text.txt', 'w') as f:
    f.write(nanogpt_tokenizer.decode(m.generate(context,  block_size=block_size, max_new_tokens=1000)[0].tolist()))
    

