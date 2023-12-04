import torch
import torch.nn as nn
from torch.nn import functional as F

from femtotransformers.models.nanogpt.nanogpt_tokenizer import NanoGptTokenizer
from femtotransformers.models.nanogpt.nanogpt_modeling import GPTLanguageModel


# hyperparameters
batch_size = 4 #64  # how many independent sequences will we process in parallel?
block_size = 8 #256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
num_heads = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('examples/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# char tokenization
nanogpt_tokenizer = NanoGptTokenizer()
nanogpt_tokenizer.fit(chars)
data = torch.tensor(nanogpt_tokenizer.encode(text), dtype=torch.long)

# Train and test splits
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# few notes on batch vs block size

# Block size is the context size (therefore you associated the time-dimension with it) that the model sees at each timestep
# During training we apply teacher forcing, so the model sees the true
# sequence of tokens as input. Each token of the block can be fed into the model one at a time to allow the tansformer 
# to learn to do predictions with tokens up to block size.  As such it is the maximum context length. 

# The batch size is obviously the number of block_size sequences that are fed to the transformer at a time for training, to improve the GPU 
# utilization.  

# data loading
def get_batch(split, block_size=block_size, batch_size=batch_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    
    # generate the batch size indices 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # pick up the x and the y - they y is offset by 1 as we predict the next token
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # store on the device (CPU/GPU)
    x, y = x.to(device), y.to(device)
    return x, y


# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out

# model = GPTLanguageModel(n_layer=n_layer, vocab_size=vocab_size, n_embd=n_embd,block_size=block_size, dropout=dropout, num_heads=num_heads)
# m = model.to(device)
# # print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(
#             f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(nanogpt_tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
