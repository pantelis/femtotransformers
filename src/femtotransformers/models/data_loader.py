import torch

# data loading
def get_batch(data, block_size: int, batch_size:int, device:str):
    # generate a small batch of data of inputs x and targets y
    
    # generate the batch size indices 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # pick up the x and the y - they y is offset by 1 as we predict the next token
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # store on the device (CPU/GPU)
    x, y = x.to(device), y.to(device)
    
    return x, y
