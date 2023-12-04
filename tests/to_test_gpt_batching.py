import torch
from examples.nanogpt import get_batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def test_get_batch_train():
#     # Mocking the train_data and block_size
#     train_data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     block_size = 3
#     batch_size = 2
    
#     # Call the get_batch function with 'train' split
#     x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size, device=device)

#     # Assert the shape and values of x and y
#     assert x.shape == (batch_size, block_size)
#     assert y.shape == (batch_size, block_size)
#     assert torch.allclose(x, torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device))
#     assert torch.allclose(y, torch.tensor([[2, 3, 4], [5, 6, 7]]).to(device))
#     assert x.device == device
#     assert y.device == device

# def test_get_batch_val():
#     # Mocking the val_data and block_size
#     val_data = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#     block_size = 3
#     batch_size = 2
#     device = torch.device('cpu')

#     # Call the get_batch function with 'val' split
#     x, y = get_batch(val_data, block_size=block_size, batch_size=batch_size, device=device)

#     # Assert the shape and values of x and y
#     assert x.shape == (batch_size, block_size)
#     assert y.shape == (batch_size, block_size)
#     assert torch.allclose(x, torch.tensor([[11, 12, 13, 14], [14, 15, 16, 17], [15, 16, 17, 18]]))
#     assert torch.allclose(y, torch.tensor([[12, 13, 14, 15], [15, 16, 17, 18], [16, 17, 18, 19]]))
#     assert x.device == device
#     assert y.device == device

# # Run the test functions
# test_get_batch_train()
# test_get_batch_val()