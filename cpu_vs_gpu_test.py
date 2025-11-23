import torch, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on:", device)

# Matrix multiplication timing test
size = 4096
x = torch.randn(size, size, device=device)
y = torch.randn(size, size, device=device)

start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
print("Time:", time.time() - start, "seconds")
