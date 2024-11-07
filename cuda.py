import torch

print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())  # For Apple Silicon
