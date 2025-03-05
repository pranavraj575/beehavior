import torch
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda) # Check CUDA version used by PyTorch
print(torch.cuda.is_available()) # Should return True
