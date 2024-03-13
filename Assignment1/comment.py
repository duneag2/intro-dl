import torch
import torch.nn.functional as F

matrix1 = torch.tensor([[1., 2.], [3., 4.]])
matrix2 = torch.tensor([[5., 6.], [7., 8.]])

# L1-norm
l1_norm = torch.norm(matrix1 - matrix2, p=1)
print("L1-norm:", l1_norm.item())

# L2-norm
l2_norm = torch.norm(matrix1 - matrix2)
print("L2-norm:", l2_norm.item())

# Cosine similarity
dot_product = torch.sum(matrix1 * matrix2)
norm_product = torch.norm(matrix1) * torch.norm(matrix2)
cos_sim = dot_product / norm_product
print("Cosine similarity:", cos_sim.item())
