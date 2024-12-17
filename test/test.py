import torch
import pmpp

A = torch.tensor([1, 2, 3], dtype=torch.float32)
B = torch.tensor([4, 5, 6], dtype=torch.float32)

print(torch.ops.pmpp.vector_add(A, B))

A = A.cuda()
B = B.cuda()

print(torch.ops.pmpp.vector_add(A, B))
