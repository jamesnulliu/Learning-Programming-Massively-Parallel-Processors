import torch
import pmpp

A = torch.tensor([1, 2, 3], dtype=torch.float32)
B = torch.tensor([4, 5, 6], dtype=torch.float32)

print(torch.ops.pmpp.vector_add(A, B))

A = A.cuda()
B = B.cuda()

print(torch.ops.pmpp.vector_add(A, B))

pic_in = torch.randint(0, 256, size=(600, 800, 3), dtype=torch.uint8)
pic_out_cpu = torch.ops.pmpp.cvt_rgb_to_gray(pic_in)
print(pic_out_cpu)

pic_in = pic_in.cuda()
pic_out_cuda = torch.ops.pmpp.cvt_rgb_to_gray(pic_in)
print(pic_out_cuda.cpu())

print(torch.ops.pmpp.matmul(torch.ones((32, 32)).cuda(), torch.ones((32, 32)).cuda()))

