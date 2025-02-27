import torch
import math
import torch.utils.benchmark as benchmark
import cutlass_gemm

M = K = N = 16384
cuda = torch.device('cuda')
torch.set_float32_matmul_precision('high')

A = torch.normal(0,1,size=(M, K)).to(device=cuda).to(dtype=torch.float32)/math.sqrt(K)
B = torch.normal(0,1,size=(K, N)).to(device=cuda).to(dtype=torch.float32)/math.sqrt(K)

C1 = cutlass_gemm.mm(A,B)
C2 = torch.mm(A,B)

t0 = benchmark.Timer(
    stmt='cutlass_gemm.mm(A, B)',
    setup='import cutlass_gemm',
    globals={'A': A, 'B': B})

t1 = benchmark.Timer(
    stmt='torch.mm(A, B)',
    setup='import torch',
    globals={'A': A, 'B': B})

t2 = benchmark.Timer(
    stmt='torch.compile(torch.mm)(A, B)',
    setup='import torch',
    globals={'A': A, 'B': B})

print("### Warmup ###")
print("Benchmarking CUTLASS GEMM:")
print(t0.timeit(100))
print("\nBenchmarking PyTorch MM:")
print(t1.timeit(100))
print("\nBenchmarking Torch Compile:")
print(t2.timeit(100))
print("### End of Warmup ###")

print("### Benchmarking ###")
print("Benchmarking CUTLASS GEMM:")
print(t0.timeit(100))
print("\nBenchmarking PyTorch MM:")
print(t1.timeit(100))
print("\nBenchmarking Torch Compile:")
print(t2.timeit(100))
print("### End of Benchmarking ###")

print("\nResults comparison:")
print("max deviation: {:.10f}".format(torch.max(torch.abs(C2-C1))))