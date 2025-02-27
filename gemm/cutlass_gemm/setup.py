import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Our example needs CUTLASS. Luckily it is header-only library, so all we need to do is include
cutlass_dir = os.environ.get("CUTLASS_DIR", "")
if not os.path.isdir(cutlass_dir):
  raise Exception("Environment variable CUTLASS_DIR must point to the CUTLASS installation") 
_cutlass_include_dirs = ["tools/util/include","include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]

# Get PyTorch's lib directory for RPATH
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

# Set additional flags needed for compilation here
nvcc_flags=["-O3","-DNDEBUG","-std=c++17"]
ld_flags=[]

# Add RPATH handling
extra_link_args = ["-Wl,-rpath,{}".format(torch_lib_dir)]

setup(
    name='cutlass_gemm',
    version='0.1.0',
    install_requires=['torch>=2.0.0'],
    ext_modules=[
        CUDAExtension(
                name="cutlass_gemm",  
                sources=["cutlass_gemm.cu"],
                include_dirs=cutlass_include_dirs,
                extra_compile_args={'nvcc': nvcc_flags},
                extra_link_args=extra_link_args,
                libraries=ld_flags)
   ],
    cmdclass={
        'build_ext': BuildExtension
    })